#!/usr/bin/env python3
"""Stage 4 Dieharder (Full 114-Subtest `-a` + 27-Test Good Battery) and
PractRand Deep Stream (8 GiB Continuous + 2 GiB Multi-Key Sub-Orbit Boundaries)
on the promoted production `src/ra_core/c/ra_core_engine.c` (`Opsi_A3` + `eng_v2_E2`).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM_BIN = HERE / "stream_v2_final"
PRACTRAND_BIN = Path(os.environ.get("PRACTRAND_BIN", "RNG_test"))
LOG_DIR = HERE / "logs"
DATA_DIR = HERE / "data"
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

ALL_DIEHARDER_D = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    100, 101, 102,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
]

GOOD_27_SET = {
    0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 15, 16, 17,
    100, 101, 102,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
}

# Longest tests first for parallel scheduling
PRIORITY_ORDER = [
    201, 17, 204, 200, 202, 203, 102, 101, 100,
    15, 16, 10, 11, 12, 13, 5, 6, 7, 14, 0, 1, 2, 3, 4, 8, 9,
    205, 206, 207, 208, 209,
]

ROW_RE = re.compile(
    r"^\s*([a-zA-Z0-9_]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(PASSED|WEAK|FAILED)\s*$"
)


def run_dieharder_d(mode_args: list[str], d: int) -> str:
    gen = subprocess.Popen(
        [str(STREAM_BIN), *mode_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    cmd = ["dieharder", "-g", "200", "-d", str(d)]
    res = subprocess.run(
        cmd,
        stdin=gen.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert gen.stdout is not None
    gen.stdout.close()
    gen.terminate()
    try:
        gen.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        gen.kill()
    return res.stdout.decode(errors="replace")


def run_dieharder_battery(label: str, mode_args: list[str], workers: int = 8) -> dict:
    t0 = time.monotonic()
    outputs: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {
            pool.submit(run_dieharder_d, mode_args, d): d
            for d in PRIORITY_ORDER
        }
        for fut in concurrent.futures.as_completed(fut_map):
            d = fut_map[fut]
            outputs[d] = fut.result()

    raw_path = LOG_DIR / f"dieharder_{label}.log"
    all_pass = all_weak = all_fail = 0
    good_pass = good_weak = good_fail = 0
    good_fam_clean = good_fam_weak = good_fam_fail = 0
    weak_list: list[str] = []
    fail_list: list[str] = []

    with open(raw_path, "w") as f:
        for d in ALL_DIEHARDER_D:
            txt = outputs[d]
            f.write(f"=== -d {d} ===\n{txt}\n")
            fam_assessments = []
            for line in txt.splitlines():
                m = ROW_RE.match(line)
                if m:
                    tname, ntup, tsamples, psamples, pval, assessment = m.groups()
                    fam_assessments.append(assessment)
                    if assessment == "PASSED":
                        all_pass += 1
                        if d in GOOD_27_SET:
                            good_pass += 1
                    elif assessment == "WEAK":
                        all_weak += 1
                        weak_list.append(f"-d {d} ({tname}|ntup={ntup}): p={pval}")
                        if d in GOOD_27_SET:
                            good_weak += 1
                    elif assessment == "FAILED":
                        all_fail += 1
                        fail_list.append(f"-d {d} ({tname}|ntup={ntup}): p={pval}")
                        if d in GOOD_27_SET:
                            good_fail += 1
            if d in GOOD_27_SET:
                if "FAILED" in fam_assessments:
                    good_fam_fail += 1
                elif "WEAK" in fam_assessments:
                    good_fam_weak += 1
                else:
                    good_fam_clean += 1

    dt = time.monotonic() - t0
    res = {
        "label": label,
        "seconds": round(dt, 2),
        "all_114_subtests": {
            "total": all_pass + all_weak + all_fail,
            "passed": all_pass,
            "weak": all_weak,
            "failed": all_fail,
        },
        "good_27_battery": {
            "families_total": 27,
            "families_clean_pass": good_fam_clean,
            "families_with_weak": good_fam_weak,
            "families_failed": good_fam_fail,
            "subtests_total": good_pass + good_weak + good_fail,
            "subtests_passed": good_pass,
            "subtests_weak": good_weak,
            "subtests_failed": good_fail,
        },
        "weak_details": weak_list,
        "fail_details": fail_list,
        "log_file": raw_path.name,
    }
    print(
        f"[Dieharder {label}] 114-subtest: {all_pass} PASS / {all_weak} WEAK / {all_fail} FAIL | "
        f"27-Good: {good_pass} PASS / {good_weak} WEAK / {good_fail} FAIL ({dt:.1f}s)",
        flush=True,
    )
    return res


def run_practrand_regime(label: str, mode_args: list[str], tlmax: str) -> dict:
    t0 = time.monotonic()
    gen = subprocess.Popen(
        [str(STREAM_BIN), *mode_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    cmd = [
        str(PRACTRAND_BIN),
        "stdin32",
        "-tlmin", "16MB",
        "-tlmax", tlmax,
        "-multithreaded",
    ]
    res = subprocess.run(
        cmd,
        stdin=gen.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert gen.stdout is not None
    gen.stdout.close()
    gen.terminate()
    try:
        gen.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        gen.kill()

    txt = res.stdout.decode(errors="replace")
    log_path = LOG_DIR / f"practrand_{label}_{tlmax}.log"
    log_path.write_text(txt)

    checkpoints = []
    cur_len = None
    anomalies = []
    status = "CLEAN"
    for line in txt.splitlines():
        if "length=" in line:
            if cur_len is not None:
                checkpoints.append({"length": cur_len, "anomalies": list(anomalies)})
            m = re.search(r"length=\s*([0-9]+\s*[A-Za-z]+)", line)
            cur_len = m.group(1).replace(" ", "") if m else line.strip()
            anomalies = []
        elif "no anomalies" in line:
            pass
        elif any(k in line for k in ("unusual", "suspicious", "FAIL")):
            anomalies.append(line.strip())
            if "FAIL" in line:
                status = "FAIL"
            elif "SUSPICIOUS" in line.upper() and status != "FAIL":
                status = "PASS (minor flag)"
            elif "unusual" in line.lower() and status == "CLEAN":
                status = "PASS (unusual)"
    if cur_len is not None:
        checkpoints.append({"length": cur_len, "anomalies": list(anomalies)})

    dt = time.monotonic() - t0
    out = {
        "label": label,
        "tlmax": tlmax,
        "seconds": round(dt, 2),
        "status": status,
        "checkpoints": checkpoints,
        "log_file": log_path.name,
    }
    print(f"[PractRand {label} ({tlmax})] status={status} ({dt:.1f}s)", flush=True)
    return out


def main():
    print("=== Running Stage 4 Dieharder Full Battery (114 Subtests) ===")
    dh_cont = run_dieharder_battery("continuous_key42", ["continuous", "42", "0"], workers=8)
    dh_k1 = run_dieharder_battery("k1_weyl", ["k1_weyl"], workers=8)

    print("\n=== Running Stage 4 PractRand Deep Stream (8GB Continuous + 2GB Sub-Orbit Boundaries) ===")
    pr_results = []
    # Run the 2GB boundary checks and 8GB continuous check
    regimes = [
        ("continuous_key42", ["continuous", "42", "0"], "8GB"),
        ("k1_weyl",          ["k1_weyl"],               "2GB"),
        ("k16_seq_key",      ["k16_seq_key"],           "2GB"),
        ("k64_seq_key",      ["k64_seq_key"],           "2GB"),
        ("k255_seq_key",     ["k255_seq_key"],          "2GB"),
        ("k255_seq_addr",    ["k255_seq_addr"],         "2GB"),
    ]
    for label, margs, tlmax in regimes:
        pr_results.append(run_practrand_regime(label, margs, tlmax))

    summary = {
        "dieharder": {
            "continuous_key42": dh_cont,
            "k1_weyl": dh_k1,
        },
        "practrand": pr_results,
    }
    out_json = DATA_DIR / "dieharder_practrand_stage4.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
