#!/usr/bin/env python3
"""Rebuild, run, and audit the measured shootout regimes on one CPU core.

Run from any directory: python3 run_fairness.py. No network or paid service is used.
The separate Group 3 tree result is imported and explicitly marked exploratory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FLAGS = ["-O3", "-march=native", "-mavx2", "-fno-tree-vectorize", "-ffp-contract=off"]
PROGRAMS = {
    "micro": ("run_shootout_microbench.c", "run_shootout_microbench", "shootout_microbench.log", "shootout_microbench.json"),
    "photon": ("mc_workload1_photon_v2.c", "mc_workload1_photon_v2", "mc_workload1.log", "mc_workload1_v2.json"),
    "pi_bernoulli": ("mc_workload2_pi_ising_v2.c", "mc_workload2_pi_ising_v2", "mc_workload2.log", "mc_workload2_v2.json"),
    "gbm": ("mc_workload3_gbm_v2.c", "mc_workload3_gbm_v2", "mc_workload3.log", "mc_workload3_v2.json"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_dependencies(source: str) -> dict[str, str]:
    text = subprocess.check_output(["gcc", *FLAGS, "-MM", source], cwd=HERE, text=True)
    names = text.replace("\\\n", " ").split(":", 1)[1].split()
    return {name: sha256(HERE / name) for name in names}


def run(cmd: list[str], *, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(cmd, cwd=HERE, check=True)
    else:
        with log.open("w", encoding="utf-8") as out:
            subprocess.run(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, check=True)


def find_metric(data: dict, suite: str, regime: str, generator: str) -> dict:
    if suite == "A":
        return next(row[regime] for row in data["suite_a_vigna_shootout"] if row["name"] == generator)
    root = {"B": "suite_b_scalar_buffer", "C": "suite_c_avx2_x8", "D": "suite_d_stateful_engine"}[suite]
    rows = data[root]["engines" if suite == "D" else "generators"]
    return next(row["regimes"][regime] for row in rows if row["name"] == generator)


def audit_micro_samples() -> dict:
    dataset = json.loads((DATA / "shootout_microbench.json").read_text())
    with (DATA / "shootout_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["suite"], row["regime"], row["generator"]), []).append(row)

    summary = []
    for (suite, regime, generator), items in sorted(groups.items()):
        if len(items) != 7 or {int(x["round"]) for x in items} != set(range(1, 8)):
            raise ValueError(f"expected seven measured rounds: {suite}/{regime}/{generator}")
        raw = sorted((float(x["elapsed_s"]), float(x["cycles"]), float(x["output_bytes"])) for x in items)
        min_sec, min_cycles, min_bytes = raw[0]
        metric = find_metric(dataset, suite, regime, generator)
        if not abs((min_bytes / 1048576.0) / min_sec - metric["mibs"]) <= max(0.1, metric["mibs"] * 0.003):
            raise ValueError(f"MiB/s disagrees with raw sample: {suite}/{regime}/{generator}")
        if not abs(min_cycles / min_bytes - metric["cpb"]) <= max(0.002, metric["cpb"] * 0.003):
            raise ValueError(f"cpb disagrees with raw sample: {suite}/{regime}/{generator}")
        med_sec = statistics.median(x[0] for x in raw)
        summary.append({
            "suite": suite, "regime": regime, "generator": generator,
            "rounds": 7, "output_bytes_per_round": min_bytes,
            "min_s": min_sec, "median_s": med_sec,
            "min_mibs": metric["mibs"], "median_mibs": (min_bytes / 1048576.0) / med_sec,
            "short_sample_under_5ms": min_sec < 0.005,
        })
    if not summary:
        raise ValueError("no raw microbenchmark samples")
    (DATA / "shootout_sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"regime_rows": len(summary), "raw_samples": len(rows),
            "short_rows_under_5ms": sum(row["short_sample_under_5ms"] for row in summary)}


def audit_mc_outputs() -> dict:
    photon = json.loads((DATA / "mc_workload1_v2.json").read_text())
    pi = json.loads((DATA / "mc_workload2_v2.json").read_text())
    gbm = json.loads((DATA / "mc_workload3_v2.json").read_text())
    if pi.get("pi_conversion") != "top24_float_midpoint":
        raise ValueError("Pi scalar/AVX2 conversion contract is missing")
    sample_rows = 0

    def check_samples(row: dict, sample_key: str, min_key: str, med_key: str, label: str) -> None:
        nonlocal sample_rows
        samples = row.get(sample_key)
        if not isinstance(samples, list) or len(samples) != 5 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError(f"Expected five positive raw MC samples: {label}")
        # Summary timings are printed to five decimal places in the C writers.
        if abs(min(samples) - row[min_key]) > 0.0000051 or abs(statistics.median(samples) - row[med_key]) > 0.0000051:
            raise ValueError(f"MC timing summary disagrees with raw samples: {label}")
        sample_rows += 1

    for rejection in photon["photon_transport"]:
        for row in rejection["generators"]:
            check_samples(row, "samples_s", "wall_min", "wall_med", f"photon/{rejection['p_reject']}/{row['name']}")
    for row in photon["rare_event_trigger"]:
        check_samples(row, "samples_s", "wall_min", "wall_med", f"rare/{row['name']}")

    pi_rows = 0
    for key, rows in pi.items():
        if not key.endswith(("k255", "k63", "k31", "k15")) or "consumer" not in key:
            continue
        for row in rows:
            check_samples(row, "samples_s", "wall_min", "wall_med", f"{key}/{row['name']}")
            k = row["k"]
            if row["generated_words"] != 1_000_000 * k or row["consumed_words"] != 1_000_000 * (k // 2) * 2:
                raise ValueError(f"Pi word accounting failed: {key}/{row['name']}")
            if row["err"] > 4 * row["se"]:
                raise ValueError(f"Pi estimate outside four standard errors: {key}/{row['name']}")
            pi_rows += 1
    for key, rows in pi.items():
        if not key.startswith("bernoulli_flip_"):
            continue
        for row in rows:
            check_samples(row, "samples_s", "wall_min", "wall_med", f"{key}/{row['name']}")
    if pi_rows == 0:
        raise ValueError("No Pi rows found")
    if len(photon["photon_transport"]) != 3 or not gbm["asian_option_gbm"]:
        raise ValueError("Missing photon or GBM rows")
    for row in gbm["asian_option_gbm"]:
        check_samples(row, "gen_samples_s", "gen_min", "gen_med", f"gbm/gen/{row['name']}")
        check_samples(row, "e2e_samples_s", "e2e_min", "e2e_med", f"gbm/e2e/{row['name']}")
        if row["gen_min"] <= 0 or row["e2e_min"] < row["gen_min"]:
            raise ValueError(f"Invalid GBM timing: {row['name']}")
    return {"photon_rejection_regimes": 3, "pi_rows_checked": pi_rows,
            "timing_rows_checked": sample_rows, "raw_timing_samples": sample_rows * 5,
            "gbm_rows_checked": len(gbm["asian_option_gbm"]),
            "tree_cross_family_speedup": "withheld_different_work"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=PROGRAMS, default=list(PROGRAMS))
    parser.add_argument("--core", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    core = args.core if args.core is not None else allowed[0]
    if core not in allowed:
        raise SystemExit(f"CPU {core} is outside affinity mask {allowed}")

    previous_path = DATA / "fairness_manifest.json"
    if args.audit_only:
        if not previous_path.exists():
            raise SystemExit("No fairness manifest to audit")
        previous = json.loads(previous_path.read_text())
        if previous.get("core") != core or previous.get("compiler") != subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]:
            raise ValueError("Current core/compiler differs from recorded run")
        for name, (source, binary, log_name, json_name) in PROGRAMS.items():
            record = previous.get("programs", {}).get(name)
            if not record:
                raise ValueError(f"Missing recorded program: {name}")
            paths = (HERE / source, HERE / binary, HERE / log_name, DATA / json_name)
            for path, field in zip(paths, ("source_sha256", "binary_sha256", "log_sha256", "json_sha256")):
                if not path.exists() or record.get(field) != sha256(path):
                    raise ValueError(f"Changed or missing {field}: {name}")
            if record.get("source_dependencies_sha256") != source_dependencies(source):
                raise ValueError(f"Changed source dependency: {name}")
        print(json.dumps({"micro": audit_micro_samples(), "mc": audit_mc_outputs()}, indent=2), flush=True)
        return

    manifest = {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "policy": "scalar auto-vectorization off; explicit AVX2 x8 retained; FP contraction off",
        "core": core, "allowed_cores": allowed,
        "platform": platform.platform(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
        "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "common_header_sha256": sha256(HERE / "baselines_shootout.h"),
        "programs": {},
        "tree_data_status": "imported exploratory; cross-family timings use different node/draw counts",
    }
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if (previous.get("common_header_sha256") == manifest["common_header_sha256"] and
                previous.get("compiler") == manifest["compiler"] and
                previous.get("core") == core):
            for name, record in previous.get("programs", {}).items():
                if name in args.only or name not in PROGRAMS:
                    continue
                source, binary, log_name, json_name = PROGRAMS[name]
                paths = (HERE / source, HERE / binary, HERE / log_name, DATA / json_name)
                if not all(path.exists() for path in paths):
                    continue
                if (record.get("source_sha256") == sha256(paths[0]) and
                    record.get("binary_sha256") == sha256(paths[1]) and
                    record.get("log_sha256") == sha256(paths[2]) and
                    record.get("json_sha256") == sha256(paths[3]) and
                    record.get("compile_command") == ["gcc", *FLAGS, source, "-lm", "-o", binary] and
                    record.get("source_dependencies_sha256") == source_dependencies(source)):
                    manifest["programs"][name] = record
    if not args.audit_only:
        for name in args.only:
            source, binary, log_name, json_name = PROGRAMS[name]
            compile_cmd = ["gcc", *FLAGS, source, "-lm", "-o", binary]
            run_cmd = ["taskset", "-c", str(core), f"./{binary}"]
            print(f"[{name}] compile", flush=True)
            run(compile_cmd)
            print(f"[{name}] run on CPU {core}", flush=True)
            run(run_cmd, log=HERE / log_name)
            json.loads((DATA / json_name).read_text())
            manifest["programs"][name] = {
                "compile_command": compile_cmd, "run_command": run_cmd,
                "source_sha256": sha256(HERE / source),
                "source_dependencies_sha256": source_dependencies(source),
                "binary_sha256": sha256(HERE / binary),
                "log_sha256": sha256(HERE / log_name),
                "json_sha256": sha256(DATA / json_name),
            }
    if (DATA / "shootout_samples.csv").exists():
        manifest["micro_sample_audit"] = audit_micro_samples()
    if all((DATA / name).exists() for name in ("mc_workload1_v2.json", "mc_workload2_v2.json", "mc_workload3_v2.json")):
        manifest["mc_audit"] = audit_mc_outputs()
    manifest["utc_finished"] = datetime.now(timezone.utc).isoformat()
    (DATA / "fairness_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.audit_only and set(manifest["programs"]) == set(PROGRAMS):
        print("[render] regenerate plots and dashboard", flush=True)
        run(["python3", "build_dashboard_and_plots.py"])
        print("[render] regenerate RESULTS.md", flush=True)
        run(["python3", "render_fairness_results.py"])
    print(json.dumps(manifest.get("micro_sample_audit", {})), flush=True)


if __name__ == "__main__":
    main()
