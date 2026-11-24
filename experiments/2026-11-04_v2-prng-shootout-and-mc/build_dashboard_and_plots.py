#!/usr/bin/env python3
"""
build_dashboard_and_plots.py
Consolidates all JSON benchmark outputs (`shootout_microbench.json`, `mc_workload1_v2.json`,
`mc_workload2_v2.json`, `mc_workload3_v2.json`, and `group3_fork_determinism.json`),
writes `data/mc_workloads_v2.json` and `data/shootout_feature_matrix.json`,
generates 4 publication-grade figures (`.png` + `.pdf`), and builds the self-contained
interactive HTML dashboard `shootout_dashboard.html`.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
GROUP3_JSON = BASE_DIR.parent / "2026-11-04_v2-comprehensive-stat-and-harness-battery" / "data" / "group3_fork_determinism.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_matrix():
    return [
        {
            "generator": "ra_core v2 (v2_W2_D2_xag_opt)",
            "family": "Bipartite Perm(L0) x Weyl M[] (Lag-2 + 2-Tap Ramp-Up)",
            "state_bytes": "128–2,048 B (0 B stack for K<=8; 32 B OTF ring for K<=255)",
            "period": ">= 2^64 streams x |Perm(N) x Z_2^32^N| (Toy exact active image: 60.0% / 38.85%)",
            "random_access_seek": "O(1) 2D (key, addr_hi) + O(1) K=1..8 closed-form register fast-path",
            "simd_parallelism": "Native 8-Way AVX2 u32x8 (MUL-free permutation cycle)",
            "practrand_full": "PASS >= 16 GiB (10/10 lanes, 9/10 CLEAN) & 1 TB family",
            "practrand_low_bitrev": "PASS 4 GiB CLEAN (4E1 Low-Byte d_t & 4E2 Bit-Reversed)",
            "dieharder_bigcrush": "PASS (27/27 Dieharder 0 FAILED; 99.5–100% NIST STS 188/188 p_T)",
            "sac_bic": "PASS (SAC min 0.4991; BIC Lag-1/Lag-2 max |r| <= 0.0074)",
            "safe_k_floor": "K_min = 1 across all N in {256, 128, 64, 32, 16}",
        },
        {
            "generator": "ra_core v1 (W0_D0 XAG Promoted)",
            "family": "Bipartite Perm(L0) x Weyl M[] (Lag-1 + 8-Tap Circular + Batch Reseed)",
            "state_bytes": "256–2,048 B (32 B OTF ring for K<=255)",
            "period": ">= 2^64 streams (Toy exact active image: 6.25% / 0.39% due to linear ra_hash)",
            "random_access_seek": "O(1) 2D (key, addr_hi) + O(1) K=1 (8-tap evaluation)",
            "simd_parallelism": "Native 8-Way AVX2 u32x8 (MUL-free permutation cycle)",
            "practrand_full": "PASS >= 16 GiB (N>=32) / 1 TB",
            "practrand_low_bitrev": "PASS (N=256)",
            "dieharder_bigcrush": "PASS (0 FAILED Dieharder; 99.4% NIST STS)",
            "sac_bic": "PASS at N>=32 (Defective at N=16 K=1..7)",
            "safe_k_floor": "K_min = 1 (N>=32); FAILS K=1..7 at N=16",
        },
        {
            "generator": "Philox4x32-10",
            "family": "Counter-Based Bijective Feistel + 32x32->64 Multiply (Random123)",
            "state_bytes": "24 B (16 B counter + 8 B key)",
            "period": "2^128 per 64-bit key (4 words/block)",
            "random_access_seek": "O(1) direct counter indexing (10-round block per 4 words)",
            "simd_parallelism": "8-Way AVX2 requires even/odd 64-bit multiply splitting",
            "practrand_full": "PASS > 32 TB",
            "practrand_low_bitrev": "PASS > 32 TB",
            "dieharder_bigcrush": "PASS (BigCrush + Dieharder)",
            "sac_bic": "PASS (Full 10-round avalanche; correlated if ctr=0 uniform with key-only)",
            "safe_k_floor": "K_min = 1 (wastes 3/4 of 10-round block at K=1)",
        },
        {
            "generator": "ChaCha8 / ChaCha12",
            "family": "Counter-Based ARX Stream Cipher (RFC 7539 / Bernstein)",
            "state_bytes": "64 B (512-bit block: 32 B key + 12 B nonce + 8 B ctr + 16 B const)",
            "period": "2^70 bytes per key/nonce",
            "random_access_seek": "O(1) block counter (computes 16-word / 64 B block per seek)",
            "simd_parallelism": "Excellent 8-Way AVX2 at bulk K>=16; 16x wasted work at K=1",
            "practrand_full": "PASS > 32 TB (Cryptographic margin at 12/20 rounds)",
            "practrand_low_bitrev": "PASS > 32 TB",
            "dieharder_bigcrush": "PASS (BigCrush + Dieharder + NIST STS)",
            "sac_bic": "PASS",
            "safe_k_floor": "K_min = 1 (wastes 15/16 of 64-byte block at K=1)",
        },
        {
            "generator": "xoshiro256**",
            "family": "GF(2)-Linear xoshiro + ** Non-Linear Scrambler (Blackman & Vigna)",
            "state_bytes": "32 B (256-bit state, requires 4x SplitMix64 init per seed)",
            "period": "2^256 - 1",
            "random_access_seek": "NO O(1) arbitrary seek (polynomial jump takes ~O(256) steps)",
            "simd_parallelism": "4x64-bit per YMM register (2x YMM for 8 streams)",
            "practrand_full": "PASS > 32 TB",
            "practrand_low_bitrev": "PASS (unlike + scrambler which fails MatrixRank/LinearComp)",
            "dieharder_bigcrush": "PASS (BigCrush + Dieharder)",
            "sac_bic": "PASS (after SplitMix64 seeding; raw state has GF(2) structure)",
            "safe_k_floor": "K_min = 1 (pays 4x SplitMix64 init overhead)",
        },
        {
            "generator": "xoroshiro128++",
            "family": "GF(2)-Linear xoroshiro + ++ Rotate-Add Scrambler (Blackman & Vigna)",
            "state_bytes": "16 B (128-bit state, requires 2x SplitMix64 init per seed)",
            "period": "2^128 - 1",
            "random_access_seek": "NO O(1) arbitrary seek (polynomial jump only)",
            "simd_parallelism": "4x64-bit per YMM register (2x YMM for 8 streams)",
            "practrand_full": "PASS > 32 TB",
            "practrand_low_bitrev": "PASS",
            "dieharder_bigcrush": "PASS (BigCrush + Dieharder)",
            "sac_bic": "PASS (after SplitMix64 seeding)",
            "safe_k_floor": "K_min = 1 (pays 2x SplitMix64 init overhead)",
        },
        {
            "generator": "PCG32 (pcg_oneseq_64_xsh_rr_32)",
            "family": "64-bit LCG + XSH-RR Output Permutation (O'Neill)",
            "state_bytes": "16 B (8 B state + 8 B stream increment)",
            "period": "2^64",
            "random_access_seek": "O(log N) jump-ahead (~64 64-bit multiplies); no O(1) tree indexing",
            "simd_parallelism": "Slow in AVX2 (lacks native vpmullq 64-bit vector multiply)",
            "practrand_full": "PASS 32 TB (single stream); known pair correlations on close increments",
            "practrand_low_bitrev": "PASS",
            "dieharder_bigcrush": "PASS (BigCrush + Dieharder)",
            "sac_bic": "PASS",
            "safe_k_floor": "K_min = 1 (2 LCG warm-up steps on srandom)",
        },
        {
            "generator": "SplitMix64",
            "family": "64-bit Weyl Counter + Stafford Mix13 Bijective Finalizer (Vigna)",
            "state_bytes": "8 B (64-bit state)",
            "period": "2^64",
            "random_access_seek": "O(1) additive step (1D only; lacks native 2D key/addr domain separation)",
            "simd_parallelism": "Slow in AVX2 (requires 2x emulated 64-bit vector multiplies per word)",
            "practrand_full": "PASS > 32 TB",
            "practrand_low_bitrev": "PASS",
            "dieharder_bigcrush": "PASS (BigCrush)",
            "sac_bic": "PASS (1D bijective hash)",
            "safe_k_floor": "K_min = 1",
        },
        {
            "generator": "WyRand",
            "family": "64-bit Weyl Counter + 64x64->128 Folded Multiply (Wang Yi)",
            "state_bytes": "8 B (64-bit state)",
            "period": "2^64",
            "random_access_seek": "O(1) additive step (1D only)",
            "simd_parallelism": "No native 64x64->128 SIMD instruction in AVX2/AVX-512",
            "practrand_full": "PASS > 32 TB",
            "practrand_low_bitrev": "PASS",
            "dieharder_bigcrush": "PASS (BigCrush)",
            "sac_bic": "PASS",
            "safe_k_floor": "K_min = 1",
        },
        {
            "generator": "MT19937 (Mersenne Twister)",
            "family": "Twisted GFSR over GF(2) (Matsumoto & Nishimura)",
            "state_bytes": "2,500 B (624 x 32-bit state + index)",
            "period": "2^19937 - 1",
            "random_access_seek": "NONE (624-word sequential LCG init per seed + 624-word twist)",
            "simd_parallelism": "Poor per-seed SIMD (SFMT requires large contiguous blocks)",
            "practrand_full": "FAILS PractRand BRank(18):256 / LinearComp at 32–256 GiB",
            "practrand_low_bitrev": "FAILS LinearComp / MatrixRank",
            "dieharder_bigcrush": "FAILS 2 BigCrush subtests (LinearComp)",
            "sac_bic": "FAILS GF(2) linear rank & high seeding correlation",
            "safe_k_floor": "Catastrophic at small K (2.6 MiB/s at K=1 due to 624-word init)",
        },
    ]


def generate_plots(micro, mc1, mc2, mc3, g3):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 300,
    })

    # -------------------------------------------------------------------------
    # Figure 1: Throughput & Cycles/Byte across K (Scalar vs 8-Way AVX2 SIMD)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Panel A: Scalar Direct-Buffer Throughput (MiB/s) across K
    ax = axes[0]
    k_labels_b = micro["suite_b_scalar_buffer"]["k_labels"]
    x_idx = np.arange(len(k_labels_b))
    gens_b = {g["name"]: g["regimes"] for g in micro["suite_b_scalar_buffer"]["generators"]}

    styles_b = [
        ("ra_core_v2 (N=256)", "#d95f02", "-o", 2.4, "ra_core v2 (N=256, W2_D2)"),
        ("ra_core_v1 (N=256)", "#7570b3", "--s", 1.8, "ra_core v1 (N=256, W0_D0)"),
        ("Philox4x32-10",      "#1b9e77", "-^", 2.0, "Philox4x32-10 (CBRNG)"),
        ("ChaCha8",            "#e7298a", "-v", 1.8, "ChaCha8 (ARX CBRNG)"),
        ("xoshiro256**",       "#66a61e", "-D", 1.7, "xoshiro256** (Linear)"),
        ("PCG32",              "#e6ab02", "-P", 1.7, "PCG32 (LCG)"),
        ("MT19937",            "#666666", ":x", 1.5, "MT19937 (Legacy)"),
    ]
    for gname, color, fmt, lw, label in styles_b:
        reg = gens_b[gname]
        y = [reg[kl]["mibs"] if reg[kl] is not None else np.nan for kl in k_labels_b]
        ax.plot(x_idx, y, fmt, color=color, linewidth=lw, markersize=5, label=label)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(k_labels_b, rotation=38, ha="right")
    ax.set_ylabel("Throughput (MiB/s, Materialized Buffer)")
    ax.set_title("(a) Scalar Direct-Buffer Throughput Across Stream Length K")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.92)

    # Panel B: 8-Way AVX2 SIMD Throughput (MiB/s) across K
    ax = axes[1]
    k_labels_c = micro["suite_c_avx2_x8"]["k_labels"]
    x_idx_c = np.arange(len(k_labels_c))
    gens_c = {g["name"]: g["regimes"] for g in micro["suite_c_avx2_x8"]["generators"]}

    styles_c = [
        ("ra_core_v2_x8 (N=256)", "#d95f02", "-o", 2.5, "ra_core v2 x8 (N=256)"),
        ("ra_core_v2_x8 (N=16)",  "#ff7f00", "--*", 2.0, "ra_core v2 x8 (N=16, 128B)"),
        ("ra_core_v1_x8 (N=256)", "#7570b3", "--s", 1.8, "ra_core v1 x8 (N=256)"),
        ("Philox4x32-10_x8",      "#1b9e77", "-^", 2.0, "Philox4x32-10 x8"),
        ("ChaCha8_x8",            "#e7298a", "-v", 1.8, "ChaCha8 x8"),
        ("xoshiro256**_x8",       "#66a61e", "-D", 1.7, "xoshiro256** x8"),
        ("PCG32_x8",              "#e6ab02", "-P", 1.7, "PCG32 x8 (emul mul64)"),
        ("SplitMix64_x8",         "#a6761d", ":d", 1.6, "SplitMix64 x8 (emul mul64)"),
    ]
    for gname, color, fmt, lw, label in styles_c:
        reg = gens_c[gname]
        y = [reg[kl]["mibs"] if reg[kl] is not None else np.nan for kl in k_labels_c]
        ax.plot(x_idx_c, y, fmt, color=color, linewidth=lw, markersize=5, label=label)

    ax.set_xticks(x_idx_c)
    ax.set_xticklabels(k_labels_c, rotation=25, ha="right")
    ax.set_ylabel("Throughput (MiB/s, 8-Way AVX2 SIMD)")
    ax.set_title("(b) Apples-to-Apples 8-Way AVX2 SIMD Throughput Across K")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shootout_fig1_throughput_across_K.png")
    fig.savefig(PLOTS_DIR / "shootout_fig1_throughput_across_K.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: shootout-inspired inlined register vs materialized buffer
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    suite_a = micro["suite_a_vigna_shootout"]
    rows_a = [r for r in suite_a if r["name"] != "MT19937"]
    height = 0.36

    # Panel A: Continuous Stream Cycles per Byte (cpb) - Sorted Fastest (Lowest cpb) First
    ax = axes[0]
    rows_a_stream = sorted(rows_a, key=lambda r: r["buffer_stream"]["cpb"])
    names_a_stream = [r["name"].replace(" (N=256)", "") for r in rows_a_stream]
    y_pos = np.arange(len(names_a_stream))
    cpb_inl = [r["inlined_stream"]["cpb"] for r in rows_a_stream]
    cpb_buf = [r["buffer_stream"]["cpb"] for r in rows_a_stream]
    ax.barh(y_pos - height / 2, cpb_inl, height, color="#2b83ba", label="Inlined Accumulator (shootout-inspired)")
    ax.barh(y_pos + height / 2, cpb_buf, height, color="#fdae61", label="Materialized Direct-Buffer")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_a_stream)
    for tick, nm in zip(ax.get_yticklabels(), names_a_stream):
        if "ra_core_v2" in nm:
            tick.set_fontweight("bold")
            tick.set_color("#d95f02")
    ax.invert_yaxis()
    ax.set_xlabel("Cycles per Byte (cpb) — Lower is Faster (Sorted by Buffer cpb)")
    ax.set_title("(a) Continuous Bulk Stream: Inlined vs. Materialized Buffer (cpb)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    # Panel B: Addressed K=1 cost per 32-bit output; these APIs return u32.
    ax = axes[1]
    rows_a_k1 = sorted(rows_a, key=lambda r: r["buffer_k1"]["ns_word"])
    names_a_k1 = [r["name"].replace(" (N=256)", "") for r in rows_a_k1]
    y_pos_k1 = np.arange(len(names_a_k1))
    ns_inl = [r["inlined_k1"]["ns_word"] for r in rows_a_k1]
    ns_buf = [r["buffer_k1"]["ns_word"] for r in rows_a_k1]
    ax.barh(y_pos_k1 - height / 2, ns_inl, height, color="#2b83ba", label="Inlined K=1 (Register Eval)")
    ax.barh(y_pos_k1 + height / 2, ns_buf, height, color="#d7191c", label="Function-Call Buffer K=1")
    ax.set_yticks(y_pos_k1)
    ax.set_yticklabels(names_a_k1)
    for tick, nm in zip(ax.get_yticklabels(), names_a_k1):
        if "ra_core_v2" in nm:
            tick.set_fontweight("bold")
            tick.set_color("#d95f02")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("K=1 cost (ns / 32-bit-equivalent output, log scale) — Lower is Faster")
    ax.set_title("(b) Per-Seed K=1 First-Draw Cost (Sorted by Buffer K=1)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    fig.suptitle("Continuous inlined streams count native 64-bit outputs; K=1 and direct buffers return u32", fontsize=10, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOTS_DIR / "shootout_fig2_inlined_vs_buffer_cpb.png")
    fig.savefig(PLOTS_DIR / "shootout_fig2_inlined_vs_buffer_cpb.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: three Monte Carlo workloads plus a Bernoulli flip diagnostic
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))

    # Panel (a): Workload 1 Photon Transport across P_reject in {0.50, 0.80, 0.90, 0.98}
    ax = axes[0, 0]
    p_labels = ["P_rej=0.50\n(1M phot)", "P_rej=0.80\n(1M phot)", "P_rej=0.90\n(1M phot)", "P_rej=0.98\n(5M rare)"]
    x_p = np.arange(len(p_labels))
    w_bar = 0.15

    def get_w1_times(gname, rare_name=None):
        vals = []
        for p_entry in mc1["photon_transport"]:
            m = next(g["wall_min"] for g in p_entry["generators"] if g["name"] == gname)
            vals.append(m * 1000.0)
        r_m = next(g["wall_min"] for g in mc1["rare_event_trigger"] if g["name"] == (rare_name or gname))
        vals.append(r_m * 1000.0)
        return vals

    w1_series = [
        ("ra_core_v2_otf (3-Stage W2_D2)", "#d95f02", "ra_core v2 OTF (W2_D2)"),
        ("ra_core_v1_otf (3-Stage W0_D0)", "#7570b3", "ra_core v1 OTF (W0_D0)"),
        ("Philox4x32-10",                  "#1b9e77", "Philox4x32-10"),
        ("ChaCha8",                        "#e7298a", "ChaCha8"),
        ("xoshiro256**",                   "#66a61e", "xoshiro256**"),
    ]
    for idx, (gname, col, lbl) in enumerate(w1_series):
        t_ms = get_w1_times(gname)
        ax.bar(x_p + (idx - 2) * w_bar, t_ms, w_bar, color=col, label=lbl)
    ax.set_xticks(x_p)
    ax.set_xticklabels(p_labels)
    ax.set_ylabel("Wall-Clock Time (ms) — Lower is Faster")
    ax.set_title("(a) Photon transport: natural paths, differing draw counts")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    # Panel (b): scheduler comparison within RA only. Philox traverses a
    # different stochastic tree and cannot enter a same-work speed chart.
    ax = axes[0, 1]
    tree_labels = [
        "ra_core v2\nHybrid AVX2 x8",
        "ra_core v2\nScalar DFS",
        "ra_core v2\nBFS Compacted",
        "ra_core v2\nWork-stealing"
    ]
    tree_ms = [
        g3["schedules"]["S4_Hybrid_AVX2_Deferred_Promotion"]["time_ms"],
        g3["schedules"]["S1_DFS_Recurrence"]["time_ms"],
        g3["schedules"]["S2_BFS_Stream_Compaction"]["time_ms"],
        g3["schedules"]["S3_Random_Work_Stealing"]["time_ms"],
    ]
    tree_cols = ["#d95f02", "#fc8d62", "#8da0cb", "#999999"]
    x_t = np.arange(len(tree_labels))
    bars = ax.bar(x_t, tree_ms, color=tree_cols, width=0.58)
    ax.set_yscale("log")
    for b, v in zip(bars, tree_ms):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.2f} ms", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax.set_xticks(x_t)
    ax.set_xticklabels(tree_labels, rotation=18, ha="right")
    ax.set_ylim(0.3, 80.0)
    ax.set_ylabel("Tree Traversal Time (ms, log scale)")
    ax.set_title("(b) RA scheduler invariance on the same 22,446-node tree")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Panel (c): Workload 2C eight-lane Bernoulli flip diagnostic (AVX2)
    ax = axes[1, 0]
    ising_ks = ["K=255\n(N=256)", "K=63\n(N=64)", "K=31\n(N=32)", "K=15\n(N=16)"]
    ising_keys = ["bernoulli_flip_avx2_k255", "bernoulli_flip_avx2_k63", "bernoulli_flip_avx2_k31", "bernoulli_flip_avx2_k15"]
    x_is = np.arange(len(ising_ks))
    w_is = 0.13

    def get_ising_mibs(prefix):
        out = []
        for ik in ising_keys:
            row = next(r for r in mc2[ik] if r["name"].startswith(prefix))
            out.append(row["mibs"])
        return out

    is_series = [
        ("ra_core_v2_x8",    "#d95f02", "ra_core v2 x8 (Matched N)"),
        ("ra_core_v1_x8",    "#7570b3", "ra_core v1 x8 (Matched N)"),
        ("Philox4x32-10_x8", "#1b9e77", "Philox4x32-10 x8"),
        ("ChaCha8_x8",       "#e7298a", "ChaCha8 x8"),
        ("xoshiro256**_x8",  "#66a61e", "xoshiro256** x8"),
        ("PCG32_x8",         "#e6ab02", "PCG32 x8"),
    ]
    for idx, (pref, col, lbl) in enumerate(is_series):
        vals = get_ising_mibs(pref)
        ax.bar(x_is + (idx - 2.5) * w_is, vals, w_is, color=col, label=lbl)
    ax.set_xticks(x_is)
    ax.set_xticklabels(ising_ks)
    ax.set_ylabel("Generated throughput (MiB/s) — higher is faster")
    ax.set_title("(c) Workload 2C: eight-lane Bernoulli flip diagnostic (AVX2)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=2)

    # Panel (d): Workload 3 Asian Option GBM (1,000 x 10,000) Pure Generation vs End-to-End
    ax = axes[1, 1]
    gbm_rows = mc3["asian_option_gbm"]
    gbm_names = [r["name"].replace("_orbit_into", "").replace(" (Stateful)", " OTF").replace(" (ARX counter)", "") for r in gbm_rows]
    y_g = np.arange(len(gbm_names))
    gen_ms = [r["gen_min"] * 1000.0 for r in gbm_rows]
    box_ms = [(r["e2e_min"] - r["gen_min"]) * 1000.0 for r in gbm_rows]

    ax.barh(y_g, gen_ms, color="#d95f02", label="Pure PRNG Generation Time (ms)")
    ax.barh(y_g, box_ms, left=gen_ms, color="#abd9e9", label="Remaining end-to-end work (ms)")
    ax.set_yticks(y_g)
    ax.set_yticklabels(gbm_names)
    ax.invert_yaxis()
    ax.set_xlabel("Time for 10,000,000 Words (ms) — Lower is Faster")
    ax.set_title("(d) Workload 3: Asian Option GBM (1,000 Paths x 10,000 Steps)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shootout_fig3_montecarlo_workloads_v2.png")
    fig.savefig(PLOTS_DIR / "shootout_fig3_montecarlo_workloads_v2.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 4: Architectural Tradeoff Scatter (K=1 Latency vs Bulk AVX2 Speed & N-Scaling)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))

    # Panel A: every point uses the same K=1 and K=15 AVX2 regimes.
    ax = axes[0]
    def c_mibs(name, k):
        return gens_c[name][k]["mibs"]
    pts = [
        ("ra_core v2 (N=256)", c_mibs("ra_core_v2_x8 (N=256)", "K=1"), c_mibs("ra_core_v2_x8 (N=256)", "K=15"), "#d95f02", "o", 130),
        ("ra_core v2 (N=64)",  c_mibs("ra_core_v2_x8 (N=64)", "K=1"), c_mibs("ra_core_v2_x8 (N=64)", "K=15"), "#fc8d62", "o", 105),
        ("ra_core v2 (N=16)",  c_mibs("ra_core_v2_x8 (N=16)", "K=1"), c_mibs("ra_core_v2_x8 (N=16)", "K=15"), "#fdb863", "o", 95),
        ("ra_core v1 (N=256)", c_mibs("ra_core_v1_x8 (N=256)", "K=1"), c_mibs("ra_core_v1_x8 (N=256)", "K=15"), "#7570b3", "s", 110),
        ("Philox4x32-10", c_mibs("Philox4x32-10_x8", "K=1"), c_mibs("Philox4x32-10_x8", "K=15"), "#1b9e77", "^", 120),
        ("ChaCha8", c_mibs("ChaCha8_x8", "K=1"), c_mibs("ChaCha8_x8", "K=15"), "#e7298a", "v", 120),
        ("xoshiro256**", c_mibs("xoshiro256**_x8", "K=1"), c_mibs("xoshiro256**_x8", "K=15"), "#66a61e", "D", 100),
        ("PCG32", c_mibs("PCG32_x8", "K=1"), c_mibs("PCG32_x8", "K=15"), "#e6ab02", "P", 110),
        ("SplitMix64", c_mibs("SplitMix64_x8", "K=1"), c_mibs("SplitMix64_x8", "K=15"), "#a6761d", "d", 100),
    ]
    label_offsets = {
        "ra_core v2 (N=256)": (15, 20),
        "ra_core v2 (N=64)": (12, -32),
        "ra_core v2 (N=16)": (12, -45),
        "ra_core v1 (N=256)": (-18, 15),
        "Philox4x32-10": (8, 12),
        "ChaCha8": (8, -16),
        "xoshiro256**": (-55, -34),
        "PCG32": (-55, 25),
        "SplitMix64": (-80, -28),
    }
    for lbl, x_k1, y_bulk, col, mk, sz in pts:
        ax.scatter(x_k1, y_bulk, color=col, marker=mk, s=sz, edgecolors="black", linewidths=0.7, zorder=4)
        ax.annotate(lbl, (x_k1, y_bulk), xytext=label_offsets[lbl], textcoords="offset points",
                    fontsize=7.8, fontweight="bold" if "v2" in lbl else "normal",
                    arrowprops={"arrowstyle": "-", "color": "#888", "lw": 0.55} if "ra_core v2" in lbl or "xoshiro" in lbl else None)

    ax.set_xlabel("8-Way AVX2 per-seed K=1 throughput (MiB/s)")
    ax.set_ylabel("8-Way AVX2 K=15 throughput (MiB/s)")
    ax.set_title("(a) Same-regime AVX2 throughput: K=1 versus K=15")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel B: State-Size Scaling N in {256, 128, 64, 32, 16} for `ra_core v2`
    ax = axes[1]
    n_sizes = ["N=256\n(2048 B, K<=255)", "N=128\n(1024 B, K<=127)", "N=64\n(512 B, K<=63)", "N=32\n(256 B, K<=31)", "N=16\n(128 B, K<=15)"]
    n_names_c = ["ra_core_v2_x8 (N=256)", "ra_core_v2_x8 (N=128)", "ra_core_v2_x8 (N=64)", "ra_core_v2_x8 (N=32)", "ra_core_v2_x8 (N=16)"]
    max_ks_c  = ["K=255", "K=127", "K=63", "K=31", "K=15"]
    k1_avx2   = [gens_c[nm]["K=1"]["mibs"] for nm in n_names_c]
    k8_avx2   = [gens_c[nm]["K=8"]["mibs"] for nm in n_names_c]
    kmax_avx2 = [gens_c[nm][mk]["mibs"] for nm, mk in zip(n_names_c, max_ks_c)]

    x_n = np.arange(len(n_sizes))
    w_n = 0.25
    ax.bar(x_n - w_n, k1_avx2,   w_n, color="#fdb863", label="K=1 (0 B stack, 2-tap closed form)")
    ax.bar(x_n,       k8_avx2,   w_n, color="#fc8d62", label="K=8 (0 B stack, register fast-path)")
    ax.bar(x_n + w_n, kmax_avx2, w_n, color="#d95f02", label="K = N-1 (Full Cycle Peak AVX2)")
    ax.set_xticks(x_n)
    ax.set_xticklabels(n_sizes)
    ax.set_ylabel("8-Way AVX2 Throughput (MiB/s)")
    ax.set_title("(b) RA state-size scaling (K=1, K=8, and each N's maximum K)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shootout_fig4_pareto_radar_matrix.png")
    fig.savefig(PLOTS_DIR / "shootout_fig4_pareto_radar_matrix.pdf")
    plt.close(fig)


def build_html_dashboard(micro, mc_all, feat_matrix):
    def row_with_prefix(rows, prefix):
        return next(row for row in rows if row["name"].startswith(prefix))

    manifest = json.loads((DATA_DIR / "fairness_manifest.json").read_text())
    core = manifest["core"]
    compile_flags = " ".join(arg for arg in manifest["programs"]["micro"]["compile_command"][1:] if arg.startswith("-") and arg not in ("-lm", "-o"))
    micro_samples = json.loads((DATA_DIR / "shootout_sample_summary.json").read_text())
    unstable_rows = sum(row["median_s"] / row["min_s"] > 1.10 for row in micro_samples)

    a_rows = micro["suite_a_vigna_shootout"]
    b_rows = micro["suite_b_scalar_buffer"]["generators"]
    c_rows = micro["suite_c_avx2_x8"]["generators"]
    a_v2 = row_with_prefix(a_rows, "ra_core_v2")
    a_philox = row_with_prefix(a_rows, "Philox4x32")
    b_v2 = row_with_prefix(b_rows, "ra_core_v2 (N=256)")
    c_v2 = row_with_prefix(c_rows, "ra_core_v2_x8 (N=256)")
    c_philox = row_with_prefix(c_rows, "Philox4x32")
    rare_rows = mc_all["workload1"]["rare_event_trigger"]
    rare_v2 = row_with_prefix(rare_rows, "ra_core_v2_otf")
    rare_philox = row_with_prefix(rare_rows, "Philox4x32")
    gbm_rows = mc_all["workload3"]["asian_option_gbm"]
    gbm_v2 = row_with_prefix(gbm_rows, "ra_core_v2_orbit_into")
    gbm_philox = row_with_prefix(gbm_rows, "Philox4x32-10 (2D")
    tree = mc_all["group3_branching_tree"]
    tree_s4 = tree["schedules"]["S4_Hybrid_AVX2_Deferred_Promotion"]
    k1_ratio = a_v2["buffer_k1"]["mibs"] / a_philox["buffer_k1"]["mibs"]
    bulk_ratio = c_v2["regimes"]["K=255"]["mibs"] / c_philox["regimes"]["K=255"]["mibs"]
    gbm_ratio = gbm_philox["gen_min"] / gbm_v2["gen_min"]
    embedded_json = json.dumps({
        "micro": micro,
        "mc": mc_all,
        "features": feat_matrix,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unified PRNG Shootout & Monte Carlo Benchmark — ra_core v2</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-alt: #263449;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #f97316;
    --accent2: #38bdf8;
    --good: #22c55e;
    --warn: #eab308;
    --bad: #ef4444;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px 32px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 1.65rem; color: #fff; }}
  h2 {{ margin: 28px 0 10px 0; font-size: 1.2rem; color: var(--accent2); border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  p.sub {{ margin: 0 0 18px 0; color: var(--muted); font-size: 0.92rem; }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 22px;
  }}
  .kpi-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
  }}
  .kpi-title {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }}
  .kpi-val {{ font-size: 1.35rem; font-weight: 700; color: #fff; margin: 4px 0; }}
  .kpi-sub {{ font-size: 0.8rem; color: var(--good); }}
  .tabs {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .tab-btn {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.88rem;
  }}
  .tab-btn.active {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }}
  .controls {{
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
    background: var(--panel);
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 14px;
    font-size: 0.88rem;
  }}
  select, input[type="text"] {{
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 0.86rem;
  }}
  .table-wrap {{
    overflow-x: auto;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    white-space: nowrap;
  }}
  th, td {{
    padding: 8px 11px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }}
  th:first-child, td:first-child {{ text-align: left; position: sticky; left: 0; background: var(--panel); z-index: 2; }}
  th {{
    background: var(--panel-alt);
    color: #e2e8f0;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
  }}
  tr.highlight-v2 td {{
    background: rgba(249, 115, 22, 0.14);
    font-weight: 600;
  }}
  tr.highlight-v2 td:first-child {{
    background: #332018;
    border-left: 3px solid var(--accent);
  }}
  tr:hover td {{ background: rgba(56, 189, 248, 0.1); }}
  .badge-pass {{ background: rgba(34,197,94,0.18); color: #4ade80; padding: 2px 7px; border-radius: 4px; font-weight: 600; }}
  .badge-fail {{ background: rgba(239,68,68,0.2); color: #f87171; padding: 2px 7px; border-radius: 4px; font-weight: 600; }}
  .badge-warn {{ background: rgba(234,179,8,0.2); color: #facc15; padding: 2px 7px; border-radius: 4px; font-weight: 600; }}
  .tab-pane {{ display: none; }}
  .tab-pane.active {{ display: block; }}
</style>
</head>
<body>

<h1>Unified PRNG Shootout & Monte Carlo Benchmark Dashboard — <code>ra_core v2</code></h1>
<p class="sub">
  Single-core pinned execution (<code>taskset -c {core}</code>, <code>gcc {compile_flags}</code>).
  Suite A/B/D measure scalar paths; Suite C measures explicit 8-lane AVX2. Pi uses top-24-bit float conversion, and MiB/s counts generated words (one final word per odd-K burst is unused).
  Microbenchmark timing uses seven measured rounds after one warmup, with raw samples in <code>data/shootout_samples.csv</code>; Monte Carlo rows retain five raw timings each.
  The implementation KAT and parity gates passed with <code>0</code> mismatches.
</p>
<p class="sub"><strong>Timing variability:</strong> {unstable_rows}/{len(micro_samples)} microbenchmark rows have median time more than 10% above the minimum. Headline rates are best cases; inspect the saved medians before interpreting small differences.</p>

<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-title">Per-Seed K=1 (Buffer / Inlined)</div>
    <div class="kpi-val">{a_v2['buffer_k1']['mibs']:,.1f} / {a_v2['inlined_k1']['mibs']:,.1f} MiB/s</div>
    <div class="kpi-sub">Buffer: {k1_ratio:.2f}x Philox4x32-10 in the same regime</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Bulk Burst K=255 (Scalar / AVX2 x8)</div>
    <div class="kpi-val">{b_v2['regimes']['K=255']['mibs']:,.0f} / {c_v2['regimes']['K=255']['mibs']:,.0f} MiB/s</div>
    <div class="kpi-sub">AVX2: {bulk_ratio:.2f}x Philox_x8 in the same regime</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Branching Particle Tree (22,446 Nodes)</div>
    <div class="kpi-val">{tree_s4['time_ms']:.3f} ms</div>
    <div class="kpi-sub">RA scheduler parity: {tree_s4['mismatches_vs_S1']} mismatches; cross-family speedup withheld</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Photon Transport &amp; Rare Trigger</div>
    <div class="kpi-val">{rare_v2['wall_min'] * 1000:.1f} ms (P=0.98)</div>
    <div class="kpi-sub">Philox: {rare_philox['wall_min'] * 1000:.1f} ms; draw counts differ across streams</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Asian Option GBM (1,000 x 10,000)</div>
    <div class="kpi-val">{gbm_v2['gen_min'] * 1000:.1f} ms ({gbm_v2['gen_mibs']:,.0f} MiB/s)</div>
    <div class="kpi-sub">Pure generation: {gbm_ratio:.2f}x Philox 2D; see end-to-end table</div>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('tab-shootout', this)">1. Shootout-Inspired (Inlined vs Buffer)</button>
  <button class="tab-btn" onclick="switchTab('tab-scalar', this)">2. Scalar Direct-Buffer (K=1..Stream)</button>
  <button class="tab-btn" onclick="switchTab('tab-avx2', this)">3. 8-Way AVX2 SIMD (K=1..255)</button>
  <button class="tab-btn" onclick="switchTab('tab-engine', this)">4. Stateful OTF Engine (next_u32)</button>
  <button class="tab-btn" onclick="switchTab('tab-mc', this)">5. 3 Monte Carlo Workloads + Tree</button>
  <button class="tab-btn" onclick="switchTab('tab-features', this)">6. Statistical &amp; Architecture Matrix</button>
</div>

<div class="controls" id="metric-controls">
  <label><strong>Metric Unit:</strong>
    <select id="unit-select" onchange="renderAll()">
      <option value="mibs">Throughput (MiB/s) — Higher is Better</option>
      <option value="gbs">Throughput (GB/s) — Higher is Better</option>
      <option value="cpb">Cycles per Byte (cpb) — Lower is Better</option>
      <option value="cyc_word">Cycles per 32-bit Word — Lower is Better</option>
      <option value="ns_word">Nanoseconds per 32-bit-Equivalent Output — Lower is Better</option>
      <option value="ns_64b">Nanoseconds per 64-bit-Equivalent Output — Lower is Better</option>
    </select>
  </label>
  <label><strong>Row Ordering:</strong>
    <select id="sort-mode-select" onchange="onSortModeChange()">
      <option value="rank">Rank by Best Performance (Fastest on Top)</option>
      <option value="category">Category Order (ra_core v2 Pinned on Top)</option>
    </select>
  </label>
  <label><strong>Filter Generator:</strong>
    <input type="text" id="filter-input" placeholder="e.g. ra_core, Philox, ChaCha..." oninput="renderAll()">
  </label>
  <span style="color:var(--muted);font-size:0.82rem;">Tip: Click any column header (<code>▲/▼</code>) to sort by that column.</span>
</div>

<div id="tab-shootout" class="tab-pane active">
  <h2>Suite A: Shootout-Inspired Inlined Register-Accumulator vs. Materialized Direct-Buffer</h2>
  <p>Four families emit native 64-bit values in the continuous inlined stream; K=1 and direct-buffer APIs return 32-bit words. Rates and per-word costs count the actual output bytes.</p>
  <div class="table-wrap"><table id="tbl-shootout"></table></div>
</div>

<div id="tab-scalar" class="tab-pane">
  <h2>Suite B: Materialized Direct-Buffer Scalar Throughput across Burst Size K and State Size N</h2>
  <div class="table-wrap"><table id="tbl-scalar"></table></div>
</div>

<div id="tab-avx2" class="tab-pane">
  <h2>Suite C: Apples-to-Apples 8-Way AVX2 SIMD (<code>u32x8</code>) Across Burst Size K and State Size N</h2>
  <div class="table-wrap"><table id="tbl-avx2"></table></div>
</div>

<div id="tab-engine" class="tab-pane">
  <h2>Suite D: Stateful Streaming API (<code>next_u32()</code> per draw) Across Trajectory Length K</h2>
  <div class="table-wrap"><table id="tbl-engine"></table></div>
</div>

<div id="tab-mc" class="tab-pane">
  <h2>Workload 1: Variable-Length Photon Transport in Turbid Slab &amp; Rare-Event Trigger</h2>
  <p>Natural stochastic paths differ across generator families. Compare observed total words and rejected events with runtime; this is end-to-end workload timing.</p>
  <div class="table-wrap"><table id="tbl-mc1"></table></div>
  <h2>Workload 2B &amp; 2C: Matched 8-Way AVX2 Consumer Bounded-Burst &pi; &amp; Bernoulli Flip Diagnostic</h2>
  <p>Pi uses the same top-24-bit float conversion in scalar and AVX2 paths. MiB/s counts generated words; one word per odd-K burst is not consumed. The Bernoulli diagnostic is not an Ising simulation.</p>
  <div class="table-wrap"><table id="tbl-mc2"></table></div>
  <h2>Workload 3: Asian Call Option GBM (1,000 Paths &times; 10,000 Steps = 10M Words) &amp; Cross-Stream Independence</h2>
  <div class="table-wrap"><table id="tbl-mc3"></table></div>
</div>

<div id="tab-features" class="tab-pane">
  <h2>Comprehensive PRNG Shootout Feature, Complexity &amp; Statistical Quality Matrix</h2>
  <p>Statistical-quality entries are manually curated from earlier experiments and reference claims; this speed audit did not rerun or independently verify them.</p>
  <div class="table-wrap"><table id="tbl-features"></table></div>
</div>

<script>
const DATA = {embedded_json};
const tableSortState = {{}};

function switchTab(id, btn) {{
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if (btn) btn.classList.add('active');
}}

function fmtMetric(obj, unit) {{
  if (!obj) return '—';
  const v = obj[unit];
  if (unit === 'mibs') return v.toFixed(1);
  if (unit === 'gbs')  return v.toFixed(2);
  if (unit === 'cpb')  return v.toFixed(2);
  if (unit === 'cyc_word') return v.toFixed(2);
  if (unit === 'ns_word' || unit === 'ns_64b') return v.toFixed(2);
  return v;
}}

function isHigherBetter(unit) {{
  return unit === 'mibs' || unit === 'gbs';
}}

function isV2(name) {{
  return name.includes('ra_core_v2') || name.includes('ra_engine_v2') || name.includes('ra_core v2');
}}

function parseCellNumber(text) {{
  const cleaned = text.replace(/,/g, '').trim();
  if (!cleaned || cleaned === '—') return null;
  const m = cleaned.match(/^[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?/);
  return m ? parseFloat(m[0]) : null;
}}

function applySortToTable(tableId, colIdx, desc) {{
  const tbl = document.getElementById(tableId);
  if (!tbl || !tbl.tBodies[0]) return;
  const tbody = tbl.tBodies[0];
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    const ta = a.cells[colIdx] ? a.cells[colIdx].innerText : '';
    const tb = b.cells[colIdx] ? b.cells[colIdx].innerText : '';
    const na = parseCellNumber(ta);
    const nb = parseCellNumber(tb);
    if (na === null && nb === null) return ta.localeCompare(tb);
    if (na === null) return 1;
    if (nb === null) return -1;
    return desc ? (nb - na) : (na - nb);
  }});
  rows.forEach(r => tbody.appendChild(r));
  const ths = tbl.querySelectorAll('thead th');
  ths.forEach((th, idx) => {{
    const base = th.getAttribute('data-label') || th.innerText.replace(/\\s*[▲▼]$/, '');
    th.setAttribute('data-label', base);
    th.innerText = idx === colIdx ? (base + (desc ? ' ▼' : ' ▲')) : base;
  }});
}}

function bindSortableTables(unit) {{
  const tblIds = ['tbl-shootout', 'tbl-scalar', 'tbl-avx2', 'tbl-engine', 'tbl-mc1', 'tbl-mc2', 'tbl-mc3', 'tbl-features'];
  const sortMode = document.getElementById('sort-mode-select').value;
  const hb = isHigherBetter(unit);

  const defaultRankCols = {{
    'tbl-shootout': {{ col: 3, desc: hb }},
    'tbl-scalar':   {{ col: 10, desc: hb }},
    'tbl-avx2':     {{ col: 9, desc: hb }},
    'tbl-engine':   {{ col: 1, desc: hb }},
    'tbl-mc1':      {{ col: 7, desc: false }},
    'tbl-mc2':      {{ col: 2, desc: false }},
    'tbl-mc3':      {{ col: 6, desc: false }},
  }};

  tblIds.forEach(id => {{
    const tbl = document.getElementById(id);
    if (!tbl) return;
    const ths = tbl.querySelectorAll('thead th');
    ths.forEach((th, idx) => {{
      const base = th.innerText.replace(/\\s*[▲▼]$/, '');
      th.setAttribute('data-label', base);
      th.onclick = () => {{
        const prev = tableSortState[id];
        const desc = (prev && prev.col === idx) ? !prev.desc : (idx >= 2 ? hb : false);
        tableSortState[id] = {{ col: idx, desc, unit }};
        applySortToTable(id, idx, desc);
      }};
    }});

    if (tableSortState[id]) {{
      if (['tbl-shootout', 'tbl-scalar', 'tbl-avx2', 'tbl-engine'].includes(id) && tableSortState[id].unit !== unit) {{
        if (isHigherBetter(tableSortState[id].unit) !== hb) {{
          tableSortState[id].desc = !tableSortState[id].desc;
        }}
        tableSortState[id].unit = unit;
      }}
      applySortToTable(id, tableSortState[id].col, tableSortState[id].desc);
    }} else if (sortMode === 'rank' && defaultRankCols[id]) {{
      applySortToTable(id, defaultRankCols[id].col, defaultRankCols[id].desc);
    }}
  }});
}}

function onSortModeChange() {{
  Object.keys(tableSortState).forEach(k => delete tableSortState[k]);
  renderAll();
}}

function renderAll() {{
  const unit = document.getElementById('unit-select').value;
  const q = document.getElementById('filter-input').value.toLowerCase();
  if (tableSortState['tbl-shootout'] || tableSortState['tbl-scalar'] || tableSortState['tbl-avx2'] || tableSortState['tbl-engine']) {{
    // Update direction automatically if user switched between Higher-is-Better (MiB/s) and Lower-is-Better (cpb/ns)
  }}

  // 1. Suite A
  let htmlA = `<thead><tr>
    <th>Generator</th><th>Category</th><th>State</th>
    <th>Inlined Stream (${{unit}})</th><th>Buffer Stream (${{unit}})</th>
    <th>Inlined K=1 (${{unit}})</th><th>Buffer K=1 (${{unit}})</th>
  </tr></thead><tbody>`;
  DATA.micro.suite_a_vigna_shootout.filter(r => r.name.toLowerCase().includes(q)).forEach(r => {{
    htmlA += `<tr class="${{isV2(r.name) ? 'highlight-v2' : ''}}">
      <td>${{r.name}}</td><td style="text-align:left">${{r.category}}</td><td>${{r.state_bytes}} B</td>
      <td>${{fmtMetric(r.inlined_stream, unit)}}</td>
      <td>${{fmtMetric(r.buffer_stream, unit)}}</td>
      <td>${{fmtMetric(r.inlined_k1, unit)}}</td>
      <td>${{fmtMetric(r.buffer_k1, unit)}}</td>
    </tr>`;
  }});
  htmlA += `</tbody>`;
  document.getElementById('tbl-shootout').innerHTML = htmlA;

  // 2. Suite B
  const kB = DATA.micro.suite_b_scalar_buffer.k_labels;
  let htmlB = `<thead><tr><th>Generator</th><th>State</th>` + kB.map(k => `<th>${{k}}</th>`).join('') + `</tr></thead><tbody>`;
  DATA.micro.suite_b_scalar_buffer.generators.filter(g => g.name.toLowerCase().includes(q)).forEach(g => {{
    htmlB += `<tr class="${{isV2(g.name) ? 'highlight-v2' : ''}}"><td>${{g.name}}</td><td>${{g.state_bytes}} B</td>`;
    kB.forEach(k => {{
      htmlB += `<td>${{fmtMetric(g.regimes[k], unit)}}</td>`;
    }});
    htmlB += `</tr>`;
  }});
  htmlB += `</tbody>`;
  document.getElementById('tbl-scalar').innerHTML = htmlB;

  // 3. Suite C
  const kC = DATA.micro.suite_c_avx2_x8.k_labels;
  let htmlC = `<thead><tr><th>Generator (8-Way AVX2)</th>` + kC.map(k => `<th>${{k}}</th>`).join('') + `</tr></thead><tbody>`;
  DATA.micro.suite_c_avx2_x8.generators.filter(g => g.name.toLowerCase().includes(q)).forEach(g => {{
    htmlC += `<tr class="${{isV2(g.name) ? 'highlight-v2' : ''}}"><td>${{g.name}}</td>`;
    kC.forEach(k => {{
      htmlC += `<td>${{fmtMetric(g.regimes[k], unit)}}</td>`;
    }});
    htmlC += `</tr>`;
  }});
  htmlC += `</tbody>`;
  document.getElementById('tbl-avx2').innerHTML = htmlC;

  // 4. Suite D
  const kD = DATA.micro.suite_d_stateful_engine.k_values.map(v => 'K=' + v);
  let htmlD = `<thead><tr><th>Stateful Engine</th>` + kD.map(k => `<th>${{k}}</th>`).join('') + `</tr></thead><tbody>`;
  DATA.micro.suite_d_stateful_engine.engines.filter(e => e.name.toLowerCase().includes(q)).forEach(e => {{
    htmlD += `<tr class="${{isV2(e.name) ? 'highlight-v2' : ''}}"><td>${{e.name}}</td>`;
    kD.forEach(k => {{
      htmlD += `<td>${{fmtMetric(e.regimes[k], unit)}}</td>`;
    }});
    htmlD += `</tr>`;
  }});
  htmlD += `</tbody>`;
  document.getElementById('tbl-engine').innerHTML = htmlD;

  // 5. Monte Carlo Tables
  const p0 = DATA.mc.workload1.photon_transport[0].generators;
  const p1 = DATA.mc.workload1.photon_transport[1].generators;
  const p2 = DATA.mc.workload1.photon_transport[2].generators;
  const rareMap = {{}};
  DATA.mc.workload1.rare_event_trigger.forEach(r => {{ rareMap[r.name] = r; }});
  let htmlMC1 = `<thead><tr>
    <th>Generator</th>
    <th>P_rej=0.50 min(5)</th><th>P_rej=0.50 R_d</th>
    <th>P_rej=0.80 min(5)</th><th>P_rej=0.80 R_d</th>
    <th>P_rej=0.90 min(5)</th><th>P_rej=0.90 cyc/part</th>
    <th>Rare Trigger P=0.98 (5M ev)</th><th>Rare cyc/ev</th>
  </tr></thead><tbody>`;
  for (let i = 0; i < p0.length; i++) {{
    if (!p0[i].name.toLowerCase().includes(q)) continue;
    const rr = rareMap[p0[i].name];
    htmlMC1 += `<tr class="${{isV2(p0[i].name) ? 'highlight-v2' : ''}}">
      <td>${{p0[i].name}}</td>
      <td>${{(p0[i].wall_min * 1000).toFixed(2)}} ms</td><td>${{p0[i].rd_mean.toFixed(6)}}</td>
      <td>${{(p1[i].wall_min * 1000).toFixed(2)}} ms</td><td>${{p1[i].rd_mean.toFixed(6)}}</td>
      <td>${{(p2[i].wall_min * 1000).toFixed(2)}} ms</td><td>${{p2[i].cyc_per_particle.toFixed(1)}}</td>
      <td>${{rr ? (rr.wall_min * 1000).toFixed(2) + ' ms' : '—'}}</td>
      <td>${{rr ? rr.cyc_per_event.toFixed(1) : '—'}}</td>
    </tr>`;
  }}
  htmlMC1 += `</tbody>`;
  document.getElementById('tbl-mc1').innerHTML = htmlMC1;

  // MC2 Table (AVX2 Pi + Bernoulli flip diagnostic)
  const is255 = DATA.mc.workload2.bernoulli_flip_avx2_k255;
  const is63  = DATA.mc.workload2.bernoulli_flip_avx2_k63;
  const is31  = DATA.mc.workload2.bernoulli_flip_avx2_k31;
  const is15  = DATA.mc.workload2.bernoulli_flip_avx2_k15;
  const pi255 = DATA.mc.workload2.avx2_consumer_k255;
  let htmlMC2 = `<thead><tr>
    <th>8-Way AVX2 Generator Family</th>
    <th>AVX2 &pi; K=255 (ms / MiB/s)</th>
    <th>Bernoulli K=255 N=256 (ms / MiB/s)</th>
    <th>Bernoulli K=63 N=64 (ms / MiB/s)</th>
    <th>Bernoulli K=31 N=32 (ms / MiB/s)</th>
    <th>Bernoulli K=15 N=16 (ms / MiB/s)</th>
  </tr></thead><tbody>`;
  for (let i = 0; i < is255.length; i++) {{
    if (!is255[i].name.toLowerCase().includes(q)) continue;
    htmlMC2 += `<tr class="${{isV2(is255[i].name) ? 'highlight-v2' : ''}}">
      <td>${{is255[i].name.replace(' (N=256)', ' (Matched N)')}}</td>
      <td>${{(pi255[i].wall_min*1000).toFixed(1)}} ms (${{pi255[i].mibs.toFixed(0)}})</td>
      <td>${{(is255[i].wall_min*1000).toFixed(1)}} ms (${{is255[i].mibs.toFixed(0)}})</td>
      <td>${{(is63[i].wall_min*1000).toFixed(1)}} ms (${{is63[i].mibs.toFixed(0)}})</td>
      <td>${{(is31[i].wall_min*1000).toFixed(1)}} ms (${{is31[i].mibs.toFixed(0)}})</td>
      <td>${{(is15[i].wall_min*1000).toFixed(1)}} ms (${{is15[i].mibs.toFixed(0)}})</td>
    </tr>`;
  }}
  htmlMC2 += `</tbody>`;
  document.getElementById('tbl-mc2').innerHTML = htmlMC2;

  // MC3 Table (Asian Option GBM)
  let htmlMC3 = `<thead><tr>
    <th>Generator</th><th>Mean Payoff</th><th>SE</th>
    <th>Viol (n=50)</th><th>Viol (n=500)</th><th>Max |r| (n=500)</th>
    <th>Pure Gen Time</th><th>Pure Gen MiB/s</th><th>End-to-End Time</th>
  </tr></thead><tbody>`;
  DATA.mc.workload3.asian_option_gbm.filter(r => r.name.toLowerCase().includes(q)).forEach(r => {{
    const v500_pct = (100.0 * r.viol_500 / 500.0).toFixed(1);
    const badge = r.viol_500 <= 35 ? `<span class="badge-pass">${{r.viol_500}}/500 (${{v500_pct}}%)</span>` : `<span class="badge-fail">${{r.viol_500}}/500 (${{v500_pct}}%)</span>`;
    htmlMC3 += `<tr class="${{isV2(r.name) ? 'highlight-v2' : ''}}">
      <td>${{r.name}}</td>
      <td>${{r.mean_payoff.toFixed(6)}}</td><td>${{r.se.toFixed(6)}}</td>
      <td>${{r.viol_50}}/50</td><td>${{badge}}</td><td>${{r.max_r_500.toFixed(4)}}</td>
      <td>${{(r.gen_min*1000).toFixed(2)}} ms</td><td>${{r.gen_mibs.toFixed(1)}} MiB/s</td>
      <td>${{(r.e2e_min*1000).toFixed(2)}} ms</td>
    </tr>`;
  }});
  htmlMC3 += `</tbody>`;
  document.getElementById('tbl-mc3').innerHTML = htmlMC3;

  // 6. Features Table
  let htmlF = `<thead><tr>
    <th>Generator</th><th>Architecture</th><th>State Footprint</th>
    <th>O(1) Seek / Jump</th><th>SIMD / AVX2 Profile</th>
    <th>PractRand (Full &amp; Low/BitRev)</th><th>Dieharder / BigCrush / NIST</th><th>Safe-K Floor</th>
  </tr></thead><tbody>`;
  DATA.features.filter(f => f.generator.toLowerCase().includes(q)).forEach(f => {{
    htmlF += `<tr class="${{isV2(f.generator) ? 'highlight-v2' : ''}}">
      <td>${{f.generator}}</td>
      <td style="text-align:left">${{f.family}}</td>
      <td style="text-align:left">${{f.state_bytes}}</td>
      <td style="text-align:left">${{f.random_access_seek}}</td>
      <td style="text-align:left">${{f.simd_parallelism}}</td>
      <td style="text-align:left">${{f.practrand_full}}<br><small style="color:var(--muted)">${{f.practrand_low_bitrev}}</small></td>
      <td style="text-align:left">${{f.dieharder_bigcrush}}</td>
      <td style="text-align:left">${{f.safe_k_floor}}</td>
    </tr>`;
  }});
  htmlF += `</tbody>`;
  document.getElementById('tbl-features').innerHTML = htmlF;

  bindSortableTables(unit);
}}

renderAll();
</script>
</body>
</html>
"""
    (BASE_DIR / "shootout_dashboard.html").write_text(html, encoding="utf-8")


def main():
    micro = json.loads((DATA_DIR / "shootout_microbench.json").read_text())
    mc1   = json.loads((DATA_DIR / "mc_workload1_v2.json").read_text())
    mc2   = json.loads((DATA_DIR / "mc_workload2_v2.json").read_text())
    mc3   = json.loads((DATA_DIR / "mc_workload3_v2.json").read_text())
    g3    = json.loads(GROUP3_JSON.read_text())

    mc_all = {
        "workload1": mc1,
        "workload2": mc2,
        "workload3": mc3,
        "group3_branching_tree": g3,
    }
    (DATA_DIR / "mc_workloads_v2.json").write_text(json.dumps(mc_all, indent=2))

    feat_matrix = build_feature_matrix()
    (DATA_DIR / "shootout_feature_matrix.json").write_text(json.dumps(feat_matrix, indent=2))

    generate_plots(micro, mc1, mc2, mc3, g3)
    build_html_dashboard(micro, mc_all, feat_matrix)
    print("Successfully generated plots/shootout_fig1..4.{png,pdf}, data/mc_workloads_v2.json, data/shootout_feature_matrix.json, and shootout_dashboard.html!")


if __name__ == "__main__":
    main()
