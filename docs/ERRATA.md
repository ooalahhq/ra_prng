# ERRATA

Official record of production bugs, benchmark instrumentation flaws, reference baseline errors, and methodological corrections discovered and resolved across `ra_prng2` and `ra_core` (v1 through `ra_core v2 Final`). Its purpose is to ensure that anyone encountering earlier claims, commits, or drafts can trace every retraction and correction to verified evidence.

Each entry records the discovery/resolution date, what was wrong, root cause, verified correction, and supporting sources.

> **Archival Path Note:** Pre-v2 exploratory experiment directories (`experiments/2026-8-*`, `experiments/2026-9-*`, and internal `experiments/2026-11-02_*` logs) are omitted from the cleaned public `ra_core v2` working tree and preserved in pre-v2 git history / local research archives. Consequently, pre-v2 directory paths appear below as plain inline code, while clickable hyperlinks target tracked `ra_core v2` repository files only. See [`../README.md`](../README.md) and [`../research/2026-9_ra-core-paper/REPRODUCIBILITY.md`](../research/2026-9_ra-core-paper/REPRODUCIBILITY.md) for the tracked public evidence layout.

## Summary Index

| Date | Component / Claim | Issue Summary | Status |
|---|---|---|---|
| 2026-11-04 | `ra_core v2` key schedule (`Opsi_A3`) & Shootout Harness | `v1` sequential `addr_hi` PractRand failure; laptop turbo jitter, MC draw drift, and Workload 2b "Ising" misnomer | Resolved in `ra_core v2` & audited harnesses |
| 2026-11-02 | `singleblock K=255` `20,018.1 MB/s` (`0.191 ns/word`) claim & `rot32` UB | GCC `-O3` Dead-Code Elimination when `raw_stream == NULL` skipped all 255 permutation rounds; `(n >> (32 - s))` shift UB when `s == 0` | Retracted; replaced by materialized-buffer v2 numbers & UB-safe `rot32` |
| 2026-09-03 – 2026-11-02 | `glibc fmemopen(..., "wb")` buffer writes | Exact-sized `N * sizeof(uint32_t)` buffers lost the MSB of the final `uint32_t` word to trailing-NUL clobber | Removed from all production & engine paths |
| 2026-09-12 (resolved 2026-11-02) | Monte Carlo Asian Option bias & Philox correlation flags | Concatenating 40 `singleblock` blocks caused Box-Muller chunking bias; 1D Philox seeding (`ctr=[0,0,0,0]`) caused cross-stream correlation | Clarified (`orbit`/engine required for `>255` words); fixed 2D Philox counter |
| 2026-09-12 | `ra_core_orbit()` multi-cycle output length | Emitted 1 fewer word whenever `rng ≡ 1 (mod 255)` and `rng > 255` | Fixed via explicit `bool` cycle termination |
| 2026-09-12 | Ad-hoc microbenchmark (`singleblock` vs. `orbit`) | Wrote `volatile` anti-DCE sink 255× per call instead of 1×, masking true algorithmic behavior (later superseded by 2026-11-02 DCE audit) | Superseded by materialized-buffer harnesses |
| 2026-08-31 – 2026-09-03 | `30,303.0 MB/s` throughput claim & `MB/s` vs. `MiB/s` | Algebraic `4 / ns_per_word` extrapolation from sub-nanosecond micro-loop; binary MiB/s mislabeled as MB/s | Retracted; standardized on `MiB/s` & `cpb` |
| 2026-08-30 – 2026-08-31 | Historical `benchmarks/comparisons/` baselines | `Philox4x32` used `Philox2x32` multiplier `0xD256D193U`; `ChaCha20` test nonce had big-endian word byte order | Legacy files kept frozen for log consistency; all harnesses from 2026-08-30 use KAT-verified RFC/Random123 constants |

---

## 1. 2026-11-04 — Shootout Fairness Audit, Workload 2b ("Ising") Split, and `ra_core v2` Key-Schedule Healing (`Opsi_A3`)

### 1.1 `addr_hi` Sequential Chaining Failure in `v1` vs. `Opsi_A3` 2D Cross-Pre-Mix in `v2`

**What was wrong:**
- In `ra_core v1` (introduced during 64-bit `(key, addr_hi)` widening on 2026-09-05/06; note that `addr_hi` itself is a 32-bit coordinate paired with 32-bit `key` to form a 64-bit address), `key` and `addr_hi` were hashed independently (`km_k = fmix32(GUARD_M ^ (0x06a0dd9bu * key))` and `km_p = fmix32(GUARD_M2 ^ (0x9e3779b1u * addr_hi))`) and combined by multiplication `step_m = 0x06a0dd9bu * (km_k * km_p)`. Because the product `km_k * km_p` was even 75% of the time, index stride subgroup order collapsed frequently; moreover, `cons = key`, `r = key ^ i`, and index `0` had a universal fixed point `M[0] = L[0] = 0` across all $2^{64}$ seed pairs.
- When tested under **sequential `addr_hi` chaining** with a fixed `key` (`key = 111222`, `addr_hi = 0, 1, 2, ...`, documented in pre-v2 `experiments/2026-9-7_addr-hi-chaining-validation/RESULTS.md`), `v1` **failed PractRand catastrophically at the first 256 MiB checkpoint** (`K=1`: `FPF-14+6/16:all` `R=+47.8, p=2.3e-44` and `DC6` `R=+32.3`; `K=255`: `BCFN(2+0,13-2,T)` `R=+66.1, p=2.1e-33`).
- Intermediate `v2_base` designs that added `k_mix` without cross-mixing before `fmix32` still froze one `fmix32` branch whenever one coordinate was held constant, causing failures on sequential `key`/`addr_hi`, Gray-code, and grid bit-plane sweeps (`2C_seq`, `2D1_gray`, `2E1_grid_bp`, `2F_addr_seq_ctrl`, `4A3_contig_seq`).

**What is right (`ra_core v2 Final`):**
- Promoted on 2026-11-04 as **`Opsi_A3` + `W2_D2_xag_opt` + `eng_v2_E2`** in [`../src/ra_core/c/ra_core.c`](../src/ra_core/c/ra_core.c) and [`../src/ra_core/c/ra_core_engine.c`](../src/ra_core/c/ra_core_engine.c):
  1. **2D Cross-Pre-Mix before `fmix32`:** Both `key` and `addr_hi` feed into both pre-mix channels (`u = modular addition`, `v = rotated XOR`) prior to `fmix32` in `ra_derive_m_A3` and `ra_derive_l_A3`, producing `k_mix = km_k ^ rot32(km_p, 13)` and `cons_0 = k_mix`.
  2. **Odd-forced bijective stride & `+1` offset:** `step = (const * (km_k + km_p)) | 1u` and `(i + 1) * step` eliminate even-stride subgroup collapse and remove the `M[0] = L[0] = 0` fixed point.
- **Empirical resolution:**
  - Sequential `addr_hi` chaining with fixed `key = 0` (`2F_addr_seq_ctrl` at `K=1` and `k255_seq_addr` at `K=255`) passes PractRand cleanly to **2 GiB (`CLEAN`)**.
  - Across the full `ra_core v2` statistical battery, **57/57** non-repeating PractRand regimes pass (`0 FAIL`, `0 SUSP`), canonical orbit passes **8 GiB (`CLEAN`)**, `N=16` parametric scaling passes **15/15 (`4 GiB CLEAN` at `K=14`)**, and Dieharder passes **113 PASS / 1 WEAK / 0 FAIL (`27/27` Good)**.
- **Breaking compatibility note:** `ra_core v2` intentionally breaks bit-exact output and frozen KAT checksum compatibility relative to `v1` (`SINGLEBLOCK_KAT_CHECKSUMS` and `ORBIT_KAT_CHECKSUMS` were regenerated for `v2` in [`../src/ra_core/CHANGELOG.md`](../src/ra_core/CHANGELOG.md)). Also note that period estimates (such as `~2^4969.65` in earlier `ra_prng2` work) are heuristic state-space estimates rather than formal mathematical lower bounds; `v2` reports empirical test batteries and reduced toy-model Brent cycle lengths.
- **Finite-pool note on `2D2_sparse_K1`:** Enumerating all pairs of 32-bit coordinates with Hamming weight $\le 3$ produces only $5{,}489^2 = 30{,}129{,}121$ unique addresses ($114.93\text{ MiB}$ at $K=1$), causing any deterministic PRNG to wrap around and repeat verbatim before PractRand's `128–256 MiB` boundary. Widening the sparse pool to $\text{HW} \le 4$ ($6.40\text{ GiB}$ pool) passes PractRand to **2 GiB (`CLEAN`)** ([`../research/2026-9_ra-core-paper/DRAFT_v2.md`](../research/2026-9_ra-core-paper/DRAFT_v2.md) §3.1).

### 1.2 Shootout Fairness Audit, Work Drift, & Workload 2b ("Ising") Misnomer

**What was wrong:**
- Preliminary v2 shootout runs (`RESULTS_PRE_AUDIT.md`, excluded from the public tree) exhibited three methodological defects documented in [`../experiments/2026-11-04_v2-prng-shootout-and-mc/FAIRNESS_AUDIT.md`](../experiments/2026-11-04_v2-prng-shootout-and-mc/FAIRNESS_AUDIT.md):
  1. **Laptop turbo/load jitter:** 221/343 microbenchmark rows had `median > min * 1.10`, making minimum-time comparisons sensitive to transient frequency spikes.
  2. **Variable draw-count drift:** Stochastic branching in variable-length Monte Carlo workloads (`3A` / `E1`) caused `0.1%–0.5%` differences in total RNG draws between generators.
  3. **Workload 2b misnomer:** A kernel labeled "2D Ising Metropolis" executed a neighbor-free Bernoulli threshold flip (`spin ^= (u < thresh)`) rather than an actual nearest-neighbor spin-coupling Metropolis update.

**What is right:**
- Headline throughput comparisons were standardized on **medians** (rather than minimums) and cross-validated on a fixed-clock `1.995 GHz` Xeon KVM host (`CoV 1.0%–4.8%`) in [`../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md`](../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md).
- Variable-work Monte Carlo kernels were paired with a Dual-Mode **fixed-topology / same-work trace (`3B` / `E2`)** with `0.000%` draw drift in [`../experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md`](../experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md).
- Workload 2b was split into **Workload 2b-1 (`2D Lattice Bernoulli Sweep`)** and **Workload 2b-2 (`True 2D Ising Checkerboard Metropolis` at $T_c$)** in [`../experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md`](../experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md), where `Opsi_A3 Direct` outperforms `Philox4x32-10` by `1.250×–1.276×` at `K=1` while exposing an ad-hoc `WyRand` per-site seeding anomaly ($\langle |M| \rangle = 0.288757$).

**Verified sources:** [`../research/2026-9_ra-core-paper/DRAFT_v2.md`](../research/2026-9_ra-core-paper/DRAFT_v2.md), [`../experiments/2026-11-04_v2-final-battery/RESULTS.md`](../experiments/2026-11-04_v2-final-battery/RESULTS.md), [`../experiments/2026-11-04_v2-prng-shootout-and-mc/FAIRNESS_AUDIT.md`](../experiments/2026-11-04_v2-prng-shootout-and-mc/FAIRNESS_AUDIT.md), [`../experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md`](../experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md), [`../experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md`](../experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md), and [`../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md`](../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md).

---

## 2. 2026-11-02 — GCC `-O3` Dead-Code Elimination (DCE) Audit Invalidating `20,018.1 MB/s` (`0.191 ns/word`) & `rot32` UB Fix

**What was wrong:**
- The headline throughput claim of **`20,018.1 MB/s` (`0.191 ns/word`)** (`19.9×` faster than Philox4x32-10, `6.8×` faster than xoshiro256**) reported for `ra_core_singleblock K=255` on 2026-09-12 (as well as `21,189.03 MiB/s` / `0.180 ns/word` in `bench_param.c` on 2026-09-11 and pre-v2 `README.md` / `bench_release.c` tables) is **retracted**.
- The 2026-09-12 microbenchmark self-correction concluding that a genuine ~17–21× pure-generate speedup separated `singleblock` from `orbit` is likewise **retracted** (see also Entry 6).

**Root cause (Disassembly Proof):**
- On 2026-09-11, `static inline __attribute__((always_inline))` was added to `ra_init_state_singleblock` and `ra_permutation_cycle_singleblock` to match the inlining of `static` baseline functions in `bench_release.c`.
- Because `bench_release.c` called `ra_core_singleblock(key, addr_hi, 255, NULL)` with `raw_stream == NULL` and accumulated only the returned `uint32_t cons`, GCC `-O3` proved that when `raw_stream == NULL` the `singleblock` permutation loop produced zero observable side effects (unlike `orbit`, which mutates `L[i] <-> L[d]`) and `cons` remained invariant. `objdump -d` on `<ra_core_singleblock.part.0>` confirmed GCC emitted an explicit branch skipping all 255 permutation rounds:
  ```asm
  21e8:  48 83 7c 24 28 00    cmpq   $0x0, 0x28(%rsp)   # Check raw_stream == NULL
  21ee:  0f 84 4c 01 00 00    je     2340               # Jump straight to vzeroupper; ret
  ```
- Consequently, `20,018–21,189 MB/s` measured only state initialization divided by 255 with **zero permutation work executed**. (Separately, pre-v2 `experiments/2026-11-02_xag-opt-ra-core/RESULTS.md` also noted that `(n << s) | (n >> (32 - s))` in `rot32` triggered C Undefined Behavior when `s = r & 31 == 0`, fixed across all `v2` sources to `(n << s) | (n >> ((-s) & 31u))`.)

**What is right (Verified Materialized-Buffer Throughput):**
All valid benchmarks write generated words into a materialized buffer (`uint32_t *out`) whose contents are consumed by an anti-DCE checksum sink:
- **True `ra_core v1` (`W0_D0`, materialized buffer, Intel Core i3-1115G4):** `~270.8–499.4 MiB/s` at `K=1` (`114.7 MiB/s` when `noinline`), and **`942.4–1,284.4 MiB/s`** at `K=255` (`singleblock` `1,284.4 MiB/s` vs. `orbit` `1,253.2 MiB/s` in `experiments/2026-11-02_xag-opt-ra-core/RESULTS.md`, matching their true algorithmic cost).
- **True `ra_core v2 Final` (`Opsi_A3` + `W2_D2_xag_opt` + `eng_v2_E2`):**
  - **Machine A (Intel Core i3-1115G4, GCC 16.2.1 `-O3 -march=native`, Stage 4 median-of-7):**
    - Direct Stateless (`ra_core_default_into` / `ra_core_eval_k1`): **`710.2 MiB/s`** (`K=1`), **`1,292.1 MiB/s`** (`K=8`), **`1,596.5 MiB/s`** (`K=255`), **`1,661.3 MiB/s`** (`K=1020`).
    - Stateful OTF (`eng_v2_E2` / `ra_engine_next_u32`): **`570.2 MiB/s`** (`K=1`), **`1,103.3 MiB/s`** (`K=255` Stage 4 median; `1,219.4 MiB/s` Stage 2 best), **`966.6 MiB/s`** (continuous `next_u32`).
    - 8-Way AVX2 SIMD (`ra_core_singleblock_x8_into`): **`1,329.2 MiB/s`** (`K=1`), **`5,648.4 MiB/s`** (`K=8`), **`10,687.9 MiB/s` (`10.44 GiB/s`)** (`K=255`).
  - **Machine B (Intel Xeon KVM, fixed `1.995 GHz`, GCC 14.2.0):**
    - Direct Stateless: **`425.84–461.09 MiB/s`** (`K=1`, `2.66×` faster than `Philox4x32-10` at `173.15 MiB/s`) up to **`917.39–926.70 MiB/s`** (`K=255`, `1.20×` faster than `Philox4x32-10` at `766.49 MiB/s`).
    - Stateful OTF (`eng_v2_E2`): **`352.36 MiB/s`** (`K=1`) to **`583.21 MiB/s`** (`K=255`).
    - 8-Way AVX2 SIMD: **`883.12 MiB/s`** (`K=1`) to **`6,180.36 MiB/s` (`6.04 GiB/s`)** (`K=255`).

**Verified sources:** [`../research/2026-9_ra-core-paper/DRAFT_v2.md`](../research/2026-9_ra-core-paper/DRAFT_v2.md), [`../experiments/2026-11-04_v2-final-battery/RESULTS.md`](../experiments/2026-11-04_v2-final-battery/RESULTS.md), [`../experiments/2026-11-04_v2-engine-opt/RESULTS.md`](../experiments/2026-11-04_v2-engine-opt/RESULTS.md), [`../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md`](../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md), and pre-v2 archive `experiments/2026-11-02_xag-opt-ra-core/RESULTS.md`.

---

## 3. 2026-09-03 to 2026-11-02 — `glibc fmemopen(..., "wb")` Trailing-NUL Slack Clobber

**What was wrong:**
- Under glibc, `fmemopen(buf, size, "wb")` writes a terminating NUL byte (`\0`) upon flush/close; when `size` equals the exact output byte count (`rng * sizeof(uint32_t)`) with zero slack bytes, glibc overwrites the final byte of the buffer. On little-endian x86-64, this silently corrupts the **most significant byte (top 8 bits) of the final `uint32_t` word** (e.g., `0x17ab341f -> 0x00ab341f`).
- **Historical manifestations:**
  1. **2026-09-03 (`scrambler_ra_core_singleblock --mode k1`):** Opening a 4-byte exact `fmemopen` buffer per `K=1` draw zeroed the top 8 bits of every generated word, collapsing Fisher-Yates shuffles (`runs z = -3528`, `r = 0.976`). Temporarily mitigated with `+1` slack byte (`n * sizeof(uint32_t) + 1`), which also regenerated `SINGLEBLOCK_KAT_CHECKSUMS` (pre-v2 `experiments/2026-9-1_production-candidate-battery/ADDENDUM_POST_FIX_STATUS.md`).
  2. **2026-09-12 (`ra_core_orbit_into` / `pip-ra-core` v1 & Asian Option MC Workload B):** In `v1`, `ra_core_orbit_into` wrapped `fmemopen` assuming caller slack whereas `ctypes` buffers and `workload_b_orbit_asian.c` allocated exact `N * sizeof(uint32_t)` arrays, shifting the 10,000th word's MSB (`mean payoff = 5.596228` vs. true `5.596196`).

**What is right (Permanent Removal in 2026-11-02 / `ra_core v2`):**
- `fmemopen` was **completely removed** from all library buffer and engine paths (`ra_core_singleblock_into`, `ra_core_orbit_into`, `ra_core_default_into`, and `RaCoreEngine` in [`../src/ra_core/c/ra_core_engine.c`](../src/ra_core/c/ra_core_engine.c) now write directly into `uint32_t *out` with zero slack bytes required).
- Only the internal CLI self-test helper `checksum_key_core()` in [`../src/ra_core/c/ra_core.c`](../src/ra_core/c/ra_core.c) retains `fmemopen(buf, n * sizeof(uint32_t) + 1, "wb")` inside an internal 256-word stack buffer solely to test the `FILE *` CLI wrappers ([`../src/ra_core/CHANGELOG.md`](../src/ra_core/CHANGELOG.md)).

---

## 4. 2026-09-12 (Resolved 2026-11-02) — Monte Carlo `singleblock` Chunking Bias & Philox4x32-10 1D Counter Artifact

**What was wrong:**
1. **Asian Option Box-Muller Bias with `singleblock` Chunking:** Concatenating 40 independent `singleblock K=255` blocks into a 10,000-step geometric Brownian motion trajectory (`experiments/2026-9-12_ra-core-montecarlo-validation/RESULTS.md`) produced severe payoff bias (`7.505–7.685` vs. reference `~5.60`) and `20.0%–24.4%` cross-stream correlation violations, initially suspected to stem from `addr_hi` collisions.
2. **Philox4x32-10 Cross-Stream Correlation Flags:** In the same 2026-09-12 run, `Philox4x32-10` triggered `8.2%–16.0%` cross-stream correlation flags ($\alpha = 0.05$, nominal expectation `~5%`).

**What is right (2026-11-02 Audit):**
1. Re-running with all four addressing schemes (Weyl key-increment, 2D `addr_hi = path`, flat counter, and `fmix32`-hashed coordinates) yielded identical `~7.51–7.68` bias whenever 40 `singleblock` blocks were concatenated, proving the artifact came from periodic 255-word block resets interacting with Box-Muller pairs—not addressing collisions. Streams exceeding 255 words must use continuous `orbit` (`ra_core_default_into` with `K > 255`) or stateful `RaCoreEngine` (`eng_v2_E2`), both of which match `xoshiro256**` and `PCG32` (`mean payoff = 5.596196`).
2. The Philox4x32-10 correlation flag was caused by seeding all 1,000 streams with an identical counter `ctr = [0, 0, 0, 0]` and linear `key[0] = 0..999`. Switching to the idiomatic Random123 2D counter layout `ctr = [block_idx, path_id, 0, 0]` restored Philox4x32-10 to a nominal `4.0%` violation rate (pre-v2 `experiments/2026-11-02_montecarlo-xag-validation/RESULTS.md` and tracked [`../experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md`](../experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md)).

---

## 5. 2026-09-12 — `ra_core_orbit()` Multi-Cycle Off-by-One Word Truncation

**What was wrong:**
- In `ra_core v1`, `ra_core_orbit()` wrote **1 fewer word** than requested whenever `rng ≡ 1 (mod 255)` and `rng > 255` (`rng ∈ {256, 511, 766, 1021, ...}`). Through `ra_core_orbit_into()` (and `pip-ra-core`), the final buffer slot `out[rng - 1]` remained unwritten/uninitialized.
- **Root cause:** The pointer `*count` was overloaded to signal both "final word requested by user" and "natural completion of a 255-iteration cycle," colliding whenever `count == 256` at the start of a cycle (`*count` reached `1` on step 255 and triggered early exit `if (count <= 1) return cons;`). Existing KAT tests (`n ∈ [1, 255]`) missed all multi-cycle values.

**What is right:**
- `ra_permutation_cycle_orbit` was refactored from `void` to `bool` (`true` = stopped upon reaching user word count; `false` = cycle exhausted naturally, proceed to `ra_reseed_orbit`), documented in pre-v2 `experiments/2026-9-12_orbit-multicycle-offbyone-fix/RESULTS.md`.
- In `ra_core v2`, `ra_orbit_v2_buf` and `ra_core_orbit` emit exact word counts (`0` mismatches across `K ∈ [1, 2000]` in Gate 0 verification).

---

## 6. 2026-09-12 — Microbenchmark Instrumentation Self-Correction (`255×` vs. `1×` `volatile` Sink Writes)

**What was wrong:**
- An ad-hoc microbenchmark (`microbench_orbit_vs_singleblock.c` in pre-v2 `experiments/2026-9-12_orbit-multicycle-offbyone-fix/`) initially argued that the throughput gap between `singleblock` and `orbit` was only ~15% and that `bench_release.c`'s larger ratio was an out-of-order ILP artifact.
- **Root cause:** That microbenchmark forced a `volatile` memory store **255 times per call** (inside the inner loop) rather than **once per call**, saturating register-resident `singleblock` with memory writes while scarcely affecting swap-heavy `orbit`.
- **Subsequent supersession:** Correcting the sink frequency to 1× temporarily appeared to validate `bench_release.c`'s ~17–21× ratio (`20,018 MB/s`), before the **2026-11-02 GCC `-O3` DCE audit (Entry 2)** proved that `bench_release.c` itself was skipping the `singleblock` loop via `raw_stream == NULL`. Both findings establish the `ra_core v2` rule: generators must materialize all outputs into `uint32_t *out`, and the anti-DCE checksum runs over the output buffer after the call.

---

## 7. 2026-08-31 to 2026-09-03 — Algebraic Throughput Extrapolation (`30,303 MB/s`) & Unit Labeling

**What was wrong:**
- Early tables in `experiments/2026-8-31_prng-family-benchmark/RESULTS.md` reported **`30,303.0 MB/s`** for `singleblock` (alongside `16,326.5 MB/s` for `splitmix64` and `4,268.9 MB/s` for `xoshiro256**`).
- Earlier `bench_release` logs also labeled binary `1024 * 1024`-byte rates (`MiB/s`) as `MB/s`.

**Root cause & correction:**
- `30,303.0 MB/s` was an algebraic extrapolation (`4 / 0.132 ns/word`) from a sub-nanosecond in-process timing loop rather than sustained stream throughput. Under real syscall-inclusive `--multistream` (`fwrite` to `/dev/null`), throughput measured `154.8–387.6 MB/s` (`experiments/2026-9-3_orbit-continuous-bench/RESULTS.md`).
- As noted in [`../README.md`](../README.md) and [`../research/2026-9_ra-core-paper/DRAFT_v2.md`](../research/2026-9_ra-core-paper/DRAFT_v2.md), both `30,303 MB/s` and `20,018 MB/s` are **invalidated**; all `ra_core v2` benchmarks report direct materialized-buffer measurements strictly in `MiB/s` ($2^{20}$ bytes/s) and cycles per byte (`cpb`) (see Entry 2).

---

## 8. 2026-08-30 & 2026-08-31 — External Baseline Reference Bugs in `benchmarks/comparisons/`

**What was wrong:**
1. **Philox4x32-10 Wrong Multiplier (`PHILOX_M0`):** [`../benchmarks/comparisons/source/philox.c`](../benchmarks/comparisons/source/philox.c) and [`../benchmarks/comparisons/RNGing_speed/src/philox.c`](../benchmarks/comparisons/RNGing_speed/src/philox.c) define `#define PHILOX_M0 0xD256D193U`. That constant is the **Philox2x32** multiplier (`PHILOX_M2x32_0`), not the official Random123 **Philox4x32** multiplier `PHILOX_M4x32_0 = 0xD2511F53U` (discovered 2026-08-30 in `experiments/2026-8-30_addressable-init-research/RESULTS.md`).
2. **ChaCha20 Test Nonce Byte-Order Swap:** [`../benchmarks/comparisons/RNGing_speed/src/chacha20.c`](../benchmarks/comparisons/RNGing_speed/src/chacha20.c) encodes the RFC 8439 §2.3.2 test nonce as big-endian words `{ 0x00000009, 0x0000004a, 0x00000000 }` instead of little-endian 32-bit words `{ 0x09000000u, 0x4a000000u, 0x00000000u }` (discovered 2026-08-31 in `experiments/2026-8-31_prng-family-benchmark/RESULTS.md`; the `chacha20_block()` permutation itself was unaffected).

**What is right:**
- The legacy comparison files under `benchmarks/comparisons/` are intentionally preserved unmodified so that their historical `.txt` output logs remain reproducible against the exact code that generated them.
- Every benchmark suite from `2026-08-30` onward—including all `ra_core v2` shootout, Monte Carlo, and GPU benchmarks in [`../experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md`](../experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md) and [`../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md`](../experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md)—uses `PHILOX_M4x32_0 = 0xD2511F53U` and RFC 8439 little-endian word order, gated by official Random123 and RFC 8439 Known-Answer Tests (KATs) prior to measurement.
