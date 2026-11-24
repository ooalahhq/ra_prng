# Changelog (`ra_core`)

## 2026-11-04 — Production Promotion of `ra_core v2 Final` (`W2_D2_xag_opt` + `Opsi_A3` + `eng_v2_E2`)

- **`src/ra_core/c/ra_core.h` & `src/ra_core/c/ra_core.c`**:
  - Promoted **`ra_core v2` (`W2_D2_xag_opt`)** recurrence core:
    - **2-Tap Causal Ramp-Up Window (`W2`)**: initializes $o_{255} = M[255] \oplus (M[1] \ll 1)$ with $M[257..262] = 0$, widening to the full 8-tap sliding window by step 7 with zero main-loop branches.
    - **Lag-2 Delayed Swap Feedback (`D2`)**: decouples the data-dependent $L[i] \leftrightarrow L[d_{\text{use}}]$ address lookup by one step ($d_{\text{use}, t} = d_{t-1}$), enabling **2-step loop unrolling** and hiding L1 load-to-use latency.
    - **In-Place Per-Step $L \to M$ Coupling & $O(1)$ End-of-Cycle Guard Fold (`END_CYCLE_M_W2`)**: eliminates the $O(N)$ `ra_hash`/`ra_reseed` scan at cycle boundaries (`+86%` speedup at $K=256$).
  - Promoted **`Opsi_A3` (2D Cross-Pre-Mix Key-Mixing)**:
    - Mixes `(key, addr_hi)` into both $u$ and $v$ **before** the two `fmix32` calls (`u_m = (0x06a0dd9b * key) + p_addr`, `v_m = p_addr ^ rot32(key, 17)`), derives `k_mix = km_k ^ rot32(km_p, 13)`, forces odd bijective step `step_m = (0x06a0dd9b * (km_k + km_p)) | 1u`, and shifts the index offset by `+1` (`(idx + 1u) * step_m`).
    - Zero extra ALU instructions inside the generation loop; passes **57 / 57** non-repeating statistical/adversarial seeding regimes (`0 FAIL`) and full 8 GiB PractRand + 114-subtest Dieharder.
  - Added register-only fast paths for $K=1$ (`ra_eval_k1_word`), $K=2$ (`ra_eval_k2_words`), and $K \in [3, 8]$ (`ra_eval_k3_to_k8_words`).
  - Updated `SINGLEBLOCK_KAT_CHECKSUMS[39]` and `ORBIT_KAT_CHECKSUMS[39]` (`./build/bin/ra_core validate`: `0` mismatches across all 39 canonical test keys).
- **`src/ra_core/c/ra_core_engine.h` & `src/ra_core/c/ra_core_engine.c`**:
  - Promoted **`eng_v2_E2` (Even-$K$-Aligned 2-Step Coupled OTF Pair Evaluation)** for `ra_core_engine_t`:
    - Zero wasted words at $K=1$, $K=2$, all even $K \in \{4, 6, \dots, 254\}$, and $K=255$, achieving `1.13x–1.30x` higher throughput than `eng_v2_old` across $K \in [2, 255]$.
    - Automatic Stage 1 $\to$ Stage 2 promotion at word `#256` with exact $L[i] \leftrightarrow L[d_{\text{use}}]$ and $M[i+7] \mathrel{+}= L[d_{\text{use}}]$ replay.
  - Updated 8-way lockstep AVX2 `ra_core_singleblock_x8_into` / `ra_core_x8` (`u32x8`) to `Opsi_A3` + `W2_D2` with dedicated $K=1$, $K=2$, and $K \le 4$ fast paths (`100%` bit-exact with scalar).

## 2026-11-02 — XAG Algebraic Optimizations & 3-Stage On-The-Fly Streaming Engine (`ra_core_engine_t`)

- **`src/ra_core/c/ra_core.c`**:
  - Added `ra_eval_k1_word(key, addr_hi)` register-only algebraic fast-path for $K=1$.
  - Added XAG Dead-Cone Elimination in `ra_core_orbit` (`rng <= 255` delegates to `ra_core_singleblock`).
- **`src/ra_core/c/ra_core_engine.c` & `src/ra_core/c/ra_core_engine.h`**:
  - Replaced legacy `fmemopen` shims with direct memory-buffer cycle helpers and added 3-Stage OTF `ra_core_engine_t` and `ra_core_singleblock_x8_into`.
