# Stage 4 Final Report: Production Promotion, Golden KAT Freeze, & Full Statistical Battery for `ra_core v2 Final` (`Opsi_A3` + `eng_v2_E2`)

**Date:** 2026-11-04  
**Directory:** `experiments/2026-11-04_v2-final-battery/`  
**Promoted Production Files:**
- `src/ra_core/c/ra_core.h`
- `src/ra_core/c/ra_core.c`
- `src/ra_core/c/ra_core_engine.h`
- `src/ra_core/c/ra_core_engine.c`
- `src/ra_core/CHANGELOG.md`
- `pip-ra-core/ra_core/_bitgen.c`, `pip-ra-core/setup.py`, `pip-ra-core/tests/test_ra_core.py`, `Makefile`

---

## 1. Executive Summary

`ra_core v2 Final` — combining the **2-Step Unrolled XAG Recurrence Core (`W2_D2_xag_opt`)**, the **Stage 1 Key-Mixing Winner (`Opsi_A3`: 2D Cross-Pre-Mix before `fmix32` + Odd-Forced Bijective Step + `+1` Index Offset)**, and the **Stage 2 Stateful OTF Engine Winner (`eng_v2_E2`: Even-$K$-Aligned 2-Step Coupled OTF Pair Evaluation)** — has been promoted to the canonical production codebase (`src/ra_core/c/` and `pip-ra-core/`) and verified across all five Stage 4 gates with **zero failures**:

1. **Gate 0 — Exhaustive Structural Parity & Unit Test Suite (`0 Mismatches`, `53 / 53 Pytest PASS`):**
   - Cross-verified **`2,720,934` output words** across all 39 canonical test keys and burst lengths $K \in [1, 2000]$:
     - `ra_core_singleblock` (`FILE*`) == `ra_core_singleblock_into` (`uint32_t*`) == 1-step unoptimized reference == `ra_core_orbit_into` prefix (`0 mismatches`).
     - 8-Way AVX2 `ra_core_singleblock_x8_into` / `ra_core_x8` (`u32x8`) == scalar `ra_core_singleblock_into` across all $K \in [1, 255]$ (`0 mismatches`).
     - Stateful `ra_core_engine_t` (`eng_v2_E2`: `next_u32`, `next_u64`, `next_double`, `fill_u32`, `discard`, `position`, `fork`, `save`/`load`) == `ra_core_default_into` across all mode transitions (`mode 0 -> 1 -> 3 -> 2 -> ... -> 4`) and multi-cycle boundaries (`0 mismatches`).
   - `./build/bin/ra_core validate`: **`PASS` (`39 / 39` keys, `0` mismatches on both `singleblock` and `orbit`)**.
   - Full repository `pytest` suite (`tests/` + `pip-ra-core/tests/`): **`53 passed, 0 failed`**.

2. **2D Strict Avalanche Criterion (SAC $64 \times 32$, $N = 100{,}000$ Pairs), Bit Independence Criterion (BIC $32 \times 32$), & Minimum Perturbation Distance:**
   - **2D SAC ($64\text{ input bits} \times 32\text{ output bits} = 2{,}048\text{ cells}$, $N=100{,}000$ pairs):**
     - $K=1$ (`word #0`): $\bar{p} = 0.499959$, $\min p = 0.494200$, $\max p = 0.506620$, $\max |p - 0.5| = 0.006620$, $\chi^2 / \text{dof} = 1.1070$.
     - $K=255$ (`word #127`, mid-Cycle 0): $\bar{p} = 0.500065$, $\min p = 0.495000$, $\max p = 0.505970$, $\max |p - 0.5| = 0.005970$, $\chi^2 / \text{dof} = 0.9723$.
     - $K=255$ (`word #254`, end of Cycle 0): $\bar{p} = 0.499916$, $\min p = 0.494440$, $\max p = 0.505200$, $\max |p - 0.5| = 0.005560$, $\chi^2 / \text{dof} = 1.0113$.
     - $K=256$ (`word #255`, Cycle 1 post-reseed): $\bar{p} = 0.499970$, $\min p = 0.494860$, $\max p = 0.505460$, $\max |p - 0.5| = 0.005460$, $\chi^2 / \text{dof} = 1.0084$.
   - **BIC ($32 \times 32$ Off-Diagonal Output Bit Pair Correlations, $M = 10{,}000$ Base Seeds):**
     - Across all $64$ input bit flips (`key` and `addr_hi`), maximum off-diagonal correlation is $\max |\rho| \le 0.00477$ (`mean` $|\rho| = 0.00098\text{--}0.00103$, matching the theoretical $1/\sqrt{640{,}000} \approx 0.00125$ sampling floor) for both intra-word (`Word #0, #1, #7, #127, #254, #255`) and cross-step (`Lag-1` and `Lag-2`) pairs.
   - **Minimum Perturbation Distance (`1-bit flip`, `+1 increment`, `Gray code step`):**
     - Across all 6 perturbation schemes on `key` and `addr_hi`, mean Hamming distance is **`15.96–16.03` bits** out of 32 at $K=1$ (`min = 4..7`, `max = 25..27`) and **`4,079.5–4,080.4` bits (`0.49993–0.50005`)** out of $8{,}160$ bits across Cycle 0 ($K=255$).

3. **Dieharder Full 114-Subtest (`-a -g 200`) & 27-Test "Good"-Reliability Battery:**
   - **Continuous Stream (`key=42, addr_hi=0`):** **`113 PASS / 1 WEAK / 0 FAIL` (`114` subtests)**; **`27 / 27` Good families PASS (`109 PASS / 1 WEAK / 0 FAIL`)**.
   - **Stateless $K=1$ Weyl Stream (`k1_weyl`):** **`113 PASS / 1 WEAK / 0 FAIL` (`114` subtests)**; **`27 / 27` Good families PASS (`109 PASS / 1 WEAK / 0 FAIL`)**.

4. **PractRand 8 GiB Deep Stream & Combined 57-Regime Multi-Key Summary:**
   - **Canonical Continuous Stream (`key=42, addr_hi=0`, $K=255$ orbit) to 8 GiB:** **`PASS` (`0 FAIL`, completely `CLEAN` with `no anomalies` at `128 MiB`, `256 MiB`, `512 MiB`, `1 GiB`, `2 GiB`, `4 GiB`, and `8 GiB`)**.
   - **Production Sub-Orbit / Multi-Key Boundary Streams (`2 GiB` each):** `k1_weyl` ($K=1$), `k16_seq_key` ($K=16$), `k64_seq_key` ($K=64$), `k255_seq_key` ($K=255$), and `k255_seq_addr` ($K=255$, `key=0` fixed, sequential `addr_hi`) all achieve **`2 GiB CLEAN` (`0 anomalies` across every checkpoint from `16 MiB` to `2 GiB`)**.
   - **Combined 57-Regime Matrix:** Across all **57 non-repeating statistical & adversarial seeding regimes** (42 baseline regimes + 15 adversarial low-entropy/correlated seeding regimes healed by `Opsi_A3`), `ra_core v2 Final` achieves **`57 / 57 PASS` (`0 FAIL`)**.

5. **Production Microbenchmark Sanity Check (`src/ra_core/c/`):**
   - Direct Stateless (`ra_core_default_into` / `ra_core_eval_k1`): **`710.2 MiB/s`** at $K=1$, **`1,292.1 MiB/s`** at $K=8$, **`1,596.5 MiB/s`** at $K=255$, **`1,661.3 MiB/s`** at $K=1020$.
   - Stateful Engine (`eng_v2_E2` `ra_engine_next_u32`): **`570.2 MiB/s`** at $K=1$, **`670.2 MiB/s`** at $K=2$, **`934.0 MiB/s`** at $K=8$, **`1,103.3 MiB/s`** at $K=255$, **`966.6 MiB/s`** on continuous 1-word-at-a-time streaming.
   - 8-Way AVX2 SIMD (`ra_core_singleblock_x8_into`): **`1,329.2 MiB/s`** at $K=1$, **`5,648.4 MiB/s`** at $K=8$, **`10,687.9 MiB/s` (`10.44 GiB/s`)** at $K=255$.

---

## 2. Promoted Production Architecture (`Opsi_A3` + `eng_v2_E2`)

### 2.1 `Opsi_A3` Key-Mixing & Step Derivation (`src/ra_core/c/ra_core.h`)

```c
static inline __attribute__((always_inline)) void ra_derive_m_A3(
    uint32_t key, uint32_t addr_hi, uint32_t *step_m_out, uint32_t *k_mix_out) {
    uint32_t p_addr = 0x9e3779b1u * addr_hi;
    uint32_t u_m = (0x06a0dd9bu * key) + p_addr;
    uint32_t v_m = p_addr ^ rot32(key, 17);
    uint32_t km_k = fmix32(GUARD_M  ^ u_m);
    uint32_t km_p = fmix32(GUARD_M2 ^ v_m);
    *k_mix_out  = km_k ^ rot32(km_p, 13);
    *step_m_out = (0x06a0dd9bu * (km_k + km_p)) | 1u;
}

static inline __attribute__((always_inline)) uint32_t ra_derive_l_A3(
    uint32_t key, uint32_t addr_hi) {
    uint32_t p_addr = 0x85ebca77u * addr_hi;
    uint32_t u_l = (0x9e3779b7u * key) + p_addr;
    uint32_t v_l = p_addr ^ rot32(key, 15);
    uint32_t kl_k = fmix32(GUARD_L  ^ u_l);
    uint32_t kl_p = fmix32(GUARD_L2 ^ v_l);
    return (0x9e3779b7u * (kl_k + kl_p)) | 1u;
}
```
With initial state entries for $i \in [0, 255]$:
$$M[i] = \text{rot32}\big((i + 1) \cdot \text{step\_m},\; k_{\text{mix}} \oplus i\big), \qquad L[i] = \text{rot32}\big((i + 1) \cdot \text{step\_l},\; k_{\text{mix}} \oplus i\big), \qquad \text{cons}_0 = k_{\text{mix}}.$$

---

## 3. Gate 0 Verification & Frozen Golden KAT v2 Final

### 3.1 Gate 0 Bit-Exactness Summary (`data/gate0_kat_and_bench.json`)

| Verification Check | Scope | Words Verified | Mismatches | Status |
| :--- | :--- | :---: | :---: | :---: |
| **`singleblock_into` vs 1-Step Reference** | $39\text{ keys} \times K \in [1, 255]$ | `1,273,740` | `0` | **PASS** |
| **`singleblock_into` vs `orbit_into` Prefix** | $39\text{ keys} \times K \in [1, 255]$ + `eval_k1` | `1,273,779` | `0` | **PASS** |
| **`ra_core` (`FILE*`) vs `*_into` (`uint32_t*`)** | $39\text{ keys} \times K \in \{255, 1000\}$ | `48,945` | `0` | **PASS** |
| **8-Way AVX2 `singleblock_x8_into` vs Scalar** | $32\text{ lanes} \times K \in [1, 255]$ | `1,044,480` | `0` | **PASS** |
| **Stateful `eng_v2_E2` `next_u32` vs `orbit_into`** | $39\text{ keys} \times 2000\text{ words}$ (`mode 0..4`) | `78,000` | `0` | **PASS** |
| **Stateful `eng_v2_E2` `fill_u32` & `discard` & `position`** | $39\text{ keys} \times 28\text{ burst lengths } K \in [1, 2000]$ | `275,730` | `0` | **PASS** |
| **`./build/bin/ra_core validate` (39-Key KAT)** | `SINGLEBLOCK_KAT_CHECKSUMS` & `ORBIT_KAT_CHECKSUMS` | `2,547,480` | `0` | **PASS** |
| **Repository Pytest Suite (`tests/` + `pip-ra-core/tests/`)** | 53 unit & cross-language equivalence tests | — | `0` (`53/53`) | **PASS** |

### 3.2 Frozen Golden KAT v2 Final Vectors (`Opsi_A3` + `eng_v2_E2`)

| Seed Label | `(key, addr_hi)` | Derived `(k_mix, step_m, step_l)` | Words `[0..3]` (`K=1..4`) | Words `[4..7]` (`K=5..8`) | Cycle Boundary Words `[254, 255, 256]` |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **`key0_addr0`** | `(0x00000000, 0x00000000)` | `(0x4ea2c4a4, 0x24416e35, 0x2785f64d)` | `0xc1dbbb12, 0xfbede3b9, 0x93b1e2c8, 0x2067ac16` | `0xc477241f, 0x982a158d, 0xd77dd2e4, 0x89763004` | `0x9297471c, 0xeb5f03e7, 0x8970ad1b` |
| **`key42_addr0`** | `(0x0000002a, 0x00000000)` | `(0xc14755a0, 0x0ec3467d, 0xde211303)` | `0x9533450c, 0x288d3bd2, 0x7e833e19, 0xff96b720` | `0x8cd7be6b, 0xe91267f5, 0xacc46c12, 0xe7271ff4` | `0xb565ef5b, 0x50c62f0b, 0xf7c65da2` |
| **`key1_addr0`** | `(0x00000001, 0x00000000)` | `(0x5bf526f5, 0xafed9981, 0x196abc55)` | `0xd8d07063, 0xd78f6289, 0x180d4a60, 0xe7ddc2c3` | `0x508aa139, 0x11c8ea1e, 0x9d8ce965, 0x37953664` | `0x9b622c40, 0x1bc8ae33, 0x274436e3` |
| **`deadbeef_golden`** | `(0xdeadbeef, 0x9e3779b9)` | `(0xf28e02c1, 0x1b0f916d, 0x9a78d4e5)` | `0x5612413d, 0xc09648c9, 0xd533833f, 0xee2fc833` | `0x09e9cec7, 0x3b89f8af, 0xf08fbf4a, 0x648a24c5` | `0x59bec6de, 0xfcb0cbf3, 0xbc40ce77` |
| **`max_u32_all_ones`** | `(0xffffffff, 0xffffffff)` | `(0x49554d20, 0x1833d849, 0xe8337575)` | `0x4ada0f26, 0x78b8b5de, 0xc07bf57c, 0xb049a08e` | `0xd1fa8279, 0xac4e4438, 0x66c12ecf, 0x90e051f1` | `0xb8e535c0, 0x519cf7b7, 0x34b20631` |

### 3.3 Frozen 64-Bit FNV-1a Stream Checksums (`v2 Final`)

| Seed Label | `(key, addr_hi)` | `FNV1a-64 (K=1)` | `FNV1a-64 (K=8)` | `FNV1a-64 (K=255)` | `FNV1a-64 (K=256)` | `FNV1a-64 (K=1000)` | `FNV1a-64 (K=10000)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`key0_addr0`** | `(0, 0)` | `0x92565b17647cb906` | `0x2d031521ec86b3e1` | `0xcf914a65c5bfc357` | `0x6415e0ce86e813cb` | `0x1cf49952f8feb78e` | `0xc64c15cff8f319a8` |
| **`key42_addr0`** | `(42, 0)` | `0x113204a23f2c5aaa` | `0x14ea7a42ef2b4bc9` | `0xa2a2ab8dee50bccf` | `0x537ab3a8a4916187` | `0x44c2d9808184e9ca` | `0x49bc0a51ddac4419` |
| **`key1_addr0`** | `(1, 0)` | `0x1ff90b90933eae2e` | `0x55712640c10fd485` | `0xc8db90b4a0564280` | `0x1c63f0ebe8e58f04` | `0xa91fb6f8f9fb2cc8` | `0xa02e21917ed13343` |
| **`deadbeef_golden`** | `(0xdeadbeef, 0x9e3779b9)` | `0x0be53b12082495b7` | `0x8c87e99854132245` | `0xa81bf05044ea4b69` | `0x06c36fc913afaa13` | `0x8e5bee39f3f10c3e` | `0x349bbbf26797e412` |
| **`max_u32_all_ones`** | `(0xffffffff, 0xffffffff)` | `0xf502aa641cebbf06` | `0x0264fa527ee0e617` | `0x7fb358921bbf86c3` | `0x5d7c3c982538c232` | `0x8e06d454a2ca4770` | `0x0568ba61cd5b4993` |

---

## 4. Cryptographic Diffusion: 2D SAC, $32 \times 32$ BIC, & Perturbation Distance

Raw data in `data/sac_bic_perturbation.json` (`logs/sac_bic_perturbation.log`).

### 4.1 2D Strict Avalanche Criterion ($64 \text{ Input Bits} \times 32 \text{ Output Bits}$, $N = 100{,}000$ Pairs)

| Word Position | Input Subset | Cells | Mean Flip Prob $\bar{p}$ | Min Cell $p$ | Max Cell $p$ | Max Cell Bias $|p - 0.5|$ | RMS Bias | $\chi^2$ | $\chi^2 / \text{dof}$ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=1$ (`word #0`)** | `key` (`32x32`) | 1,024 | `0.499964` | `0.495040` | `0.506620` | `0.006620` | `0.001639` | `1101.30` | `1.0755` | **PASS** |
| **$K=1$ (`word #0`)** | `addr_hi` (`32x32`) | 1,024 | `0.499954` | `0.494200` | `0.505250` | `0.005800` | `0.001687` | `1165.74` | `1.1384` | **PASS** |
| **$K=1$ (`word #0`)** | **Combined (`64x32`)** | **2,048** | **`0.499959`** | **`0.494200`** | **`0.506620`** | **`0.006620`** | **`0.001663`** | **`2267.04`** | **`1.1070`** | **PASS** |
| **$K=255$ (`word #127`)** | `key` (`32x32`) | 1,024 | `0.500083` | `0.495910` | `0.505970` | `0.005970` | `0.001541` | `972.12` | `0.9493` | **PASS** |
| **$K=255$ (`word #127`)** | `addr_hi` (`32x32`) | 1,024 | `0.500047` | `0.495000` | `0.504830` | `0.005000` | `0.001577` | `1019.09` | `0.9952` | **PASS** |
| **$K=255$ (`word #127`)** | **Combined (`64x32`)** | **2,048** | **`0.500065`** | **`0.495000`** | **`0.505970`** | **`0.005970`** | **`0.001559`** | **`1991.21`** | **`0.9723`** | **PASS** |
| **$K=255$ (`word #254`)** | `key` (`32x32`) | 1,024 | `0.499977` | `0.494820` | `0.505200` | `0.005200` | `0.001609` | `1060.38` | `1.0355` | **PASS** |
| **$K=255$ (`word #254`)** | `addr_hi` (`32x32`) | 1,024 | `0.499855` | `0.494440` | `0.505010` | `0.005560` | `0.001571` | `1010.77` | `0.9871` | **PASS** |
| **$K=255$ (`word #254`)** | **Combined (`64x32`)** | **2,048** | **`0.499916`** | **`0.494440`** | **`0.505200`** | **`0.005560`** | **`0.001590`** | **`2071.15`** | **`1.0113`** | **PASS** |
| **$K=256$ (`word #255`, Cyc 1)** | `key` (`32x32`) | 1,024 | `0.500002` | `0.494870` | `0.504870` | `0.005130` | `0.001575` | `1016.48` | `0.9927` | **PASS** |
| **$K=256$ (`word #255`, Cyc 1)** | `addr_hi` (`32x32`) | 1,024 | `0.499938` | `0.494860` | `0.505460` | `0.005460` | `0.001600` | `1048.64` | `1.0241` | **PASS** |
| **$K=256$ (`word #255`, Cyc 1)** | **Combined (`64x32`)** | **2,048** | **`0.500070`** | **`0.494860`** | **`0.505460`** | **`0.005460`** | **`0.001588`** | **`2065.11`** | **`1.0084`** | **PASS** |

### 4.2 $32 \times 32$ Bit Independence Criterion (BIC, $M = 10{,}000$ Base Seeds)

| Word / Lag Position | `key` Flips Max / Mean $|\rho|$ | `addr_hi` Flips Max / Mean $|\rho|$ | Combined 64b Flips Max / Mean $|\rho|$ | Status |
| :--- | :---: | :---: | :---: | :---: |
| **`Word #0` ($K=1$)** | `0.00640 / 0.00151` | `0.00484 / 0.00144` | **`0.00370 / 0.00100`** | **PASS** |
| **`Word #1` ($K=2$)** | `0.00555 / 0.00145` | `0.00526 / 0.00142` | **`0.00371 / 0.00101`** | **PASS** |
| **`Word #7` ($K=8$)** | `0.00598 / 0.00141` | `0.00547 / 0.00136` | **`0.00432 / 0.00098`** | **PASS** |
| **`Word #127` (Mid-Cycle 0)** | `0.00684 / 0.00139` | `0.00518 / 0.00139` | **`0.00353 / 0.00101`** | **PASS** |
| **`Word #254` (End Cycle 0)** | `0.00614 / 0.00147` | `0.00577 / 0.00142` | **`0.00477 / 0.00102`** | **PASS** |
| **`Word #255` (Start Cycle 1)** | `0.00649 / 0.00147` | `0.00571 / 0.00145` | **`0.00422 / 0.00103`** | **PASS** |
| **`Lag-1 Cross-Step` ($t=0, 127, 254$)** | — | — | **`0.00364..0.00405 / 0.00098`** | **PASS** |
| **`Lag-2 Cross-Step` ($t=0, 127, 254$)** | — | — | **`0.00436..0.00462 / 0.00103`** | **PASS** |

### 4.3 Minimum Perturbation Hamming Distance (`1-Bit Flip`, `+1 Increment`, `Gray Code`)

| Perturbation Mode | Pairs Tested | $K=1$ (`Word #0`, out of 32 bits) Mean `[Min..Max]` | Cycle-0 ($K=255$, out of 8,160 bits) Mean (`Fraction`) `[Min..Max]` | Status |
| :--- | :---: | :---: | :---: | :---: |
| **`key` 1-Bit Flip (`b = 0..31`)** | `25,500` | `16.0138 bits (0.50043)` `[4..27]` | `4080.04 bits (0.500005)` `[3914..4268]` | **PASS** |
| **`addr_hi` 1-Bit Flip (`b = 0..31`)** | `25,500` | `16.0298 bits (0.50093)` `[5..26]` | `4080.14 bits (0.500017)` `[3876..4248]` | **PASS** |
| **`key + 1` Increment** | `10,000` | `16.0138 bits (0.50043)` `[5..25]` | `4079.84 bits (0.499980)` `[3888..4245]` | **PASS** |
| **`addr_hi + 1` Increment** | `10,000` | `15.9628 bits (0.49884)` `[6..26]` | `4079.46 bits (0.499934)` `[3918..4260]` | **PASS** |
| **`key` Gray-Code Step** | `10,000` | `15.9702 bits (0.49907)` `[6..26]` | `4080.40 bits (0.500049)` `[3911..4261]` | **PASS** |
| **`addr_hi` Gray-Code Step** | `10,000` | `16.0177 bits (0.50055)` `[7..27]` | `4080.03 bits (0.500004)` `[3901..4245]` | **PASS** |

---

## 5. Dieharder Full 114-Subtest (`-a -g 200`) & PractRand 8 GiB Deep Stream

Raw data in `data/dieharder_practrand_stage4.json` and logs in `logs/dieharder_*_full114.log` & `logs/practrand_*.log`.

### 5.1 Dieharder Battery Results (`114` Total Subtests / `27` "Good"-Reliability Families)

| Stream Mode | 114-Subtest (`dieharder -a -g 200`) `PASS / WEAK / FAIL` | 27-Test "Good" Battery (`110` Subtests) `PASS / WEAK / FAIL` | 27 "Good" Families (`Clean / w/WEAK / FAIL`) | WEAK Subtest Detail (Expected $\sim 1$ by chance at $\alpha = 0.01$) | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **`continuous_key42` (`key=42, addr=0`)** | **`113 / 1 / 0`** | **`109 / 1 / 0`** | **`26 / 1 / 0`** | `-d 102 (sts_serial|ntup=8): p=0.99787` | **PASS (`0 FAIL`)** |
| **`k1_weyl` ($K=1$ Stateless Weyl)** | **`113 / 1 / 0`** | **`109 / 1 / 0`** | **`26 / 1 / 0`** | `-d 203 (rgb_lagged_sum|ntup=18): p=0.99634` | **PASS (`0 FAIL`)** |

### 5.2 PractRand Deep Stream & Multi-Key Sub-Orbit Boundary Verification

| Stream / Addressing Regime | $K$ | Max Length Tested | Checkpoints `16MB..64MB` | Checkpoints `128MB..2GB` | Checkpoints `4GB..8GB` | Final Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`continuous_key42` (`key=42, addr_hi=0`)** | `255` (Orbit) | **8 GiB** | Minor `unusual` ($p = 1.8\times 10^{-3}$) | **`CLEAN` (`no anomalies`)** | **`CLEAN` (`no anomalies` @ 4GB & 8GB)** | **PASS (8 GiB)** |
| **`k1_weyl` (`key += gold, addr += silver`)** | `1` | **2 GiB** | **`CLEAN` (`no anomalies`)** | **`CLEAN` (`no anomalies`)** | — | **CLEAN (2 GiB)** |
| **`k16_seq_key` (`key = 1, 2, ..., addr = 0`)** | `16` | **2 GiB** | **`CLEAN` (`no anomalies`)** | **`CLEAN` (`no anomalies`)** | — | **CLEAN (2 GiB)** |
| **`k64_seq_key` (`key = 1, 2, ..., addr = 0`)** | `64` | **2 GiB** | **`CLEAN` (`no anomalies`)** | **`CLEAN` (`no anomalies`)** | — | **CLEAN (2 GiB)** |
| **`k255_seq_key` (`key = 1, 2, ..., addr = 0`)** | `255` | **2 GiB** | **`CLEAN` (`no anomalies`)** | **`CLEAN` (`no anomalies`)** | — | **CLEAN (2 GiB)** |
| **`k255_seq_addr` (`key = 0, addr = 0, 1, ...`)** | `255` | **2 GiB** | **`CLEAN` (`no anomalies`)** | **`CLEAN` (`no anomalies`)** | — | **CLEAN (2 GiB)** |

### 5.3 Combined 57-Regime Multi-Regime Matrix Summary (`v2_base` vs `ra_core v2 Final (Opsi_A3)`)

Combining the 42 non-adversarial baseline regimes from the comprehensive battery with the 15 non-repeating adversarial/correlated seeding regimes from Stage 1 (`experiments/2026-11-04_v2-keymix-optA/RESULTS.md`) and the Stage 4 deep-stream boundary suite:

| Category | Regimes Tested | `v2_base` (Pre-Stage 1) | **`ra_core v2 Final (Opsi_A3)`** | Key Improvement |
| :--- | :---: | :---: | :---: | :--- |
| **Continuous Orbit (`1A`–`1D`, up to 8 GiB)** | `8` | `8 / 8 PASS` | **`8 / 8 PASS`** (`8 GiB CLEAN` from 128MB..8GB) | Canonical 1 GiB ENT $\chi^2$ improved from $324.08$ ($p=0.0022$) to **$250.87$ ($p=0.5614$, 32/32 seeds in $[0.01,0.99]$)** |
| **Weyl / Golden-Stepped Multi-Key (`2A`, `2B`, `4A1`)** | `12` | `12 / 12 PASS` | **`12 / 12 PASS`** (`K=1..255` `CLEAN`) | Preserves 100% clean Weyl addressability across all $K \in [1, 255]$ |
| **Sequential / Unit-Increment `key` (`2C_seq`, `4A3_key`)** | `4` | `1 / 4 PASS` (`3 FAIL @ 64MB`) | **`4 / 4 CLEAN (2 GiB)`** | Eliminates all `BCFN`/`FPF` failures at $K \in \{1, 16, 64, 255\}$ |
| **Gray-Code & Extended Sparse $\text{HW}\le 4$ (`2D1`, `2D2`)** | `5` | `4 / 5 PASS` (`1 FAIL @ 64MB`) | **`5 / 5 PASS (2 GiB)`** | Eliminates `2D1_gray_K255` `FAIL @ 64MB` (`2 GiB CLEAN`) |
| **Inverted 2D Grid `(key=blk+1, addr_hi=path)` (`2E1`)** | `2` | `0 / 2 PASS` (`2 FAIL @ 512M–1G`) | **`2 / 2 CLEAN (2 GiB)`** | Both `(key, addr_hi)` orientations now `2 GiB CLEAN` at $K=1$ and $K=255$ |
| **Fixed `key=0` + Sequential/Contiguous `addr_hi` (`2F`, `4A3`)** | `5` | `3 / 5 PASS` (`1 SUSP`, `1 VERY SUSP`) | **`5 / 5 PASS (2 GiB)`** (`4/5 CLEAN`) | Eliminates single-lane constant-`key` correlation via 2D cross-pre-mix `(u_m, v_m)` |
| **Small-State $N \in \{16, 32, 64, 128\}$ Safe-$K$ Floor** | `21` | `20 / 21 PASS` (`1 SUSP` at $N=16, K=14$) | **`21 / 21 PASS (2–4 GiB)`** | Eliminates $N=16, K=14$ `SUSP` (`4 GiB CLEAN`); toy $\bar{\lambda}$ increased by **$1.98\times\text{--}2.34\times$** |
| **Total Non-Repeating Regimes** | **`57`** | **`48 / 57 PASS` (`6 FAIL`, `3 SUSP`)** | **`57 / 57 PASS` (`0 FAIL`, `0 SUSP`)** | **100% pass rate across all 57 regimes** |

---

## 6. Production Microbenchmark Sanity Check (`src/ra_core/c/`)

Measured across 7 interleaved rounds (`-O3 -march=native -mavx2 -std=gnu17 -include stdalign.h -fno-tree-vectorize`, median throughput in `MiB/s`, `data/gate0_kat_and_bench.json`):

| Burst Length $K$ | Direct Stateless `ra_core_default_into` / `ra_core_eval_k1` (`MiB/s`) | Stateful Engine `eng_v2_E2` `ra_engine_next_u32` (`MiB/s`) | 8-Way AVX2 SIMD `ra_core_singleblock_x8_into` (`MiB/s`) |
| :--- | :---: | :---: | :---: |
| **$K = 1$** | **`710.2 MiB/s`** | **`570.2 MiB/s`** | **`1,329.2 MiB/s`** |
| **$K = 2$** | **`764.4 MiB/s`** | **`670.2 MiB/s`** | **`2,478.3 MiB/s`** |
| **$K = 4$** | **`960.8 MiB/s`** | **`822.8 MiB/s`** | **`4,119.9 MiB/s`** |
| **$K = 8$** | **`1,292.1 MiB/s`** | **`934.0 MiB/s`** | **`5,648.4 MiB/s`** |
| **$K = 16$** | **`1,416.4 MiB/s`** | **`1,007.8 MiB/s`** | **`7,578.0 MiB/s`** |
| **$K = 64$** | **`1,476.4 MiB/s`** | **`1,059.1 MiB/s`** | **`9,851.4 MiB/s`** |
| **$K = 255$** | **`1,596.5 MiB/s`** | **`1,103.3 MiB/s`** | **`10,687.9 MiB/s` (`10.44 GiB/s`)** |
| **$K = 256$** | **`1,240.1 MiB/s`** | **`524.5 MiB/s`** | — |
| **$K = 1020$** | **`1,661.3 MiB/s`** | **`857.4 MiB/s`** | — |
| **Continuous `next_u32` ($2.5 \times 10^7$ words)** | — | **`966.6 MiB/s`** | — |
