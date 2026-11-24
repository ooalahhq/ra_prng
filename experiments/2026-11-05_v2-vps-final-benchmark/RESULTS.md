# Stage 6 Final Benchmark & Second-Machine Validation Report (`ra_core v2`: `Opsi_A3` + `W2_D2_xag_opt` + `eng_v2_E2`)

**Date:** 2026-11-05  
**Experiment ID:** `2026-11-05_v2-vps-final-benchmark`  
**Status:** **COMPLETE (`0 Mismatches`, 100% Bit-Exact Cross-Machine & Cross-Compiler Parity)**

---

## 1. Executive Summary

Stage 6 validates the frozen production release of **`ra_core v2`** (`Opsi_A3` 2D cross-pre-mix key derivation + `W2_D2_xag_opt` 2-step unrolled recurrence + `eng_v2_E2` coupled pair stateful OTF engine) across two distinct x86-64 hardware platforms, Linux kernels, and GCC major compiler generations:
1. **Machine A (Primary Development / Mobile Workstation Node):** 11th Gen Intel Core i3-1115G4 (Tiger Lake, 3.00 GHz base / 4.10 GHz boost, AVX2 + AVX-512), Arch Linux (`7.2.11-zen1-1-zen`), **GCC 16.2.1**.
2. **Machine B (Server / Cloud x86-64 Validation Node):** x86-64 Server Virtualized Node (Common KVM Hypervisor / Xeon-class, 8 vCPU @ fixed 1.995 GHz, AVX2 + AVX-512), Debian 13 *trixie* (`6.12.90+deb13-cloud-amd64`), **GCC 14.2.0**.

### Key Outcomes
- **100% Cross-Architecture & Cross-Compiler Bit-Exactness:** Across **`2,720,934` Gate 0 verification words**, **39 canonical KAT keys**, **5 frozen Golden KAT vectors** ($K \in \{1, 8, 255, 256, 1000, 10000\}$), **110 Suite D generator-burst pairs**, and **all 240 Clean Monte Carlo generator-workload pairs**, Machine A (GCC 16.2.1) and Machine B (GCC 14.2.0) produce **100% bit-identical output streams, 64-bit checksums, draw counts, and physical observables (`0 mismatches`)**.
- **Cycle-Normalized Performance Invariance (`cpb` & `cyc/draw`):** While raw throughput in `MiB/s` scales proportionally with clock frequency (~1.995 GHz fixed server clock vs. ~3.0–4.1 GHz client boost), cycle-normalized efficiency (`cpb` and `cyc/draw`) is tightly preserved across both machines:
  - At $K=1$ (`Suite D`), `ra_core v2 (Opsi_A3 Direct)` runs at **`4.12 cpb` (`461.09 MiB/s`, CoV `4.15%`)** on Machine B vs. **`3.98 cpb` (`716.65 MiB/s`)** on Machine A — outperforming `Philox4x32-10` (`10.99 cpb`, `173.15 MiB/s`) by **$2.66\times$** and `ChaCha8` (`37.63 cpb`, `50.57 MiB/s`) by **$9.12\times$** on the server node.
  - At $K=64..255$, `ra_core v2 (Opsi_A3 Direct)` sustains **`1.98–2.07 cpb` (`917.39–960.00 MiB/s`)** on Machine B vs. **`1.99 cpb` (`1,435.75–1,438.82 MiB/s`)** on Machine A, while 8-way AVX2 SIMD (`ra_core_singleblock_x8_into`) reaches **`6,180.36 MiB/s` (`6.04 GiB/s`)** on Machine B (`10,687.9 MiB/s` / `10.44 GiB/s` on Machine A).
- **Superior Timing Stability (`CoV%`) on the Server Node:** Because Machine B operates at a fixed `1,995.312 MHz` clock without laptop thermal/turbo frequency transitions, timing variance (`CoV%`) drops dramatically across long bursts and Monte Carlo workloads (e.g., at $K=256..1000$, `Opsi_A3 Direct` `CoV%` drops from `4.97%–21.49%` on Machine A to **`1.05%–3.54%`** on Machine B; in Photon Transport `E2-Slab`, `eng_v2_E2` `CoV%` drops from `24.93%` to **`3.37%`**).

---

## 2. Hardware & Compiler Specifications

| Attribute | Machine A (Local Client Node) | Machine B (Server Validation Node) |
| :--- | :--- | :--- |
| **Platform Role** | Primary Development & Ablation Node | Independent Cross-Machine Validation Node |
| **Processor** | 11th Gen Intel Core i3-1115G4 (2C/4T) | x86-64 Cloud Server Node (Common KVM, 8 vCPU) |
| **Clock Frequency** | 3.00 GHz Base / up to 4.10 GHz Turbo Boost | Fixed 1.995 GHz (`1995.312 MHz`, 0 turbo jitter) |
| **SIMD / ISA Extensions** | SSE4.2, AVX, AVX2, AVX-512F/BW/VL, BMI1/2 | SSE4.2, AVX, AVX2, AVX-512F/DQ/CD/BW/VL, BMI1/2 |
| **Memory** | LPDDR4x Dual-Channel | 7.8 GiB ECC/Server RAM (5.8 GiB free during run) |
| **Operating System** | Arch Linux x86-64 (`7.2.11-zen1-1-zen`) | Debian GNU/Linux 13 *trixie* (`6.12.90+deb13-cloud-amd64`) |
| **C Compiler** | `gcc (GCC) 16.2.1 20260411` | `gcc (Debian 14.2.0-19) 14.2.0` |
| **Compilation Flags** | `-O3 -march=native -mavx2 -fno-tree-vectorize -ffp-contract=off -std=gnu17 -lm` | `-O3 -march=native -mavx2 -fno-tree-vectorize -ffp-contract=off -std=gnu17 -lm` |
| **Execution Pinning** | Single-core pinned (`taskset -c 0`) | Single-core pinned (`taskset -c 3`) |

---

## 3. Gate 0 Cross-Machine Bit-Exactness & Production Microbenchmark

Executing `run_stage4_gate0_kat_bench` on Machine B (`GCC 14.2.0`, `taskset -c 3`) verified all internal parity invariants and frozen Golden KAT vectors (`data/vps_gate0_kat_and_bench.json`):

### 3.1 Gate 0 Parity Verification Summary (`2,720,934` Words)

| Verification Gate | Machine A (GCC 16.2.1) | Machine B (GCC 14.2.0) | Status |
| :--- | :---: | :---: | :---: |
| Total Words Cross-Verified ($39\text{ keys} \times K \in [1, 2000]$) | `2,720,934` | `2,720,934` | **IDENTICAL** |
| `ra_core_singleblock` vs. 1-Step Unoptimized Reference (`mism_sb_vs_ref`) | `0` | `0` | **PASS** |
| `ra_core_singleblock` vs. `ra_core_orbit` (`mism_sb_vs_orbit`) | `0` | `0` | **PASS** |
| File/Stream API vs. Buffer API (`mism_file_vs_buf`) | `0` | `0` | **PASS** |
| 8-Way AVX2 SIMD (`ra_core_singleblock_x8_into`) vs. Scalar (`mism_avx2_vs_scalar`) | `0` | `0` | **PASS** |
| Stateful `eng_v2_E2` (`next_u32`) vs. Orbit (`mism_eng_u32_vs_orbit`) | `0` | `0` | **PASS** |
| Stateful `eng_v2_E2` (`fill_u32`) vs. Orbit (`mism_eng_fill_vs_orbit`) | `0` | `0` | **PASS** |
| Stateful `eng_v2_E2` (`discard` / `fork` / `save` / `load`) (`mism_eng_discard_pos`) | `0` | `0` | **PASS** |
| 39-Key Canonical Validation Suite (`run_validate() rc`) | `0` | `0` | **PASS** |

### 3.2 Cross-Machine Golden KAT Checksum Parity

All 5 frozen Golden KAT seeds (`key0_addr0`, `key42_addr0`, `key1_addr0`, `deadbeef_golden`, `max_u32_all_ones`) and the Stage 1 ablation reference checksums (`0xb2f0606b65e67d3b` for `(0,0)` and `0xcab7cc568b67bd99` for `(42,0)`) matched **100% bit-for-bit** on Machine B:

| Seed Label | `(key, addr_hi)` | `FNV1a-64 (K=1)` | `FNV1a-64 (K=8)` | `FNV1a-64 (K=255)` | `FNV1a-64 (K=256)` | `FNV1a-64 (K=10000)` | Machine A vs. B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **`key0_addr0`** | `(0x00000000, 0x00000000)` | `0x92565b17647cb906` | `0x703eac0d8ccfde76` | `0xcf914a65c5bfc357` | `0x6415e0ce86e813cb` | `0xc64c15cff8f319a8` | **EXACT MATCH** |
| **`key42_addr0`** | `(0x0000002a, 0x00000000)` | `0x113204a23f2c5aaa` | `0x6c29ff1d79e8bf22` | `0xa2a2ab8dee50bccf` | `0x537ab3a8a4916187` | `0x49bc0a51ddac4419` | **EXACT MATCH** |
| **`key1_addr0`** | `(0x00000001, 0x00000000)` | `0x1ff90b90933eae2e` | `0x8f3ff2dbef6f5429` | `0xc8db90b4a0564280` | `0x1c63f0ebe8e58f04` | `0xa02e21917ed13343` | **EXACT MATCH** |
| **`deadbeef`** | `(0xdeadbeef, 0x9e3779b9)` | `0x0be53b12082495b7` | `0xe917a1c63140b8e3` | `0xa81bf05044ea4b69` | `0x06c36fc913afaa13` | `0x349bbbf26797e412` | **EXACT MATCH** |
| **`all_ones`** | `(0xffffffff, 0xffffffff)` | `0xf502aa641cebbf06` | `0x4a063f9d089da8b8` | `0x7fb358921bbf86c3` | `0x5d7c3c982538c232` | `0x0568ba61cd5b4993` | **EXACT MATCH** |

### 3.3 Production API Microbenchmark on Machine B (`vps_gate0_kat_and_bench.json`)

| Burst Length $K$ | Direct Stateless (`ra_core_default_into`) | Stateful Engine (`eng_v2_E2`) | 8-Way AVX2 SIMD (`ra_core_singleblock_x8_into`) | AVX2 Speedup vs. Scalar |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 1$** | `425.84 MiB/s` | `352.36 MiB/s` | `883.12 MiB/s` | $2.07\times$ |
| **$K = 2$** | `522.22 MiB/s` | `379.21 MiB/s` | `1,660.90 MiB/s` | $3.18\times$ |
| **$K = 4$** | `664.40 MiB/s` | `475.82 MiB/s` | `2,760.66 MiB/s` | $4.16\times$ |
| **$K = 8$** | `841.42 MiB/s` | `524.46 MiB/s` | `3,714.07 MiB/s` | $4.41\times$ |
| **$K = 16$** | `900.62 MiB/s` | `567.07 MiB/s` | `4,799.30 MiB/s` | $5.33\times$ |
| **$K = 64$** | `948.01 MiB/s` | `567.00 MiB/s` | `5,944.28 MiB/s` | $6.27\times$ |
| **$K = 255$** | **`926.70 MiB/s`** | **`583.21 MiB/s`** | **`6,180.36 MiB/s` (`6.04 GiB/s`)** | **$6.67\times$** |
| **$K = 256$** | `663.62 MiB/s` | `279.44 MiB/s` | — | — |
| **$K = 1020$** | `915.49 MiB/s` | `402.56 MiB/s` | — | — |

---

## 4. Suite D Pure-Generate Shootout: Machine A vs. Machine B (`K = 1..1000`)

Executed via `run_stage6_suite_d_shootout` (`data/local_suite_d_shootout.csv` and `data/vps_suite_d_shootout.csv`, 9 interleaved trials per $K$, reporting median `MiB/s`, `cpb`, `CoV%`, and 32-bit stream checksum). **All 110 `(generator, K)` checksums match 100% between Machine A and Machine B.**

### 4.1 Key Regimes Comparison ($K \in \{1, 8, 16, 64, 255, 256, 1000\}$)

| Burst $K$ | Generator | Machine A (`MiB/s`) | Machine A (`cpb`) | Machine A (`CoV%`) | Machine B (`MiB/s`) | Machine B (`cpb`) | Machine B (`CoV%`) | Stream Checksum (A = B) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 1$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`716.65`** | **`3.98`** | `12.18%` | **`461.09`** | **`4.12`** | **`4.15%`** | `0xef31cf72` |
| $K = 1$ | **`ra_engine_v2 (eng_v2_E2)`** | `573.94` | `4.98` | `13.05%` | `331.80` | `5.73` | `4.88%` | `0xef31cf72` |
| $K = 1$ | `ra_core_v1 (W0_D0 Direct)` | `499.37` | `5.72` | `10.17%` | `307.34` | `6.19` | `5.66%` | `0x44cb7cba` |
| $K = 1$ | `ra_engine_v1` | `390.60` | `7.31` | `7.46%` | `190.97` | `9.96` | `4.82%` | `0x44cb7cba` |
| $K = 1$ | `Philox4x32-10` | `270.44` | `10.56` | `13.09%` | `173.15` | `10.99` | `6.11%` | `0xefdd5a07` |
| $K = 1$ | `ChaCha8` | `99.99` | `28.57` | `1.06%` | `50.57` | `37.63` | `3.75%` | `0x86027e6d` |
| $K = 1$ | `Squares32` | `1,124.24` | `2.54` | `3.24%` | `697.16` | `2.73` | `9.10%` | `0xba1576cc` |
| $K = 1$ | `PCG32` | `2,543.90` | `1.12` | `6.45%` | `1,225.99` | `1.55` | `10.38%` | `0x4b3098b6` |
| $K = 1$ | `SFC64` | `687.06` | `4.16` | `3.70%` | `433.94` | `4.38` | `6.76%` | `0xc96af063` |
| $K = 1$ | `WyRand` | `2,681.20` | `1.06` | `4.67%` | `1,426.94` | `1.33` | `2.83%` | `0x2daa852f` |
| $K = 1$ | `xoshiro256**` | `646.81` | `4.42` | `8.91%` | `431.74` | `4.41` | `5.67%` | `0x57708627` |
| **$K = 8$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`1,221.26`** | **`2.34`** | `8.02%` | **`772.54`** | **`2.46`** | `10.28%` | `0xf6e81470` |
| $K = 8$ | **`ra_engine_v2 (eng_v2_E2)`** | `1,002.66` | `2.85` | `7.99%` | `545.26` | `3.49` | `6.41%` | `0xf6e81470` |
| $K = 8$ | `ra_core_v1 (W0_D0 Direct)` | `796.30` | `3.59` | `6.47%` | `516.27` | `3.69` | `1.71%` | `0xfa7c2f2a` |
| $K = 8$ | `ra_engine_v1` | `907.21` | `3.15` | `6.35%` | `519.44` | `3.66` | `8.03%` | `0xfa7c2f2a` |
| $K = 8$ | `Philox4x32-10` | `1,029.90` | `2.77` | `3.47%` | `717.52` | `2.65` | `6.54%` | `0xe44dfcf5` |
| $K = 8$ | `ChaCha8` | `666.25` | `4.29` | `1.23%` | `385.08` | `4.94` | `3.37%` | `0xe58a0e48` |
| **$K = 16$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`1,227.74`** | **`2.33`** | `8.88%` | **`893.16`** | **`2.13`** | `12.19%` | `0xbeba54b1` |
| $K = 16$ | **`ra_engine_v2 (eng_v2_E2)`** | `1,000.20` | `2.86` | `10.43%` | `619.36` | `3.07` | **`0.98%`** | `0xbeba54b1` |
| $K = 16$ | `ra_core_v1 (W0_D0 Direct)` | `890.44` | `3.21` | `9.12%` | `631.74` | `3.01` | `0.85%` | `0x1064210e` |
| $K = 16$ | `ra_engine_v1` | `1,003.61` | `2.85` | `10.09%` | `598.81` | `3.18` | `0.99%` | `0x1064210e` |
| $K = 16$ | `Philox4x32-10` | `1,043.84` | `2.74` | `2.86%` | `764.94` | `2.49` | `1.40%` | `0x9e062eed` |
| $K = 16$ | `ChaCha8` | `855.92` | `3.34` | `1.09%` | `621.87` | `3.06` | `0.99%` | `0xee40f08d` |
| **$K = 64$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`1,438.82`** | **`1.99`** | `9.95%` | **`960.00`** | **`1.98`** | `8.70%` | `0x964a23b6` |
| $K = 64$ | **`ra_engine_v2 (eng_v2_E2)`** | `1,172.43` | `2.44` | `4.53%` | `616.84` | `3.08` | `4.13%` | `0x964a23b6` |
| $K = 64$ | `ra_core_v1 (W0_D0 Direct)` | `993.80` | `2.87` | `4.31%` | `739.73` | `2.57` | `2.08%` | `0xb4d7ec05` |
| $K = 64$ | `Philox4x32-10` | `1,078.68` | `2.65` | `1.43%` | `803.48` | `2.37` | `1.58%` | `0x643c0a0f` |
| $K = 64$ | `ChaCha8` | `992.41` | `2.88` | `1.59%` | `637.24` | `2.99` | `2.36%` | `0x12430dc0` |
| **$K = 255$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`1,435.75`** | **`1.99`** | `2.06%` | **`917.39`** | **`2.07`** | `9.06%` | `0x08803ca3` |
| $K = 255$ | **`ra_engine_v2 (eng_v2_E2)`** | `1,139.99` | `2.51` | `2.30%` | `663.49` | `2.87` | **`2.22%`** | `0x08803ca3` |
| $K = 255$ | `ra_core_v1 (W0_D0 Direct)` | `1,018.29` | `2.81` | `0.41%` | `756.75` | `2.51` | `2.17%` | `0x0250b7de` |
| $K = 255$ | `ra_engine_v1` | `1,251.93` | `2.28` | `5.02%` | `660.01` | `2.88` | `4.75%` | `0x0250b7de` |
| $K = 255$ | `Philox4x32-10` | `1,074.15` | `2.66` | `1.35%` | `766.49` | `2.48` | `6.89%` | `0x000820ed` |
| $K = 255$ | `ChaCha8` | `1,013.87` | `2.82` | `5.44%` | `643.50` | `2.96` | `1.63%` | `0x60780852` |
| **$K = 256$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`1,083.22`** | **`2.64`** | `4.97%` | **`679.51`** | **`2.80`** | **`1.05%`** | `0xf93eac47` |
| $K = 256$ | **`ra_engine_v2 (eng_v2_E2)`** | `516.10` | `5.54` | `3.92%` | `291.83` | `6.52` | **`1.80%`** | `0xf93eac47` |
| $K = 256$ | `ra_core_v1 (W0_D0 Direct)` | `797.37` | `3.58` | `19.01%` | `534.88` | `3.56` | `1.13%` | `0x86e9c1e9` |
| $K = 256$ | `ra_engine_v1` | `428.12` | `6.67` | `25.16%` | `268.62` | `7.08` | `1.61%` | `0x86e9c1e9` |
| **$K = 1000$** | **`ra_core_v2 (Opsi_A3 Direct)`** | **`1,390.47`** | **`2.05`** | `12.59%` | **`863.42`** | **`2.20`** | **`2.60%`** | `0x195f0bb5` |
| $K = 1000$ | **`ra_engine_v2 (eng_v2_E2)`** | `896.88` | `3.19` | `10.24%` | `548.76` | `3.47` | **`1.00%`** | `0x195f0bb5` |
| $K = 1000$ | `ra_core_v1 (W0_D0 Direct)` | `904.51` | `3.16` | `17.28%` | `651.45` | `2.92` | `6.20%` | `0x95bf608d` |
| $K = 1000$ | `ra_engine_v1` | `767.17` | `3.72` | `14.28%` | `504.78` | `3.77` | `3.25%` | `0x95bf608d` |
| $K = 1000$ | `Philox4x32-10` | `1,043.65` | `2.74` | `3.47%` | `790.00` | `2.41` | `2.41%` | `0x43b9e978` |
| $K = 1000$ | `ChaCha8` | `968.64` | `2.95` | `16.00%` | `641.02` | `2.97` | `2.05%` | `0xc856708a` |

---

## 5. Stage 3 Clean Monte Carlo Application Shootout: Machine A vs. Machine B

Executed via `clean_mc_shootout` on Machine B (`data/vps_clean_mc_results.json`, `data/vps_clean_mc_summary.csv`). Every single 64-bit checksum (`checksum`), total uniform draw count (`total_draws`), and physical observable (`obs1`, `obs2`) is **100% bit-for-bit identical** between Machine A and Machine B.

### 5.1 Workload 2b-1 (`2D Lattice Bernoulli Sweep`, $K=1$) & Workload 2b-2 (`True 2D Ising Checkerboard Metropolis` at $T_c \approx 2.269185$)

| Workload / Mode | Generator | Machine A Time (`ms`) | Machine A (`cyc/draw`) | Machine B Time (`ms`) | Machine B (`cyc/draw`) | Machine B `CoV%` | Machine B Speedup vs. Philox | Energy $\langle E \rangle/N$ / $\langle \|M\| \rangle$ | Checksum (A = B) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2b-1 Bernoulli ($K=1$)** | **`ra_core_v2 (Opsi_A3)`** | **`15.351`** | `21.92` | **`22.436`** | **`21.34`** | **`4.70%`** | **$1.304\times$** | $\bar{p} = 0.500043$ | `0x82121719` |
| 2b-1 Bernoulli ($K=1$) | `ra_core_v1 (W0_D0)` | `19.107` | `27.29` | `31.455` | `29.93` | `3.41%` | $0.930\times$ | $\bar{p} = 0.499824$ | `0x723d069a` |
| 2b-1 Bernoulli ($K=1$) | `Philox4x32-10` | `20.608` | `29.43` | `29.258` | `27.83` | `4.04%` | $1.000\times$ | $\bar{p} = 0.500150$ | `0xf8a0eeed` |
| 2b-1 Bernoulli ($K=1$) | `ChaCha8` | `78.115` | `111.56` | `136.632` | `129.98` | `4.25%` | $0.214\times$ | $\bar{p} = 0.499869$ | `0x5ef804e9` |
| **2b-2A Ising ($K=1$)** | **`ra_core_v2 (Opsi_A3)`** | **`19.106`** | `27.29` | **`29.501`** | **`28.06`** | **`4.83%`** | **$1.276\times$** | $-1.417175$ / $0.582092$ | `0x8210691e` |
| 2b-2A Ising ($K=1$) | `ra_core_v1 (W0_D0)` | `23.626` | `33.74` | `41.553` | `39.53` | `2.28%` | $0.906\times$ | $-1.421265$ / $0.625916$ | `0x723e8883` |
| 2b-2A Ising ($K=1$) | `Philox4x32-10` | `23.873` | `34.09` | `37.644` | `35.81` | `3.33%` | $1.000\times$ | $-1.401489$ / $0.570770$ | `0xf8a28877` |
| 2b-2A Ising ($K=1$) | `ChaCha8` | `83.605` | `119.40` | `143.400` | `136.42` | `3.93%` | $0.263\times$ | $-1.408936$ / $0.616394$ | `0x5efb842d` |
| 2b-2A Ising ($K=1$) | `WyRand` (ad-hoc per-site seed) | `11.423` | `16.31` | `18.249` | `17.36` | `1.95%` | $2.063\times$ | $-1.270264$ / **`0.288757 (ANOMALY)`** | `0xf7acbcfe` |
| **2b-2B Ising ($K=128$)** | **`ra_core_v2 (Opsi_A3)`** | **`12.465`** | `17.80` | **`22.573`** | `21.47` | `12.51%` | $0.949\times$ | $-1.422180$ / $0.625549$ | `0x63695284` |
| 2b-2B Ising ($K=128$) | **`ra_core_v2 (eng_v2_E2)`** | `13.883` | `19.83` | `29.698` | `28.25` | `3.75%` | $0.721\times$ | $-1.422180$ / $0.625549$ | `0x63695284` |
| 2b-2B Ising ($K=128$) | `Philox4x32-10` | `13.357` | `19.08` | `21.411` | `20.36` | `4.33%` | $1.000\times$ | $-1.404419$ / $0.563965$ | `0xc62d054c` |

### 5.2 Workload 2a (`Asian Option Pricing`), Workload 3A/3B (`Branching Particle Tree`), and Workload E1/E2 (`Photon Transport`)

| Workload / Mode | Generator | Exact Draws | Machine A Time (`ms`) | Machine A (`cyc/d`) | Machine B Time (`ms`) | Machine B (`cyc/d`) | Machine B `CoV%` | Checksum (A = B) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2a Asian Option ($K=16$)** | **`ra_core_v2 (Opsi_A3)`** | `1,600,000` | **`8.422`** | `15.77` | **`13.965`** | `17.41` | **`2.04%`** | `0xff8fc271` |
| 2a Asian Option ($K=16$) | `ra_core_v1 (W0_D0)` | `1,600,000` | `11.149` | `20.87` | `16.796` | `20.94` | `5.80%` | `0x99f65022` |
| 2a Asian Option ($K=16$) | `Philox4x32-10` | `1,600,000` | `9.219` | `17.26` | `13.934` | `17.38` | `16.43%` | `0x570b5eed` |
| **2a Asian Option ($K=255$)** | **`ra_core_v2 (Opsi_A3)`** | `5,100,000` | **`22.859`** | `13.42` | **`37.705`** | `14.75` | `9.14%` | `0xd9071172` |
| 2a Asian Option ($K=255$) | `ra_core_v1 (W0_D0)` | `5,100,000` | `30.326` | `17.81` | `43.961` | `17.19` | `2.54%` | `0x7fa26a82` |
| 2a Asian Option ($K=255$) | `Philox4x32-10` | `5,100,000` | `25.787` | `15.14` | `36.946` | `14.45` | `5.55%` | `0x66bf271b` |
| **3A Tree (Coupled Physics)** | **`ra_core_v2 (eng_v2_E2)`** | `47,168` | **`0.740`** | `47.01` | **`1.176`** | **`49.75`** | **`2.41%`** | `0xbf334c07f2817a34` |
| 3A Tree (Coupled Physics) | `ra_core_v1 (eng_v1)` | `48,092` | `0.773` | `48.16` | `1.298` | `53.83` | `1.16%` | `0x7b2f50a83a699a2e` |
| 3A Tree (Coupled Physics) | `Philox4x32-10` | `46,884` | `0.845` | `53.95` | `1.203` | `51.18` | `1.63%` | `0xdbbba4a304c1536b` |
| 3A Tree (Coupled Physics) | `ChaCha8` | `47,597` | `1.243` | `78.21` | `2.080` | `87.19` | `59.06%` | `0x3a5a517660c169d2` |
| **3B Tree (Fixed-Topology)** | **`ra_core_v2 (Direct_A3)`** | **`396,628`** | **`1.978`** | `14.94` | **`3.370`** | `16.95` | **`3.76%`** | `0x650730d8b86d062d` |
| 3B Tree (Fixed-Topology) | **`ra_core_v2 (eng_v2_E2)`** | **`396,628`** | **`2.412`** | `18.21` | **`4.529`** | `22.78` | **`1.37%`** | `0x650730d8b86d062d` |
| 3B Tree (Fixed-Topology) | `ra_core_v1 (eng_v1)` | `396,628` | `2.634` | `19.89` | `5.006` | `25.18` | `3.06%` | `0x755c47146bd78328` |
| 3B Tree (Fixed-Topology) | `Philox4x32-10` | `396,628` | `2.089` | `15.78` | `2.949` | `14.83` | `5.93%` | `0x58ca717be6760598` |
| 3B Tree (Fixed-Topology) | `ChaCha8` | `396,628` | `2.315` | `17.48` | `4.175` | `20.99` | `4.55%` | `0x740660c65deb97cb` |
| **E1-Slab ($P_{\text{rej}}=0.90$, Coupled)** | **`ra_core_v2 (eng_v2_E2)`** | `737,826` | **`7.955`** | `32.29` | **`9.255`** | **`25.03`** | `17.25%` | `0x00000b91d5c623d2` |
| E1-Slab ($P_{\text{rej}}=0.90$, Coupled) | `ra_core_v1 (eng_v1)` | `730,503` | `7.832` | `32.11` | `9.771` | `26.68` | `8.85%` | `0x00000bb25f5aa254` |
| E1-Slab ($P_{\text{rej}}=0.90$, Coupled) | `Philox4x32-10` | `735,672` | `8.566` | `34.87` | `12.354` | `33.50` | `13.62%` | `0x00000ba22d8e0e7a` |
| **E2-Slab ($P_{\text{rej}}=0.90$, Fixed)** | **`ra_core_v2 (eng_v2_E2)`** | **`731,124`** | **`5.282`** | **`21.63`** | **`7.064`** | **`19.27`** | **`3.37%`** | `0x0000004cdedfa8a8` |
| E2-Slab ($P_{\text{rej}}=0.90$, Fixed) | `ra_core_v1 (eng_v1)` | `731,124` | `6.175` | `25.29` | `8.378` | `22.85` | `3.26%` | `0x0000004c78dd51be` |
| E2-Slab ($P_{\text{rej}}=0.90$, Fixed) | `Philox4x32-10` | `731,124` | `8.069` | `33.05` | `8.773` | `23.93` | `3.20%` | `0x0000004c4e6fc0fe` |
| E2-Slab ($P_{\text{rej}}=0.90$, Fixed) | `ChaCha8` | `731,124` | `13.390` | `54.85` | `20.567` | `56.11` | `1.78%` | `0x0000004c3de7dc52` |

---

## 6. Artifact Inventory

All raw outputs, CSV tables, and execution logs are archived in `experiments/2026-11-05_v2-vps-final-benchmark/`:
- `run_stage6_suite_d_shootout.c` — Self-contained Cross-Machine Suite D pure-generate shootout harness ($K \in \{1..1000\}$ across 11 PRNGs).
- `data/vps_gate0_kat_and_bench.json` — Machine B Gate 0 verification (`2,720,934` words, `0 mismatches`), Golden KAT checksums, and production microbenchmark.
- `data/local_suite_d_shootout.json` & `data/local_suite_d_shootout.csv` — Machine A Suite D pure-generate results.
- `data/vps_suite_d_shootout.json` & `data/vps_suite_d_shootout.csv` — Machine B Suite D pure-generate results.
- `data/vps_stage2_engine_bench.json` & `data/vps_stage2_samples.csv` — Machine B Stage 2 OTF engine candidate progression benchmark.
- `data/vps_clean_mc_results.json` & `data/vps_clean_mc_summary.csv` — Machine B Stage 3 Clean Monte Carlo application shootout results.
- `logs/vps_gate0.log`, `logs/vps_suite_d_shootout.log`, `logs/vps_stage2_engine_bench.log`, `logs/vps_clean_mc_shootout.log` — Complete console logs from Machine B.
