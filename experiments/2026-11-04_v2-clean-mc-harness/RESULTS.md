# Tahap 3: Pembersihan Harness & Metodologi Benchmark Monte Carlo (`ra_core v2`)

**Tanggal:** 2026-11-04  
**Direktori Eksperimen:** `experiments/2026-11-04_v2-clean-mc-harness`  
**Arsitektur Terkunci:**  
- **Direct Stateless API (`ra_core_v2`):** `Opsi_A3` (`ra_singleblock_opt_A3` / `ra_eval_k1_A3`, 3-round Feistel key-mixing $W_0(k), W_1(k)$ + $D_0(k) \text{ OR } 1$) dari Tahap 1.  
- **Stateful OTF Engine (`ra_engine_v2`):** `eng_v2_E2` (`ra_engine_v2_E2_t`, direct in-struct scalar accumulator tanpa overhead frame/orbit-table) dari Tahap 2.  
- **Kompilasi:** `gcc -O3 -march=native -mavx2 -fno-tree-vectorize -ffp-contract=off` (pinned `taskset -c 0`, $N=9$ sampel interleaved dengan warmup penuh).

---

## 1. Ringkasan Eksekutif & Resolusi Audit Metodologi

Dalam `AUDIT_REPORT.md` dan `FAIRNESS_AUDIT.md` pada eksperimen shootout sebelumnya (`2026-11-02_v2-shootout`), teridentifikasi dua kelemahan metodologi yang berpotensi dikritik reviewer:

1. **Stochastic Trajectory Drift (Variasi Beban Kerja Acak Antar-PRNG):**
   - Pada simulasi bercabang (**Group 3: Branching Particle Tree**) dan **Suite E: Photon Transport MC**, keputusan hidup/mati partikel (`u < P_survive`) atau derajat percabangan diambil dari *random draw* masing-masing PRNG. Akibatnya, total simpul/langkah yang dikunjungi berfluktuasi $\sim 0.1\% - 0.5\%$ antar-PRNG, sehingga waktu dinding (*wall-clock*) tidak 100% *apples-to-apples* pada beban kerja yang identik bit-for-bit.
   - **Solusi Tahap 3 (Evaluasi Dual-Mode):**
     - **Mode Coupled Stochastic Physics (`3A`, `E1-Geom`, `E1-Slab`):** Simulasi MC fisik sesungguhnya di mana PRNG mengemudikan nasib partikel, sekaligus memvalidasi bahwa ekspektasi fisik ($\langle L \rangle = \frac{1}{1-P}$, reflektansi difus $R_d$, transmitansi $T_d$, dan rata-rata simpul pohon) konsisten secara statistik di semua PRNG.
     - **Mode Fixed-Topology / Fixed-Trajectory Same-Work (`3B`, `E2-Geom`, `E2-Slab`):** Topologi cabang dan panjang lintasan $L_p$ untuk setiap partikel $p$ ditentukan oleh fungsi jadwal referensi deterministik (`splitmix64`/`fmix32` dari `(particle_id, step)`), sementara PRNG yang diuji dipanggil **tepat $L_p$ kali** untuk membangkitkan payload hamburan/deposisi. Hasilnya: **`total_draws` dan `total_nodes` 100% identik bit-for-bit di seluruh PRNG yang diuji (0.000% drift)**.

2. **Kejujuran Fisika Workload 2b (`Bernoulli Threshold Flip` vs `True 2D Ising Checkerboard Metropolis`):**
   - Pada shootout sebelumnya, Workload 2b dinamai `2D Ising Checkerboard` tetapi kernelnya hanya mengevaluasi ambang batas tunggal `spin ^= (u < threshold)` tanpa interaksi spin tetangga.
   - **Solusi Tahap 3:**
     - **Workload 2b-1 (`2D Lattice Bernoulli Sweep`):** Mempertahankan kernel per-sel $K=1$ non-interaksi dengan nama yang jujur untuk kontinuitas data historis.
     - **Workload 2b-2 (`True 2D Ising Checkerboard Metropolis`):** Implementasi penuh model Ising 2D pada kisi periodik $256 \times 256$ (mask bitwise `& 255`) pada suhu kritis Onsager $T_c = \frac{2}{\ln(1+\sqrt{2})} \approx 2.2691853$, dengan dekomposisi dua sub-kisi *checkerboard* (hitam $(i+j)\bmod 2 = 0$, putih $(i+j)\bmod 2 = 1$), jumlah 4 tetangga terdekat $S_{nn} = s_{i-1,j} + s_{i+1,j} + s_{i,j-1} + s_{i,j+1} \in \{-4, -2, 0, +2, +4\}$, perubahan energi $\Delta E = 2 s_{i,j} S_{nn}$, dan tabel *lookup* Boltzmann integer 32-bit pra-hitung `BOLTZMANN_LUT_TC = {736900509u, 126432030u}` untuk $\Delta E \in \{+4, +8\}$. Diuji pada dua rezim akses:
       * **Mode 2b-2A (Per-Site Stateless Random Access, $K=1$):** Pemanggilan independen per situs `(sweep, site_id)`.
       * **Mode 2b-2B (Row-Sublattice Batched, $K=128$):** Pemanggilan 128 situs sewarna per baris `(sweep, row, parity)`.

---

## 2. Gate 0: Verifikasi Bit-Exactness (`Opsi_A3` vs `eng_v2_E2`)

Sebelum seluruh benchmark dijalankan, Gate 0 memverifikasi ekuivalensi bit-for-bit antara API langsung (`ra_singleblock_opt_A3`), API multi-frame (`ra_orbit_opt_A3`), dan mesin OTF (`ra_engine_v2_E2`):
- **Single-block vs Multi-frame vs OTF Engine ($K \in [1, 255]$, 39 kunci KAT):** `0 mismatches`.
- **Multi-frame vs OTF Engine melewati batas sub-orbit ($K \in \{255, 256, 257, 510, 511, 1024, 2000\}$):** `0 mismatches`.
- **Konsistensi Checksum pada Workload Aplikasi:** Pada Workload 2a ($K=16, 64, 255$), Workload 2b-2B ($K=128$), dan Workload 3B, `ra_core_v2 (Opsi_A3)` dan `ra_core_v2 (eng_v2_E2)` menghasilkan **checksum 64-bit dan nilai observabel fisik yang 100% identik bit-for-bit**.

---

## 3. Bukti Eliminasi Stochastic Trajectory Drift (`total_draws` Bit-Exact Match)

Tabel berikut membuktikan bahwa pada seluruh workload **Fixed-Topology / Fixed-Trajectory (`3B`, `E2-Geom`, `E2-Slab`)**, setiap PRNG mengeksekusi jumlah panggilan acak (`total_draws`) dan jumlah item/simpul (`total_items`) yang **sama persis tanpa deviasi satu pun**:

| Workload (Mode Fixed-Topology / Same-Work) | Parameter | `total_items` (Semua PRNG) | `total_draws` (Semua PRNG) | Max Drift Antar-PRNG | Mean Steps $\langle L \rangle$ | Max Steps $L_{\max}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **3B: Branching Particle Tree** | `Fixed_Topology` | `18,704` nodes | **`396,628`** draws | **0 (0.000%)** | 21.2055 words/node | 32 |
| **E2-Geom: Photon Transport** | $P_{\text{surv}} = 0.50$ | `250,000` photons | **`499,567`** draws | **0 (0.000%)** | 1.998268 | 17 |
| **E2-Geom: Photon Transport** | $P_{\text{surv}} = 0.80$ | `250,000` photons | **`1,250,082`** draws | **0 (0.000%)** | 5.000328 | 73 |
| **E2-Geom: Photon Transport** | $P_{\text{surv}} = 0.90$ | `250,000` photons | **`2,505,465`** draws | **0 (0.000%)** | 10.021860 | 105 |
| **E2-Geom: Photon Transport** | $P_{\text{surv}} = 0.98$ | `250,000` photons | **`12,493,210`** draws | **0 (0.000%)** | 49.972840 | 876 |
| **E2-Slab: Turbid Slab MC** | $P_{\text{rej}} = 0.50$ | `250,000` photons | **`2,663,753`** draws | **0 (0.000%)** | 10.655012 | 182 |
| **E2-Slab: Turbid Slab MC** | $P_{\text{rej}} = 0.80$ | `250,000` photons | **`1,215,420`** draws | **0 (0.000%)** | 4.861680 | 177 |
| **E2-Slab: Turbid Slab MC** | $P_{\text{rej}} = 0.90$ | `250,000` photons | **`731,124`** draws | **0 (0.000%)** | 2.924496 | 177 |
| **E2-Slab: Turbid Slab MC** | $P_{\text{rej}} = 0.98$ | `250,000` photons | **`346,400`** draws | **0 (0.000%)** | 1.385600 | 159 |

---

## 4. Hasil Lengkap per Workload

### 4.1 Workload 2a: Parallel Trajectory Asian Option Pricing MC (Stateless Per-Path)

Setiap lintasan opsi (`path_id`) diinisialisasi secara independen sebagai stream pendek sepanjang $K \in \{16, 64, 255\}$ langkah waktu.

| Rezim | Generator | Tipe | Median (ms) | Mean (ms) | CoV% | MiB/s | ns/draw | cyc/draw | vs Philox | vs `ra_v1` | Option Payoff ($\pm$ SE) | Checksum |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=16$** | **`ra_core_v2 (Opsi_A3)`** | **Counter/Direct** | **8.422** | 8.859 | 8.86% | **724.7** | **5.264** | **15.77** | **1.095×** | **1.324×** | $6.0183 \pm 0.0263$ | `0xff8fc271` |
| ($10^5$ paths, | **`ra_core_v2 (eng_v2_E2)`** | **Counter/OTF** | **9.715** | 9.910 | 6.15% | **628.2** | **6.072** | **18.19** | **0.949×** | **1.148×** | $6.0183 \pm 0.0263$ | `0xff8fc271` |
| $1.6\times 10^6$ | `ra_core_v1 (W0_D0)` | Counter/Direct | 11.149 | 11.199 | 2.41% | 547.5 | 6.968 | 20.87 | 0.827× | 1.000× | $6.0601 \pm 0.0264$ | `0x99f65022` |
| draws) | `Philox4x32-10` | Counter (10-rnd) | 9.219 | 9.398 | 6.64% | 662.0 | 5.762 | 17.26 | 1.000× | 1.209× | $6.0764 \pm 0.0266$ | `0x570b5eed` |
| | `Squares32` | Counter (light) | 6.434 | 6.473 | 2.03% | 948.7 | 4.021 | 12.04 | 1.433× | 1.733× | $6.0566 \pm 0.0265$ | `0x6076c780` |
| | `WyRand` | Stateful (ad-hoc) | 5.351 | 5.424 | 2.62% | 1140.7 | 3.344 | 10.02 | 1.723× | 2.084× | $6.0675 \pm 0.0264$ | `0x22797f58` |
| | `SplitMix64` | Stateful (ad-hoc) | 6.012 | 6.123 | 4.96% | 1015.2 | 3.757 | 11.25 | 1.534× | 1.854× | $6.0386 \pm 0.0265$ | `0xab382e1c` |
| | `PCG32` | Stateful (ad-hoc) | 5.923 | 5.981 | 4.48% | 1030.5 | 3.702 | 11.09 | 1.557× | 1.882× | $6.0308 \pm 0.0264$ | `0x30b3a18e` |
| | `SFC64` | Stateful (ad-hoc) | 5.942 | 6.042 | 4.90% | 1027.2 | 3.714 | 11.12 | 1.552× | 1.876× | $6.0550 \pm 0.0264$ | `0x01e8e114` |
| | `xoshiro256**` | Stateful (SM64) | 6.377 | 6.507 | 4.86% | 957.2 | 3.985 | 11.94 | 1.446× | 1.748× | $5.9704 \pm 0.0262$ | `0xe48544d4` |
| **$K=64$** | **`ra_core_v2 (Opsi_A3)`** | **Counter/Direct** | **15.588** | 16.197 | 12.86% | **783.1** | **4.871** | **14.59** | **1.073×** | **1.260×** | $5.8027 \pm 0.0359$ | `0x312ba7e3` |
| ($5\times 10^4$ paths, | **`ra_core_v2 (eng_v2_E2)`** | **Counter/OTF** | **16.979** | 17.065 | 2.70% | **719.0** | **5.306** | **15.89** | **0.985×** | **1.157×** | $5.8027 \pm 0.0359$ | `0x312ba7e3` |
| $3.2\times 10^6$ | `ra_core_v1 (W0_D0)` | Counter/Direct | 19.644 | 19.775 | 1.55% | 621.4 | 6.139 | 18.39 | 0.851× | 1.000× | $5.8800 \pm 0.0363$ | `0x1ae051f4` |
| draws) | `Philox4x32-10` | Counter (10-rnd) | 16.723 | 16.811 | 2.33% | 729.9 | 5.226 | 15.65 | 1.000× | 1.175× | $5.9473 \pm 0.0364$ | `0xa24692ca` |
| | `Squares32` | Counter (light) | 11.438 | 11.524 | 3.38% | 1067.3 | 3.574 | 10.71 | 1.462× | 1.717× | $5.8302 \pm 0.0361$ | `0xf61f4db8` |
| | `WyRand` | Stateful (ad-hoc) | 9.874 | 9.972 | 2.67% | 1236.3 | 3.086 | 9.24 | 1.694× | 1.989× | $5.7882 \pm 0.0360$ | `0x43b93712` |
| | `SplitMix64` | Stateful (ad-hoc) | 11.003 | 10.973 | 2.34% | 1109.4 | 3.439 | 10.30 | 1.520× | 1.785× | $5.7636 \pm 0.0359$ | `0x236c55ee` |
| | `PCG32` | Stateful (ad-hoc) | 11.056 | 11.472 | 10.80% | 1104.1 | 3.455 | 10.35 | 1.513× | 1.777× | $5.7929 \pm 0.0360$ | `0xe26dbc1e` |
| | `SFC64` | Stateful (ad-hoc) | 10.302 | 10.430 | 2.77% | 1184.9 | 3.219 | 9.64 | 1.623× | 1.907× | $5.8760 \pm 0.0364$ | `0xd7e7dcea` |
| | `xoshiro256**` | Stateful (SM64) | 11.141 | 11.275 | 4.99% | 1095.7 | 3.481 | 10.43 | 1.501× | 1.763× | $5.8075 \pm 0.0359$ | `0xd00ee055` |
| **$K=255$** | **`ra_core_v2 (Opsi_A3)`** | **Counter/Direct** | **22.859** | 24.599 | 14.86% | **851.1** | **4.482** | **13.42** | **1.128×** | **1.327×** | $5.8586 \pm 0.0569$ | `0xd9071172` |
| ($2\times 10^4$ paths, | **`ra_core_v2 (eng_v2_E2)`** | **Counter/OTF** | **27.059** | 27.982 | 7.85% | **719.0** | **5.306** | **15.89** | **0.953×** | **1.121×** | $5.8586 \pm 0.0569$ | `0xd9071172` |
| $5.1\times 10^6$ | `ra_core_v1 (W0_D0)` | Counter/Direct | 30.325 | 30.344 | 2.56% | 641.5 | 5.946 | 17.81 | 0.850× | 1.000× | $5.8036 \pm 0.0571$ | `0x7fa26a82` |
| draws) | `Philox4x32-10` | Counter (10-rnd) | 25.787 | 26.349 | 4.93% | 754.5 | 5.056 | 15.14 | 1.000× | 1.176× | $5.8158 \pm 0.0569$ | `0x66bf271b` |
| | `Squares32` | Counter (light) | 17.269 | 17.226 | 1.72% | 1126.6 | 3.386 | 10.14 | 1.493× | 1.756× | $5.6875 \pm 0.0562$ | `0xcd604b2e` |
| | `WyRand` | Stateful (ad-hoc) | 14.974 | 15.213 | 3.67% | 1299.3 | 2.936 | 8.79 | 1.722× | 2.025× | $5.8037 \pm 0.0569$ | `0x4f16b01d` |
| | `SplitMix64` | Stateful (ad-hoc) | 16.753 | 17.018 | 6.50% | 1161.3 | 3.285 | 9.84 | 1.539× | 1.810× | $5.7590 \pm 0.0566$ | `0x1e6206b8` |
| | `PCG32` | Stateful (ad-hoc) | 16.680 | 16.926 | 4.54% | 1166.4 | 3.271 | 9.80 | 1.546× | 1.818× | $5.8159 \pm 0.0568$ | `0x131c142d` |
| | `SFC64` | Stateful (ad-hoc) | 15.350 | 15.640 | 4.58% | 1267.4 | 3.010 | 9.01 | 1.680× | 1.976× | $5.7733 \pm 0.0564$ | `0x10b4d7c4` |
| | `xoshiro256**` | Stateful (SM64) | 16.729 | 16.824 | 3.94% | 1163.0 | 3.280 | 9.82 | 1.541× | 1.813× | $5.8797 \pm 0.0567$ | `0xcf46a80a` |

---

### 4.2 Workload 2b-1 vs 2b-2: Lattice Bernoulli Sweep vs True 2D Ising Checkerboard Metropolis ($256 \times 256$, 32 Sweeps = $2,097,152$ Site Updates)

#### A. Workload 2b-1: `2D Lattice Bernoulli Sweep` (Per-Cell Stateless $K=1$, Legacy Non-Interacting Flip)
Pada rezim $K=1$ per sel tanpa interaksi tetangga, `ra_core_v2 (Opsi_A3)` mencapai **15.351 ms** (**521.1 MiB/s**, **7.32 ns/draw**), mengalahkan `Philox4x32-10` (**20.608 ms**, **1.342× speedup**) dan `ra_core_v1` (**19.107 ms**, **1.245× speedup**).

| Generator | Median (ms) | Mean (ms) | CoV% | MiB/s | ns/draw | cyc/draw | Speedup vs Philox | Speedup vs `ra_v1` | Flip Rate | Final $\|M\|$ | Checksum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ra_core_v2 (Opsi_A3)`** | **15.351** | 16.435 | 15.15% | **521.1** | **7.320** | **21.92** | **1.342×** | **1.245×** | 0.171283 | 0.0104 | `0x82121719` |
| `ra_core_v1 (W0_D0)` | 19.107 | 19.942 | 8.12% | 418.7 | 9.111 | 27.29 | 1.079× | 1.000× | 0.171625 | 0.0018 | `0x723d069a` |
| `Philox4x32-10` | 20.608 | 20.747 | 1.37% | 388.2 | 9.827 | 29.43 | 1.000× | 0.927× | 0.171746 | 0.0034 | `0xf8a0eeed` |
| `Squares32` | 7.273 | 7.603 | 9.48% | 1100.0 | 3.468 | 10.39 | 2.834× | 2.627× | 0.171819 | 0.0014 | `0x8131f5e9` |
| `WyRand` | 5.006 | 5.212 | 7.73% | 1598.0 | 2.387 | 7.15 | 4.117× | 3.817× | 0.171367 | 0.0081 | `0xf7a1dc87` |
| `SplitMix64` | 6.656 | 6.700 | 1.81% | 1202.0 | 3.174 | 9.51 | 3.096× | 2.871× | 0.171941 | 0.0075 | `0x32bea49f` |
| `PCG32` | 5.375 | 5.394 | 0.85% | 1488.4 | 2.563 | 7.68 | 3.834× | 3.555× | 0.171406 | 0.0002 | `0x12c7da0a` |
| `SFC64` | 12.496 | 12.519 | 1.17% | 640.2 | 5.959 | 17.85 | 1.649× | 1.529× | 0.171767 | 0.0015 | `0x3e217285` |
| `xoshiro256**` | 7.475 | 7.528 | 1.29% | 1070.3 | 3.564 | 10.68 | 2.757× | 2.556× | 0.171779 | 0.0074 | `0xd22fc1cf` |
| `ChaCha8` | 78.115 | 80.268 | 7.74% | 102.4 | 37.248 | 111.56 | 0.264× | 0.245× | 0.171391 | 0.0014 | `0x5ef804e9` |

#### B. Workload 2b-2A & 2b-2B: `True 2D Ising Checkerboard Metropolis` ($256 \times 256$ Periodic Lattice, $T = T_c \approx 2.269185$)
Dengan evaluasi tetangga 4-arah penuh $\Delta E = 2 s_{i,j} (s_{i-1,j} + s_{i+1,j} + s_{i,j-1} + s_{i,j+1})$ dan kriteria Metropolis pada suhu kritis $T_c$:
- **Mode 2b-2A (Per-Site Stateless $K=1$):** `ra_core_v2 (Opsi_A3)` selesai dalam **19.106 ms** (**418.7 MiB/s**, **9.11 ns/site**), **1.250× lebih cepat dari `Philox4x32-10`** (23.873 ms) dan **1.237× lebih cepat dari `ra_core_v1`** (23.626 ms).
- **Mode 2b-2B (Row-Sublattice Batched $K=128$):** Ketika 128 situs sewarna pada setiap baris dievaluasi sebagai stream $K=128$, waktu `ra_core_v2 (Opsi_A3)` turun menjadi **12.465 ms** (**641.8 MiB/s**, **5.94 ns/site**, **1.53× lebih cepat dibanding mode $K=1$**), unggul **1.072× atas `Philox4x32-10`** (13.357 ms) dan **1.167× atas `ra_core_v1`** (14.544 ms). Engine OTF `eng_v2_E2` mencapai **13.883 ms** dengan checksum & fisika **100% identik bit-for-bit** (`0x63695284`).
- **Validasi Fisika Ising 2D:** Seluruh PRNG berkualitas tinggi membawa kisi dari kondisi awal *checkerboard* ($\langle E \rangle / N = +0.50$ setelah paritas pertama) menuju rezim feromagnetik kritis dengan energi per spin $\langle E \rangle / N \approx -1.40 \dots -1.44$, magnetisasi $\langle |M| \rangle \approx 0.57 \dots 0.67$, dan laju penerimaan Metropolis $\approx 0.214 \dots 0.223$.

| Mode Akses | Generator | Median (ms) | Mean (ms) | CoV% | MiB/s | ns/site | cyc/site | vs Philox | vs `ra_v1` | Energy/Spin $\langle E \rangle/N$ | Mag $\langle\|M\|\rangle$ | Accept Rate | Checksum |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2b-2A: Per-Site** | **`ra_core_v2 (Opsi_A3)`** | **19.106** | 19.611 | 6.52% | **418.7** | **9.111** | **27.29** | **1.250×** | **1.237×** | $-1.4172$ | $0.5821$ | $0.2194$ | `0x8210691e` |
| **Stateless ($K=1$)** | `ra_core_v1 (W0_D0)` | 23.626 | 24.061 | 5.45% | 338.6 | 11.266 | 33.74 | 1.010× | 1.000× | $-1.4213$ | $0.6259$ | $0.2173$ | `0x723e8883` |
| | `Philox4x32-10` | 23.873 | 24.251 | 6.14% | 335.1 | 11.384 | 34.09 | 1.000× | 0.990× | $-1.4015$ | $0.5708$ | $0.2217$ | `0xf8a28877` |
| | `Squares32` | 10.029 | 10.339 | 5.60% | 797.7 | 4.782 | 14.32 | 2.380× | 2.356× | $-1.4309$ | $0.6073$ | $0.2170$ | `0x81327bf6` |
| | `WyRand` | 11.423 | 11.866 | 9.13% | 700.3 | 5.447 | 16.31 | 2.090× | 2.068× | $-1.2703$ | $0.2888$ | $0.2534$ | `0xf7acbcfe` |
| | `SplitMix64` | 10.138 | 10.661 | 14.78% | 789.1 | 4.834 | 14.48 | 2.355× | 2.330× | $-1.3895$ | $0.5640$ | $0.2263$ | `0x32bc19b9` |
| | `PCG32` | 9.815 | 10.154 | 9.55% | 815.0 | 4.680 | 14.02 | 2.432× | 2.407× | $-1.4295$ | $0.6165$ | $0.2180$ | `0x12c45c1f` |
| | `SFC64` | 16.078 | 16.631 | 8.38% | 497.6 | 7.667 | 22.96 | 1.485× | 1.469× | $-1.4234$ | $0.6528$ | $0.2140$ | `0x3e22d56e` |
| | `xoshiro256**` | 10.929 | 11.026 | 2.97% | 732.0 | 5.211 | 15.61 | 2.184× | 2.162× | $-1.4416$ | $0.6732$ | $0.2145$ | `0xd22c6230` |
| | `ChaCha8` | 83.605 | 85.555 | 5.59% | 95.7 | 39.866 | 119.40 | 0.286× | 0.283× | $-1.4089$ | $0.6164$ | $0.2184$ | `0x5efb842d` |
| **2b-2B: Row-Batched** | **`ra_core_v2 (Opsi_A3)`** | **12.465** | 12.546 | 1.61% | **641.8** | **5.944** | **17.80** | **1.072×** | **1.167×** | $-1.4222$ | $0.6255$ | $0.2158$ | `0x63695284` |
| **($K=128$/half-row)** | **`ra_core_v2 (eng_v2_E2)`** | **13.883** | 14.067 | 3.42% | **576.2** | **6.620** | **19.83** | **0.962×** | **1.048×** | $-1.4222$ | $0.6255$ | $0.2158$ | `0x63695284` |
| | `ra_core_v1 (W0_D0)` | 14.544 | 15.160 | 12.12% | 550.1 | 6.935 | 20.77 | 0.918× | 1.000× | $-1.4209$ | $0.6156$ | $0.2192$ | `0x1c0c667f` |
| | `Philox4x32-10` | 13.357 | 14.103 | 14.24% | 598.9 | 6.369 | 19.08 | 1.000× | 1.089× | $-1.4044$ | $0.5640$ | $0.2235$ | `0xc62d054c` |
| | `Squares32` | 9.914 | 10.468 | 15.08% | 807.0 | 4.727 | 14.16 | 1.347× | 1.467× | $-1.4283$ | $0.6496$ | $0.2176$ | `0xd24e5b8a` |
| | `WyRand` | 9.320 | 9.656 | 11.18% | 858.4 | 4.444 | 13.31 | 1.433× | 1.561× | $-1.3654$ | $0.5619$ | $0.2248$ | `0xdf65becf` |
| | `SplitMix64` | 10.219 | 10.450 | 5.20% | 782.9 | 4.873 | 14.59 | 1.307× | 1.423× | $-1.4251$ | $0.6418$ | $0.2173$ | `0xe9f0ba57` |
| | `PCG32` | 10.723 | 10.692 | 1.21% | 746.0 | 5.113 | 15.31 | 1.246× | 1.356× | $-1.4133$ | $0.5789$ | $0.2230$ | `0x410a1f78` |
| | `SFC64` | 9.606 | 9.630 | 1.69% | 832.8 | 4.581 | 13.72 | 1.390× | 1.514× | $-1.4307$ | $0.6269$ | $0.2192$ | `0xd61ab7eb` |
| | `xoshiro256**` | 9.859 | 9.896 | 1.59% | 811.4 | 4.701 | 14.08 | 1.355× | 1.475× | $-1.4040$ | $0.5823$ | $0.2185$ | `0x1c401b95` |

> **Catatan Menarik pada Mode 2b-2A (`WyRand`):** Ketika `WyRand` dipanggil secara *stateless ad-hoc* per situs `(sweep, site_id)` tanpa key-mixing multi-putaran (`seed = key ^ (addr_hi << 32)` ditambah satu langkah `0x9E3779B97F4A7C15`), korelasi antar-situs tetangga pada kisi 2D menyebabkan magnetisasi tertahan di $\langle |M| \rangle \approx 0.289$ dan energi $-1.270$, berbeda nyata dari distribusi ekuilibrium Ising $T_c$ ($\langle |M| \rangle \approx 0.58-0.67$, $\langle E \rangle / N \approx -1.42$) yang dicapai oleh `ra_core_v2 (Opsi_A3)`, `Philox4x32-10`, dan PRNG lainnya. Ini menjadi bukti empiris langsung mengapa *key-mixing* `Opsi_A3` yang kuat sangat krusial pada akses acak kisi spasial $K=1$.

---

### 4.3 Workload 3A & 3B: Branching Particle Tree (Coupled Stochastic vs Fixed-Topology Same-Work)

- **Mode 3A (`Coupled_Stochastic`):** Setiap simpul menarik 1 kata kontrol (`w0`) dari PRNG sendiri untuk menentukan `n_children` (1..3) dan `k_sim` (1..4). Terlihat fluktuasi stokastik alami: total simpul berkisar antara `21,674` (`Squares32`) hingga `22,400` (`ra_core_v1`), dan total draws berkisar `46,539` hingga `48,366` ($\pm 1.9\%$ rentang antar-PRNG). Pada mode ini, `ra_core_v2 (eng_v2_E2)` mencapai **0.740 ms** (**1.141× lebih cepat dari Philox** 0.845 ms).
- **Mode 3B (`Fixed_Topology_SameWork`):** Fungsi jadwal deterministik `tree_schedule_ctrl(node_id, depth)` menetapkan struktur pohon yang **100% identik** (`18,704` simpul, tepat **`396,628` draws**, rata-rata $21.2055$ kata/simpul) untuk semua PRNG.
  - `ra_core_v2 (Direct_A3)` menyelesaikan pohon dalam **1.978 ms** (**765.0 MiB/s**, **4.99 ns/draw**), **1.056× lebih cepat dari `Philox4x32-10`** (2.089 ms) dan **1.331× lebih cepat dari `ra_core_v1`** (2.634 ms).
  - `ra_core_v2 (eng_v2_E2)` menyelesaikan pohon yang sama dalam **2.412 ms** (**627.4 MiB/s**), **1.092× lebih cepat dari `ra_core_v1 (eng_v1)`**, dengan checksum 64-bit yang **100% identik bit-for-bit** dengan `Direct_A3` (`0x650730d8b86d062d`).

| Mode | Generator | Total Nodes | Total Draws | Median (ms) | Mean (ms) | CoV% | MiB/s | ns/draw | cyc/draw | Speedup vs Philox | Checksum |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3A: Coupled** | **`ra_core_v2 (eng_v2_E2)`** | 21,980 | 47,168 | **0.740** | 0.743 | 1.24% | **243.0** | **15.696** | **47.01** | **1.141×** | `0xbf334c07f2817a34` |
| **Stochastic** | `ra_core_v1 (eng_v1)` | 22,400 | 48,092 | 0.773 | 0.778 | 1.68% | 237.2 | 16.079 | 48.16 | 1.092× | `0x7b2f50a83a699a2e` |
| (Drift $\sim 1.9\%$) | `Philox4x32-10` | 21,860 | 46,884 | 0.845 | 0.855 | 3.20% | 211.8 | 18.015 | 53.96 | 1.000× | `0xdbbba4a304c1536b` |
| | `Squares32` | 21,674 | 46,539 | 0.641 | 0.642 | 1.09% | 276.9 | 13.775 | 41.25 | 1.318× | `0xeb63983555e77575` |
| | `WyRand` | 22,320 | 48,366 | 0.488 | 0.501 | 5.02% | 377.8 | 10.097 | 30.24 | 1.730× | `0x4a2f59e08e7a6cf3` |
| | `SplitMix64` | 21,772 | 47,088 | 0.517 | 0.527 | 5.09% | 347.6 | 10.974 | 32.87 | 1.635× | `0x4cf3f32e8e3d3e4c` |
| | `PCG32` | 21,894 | 48,215 | 0.495 | 0.499 | 2.83% | 371.3 | 10.274 | 30.77 | 1.705× | `0xd30905b7df4cfab2` |
| | `SFC64` | 22,130 | 47,492 | 0.575 | 0.579 | 2.40% | 315.2 | 12.101 | 36.24 | 1.470× | `0xc64bd65b5aef371c` |
| | `xoshiro256**` | 22,140 | 47,791 | 0.583 | 0.595 | 4.96% | 312.5 | 12.206 | 36.56 | 1.448× | `0x63c2bf7942b8a529` |
| | `ChaCha8` | 21,806 | 47,597 | 1.243 | 1.245 | 1.35% | 146.1 | 26.113 | 78.21 | 0.680× | `0x3a5a517660c169d2` |
| **3B: Fixed** | **`ra_core_v2 (Direct_A3)`** | **18,704** | **396,628** | **1.978** | 2.040 | 8.06% | **765.0** | **4.987** | **14.94** | **1.056×** | `0x650730d8b86d062d` |
| **Topology** | **`ra_core_v2 (eng_v2_E2)`** | **18,704** | **396,628** | **2.412** | 2.415 | 2.08% | **627.4** | **6.081** | **18.21** | **0.866×** | `0x650730d8b86d062d` |
| (**0.000% Drift**, | `ra_core_v1 (eng_v1)` | **18,704** | **396,628** | 2.634 | 2.645 | 1.16% | 574.5 | 6.640 | 19.89 | 0.793× | `0x755c47146bd78328` |
| Exact Same | `Philox4x32-10` | **18,704** | **396,628** | 2.089 | 2.103 | 1.76% | 724.1 | 5.268 | 15.78 | 1.000× | `0x58ca717be6760598` |
| Work) | `Squares32` | **18,704** | **396,628** | 1.141 | 1.146 | 0.79% | 1326.2 | 2.876 | 8.62 | 1.831× | `0xbbaaa34ee4d7a5b7` |
| | `WyRand` | **18,704** | **396,628** | 0.887 | 0.893 | 1.54% | 1705.1 | 2.237 | 6.70 | 2.355× | `0x305e747b5c7eda81` |
| | `SplitMix64` | **18,704** | **396,628** | 1.012 | 1.022 | 2.61% | 1494.6 | 2.552 | 7.65 | 2.064× | `0xbd6cf177f6201fed` |
| | `PCG32` | **18,704** | **396,628** | 1.001 | 1.012 | 2.97% | 1512.2 | 2.523 | 7.56 | 2.088× | `0x1d28e4e052cb281a` |
| | `SFC64` | **18,704** | **396,628** | 1.062 | 1.082 | 4.44% | 1425.1 | 2.677 | 8.02 | 1.968× | `0xc0a1f4132ed41eb3` |
| | `xoshiro256**` | **18,704** | **396,628** | 1.056 | 1.074 | 5.28% | 1433.4 | 2.661 | 7.97 | 1.980× | `0x5ea949cc332810ae` |
| | `ChaCha8` | **18,704** | **396,628** | 2.315 | 2.415 | 8.65% | 653.5 | 5.837 | 17.48 | 0.903× | `0x740660c65deb97cb` |

---

### 4.4 Workload E1 & E2: Photon Transport MC (`ra_engine` Stateful OTF Shootout, $250,000$ Photons)

#### A. Mode E1-Geom vs E2-Geom: Geometric Survival Photon Transport ($P_{\text{surv}} \in \{0.50, 0.80, 0.90, 0.98\}$)

Perbandingan langsung antara **Mode E1-Geom (Coupled Stochastic)** dan **Mode E2-Geom (Fixed-Trajectory Same-Work)**:

| Rezim $P_{\text{surv}}$ | Generator | **E1 (Coupled)** Draws | **E1** $\langle L \rangle$ (Teori) | **E1** Med (ms) | **E1** vs Philox | **E2 (Fixed)** Draws | **E2** Med (ms) | **E2** MiB/s | **E2** ns/draw | **E2** vs Philox |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P=0.50$** | **`ra_core_v2 (eng_v2_E2)`** | 499,934 | 1.9997 (2.0) | **5.302** | **1.385×** | **`499,567`** | **3.843** | **495.8** | **7.694** | **1.280×** |
| ($\langle L \rangle = 2$) | `ra_core_v1 (eng_v1)` | 499,461 | 1.9978 (2.0) | 5.840 | 1.258× | **`499,567`** | 4.155 | 458.6 | 8.318 | 1.184× |
| | `Philox4x32-10` | 500,905 | 2.0036 (2.0) | 7.345 | 1.000× | **`499,567`** | 4.919 | 387.5 | 9.846 | 1.000× |
| | `Squares32` | 500,733 | 2.0029 (2.0) | 4.021 | 1.827× | **`499,567`** | 2.233 | 853.3 | 4.470 | 2.202× |
| | `WyRand` | 500,363 | 2.0015 (2.0) | 2.156 | 3.407× | **`499,567`** | 2.106 | 905.0 | 4.215 | 2.336× |
| | `SplitMix64` | 499,567 | 1.9983 (2.0) | 2.761 | 2.660× | **`499,567`** | 2.261 | 843.0 | 4.525 | 2.176× |
| | `PCG32` | 499,881 | 1.9995 (2.0) | 2.365 | 3.106× | **`499,567`** | 1.940 | 982.1 | 3.884 | 2.535× |
| | `SFC64` | 501,167 | 2.0047 (2.0) | 3.118 | 2.356× | **`499,567`** | 2.698 | 706.2 | 5.401 | 1.823× |
| | `xoshiro256**` | 499,775 | 1.9991 (2.0) | 3.165 | 2.321× | **`499,567`** | 2.993 | 636.7 | 5.992 | 1.643× |
| | `ChaCha8` | 500,045 | 2.0002 (2.0) | 11.641 | 0.631× | **`499,567`** | 11.958 | 159.4 | 23.936 | 0.411× |
| **$P=0.80$** | **`ra_core_v2 (eng_v2_E2)`** | 1,249,896 | 4.9996 (5.0) | **10.871** | **1.083×** | **`1,250,082`** | **8.388** | **568.5** | **6.710** | **1.124×** |
| ($\langle L \rangle = 5$) | `ra_core_v1 (eng_v1)` | 1,249,685 | 4.9987 (5.0) | 9.154 | 1.286× | **`1,250,082`** | 7.998 | 596.2 | 6.398 | 1.178× |
| | `Philox4x32-10` | 1,252,033 | 5.0081 (5.0) | 11.773 | 1.000× | **`1,250,082`** | 9.424 | 506.0 | 7.539 | 1.000× |
| | `Squares32` | 1,252,064 | 5.0083 (5.0) | 5.615 | 2.097× | **`1,250,082`** | 3.932 | 1212.9 | 3.145 | 2.397× |
| | `WyRand` | 1,253,501 | 5.0140 (5.0) | 3.304 | 3.563× | **`1,250,082`** | 3.692 | 1291.6 | 2.953 | 2.553× |
| | `SplitMix64` | 1,250,082 | 5.0003 (5.0) | 4.538 | 2.594× | **`1,250,082`** | 3.743 | 1274.0 | 2.994 | 2.518× |
| | `PCG32` | 1,248,973 | 4.9959 (5.0) | 3.818 | 3.083× | **`1,250,082`** | 3.149 | 1514.2 | 2.519 | 2.992× |
| | `SFC64` | 1,248,755 | 4.9950 (5.0) | 4.350 | 2.707× | **`1,250,082`** | 4.341 | 1098.6 | 3.472 | 2.171× |
| | `xoshiro256**` | 1,252,732 | 5.0109 (5.0) | 4.440 | 2.652× | **`1,250,082`** | 4.636 | 1028.6 | 3.709 | 2.033× |
| | `ChaCha8` | 1,250,085 | 5.0003 (5.0) | 14.195 | 0.829× | **`1,250,082`** | 15.764 | 302.5 | 12.610 | 0.598× |
| **$P=0.90$** | **`ra_core_v2 (eng_v2_E2)`** | 2,498,186 | 9.9927 (10.0) | **15.626** | **1.052×** | **`2,505,465`** | **11.922** | **801.7** | **4.759** | **1.125×** |
| ($\langle L \rangle = 10$) | `ra_core_v1 (eng_v1)` | 2,510,103 | 10.0404 (10.0) | 14.175 | 1.160× | **`2,505,465`** | 11.212 | 852.5 | 4.475 | 1.196× |
| | `Philox4x32-10` | 2,504,363 | 10.0175 (10.0) | 16.436 | 1.000× | **`2,505,465`** | 13.413 | 712.6 | 5.353 | 1.000× |
| | `Squares32` | 2,500,325 | 10.0013 (10.0) | 7.502 | 2.191× | **`2,505,465`** | 5.988 | 1596.2 | 2.390 | 2.240× |
| | `WyRand` | 2,497,445 | 9.9898 (10.0) | 4.402 | 3.734× | **`2,505,465`** | 3.903 | 2448.8 | 1.558 | 3.437× |
| | `SplitMix64` | 2,505,465 | 10.0219 (10.0) | 5.754 | 2.856× | **`2,505,465`** | 5.389 | 1773.6 | 2.151 | 2.489× |
| | `PCG32` | 2,498,083 | 9.9923 (10.0) | 5.610 | 2.930× | **`2,505,465`** | 4.935 | 1936.9 | 1.970 | 2.718× |
| | `SFC64` | 2,499,377 | 9.9975 (10.0) | 5.523 | 2.976× | **`2,505,465`** | 5.495 | 1739.2 | 2.193 | 2.441× |
| | `xoshiro256**` | 2,507,761 | 10.0310 (10.0) | 6.052 | 2.716× | **`2,505,465`** | 5.835 | 1637.9 | 2.329 | 2.299× |
| | `ChaCha8` | 2,498,212 | 9.9928 (10.0) | 18.218 | 0.902× | **`2,505,465`** | 18.481 | 517.2 | 7.376 | 0.726× |
| **$P=0.98$** | **`ra_core_v2 (eng_v2_E2)`** | 12,480,398 | 49.9216 (50.0) | **51.339** | **1.206×** | **`12,493,210`** | **60.882** | **782.8** | **4.873** | **0.930×** |
| ($\langle L \rangle = 50$) | `ra_core_v1 (eng_v1)` | 12,510,771 | 50.0431 (50.0) | 52.841 | 1.172× | **`12,493,210`** | 51.453 | 926.2 | 4.119 | 1.100× |
| | `Philox4x32-10` | 12,472,617 | 49.8905 (50.0) | 61.913 | 1.000× | **`12,493,210`** | 56.623 | 841.7 | 4.532 | 1.000× |
| | `Squares32` | 12,530,457 | 50.1218 (50.0) | 26.829 | 2.308× | **`12,493,210`** | 23.512 | 2027.0 | 1.882 | 2.408× |
| | `WyRand` | 12,514,805 | 50.0592 (50.0) | 16.103 | 3.845× | **`12,493,210`** | 14.308 | 3330.8 | 1.145 | 3.957× |
| | `SplitMix64` | 12,493,210 | 49.9728 (50.0) | 25.580 | 2.420× | **`12,493,210`** | 19.904 | 2394.4 | 1.593 | 2.845× |
| | `PCG32` | 12,484,591 | 49.9384 (50.0) | 21.720 | 2.851× | **`12,493,210`** | 21.715 | 2194.7 | 1.738 | 2.608× |
| | `SFC64` | 12,539,930 | 50.1597 (50.0) | 17.072 | 3.627× | **`12,493,210`** | 17.314 | 2752.5 | 1.386 | 3.270× |
| | `xoshiro256**` | 12,532,476 | 50.1299 (50.0) | 21.236 | 2.915× | **`12,493,210`** | 18.771 | 2539.0 | 1.502 | 3.017× |
| | `ChaCha8` | 12,488,622 | 49.9545 (50.0) | 59.578 | 1.039× | **`12,493,210`** | 61.438 | 775.7 | 4.918 | 0.922× |

#### B. Mode E1-Slab vs E2-Slab: 3D Turbid Slab Photon Transport MC ($P_{\text{rej}} \in \{0.50, 0.80, 0.90, 0.98\}$)

Simulasi transpor foton 3D di dalam medium keruh (*turbid slab*, $\mu_t d = 5.0$, albedo $= 0.9$, Russian roulette threshold $= 0.01$) dengan probabilitas *early surface rejection* $P_{\text{rej}}$:

| Rezim $P_{\text{rej}}$ | Generator | **E1 (Coupled)** Draws | **E1** $R_d / T_d$ | **E1** Med (ms) | **E1** vs Philox | **E2 (Fixed)** Draws | **E2** Med (ms) | **E2** MiB/s | **E2** ns/draw | **E2** vs Philox |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_{\text{rej}}=0.50$** | **`ra_core_v2 (eng_v2_E2)`** | 2,660,238 | $0.2561 / 0.0931$ | **20.876** | **1.196×** | **`2,663,753`** | **13.546** | **750.2** | **5.085** | **1.215×** |
| ($\langle K \rangle \approx 10.66$) | `ra_core_v1 (eng_v1)` | 2,666,521 | $0.2564 / 0.0941$ | 22.352 | 1.117× | **`2,663,753`** | 12.183 | 834.1 | 4.574 | 1.351× |
| | `Philox4x32-10` | 2,665,899 | $0.2556 / 0.0937$ | 24.973 | 1.000× | **`2,663,753`** | 16.458 | 617.4 | 6.179 | 1.000× |
| | `Squares32` | 2,661,869 | $0.2564 / 0.0928$ | 20.299 | 1.230× | **`2,663,753`** | 6.368 | 1595.7 | 2.391 | 2.585× |
| | `WyRand` | 2,660,518 | $0.2566 / 0.0925$ | 17.384 | 1.437× | **`2,663,753`** | 4.755 | 2137.2 | 1.785 | 3.462× |
| | `SplitMix64` | 2,665,970 | $0.2571 / 0.0934$ | 18.488 | 1.351× | **`2,663,753`** | 5.879 | 1728.4 | 2.207 | 2.800× |
| | `PCG32` | 2,658,940 | $0.2565 / 0.0935$ | 16.355 | 1.527× | **`2,663,753`** | 4.892 | 2077.2 | 1.836 | 3.365× |
| | `SFC64` | 2,650,011 | $0.2559 / 0.0932$ | 15.187 | 1.644× | **`2,663,753`** | 5.727 | 1774.3 | 2.150 | 2.874× |
| | `xoshiro256**` | 2,654,916 | $0.2557 / 0.0933$ | 16.572 | 1.507× | **`2,663,753`** | 7.205 | 1410.3 | 2.705 | 2.284× |
| | `ChaCha8` | 2,656,301 | $0.2561 / 0.0940$ | 26.271 | 0.951× | **`2,663,753`** | 22.838 | 444.9 | 8.574 | 0.721× |
| **$P_{\text{rej}}=0.80$** | **`ra_core_v2 (eng_v2_E2)`** | 1,220,610 | $0.1023 / 0.0371$ | **10.423** | **1.185×** | **`1,215,420`** | **6.068** | **764.1** | **4.992** | **1.309×** |
| ($\langle K \rangle \approx 4.86$) | `ra_core_v1 (eng_v1)` | 1,212,958 | $0.1031 / 0.0372$ | 11.272 | 1.096× | **`1,215,420`** | 6.545 | 708.4 | 5.385 | 1.213× |
| | `Philox4x32-10` | 1,215,927 | $0.1025 / 0.0374$ | 12.349 | 1.000× | **`1,215,420`** | 7.942 | 583.8 | 6.535 | 1.000× |
| | `Squares32` | 1,213,786 | $0.1021 / 0.0372$ | 8.654 | 1.427× | **`1,215,420`** | 2.915 | 1590.8 | 2.398 | 2.725× |
| | `WyRand` | 1,208,820 | $0.1024 / 0.0369$ | 7.071 | 1.747× | **`1,215,420`** | 2.097 | 2210.7 | 1.726 | 3.787× |
| | `SplitMix64` | 1,211,323 | $0.1026 / 0.0373$ | 8.367 | 1.476× | **`1,215,420`** | 2.661 | 1742.6 | 2.189 | 2.985× |
| | `PCG32` | 1,216,902 | $0.1024 / 0.0377$ | 7.406 | 1.668× | **`1,215,420`** | 2.179 | 2127.6 | 1.793 | 3.645× |
| | `SFC64` | 1,207,703 | $0.1024 / 0.0371$ | 7.531 | 1.640× | **`1,215,420`** | 3.026 | 1532.2 | 2.490 | 2.625× |
| | `xoshiro256**` | 1,211,545 | $0.1021 / 0.0374$ | 7.289 | 1.694× | **`1,215,420`** | 3.328 | 1393.2 | 2.738 | 2.387× |
| | `ChaCha8` | 1,205,435 | $0.1028 / 0.0375$ | 18.725 | 0.660× | **`1,215,420`** | 14.830 | 312.6 | 12.201 | 0.536× |
| **$P_{\text{rej}}=0.90$** | **`ra_core_v2 (eng_v2_E2)`** | 737,826 | $0.0509 / 0.0185$ | **7.955** | **1.077×** | **`731,124`** | **5.282** | **528.0** | **7.224** | **1.528×** |
| ($\langle K \rangle \approx 2.92$) | `ra_core_v1 (eng_v1)` | 730,503 | $0.0514 / 0.0185$ | 7.832 | 1.094× | **`731,124`** | 6.175 | 451.7 | 8.445 | 1.307× |
| | `Philox4x32-10` | 735,672 | $0.0512 / 0.0188$ | 8.566 | 1.000× | **`731,124`** | 8.069 | 345.6 | 11.037 | 1.000× |
| | `Squares32` | 726,684 | $0.0510 / 0.0184$ | 6.834 | 1.254× | **`731,124`** | 2.421 | 1151.9 | 3.312 | 3.333× |
| | `WyRand` | 736,301 | $0.0514 / 0.0185$ | 4.206 | 2.037× | **`731,124`** | 1.753 | 1591.0 | 2.398 | 4.603× |
| | `SplitMix64` | 729,063 | $0.0513 / 0.0185$ | 6.670 | 1.284× | **`731,124`** | 2.011 | 1386.8 | 2.751 | 4.012× |
| | `PCG32` | 732,591 | $0.0509 / 0.0188$ | 5.010 | 1.710× | **`731,124`** | 1.631 | 1709.7 | 2.231 | 4.947× |
| | `SFC64` | 726,868 | $0.0511 / 0.0182$ | 4.887 | 1.753× | **`731,124`** | 2.577 | 1082.4 | 3.524 | 3.132× |
| | `xoshiro256**` | 724,148 | $0.0512 / 0.0185$ | 5.413 | 1.583× | **`731,124`** | 2.736 | 1019.4 | 3.742 | 2.949× |
| | `ChaCha8` | 728,783 | $0.0513 / 0.0188$ | 19.319 | 0.443× | **`731,124`** | 13.390 | 208.3 | 18.315 | 0.603× |
| **$P_{\text{rej}}=0.98$** | **`ra_core_v2 (eng_v2_E2)`** | 348,212 | $0.0104 / 0.0037$ | **2.568** | **1.723×** | **`346,400`** | **2.725** | **484.9** | **7.867** | **1.640×** |
| ($\langle K \rangle \approx 1.39$) | `ra_core_v1 (eng_v1)` | 346,546 | $0.0101 / 0.0037$ | 2.836 | 1.560× | **`346,400`** | 3.067 | 430.8 | 8.855 | 1.457× |
| | `Philox4x32-10` | 347,798 | $0.0105 / 0.0038$ | 4.425 | 1.000× | **`346,400`** | 4.470 | 295.6 | 12.904 | 1.000× |
| | `Squares32` | 347,822 | $0.0102 / 0.0038$ | 1.708 | 2.591× | **`346,400`** | 1.252 | 1055.5 | 3.614 | 3.571× |
| | `WyRand` | 346,350 | $0.0103 / 0.0036$ | 1.048 | 4.223× | **`346,400`** | 0.875 | 1510.9 | 2.525 | 5.111× |
| | `SplitMix64` | 346,742 | $0.0104 / 0.0037$ | 1.397 | 3.167× | **`346,400`** | 0.979 | 1349.6 | 2.827 | 4.565× |
| | `PCG32` | 347,680 | $0.0104 / 0.0038$ | 1.086 | 4.073× | **`346,400`** | 0.728 | 1815.2 | 2.102 | 6.141× |
| | `SFC64` | 347,520 | $0.0105 / 0.0035$ | 1.754 | 2.524× | **`346,400`** | 1.739 | 760.0 | 5.019 | 2.571× |
| | `xoshiro256**` | 340,886 | $0.0101 / 0.0036$ | 1.267 | 3.492× | **`346,400`** | 2.028 | 651.7 | 5.854 | 2.204× |
| | `ChaCha8` | 344,450 | $0.0104 / 0.0038$ | 10.919 | 0.405× | **`346,400`** | 11.042 | 119.7 | 31.875 | 0.405× |

---

## 5. Artefak & Berkas yang Dihasilkan

1. **Source Harness C Bersih:**  
   `experiments/2026-11-04_v2-clean-mc-harness/clean_mc_shootout.c`
2. **Binary Eksekusi:**  
   `experiments/2026-11-04_v2-clean-mc-harness/clean_mc_shootout`
3. **Ringkasan Statistik CSV ($N=9$ Interleaved Rounds):**  
   `experiments/2026-11-04_v2-clean-mc-harness/data/clean_mc_summary.csv`
4. **Sampel Mentah per Putaran CSV ($2,169$ Baris Pengukuran):**  
   `experiments/2026-11-04_v2-clean-mc-harness/data/clean_mc_samples.csv`
5. **Laporan JSON Terstruktur:**  
   `experiments/2026-11-04_v2-clean-mc-harness/data/clean_mc_results.json`
6. **Dokumen Hasil & Analisis ini:**  
   `experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md`
