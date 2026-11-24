# GPU K1 Colab probe — hasil eksploratif

Log sesi Colab diterima pada 2026-11-05. GPU: NVIDIA RTX PRO 6000 Blackwell
Server Edition, compute capability 12.0, CUDA toolkit 13.0 (`nvcc` V13.0.88
menurut output sesi). Sumber mentah: `data/gpu_compile.log` dan
`data/gpu_k1_probe_results.txt`. Notebook menjalankan satu draw per alamat
independen, `N=16,777,216`, tiga warmup dan sembilan timing per generator/mode.
Timing adalah durasi kernel CUDA event, tanpa alokasi, kompilasi, atau transfer
host/device.

| Mode | ra_core v2 Final median | Philox4x32-10 median | Rasio waktu RA / Philox | Kesimpulan |
| --- | ---: | ---: | ---: | --- |
| `raw_k1` (menulis 64 MiB) | 0.0416 ms | 0.0415 ms | 1.0024 | Sama dalam resolusi/noise pengukuran |
| `bernoulli_k1` (keputusan + reduksi) | 0.0719 ms | 0.0720 ms | 0.9986 | Sama dalam resolusi/noise pengukuran |

KAT beku ra_core K1 dan tiga KAT resmi Philox lulus. Output GPU/CPU cocok
untuk 1.024 alamat pertama pada kedua generator. `ptxas` melaporkan nol
stack frame dan nol spill pada semua kernel. RA memakai 10 register di kernel
raw dan 12 di Bernoulli; Philox masing-masing 14 dan 16. Perbedaan register
belum menghasilkan keunggulan waktu pada dua mode ini.

Selisih median hanya 0.0001 ms (0.1 mikrodetik), sedangkan dokumentasi CUDA
menyebut resolusi `cudaEventElapsedTime` sekitar 0.5 mikrodetik. Sampel raw
Philox memiliki satu outlier 0.0561 ms; median tetap 0.0415 ms. Sampel yang
dicetak diurutkan, sehingga urutan pasangan A/B asli tidak tersimpan.

Output raw 64 MiB / 0.0416 ms setara ~1.61 TB/s efektif, di sekitar
spesifikasi bandwidth memori 1.597 TB/s untuk GPU ini. Ini mengisyaratkan
biaya penulisan output/cache besar dalam uji raw, tetapi tanpa profiler tidak
boleh dinyatakan sebagai bottleneck DRAM yang terbukti. Uji Bernoulli juga
memuat satu reduksi bersama per blok, sehingga waktu tidak mengisolasi biaya
PRNG saja.

**Keputusan:** implementasi K1 GPU benar dan tidak spill, tetapi dua mode
yang diuji **tidak menunjukkan kemenangan kinerja ra_core atas Philox**.
Jangan masukkan rasio GPU sebagai klaim keunggulan paper/produk. Tidak perlu
mengulang atau men-tune microbenchmark ini hanya untuk mengejar selisih kecil.
Jika riset GPU dilanjutkan, langkah bernilai adalah satu simulasi nyata dengan
jadwal draw yang sama, misalnya sapuan Ising checkerboard K1, lalu mengukur
durasi simulasi penuh. Keputusan dan scope eksperimen berikutnya perlu
disepakati sebelum implementasi.

## Provenance dan batas

- SHA-256 sumber lokal `gpu_k1_probe.cu` yang tertanam dalam notebook:
  `0d5dc973cedd210eed939a24c3b3ed4a3cc3ec1c5d8ecc12b65bc885288046fb`.
  Log Colab tidak menyimpan hash sumber saat dikompilasi, jadi kesamaan dengan
  sumber lokal tidak dapat diverifikasi independen dari log.
- SHA-256 compile log:
  `1924d726fdaa74a828950561c9f69348d557625e4627cdfa0a6b2f0976912296`.
- SHA-256 results log:
  `24951a6f1142b62e5eacc5baa2fb26fcfa2d3243a7b1c707d79eb2b0842ef1d0`.
- Satu GPU, satu sesi Colab, sembilan sampel per mode. Belum mengukur K>1,
  workload photon/Ising penuh, kecepatan lintas GPU, biaya transfer, atau
  efisiensi energi.

Sumber eksternal: [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
dan [CUDA Runtime API: Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html).
