# GPU batch 1: Colab Blackwell result

Run: 2026-11-05 01:51 UTC, NVIDIA RTX PRO 6000 Blackwell Server Edition
(compute capability 12.0, 97,887 MiB), CUDA 13.0 / `nvcc 13.0.88`, compiled
with `-O3 -std=c++17 -arch=sm_120`. Raw files are in
[`runs/2026-11-05_colab_blackwell/`](runs/2026-11-05_colab_blackwell/).
The manifest's SHA-256 hashes match the current `batch1_core.h`, `batch1.cu`,
and archived result log. The run completed with 170 timed sample rows.

## Correctness gates

- Host RA K=1/2/4/8 golden vectors and official Philox KAT passed.
- Sampled Ising site/sweep address uniqueness passed.
- The L=256, 32-sweep CPU Ising golden matched the audited raw workload:
  RA checksum `0x8210691e`, E/N `-1.417175`, |M|/N `0.582092`;
  Philox checksum `0xf8a28877`, E/N `-1.401489`, |M|/N `0.570770`.
- GPU/CPU parity passed for the first 1,024 K=1/2/4/8 burst addresses.
- GPU/CPU true Ising parity passed for both generators at L=64, 2 sweeps:
  every final spin, draw XOR, accept count, energy, and magnetization.
- During the full batch, GPU Ising results were checked against the CPU
  reference for both generators at L=64/256/1024, 32 sweeps, on the warmup
  and each of five timed runs. No mismatch or CUDA error occurred.

The earlier notebook's `illegal memory access` came from initializing the
GPU audit vector with an instance member before that member was initialized.
The corrected `static constexpr blocks_per_pass` source is the one whose hash
appears in the manifest.

## Timing results

Medians below are in milliseconds. `GPU kernel` covers all Ising half-sweep
launches; `GPU total` also covers resetting the lattice, final-spin and audit
copies to host, and audit reduction. CPU is a separate implementation on the
Colab host. RA/Philox speedup is Philox median divided by RA median; values
above 1 mean RA is faster.

| True Ising, 32 sweeps | RA CPU | Philox CPU | RA GPU kernel | Philox GPU kernel | RA GPU total | Philox GPU total | RA/Philox GPU kernel speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L=64 | 0.888 | 1.155 | 0.132 | 0.133 | 0.157 | 0.159 | 1.010× |
| L=256 | 12.872 | 17.135 | 0.132 | 0.132 | 0.221 | 0.221 | 1.001× |
| L=1024 | 221.856 | 288.781 | 0.263 | 0.264 | 1.144 | 1.145 | 1.004× |

RA is consistently faster on this host CPU for true Ising (1.30–1.33×).
On the GPU, Ising timings are effectively tied. At L=64 and L=256 the
64-launch kernel time stays near 0.132 ms despite 16× more sites; launch
overhead is a plausible contributor, but profiling is needed to attribute the
time. The GPU/CPU ratio reflects different hardware and is not a generator
speedup.

| Burst consumer, 4,194,304 addresses | RA GPU kernel | Philox GPU kernel | RA GPU total | Philox GPU total | RA/Philox kernel speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=2 | 0.0143 | 0.0142 | 0.648 | 0.651 | 0.993× |
| K=4 | 0.0185 | 0.0144 | 0.652 | 0.649 | 0.777× |
| K=8 | 0.0271 | 0.0226 | 0.663 | 0.654 | 0.833× |

At K=4 and K=8, RA's kernel was slower in all seven paired repetitions:
about 29% and 20% longer than Philox, respectively. The device-to-host
transfer dominates `GPU total`, so the end-to-end medians differ by less than
2%. The compiler reported no register spills for these kernels; the cause
of the kernel gap has not been profiled.

| K=1 addresses | RA CPU | Philox CPU | RA GPU kernel | Philox GPU kernel | RA GPU total | Philox GPU total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.0132 | 0.0229 | 0.0044 | 0.0040 | 0.0172 | 0.0157 |
| 65,536 | 0.2097 | 0.3653 | 0.0042 | 0.0041 | 0.0294 | 0.0314 |
| 1,048,576 | 3.4988 | 5.8769 | 0.0062 | 0.0060 | 0.1831 | 0.1834 |
| 4,194,304 | 13.4846 | 23.4831 | 0.0137 | 0.0142 | 0.6415 | 0.6437 |

At K=1, RA is 1.68–1.74× faster than Philox on the Colab CPU. GPU kernel
differences are small and inconsistent across size/repetition; the largest
K=1 run favors RA by only 3.3% in kernel median and 0.3% in total median.
For RA at 4,096 addresses, the CPU median (0.0132 ms) beats GPU total
(0.0172 ms); by 65,536 addresses GPU total (0.0294 ms) beats CPU
(0.2097 ms). This is an observed crossover interval for this exact host,
GPU, output transfer, and consumer, not a general threshold.

## Interpretation and scope

This batch establishes a bit-correct CUDA port of the frozen RA v2 core and
demonstrates competitive K=1 Ising GPU execution against this direct Philox
implementation. It does **not** establish a GPU speed advantage for RA; the
K=4/K=8 kernel results favor Philox. Differences near 1% should not be
treated as wins from one Colab session. The distinct RA and Philox Ising
trajectories have different physical observables; one trajectory per
generator is not a statistical accuracy comparison.

This is an exploratory, single-device batch. The comparator is the
Random123-compatible Philox implementation in this harness, not a tuned GPU
library. The Ising audit writes extra per-block records and transfers them
to the host; an actual application might keep state and results on device.
No energy, monetary cost, photon transport, or independent-session variance
was measured. These limits should accompany any use of these numbers in a
paper or product claim.
