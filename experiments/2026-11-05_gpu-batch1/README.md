# GPU batch 1: Ising K1, short bursts, size crossover

The completed Colab Blackwell run and its archived logs are analyzed in
[`RESULTS.md`](RESULTS.md).

Open `batch1_colab.ipynb` with Colab (**File → Upload notebook**) and run all
cells in order. The notebook embeds `batch1_core.h` and `batch1.cu`; no GitHub
clone or Google Drive mount is required. It detects the assigned GPU compute
capability and compiles for that `sm_XX` target. The first CUDA failure or
correctness mismatch stops the batch.

Revision 2026-11-05: fixed the GPU Ising audit allocation. The previous
notebook initialized its audit vector from an instance member before that
member was initialized, which could produce an undersized CUDA allocation
and `illegal memory access` during the `--check` Ising parity test. Reupload
the updated notebook so Colab writes and recompiles the corrected source.

The program runs **sequentially**, never concurrently on the same GPU:

1. Frozen `ra_core v2 Final` KAT for K=1,2,4,8; official Philox KAT;
   uniqueness of the Ising address mapping; CPU Ising checksum and physical
   observables against the audited raw JSON; GPU/CPU parity for 1,024 burst
   addresses and a short true Ising simulation.
2. True 2D periodic Ising checkerboard Metropolis at critical temperature,
   32 sweeps, lattice sizes 64, 256, 1024. Each attempted spin consumes exactly
   one K1 draw, including unconditional accepts. CPU and GPU must agree on
   every final spin, draw XOR, accepted count, energy, and magnetization for
   each generator. The L=256 initial state and address schedule match the
   audited CPU workload. The GPU kernel includes a per-block draw/accept audit
   so the compiler cannot skip unused draws.
3. K=2,4,8 short bursts over 4,194,304 independent addresses. One threshold
   count per burst is materialized. Philox consumes the first K words of its
   4-word blocks; K>4 advances the counter. This is the same K-word contract
   as the CPU shootout, not an amortized comparison when K<4.
4. K1 output-size curve at 4,096; 65,536; 1,048,576; 4,194,304 addresses.
   CPU time and GPU kernel time are reported separately. GPU `total_ms`
   includes launch and device-to-host output transfer; CPU uses a preallocated
   output array. This is an exploratory crossover on Colab's assigned CPU/GPU,
   not a cost or energy measurement.

All `SAMPLE` rows appear in actual run order (alternating RA/Philox first
position). There are five GPU and three CPU Ising samples per size, seven GPU
burst samples per K/size, and three CPU K1 size-curve samples. `SUMMARY` rows
report medians. Source hashes, compiler output, GPU identity, and all raw
samples are saved by the notebook as `batch1_manifest.json`,
`batch1_compile.log`, `batch1_host_check.log`, `batch1_check.log`, and
`batch1_results.txt`.

The Philox baseline is the direct Random123-compatible function already used
in the CPU shootout, not NVIDIA cuRAND. Any future GPU paper or product speed
claim must also compare an appropriate optimized GPU library and repeat on a
controlled device. This batch does not implement photon transport, N=16/32
state-size variants, or energy/cost measurement.

## Local host gate

CUDA is not needed to check the shared formulas and CPU reference:

```bash
g++ -std=c++17 -O3 -DRA_HOST_ONLY -x c++ batch1.cu -o batch1_host
./batch1_host
g++ -std=c++17 -O3 host_parity.cpp -o host_parity
./host_parity
```

The differential check covers 10,000 `(key, addr_hi)` pairs for each of
K=1,2,4,8 against the production header. It passed locally before the
Colab notebook was prepared. CUDA compilation and execution must still be
verified in Colab because this workstation has no CUDA toolkit or GPU.

The audited raw JSON gives L=256 Ising RA checksum `0x8210691e`, energy per
spin `-1.417175`, and absolute magnetization per spin `0.582092`; Philox
checksum `0xf8a28877`, energy `-1.401489`, and magnetization `0.570770`.
These raw observables match the final 2b-2A table in the VPS `RESULTS.md`.
The batch gate uses the JSON and the original C workload.
