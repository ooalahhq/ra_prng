# Developer tools — ra_prng

## Fairness shootout runner

`experiments/2026-11-04_v2-prng-shootout-and-mc/run_fairness.py` rebuilds and runs the four benchmark binaries on one allowed CPU core, then checks every microbenchmark result against seven raw timing samples and every Monte Carlo timing row against five raw samples. It records the exact commands, compiler, CPU, hashes, and sampling summary in `data/fairness_manifest.json`. `render_fairness_results.py` regenerates `RESULTS.md` from JSON. No network service is used.

```bash
cd experiments/2026-11-04_v2-prng-shootout-and-mc
python3 run_fairness.py
```

The runner uses `gcc -O3 -march=native -mavx2 -fno-tree-vectorize -ffp-contract=off`, one warmup and seven measured rounds for the microbenchmark, and regenerates plots and `shootout_dashboard.html` after all four programs pass. `--only micro photon pi_bernoulli gbm` can select programs; `--core N` selects an allowed core. The separate Group 3 tree dataset is imported as exploratory because PRNGs can produce different node/draw counts; its cross-family speed ratios are not a same-work benchmark.

Relevant source and checks are in the experiment's `RESULTS.md` and `FAIRNESS_AUDIT.md`.

## GPU K1 Colab probe

`experiments/2026-11-05_gpu-colab-k1/gpu_k1_colab.ipynb` is a self-contained
CUDA 13.0 notebook for the first GPU comparison of `ra_core v2 Final` K1 and
Philox4x32-10 on one independent draw per address. It gates timing on frozen
KAT vectors and GPU/CPU parity, then records raw-output and fused Bernoulli
timings separately. The source and interpretation limits are in that folder's
`README.md`. This is an exploratory probe, not a paper result.

## GPU batch 1 Colab runner

`experiments/2026-11-05_gpu-batch1/batch1_colab.ipynb` is a self-contained
sequential CUDA notebook for the first follow-up batch: KAT and CPU/GPU parity,
true Ising checkerboard K1 at three lattice sizes, short bursts K2/K4/K8, and
an exploratory K1 CPU/GPU output-size crossover. It saves chronological raw
samples plus source hashes, GPU identity, and compile logs. The folder's
`README.md` defines the exact work and limits; the two CUDA source files are
embedded in the notebook. Host-only and differential checks can be run from
that directory before Colab. This is an exploratory runner, not a published
GPU result.
