# Repository Structure

`ra_prng` contains canonical C and Python implementations of the RA
array-based PRNG line (`ra_prng2`, experimental `ra_prng3`, and the current
addressable `ra_core v2`), their installable Python bindings, reproducibility
tests, manuscripts, and the `ra_core v2` validation and benchmark archive.

```
.
├── src/                                  canonical algorithm implementations
│   ├── ra_prng2/                         32-bit version (CSAI2025 manuscript)
│   │   ├── c/                            three C variants (see below) + CHANGELOG.md
│   │   └── python/                       pure-Python reference implementation (ra_prng2.py)
│   ├── ra_prng3/                         64-bit experimental version ("do not use")
│   │   ├── c/                            single-file C implementation + CHANGELOG.md
│   │   └── python/                       empty - no Python port exists yet
│   └── ra_core/                          current production addressable generator (v2)
│       ├── CHANGELOG.md                  promotion history for ra_core
│       └── c/                            ra_core.h, ra_core.c (Opsi_A3 + W2_D2_xag_opt),
│                                         ra_core_engine.h, ra_core_engine.c (eng_v2_E2)
├── bindings/cpython/                     installable Python <-> C ctypes binding for ra_prng2
├── pip-ra-core/                          installable Python <-> C package ("ra-core") for ra_core
│   ├── ra_core/                          ctypes engine wrapper + Cython numpy BitGenerator
│   ├── tests/                            pytest suite + frozen Golden KAT v2 vectors
│   └── setup.py, pyproject.toml, README.md
├── tests/                                ra_prng2 Python unit tests & C/Python equivalence tests
├── tools/battery/                        shared avalanche and PractRand test-gate helpers
├── benchmarks/
│   ├── results/                          precomputed statistical-test output (Dieharder,
│   │                                     NIST STS, TestU01, PractRand) for ra_prng2/3
│   └── comparisons/                      speed/entropy comparisons vs xoshiro256**, PCG32,
│       ├── source/                       Philox4x32, ChaCha20, /dev/urandom + derived
│       │                                 file-shuffling CLI variants (see source/README.md)
│       └── RNGing_speed/, entropy/, scrambling_speed/   raw result .txt files
├── experiments/                          preserved ra_core v2 validation & benchmark evidence
│   ├── 2026-11-04_v2-final-battery/      production promotion, Golden KATs, SAC/BIC, Dieharder, PractRand
│   ├── 2026-11-04_v2-engine-opt/         stateful OTF engine ablation (eng_v2_E2)
│   ├── 2026-11-04_v2-clean-mc-harness/   same-work Monte Carlo shootout (Ising, Photon, Asian)
│   ├── 2026-11-04_v2-prng-shootout-and-mc/  fairness-audited microbenchmark & MC shootout + dashboard
│   ├── 2026-11-05_v2-vps-final-benchmark/   second-machine cross-compiler validation
│   ├── 2026-11-05_gpu-batch1/            exploratory CUDA Blackwell parity & batch-1 study
│   └── 2026-11-05_gpu-colab-k1/          initial exploratory CUDA K1 Colab probe
├── research/
│   ├── csai2025_ra_prng/                 CSAI2025 manuscript + acceptance letter (unpublished)
│   ├── iccs2026_ra_prng/                 ICCS2026 manuscript (unpublished)
│   └── 2026-9_ra-core-paper/             ra_core v2 manuscript (DRAFT_v2.md, REPRODUCIBILITY.md,
│                                         MANIFEST.sha256, ieee/{main.tex,references.bib,main.pdf})
├── sub-projects/                         downstream application placeholders
│   ├── image_encrypting/
│   └── fingerprinting_tools/
├── docs/                                 STRUCTURE.md, ERRATA.md, api/ra_prng2.md, developer-tools/
├── pyproject.toml, setup.py, Makefile    root packaging and C/extension build targets
└── .github/workflows/ci.yml              CI build + smoke-test workflow
```

Earlier pre-v2 exploratory folders under `experiments/` are excluded from the
current working tree to keep the public release focused on `ra_core v2`, and
remain accessible in pre-v2 git history.

## `ra_prng2` vs `ra_prng3` vs `ra_core` vs `benchmarks/comparisons/source/`

These are **not duplicates of each other** — each serves a different purpose:

- `src/ra_prng2/c/ra_prng2.c` — canonical 32-bit generator from the CSAI2025
  manuscript.
- `src/ra_prng2/c/ra_prng2_thread.c` — OpenMP-enabled variant for multi-threaded
  throughput benchmarking.
- `src/ra_prng2/c/ra_prng2_struct.c` — reusable `RA_PRNG` struct API
  (`ra_prng_init`/`ra_prng_next`/`ra_prng_advance`) backing `bindings/cpython/`.
- `src/ra_prng3/c/ra_prng3.c` — 64-bit experimental successor (`ZepFold` core),
  marked "do not use".
- `src/ra_core/c/ra_core.{h,c}` and `ra_core_engine.{h,c}` — current `ra_core v2`
  addressable permutation-stream generator (`Opsi_A3` + `W2_D2_xag_opt` +
  `eng_v2_E2`) backing `pip-ra-core/`.
- `benchmarks/comparisons/source/ra_prng2.c` and `ra_prng3.c` — derived
  file-shuffling CLI tools (`--input`/`--output`/`--seed`/`--multiplier_m`/
  `--multiplier_l`), deliberately kept separate from the canonical generators
  (see `benchmarks/comparisons/source/README.md`).

## Build

See [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`README.md`](../README.md), and
the root `Makefile` / `pyproject.toml`.
