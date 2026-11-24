# ra_prng2 and Beyond: An Array-Based PRNG Architecture for Efficient Random Generation

> **Project ownership:** This repository is the personal independent research
> project of **Hamas A. Rahman** (`@hamzy-hams`), hosted at
> [`ooalahhq/ra_prng`](https://github.com/ooalahhq/ra_prng).

`ra_prng` is a deterministic, array-based PRNG architecture that makes array-index
permutation the *source* of randomness rather than something built on top of it.
Two internal 256-entry arrays continuously permute and mutate each other; entropy
generation and index shuffling happen inside the same in-place loop. That keeps
internal state small (~2 KB, L1-cache friendly) while still passing large-scale
statistical batteries, which makes it well suited to data shuffling in AI/ML
training pipelines, neural-network weight initialization, deterministic
simulations, statistical experiments, or general-purpose use.

> **Note:** Some code comments/identifiers are still in Indonesian and have not
> yet been translated to English.
> This PRNG is **not designed or intended for cryptographic use!!!**

## What's in this repository

| Path | What it is | Status |
|---|---|---|
| `src/ra_prng2/c/`, `src/ra_prng2/python/` | The 32-bit generator described in the CSAI2025 paper — canonical (`ra_prng2.c`), OpenMP-parallel (`ra_prng2_thread.c`), struct-refactored (`ra_prng2_struct.c`, backs the Python binding), and a pure-Python reference port (`ra_prng2.py`) | Stable |
| `src/ra_prng3/c/` | 64-bit experimental successor (`ZepFold` core), larger internal state, theoretical period up to 2^32832 | Experimental — **do not use** |
| `src/ra_core/c/` | Current production `ra_core v2` addressable/orbit-addressing generator (`ra_core.c`, `ra_core.h`, `ra_core_engine.c`, `ra_core_engine.h`) — see the `ra_core v2` section below | Stable |
| `pip-ra-core/` | Installable Python package (`ra-core`) wrapping `src/ra_core/c/` via `ctypes` and a Cython `numpy.random.BitGenerator` extension (see [`pip-ra-core/README.md`](pip-ra-core/README.md)) | Stable |
| `bindings/cpython/` | Installable `pip install .` package — a `ctypes` wrapper calling straight into `ra_prng2_struct.c`, no logic reimplemented in Python | Stable |
| `tests/` | Cross-language equivalence and unit tests for `ra_prng2` (paired with `pip-ra-core/tests/` for `ra_core`) | Active |
| `tools/battery/` | Shared statistical test drivers (avalanche and PractRand gate helpers) | Research tooling |
| `benchmarks/comparisons/source/` | A *derived* file-shuffling CLI tool (`--input`/`--output`/`--seed`/...) built on the same mixing ideas but with its own permutation/index math — not the paper algorithm itself | Separate tool |
| `benchmarks/results/`, `benchmarks/comparisons/` | Precomputed statistical-test output and throughput/entropy comparisons vs xoshiro256\*\*, PCG32, Philox4x32, ChaCha20, `/dev/urandom` | Reference data |
| `experiments/` | Preserved `ra_core v2` statistical validation, shootout, Monte Carlo, cross-machine, and exploratory GPU evidence folders (earlier pre-v2 investigations remain in git history) | Paper evidence |
| `sub-projects/` | Downstream applications (`image_encrypting`, `fingerprinting_tools`) | Placeholders, not yet implemented |
| `research/` | Conference paper manuscripts — CSAI2025 (accepted by reviewers for oral presentation & publication, but never actually presented or published due to lack of funding), ICCS2026, and the current `ra_core v2` manuscript in [`research/2026-9_ra-core-paper/`](research/2026-9_ra-core-paper/REPRODUCIBILITY.md) | See Citation |

See [`docs/STRUCTURE.md`](docs/STRUCTURE.md) for the full directory map and
exactly how `ra_prng2`, `ra_prng3`, `ra_core`, and the CLI shuffle variants
under `benchmarks/comparisons/source/` relate to each other.

## Research boundary

This repository owns the RA generator design, production implementation,
statistical evaluation, reproducibility checks, and manuscripts. Its application
benchmarks characterize RA's behavior and suitable workloads; they do not by
themselves establish a reason for an existing simulation project to replace its
PRNG. End-to-end audits of external scientific simulation workflows (sampling,
kernels, and build tuning independent of PRNG choice) are conducted separately
outside this repository.

## Build & install

```bash
make build-c          # -> build/bin/{ra_prng2, ra_prng2_thread, ra_prng3, ra_core}
make build-ext        # -> build/lib/libra_prng2.so (used by bindings/cpython/_ra_prng2.py)
make build-ext-core   # -> build/lib/libra_core_engine.so (used by pip-ra-core ctypes dev loop)
make install          # editable install of ra_prng2 (pip install -e .)
make test             # run tests/ (ra_prng2 Python/C equivalence tests)
make clean            # remove build/ and pip-ra-core build artifacts
```

To verify `ra_core` directly and run the full test suite (both `ra_prng2` and
`ra_core`, including frozen Golden KAT vectors):

```bash
make build-c build-ext build-ext-core
./build/bin/ra_core validate
pip install ./pip-ra-core
pytest tests/ pip-ra-core/tests/
```

Or compile the standalone C binaries manually:

```bash
# Standard ra_prng2
gcc -O3 -march=native -std=gnu17 -include stdalign.h src/ra_prng2/c/ra_prng2.c -o ra_prng2

# OpenMP-enabled ra_prng2
gcc -O3 -march=native -std=gnu17 -include stdalign.h -fopenmp src/ra_prng2/c/ra_prng2_thread.c -o ra_prng2_thread

# Production ra_core v2 CLI
gcc -O3 -march=native -std=gnu17 -include stdalign.h src/ra_core/c/ra_core.c -o ra_core
```

The default output is either the raw random stream or the result of the
folding process, depending on the binary.

### Python

```bash
pip install .              # ra_prng2 package (compiles libra_prng2.so with gcc)
pip install ./pip-ra-core  # ra_core package (compiles libra_core_engine.so + Cython BitGenerator)
```

Because compilation uses `-march=native`, install **on the machine it will
run on** — do not copy a wheel built on one machine to another with a
different CPU.

```python
from bindings.cpython._ra_prng2 import RaPrng2

r = RaPrng2(seed=42)
r.next()        # one output
r.advance(10)   # advance 10 outputs, return the last
```

For the `ra_core` Python API (`eval_k1`, `generate`, `generate_x8`,
`RaCoreEngine`, `RaCoreBitGenerator`), see [`pip-ra-core/README.md`](pip-ra-core/README.md).
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the build workflow and coding conventions.

## Statistical validation

Results below are for `ra_prng2` (`benchmarks/results/ra_prng2_original/`):

* **Dieharder**: 113 PASSED, 2 WEAK, 0 FAILED — WEAK flags are expected and
  also appear in Xoshiro, PCG, and ChaCha20
* **NIST STS**: full battery passed, all sub-tests within the expected
  pass-rate range
* **TestU01 BigCrush** (v1.2.3): all 160 statistics passed, no failures
* **PractRand**: no anomalies through 128 GB of output

For `ra_core v2` validation (57/57 PractRand regimes, 8 GiB canonical orbit,
114-subtest Dieharder, 2D SAC/BIC, and frozen Golden KATs), see the
[final battery report](experiments/2026-11-04_v2-final-battery/RESULTS.md).

## Performance

Historical `ra_prng2`/`ra_prng3` throughput comparisons on Intel Core i3-1115G4
/ Arch Linux 6.8.7 / GCC 13.2.0 are archived under `benchmarks/comparisons/`:

* `ra_prng2` outperforms ChaCha20 and `/dev/urandom` in raw throughput
* `ra_prng3` (experimental) surpasses Philox4x32 in raw RNG throughput

> **Note:** An earlier pre-2026-09 throughput claim (30,303 MB/s) turned out to
> be non-reproducible (~7,019 MB/s replaying the same formula on this machine),
> and a September 2026 `ra_core v1` `singleblock K=255` figure (20,018 MB/s)
> was later invalidated by a Dead-Code Elimination audit (see
> [`docs/ERRATA.md`](docs/ERRATA.md)). Use only the re-verified `ra_core v2`
> reports linked below for current performance claims.

### `ra_core v2` (C core and Python binding)

The current production formula is `Opsi_A3` + `W2_D2_xag_opt` with the
`eng_v2_E2` stateful engine (`src/ra_core/c/`, wrapped in `pip-ra-core/`). It
was promoted on 2026-11-04 after passing the final statistical battery and
bit-exactness gates; cross-machine validation on a second host was completed on
2026-11-05. The current manuscript is
[`research/2026-9_ra-core-paper/DRAFT_v2.md`](research/2026-9_ra-core-paper/DRAFT_v2.md)
with [`ieee/main.tex`](research/2026-9_ra-core-paper/ieee/main.tex), its
compiled PDF, and the frozen checksum index in
[`REPRODUCIBILITY.md`](research/2026-9_ra-core-paper/REPRODUCIBILITY.md).

For current measurements and their exact scope, see:

* [Final battery, KAT freeze, and production microbenchmark](experiments/2026-11-04_v2-final-battery/RESULTS.md)
* [Stateful OTF engine optimization (`eng_v2_E2`)](experiments/2026-11-04_v2-engine-opt/RESULTS.md)
* [Fairness-audited PRNG shootout and Monte Carlo baselines](experiments/2026-11-04_v2-prng-shootout-and-mc/RESULTS.md) ([fairness audit](experiments/2026-11-04_v2-prng-shootout-and-mc/FAIRNESS_AUDIT.md))
* [Clean same-work Monte Carlo shootout](experiments/2026-11-04_v2-clean-mc-harness/RESULTS.md)
* [Second-machine cross-compiler validation](experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md)
* [Exploratory GPU parity and timing study](experiments/2026-11-05_gpu-batch1/RESULTS.md) (and [initial K1 Colab probe](experiments/2026-11-05_gpu-colab-k1/RESULTS.md))

The GPU study establishes bit-exact CPU/GPU parity on one Blackwell device; it
is not a general GPU performance claim.

#### Historical development and benchmark notes

`ra_core` evolved from the September 2026 `v1` dual-mode addressable design
(`ra_core_orbit` for continuous multi-cycle streams and `ra_core_singleblock`
for bounded `K <= 255` bursts), which introduced 64-bit `(key, addr_hi)`
addressing to eliminate birthday-paradox collisions under repeated 32-bit keys.
Subsequent diagnostics showed that `v1` still suffered from 1D affine
key-mixing coupling under constant-coordinate seeding, an `O(N)` end-of-cycle
reseed scan, and a serial lag-1 feedback dependency; these motivated the
`v2` redesign (`Opsi_A3`, `W2_D2_xag_opt`, and `eng_v2_E2`).

A disassembly audit on 2026-11-02 also found that the earlier September 2026
`20,018 MB/s` (`0.191 ns/word`) throughput claim for `v1` `singleblock K=255`
was a GCC `-O3` Dead-Code Elimination artifact caused by benchmarking with
`raw_stream = NULL`. That correction and earlier methodological errata are
documented in [`docs/ERRATA.md`](docs/ERRATA.md), and the pre-v2 experiment
artifacts remain preserved in pre-v2 git history.

## Current research (`experiments/`)

The seven dated directories retained under `experiments/` preserve the complete
verification and benchmark trail backing `ra_core v2` (`src/ra_core/c/` and
`pip-ra-core/`) and its manuscript (`research/2026-9_ra-core-paper/`). Earlier
pre-v2 exploratory directories are preserved in git history.

## Roadmap

* Review the `ra_core v2` manuscript and archive a reproducible release
  snapshot of the source and evidence.
* Extend the exploratory GPU comparison to controlled devices and optimized
  library baselines if broader GPU performance claims become a goal.
* Study full-generator period and stream relationships formally; the current
  paper reports empirical batteries and reduced toy-model cycles, not a
  general period or independence proof.

## License

This project is licensed under the CC BY-NC-SA 4.0 —
https://creativecommons.org/licenses/by-nc-sa/4.0

## Citation

I'm preparing the official citation format. In the meantime, please link to
this GitHub repository and include the project name and primary author.

Note: neither paper below has actually been presented or published — both
are manuscripts written for conference submission.

* **CSAI2025** (accepted by reviewers for oral presentation & publication,
  but never presented or published due to lack of funding) — "ra_prng2 and
  Beyond: An Array-Based PRNG Architecture for Efficient Random Generation" —
  `research/csai2025_ra_prng/`
* **ICCS2026** (prepared/submitted, not published) — "Array-Native
  Randomness: An Efficient Unified Permutation-Generation Architecture with
  Deterministic and High Sensitivity Properties" — `research/iccs2026_ra_prng/`
* **`ra_core v2` manuscript (draft)** — "`ra_core v2`: Deferred Prefix
  Computation and 2D Cross-Pre-Mixed Seeding for Addressable
  Permutation-Stream Pseudorandom Number Generation" —
  [`research/2026-9_ra-core-paper/DRAFT_v2.md`](research/2026-9_ra-core-paper/DRAFT_v2.md)

## Contribution

External contributions are not officially open yet. However, technical
feedback and discussion are very welcome via the discussion forum.

## Author

This project was created and maintained by **Hamas A. Rahman** (online /
GitHub handle: **[@hamzy-hams](https://github.com/hamzy-hams)**), an
independent researcher in random number generation, computer science, and
the Collatz Conjecture problem. The repository is hosted at
`ooalahhq/ra_prng`.

- **Author**: Hamas A. Rahman
- **GitHub**: [github.com/hamzy-hams](https://github.com/hamzy-hams)
- **Discord**: `hamzyhams`
- **Contact Email**: nexthamas95@gmail.com
