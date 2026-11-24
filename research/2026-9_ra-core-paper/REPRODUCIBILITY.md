# `ra_core v2` paper snapshot

This index ties the 5 November 2026 `ra_core v2` manuscript to the production
implementation and the archived measurements in this repository. All paths
below are relative to the repository root. `MANIFEST.sha256` freezes the key
source, report, data, and manuscript files; the Git commit records the rest of
the experiment archive.

## Implementation and verification

- Production formula: `src/ra_core/c/ra_core.c` and `ra_core.h`.
- Stateful engine: `src/ra_core/c/ra_core_engine.c` and `ra_core_engine.h`.
- Python binding: `pip-ra-core/ra_core/` and `pip-ra-core/tests/test_ra_core.py`.
- Final production promotion, KAT, Dieharder, and PractRand evidence:
  `experiments/2026-11-04_v2-final-battery/RESULTS.md` and its `data/` files.
- Cross-machine and cross-compiler validation:
  `experiments/2026-11-05_v2-vps-final-benchmark/RESULTS.md`, `data/`, and `logs/`.

## Workload and GPU evidence

- Clean Monte Carlo workload definitions and raw summaries:
  `experiments/2026-11-04_v2-clean-mc-harness/`.
- Shootout methodology and fairness audit:
  `experiments/2026-11-04_v2-prng-shootout-and-mc/FAIRNESS_AUDIT.md`.
- Exploratory CPU/GPU parity and performance:
  `experiments/2026-11-05_gpu-batch1/RESULTS.md`, `batch1.cu`,
  `host_parity.cpp`, and `runs/2026-11-05_colab_blackwell/`.

## Manuscript

- Narrative draft: `research/2026-9_ra-core-paper/DRAFT_v2.md`.
- IEEE source and bibliography: `research/2026-9_ra-core-paper/ieee/main.tex`
  and `references.bib`.
- Compiled artifact: `research/2026-9_ra-core-paper/ieee/main.pdf`.

## Local checks

From the repository root:

```bash
sha256sum -c research/2026-9_ra-core-paper/MANIFEST.sha256
make build-c
./build/bin/ra_core validate
PYTHONPATH=pip-ra-core pytest -q tests/ pip-ra-core/tests/
cd research/2026-9_ra-core-paper/ieee && tectonic -X compile main.tex
```

The statistical batteries, Monte Carlo workloads, VPS measurements, and GPU
run are archived results, not tests automatically rerun by these commands.
The GPU results concern one Blackwell device and a direct Philox reference;
they do not establish a general GPU performance advantage.
