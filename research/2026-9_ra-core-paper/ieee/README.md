# Build Instructions for `ra_core` IEEE Conference Paper

This directory contains the IEEE two-column `ra_core v2` conference manuscript (`IEEEtran`, `conference` option), synchronized with `../DRAFT_v2.md` on 5 November 2026. No specific conference or journal venue or page limit has been selected; the standard `IEEEtran` conference format is used as the default baseline.

See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the production source,
measurement archive, checks, and SHA-256 manifest used for this draft.

## Files in This Directory

- `main.tex` — Primary LaTeX manuscript (`\documentclass[conference]{IEEEtran}`).
- `references.bib` — BibTeX bibliography database formatted for `IEEEtran.bst`.
- `avalanche_curve_orbit.png` — Figure 1 (from pre-v2 experiment `2026-9-6_orbit-address-widen-64bit-promotion`, preserved in pre-v2 git history).
- `heatmap_orbit_N256_LOWONLY.png` — Figure 2 (from pre-v2 experiment `2026-9-11_state-size-pareto-frontier`, preserved in pre-v2 git history).
- `main.pdf` — Compiled PDF output.
- `README.md` — These build instructions.

## How to Compile

### Option 1: Using Tectonic (User-Space Setup Installed at `~/.local/bin/tectonic`)

Tectonic automatically runs XeTeX and BibTeX passes until cross-references and citations stabilize:

```bash
cd research/2026-9_ra-core-paper/ieee
tectonic -X compile main.tex
```

To retain intermediate `.log`, `.aux`, and `.bbl` files for inspection:

```bash
tectonic -X compile main.tex --keep-logs --keep-intermediates
```

### Option 2: Using Standard TeX Live (`pdflatex` + `bibtex`)

If compiling in an environment with standard TeX Live on `PATH`:

```bash
cd research/2026-9_ra-core-paper/ieee
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
