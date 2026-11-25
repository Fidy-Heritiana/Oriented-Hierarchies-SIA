# Geometrically cohesive oriented hierarchies: code and data

Code and data for the application to real binary data (Section 7) of the paper:

> F. Andrianarivony *et al.* **Geometrically cohesive oriented hierarchies: exact quadratic computation and limits of a marginal selection.** Manuscript submitted for publication.

The repository contains:
- exact implementations of the maximum-cohesion rule of statistical implicative analysis (full scan, heap and matrix tournament);
- the two-stage greedy (marginal) selection;
- the analyses of three public data sets of test responses.

The synthetic experiments of Section 6 come from a separate code base, available from the corresponding author.

## Contents

| Path | Content |
|---|---|
| `code/ohier.py` | Implication intensities (classical and entropic), pair and class cohesion, exact rule (scan, heap, tournament), two-stage greedy (Algorithm 1), Jaccard and interval diagnostics |
| `code/fast.py` | Vectorised exact implementations used for the running times of Table 14 |
| `code/iita.py` | Inductive item tree analysis, minimized corrected version (Sargin & Ünlü 2009) |
| `code/qvalid.py` | Saturated G-DINA model (EM) and PVAF Q-matrix validation (de la Torre & Chiu 2016) |
| `code/analysis.py` | Helper functions (Q-matrix status of an implication, drawing of oriented hierarchies) |
| `code/final_analysis.py` | Protocol of Section 7 applied to the three data sets (bootstrap, B = 1000) |
| `code/section7_tables.py` | LaTeX tables and figures of Section 7, written from the JSON results |
| `code/verify.py` | Checks that the scan, the heap and the tournament return identical merge sequences |
| `code/bench_scale.py` | Running times for n up to 3200 (Table 14) |
| `data/` | Response matrices, Q-matrices and TIMSS item descriptions (see `data/README.md`) |
| `results/final/` | Results used in the paper: JSON files, bootstrap support matrices (`.npy`), figures, `section7_tables.tex` |

## Reproducing the results

Python 3.11 or later:

```bash
pip install -r requirements.txt
```

Check that the three exact implementations agree (547 inputs, exact ties included):

```bash
python3 code/verify.py
```

Run the Section 7 analyses on the three data sets (about 2 minutes on one core):

```bash
python3 code/final_analysis.py
```

Run the scaling benchmark for Table 14 (about 15 minutes):

```bash
python3 code/bench_scale.py
```

Write the Section 7 tables and figures:

```bash
python3 code/section7_tables.py
```

All random steps use fixed seeds; rerunning the commands reproduces the files in `results/final/`.

## Conventions

- **Intensity.** The entropic implication intensity (Gras et al. 2001) is used in the main analysis and the classical (Poisson) intensity as a sensitivity analysis. An implication x → y is considered only when x is solved less often than y.
- **Cohesion.** Pair cohesion is g(p) = sqrt(1 − H2(p)²) for p ≥ 1/2 and 0 otherwise. The cohesion of an ordered class is the geometric mean of its forward pair cohesions.
- **Ties.** Ties are broken on the key (−h(A,B), id(A), id(B)). Singletons have identifiers 0, …, n−1, and new classes receive n, n+1, … in order of creation.
- **Greedy selection.** The greedy uses α = 0.7; the other thresholds are in `results/final/*.json`.

## Licence

- Code: [to be chosen by the authors].
- Data: redistributed from the R package CDM under its licence, GPL (≥ 2); see `data/README.md`.
