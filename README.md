# Geometrically cohesive oriented hierarchies: code and data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fidy-Heritiana/Oriented-Hierarchies-SIA/blob/main/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Fidy-Heritiana/Oriented-Hierarchies-SIA/main?labpath=demo.ipynb)
[![Streamlit app](https://img.shields.io/badge/Streamlit-Live%20app-FF4B4B?logo=streamlit&logoColor=white)](https://oriented-hierarchies-sia.streamlit.app/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24544349.svg)](https://doi.org/10.5281/zenodo.24544349)

Code, data and figures for the paper:

> F. H. Andrianarivony *et al.* **Geometrically cohesive oriented hierarchies: exact quadratic computation and limits of a marginal selection.** Manuscript submitted for publication.

Archived version of this repository: [doi:10.5281/zenodo.24544349](https://doi.org/10.5281/zenodo.24544349).

The repository contains:
- exact implementations of the maximum-cohesion rule of statistical implicative analysis (full scan, heap and matrix tournament);
- the two-stage greedy (marginal) selection;
- the synthetic campaigns of Section 6 (regimes U, P and O);
- the analyses of three public data sets of test responses (Section 7);
- a demonstration notebook that reproduces the five figures of the paper;
- an interactive Streamlit app.

A related repository accompanies an earlier study of the greedy algorithm: [Study of the greedy algorithm for constructing an oriented hierarchy](https://github.com/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy) ([doi:10.5281/zenodo.24129903](https://doi.org/10.5281/zenodo.24129903)). The two code bases use different pair cohesions; this repository uses g(p) = sqrt(1 − H2(p)²), as in Definition 1 of the paper.

## Reproducing the figures

**In the browser, without installation:** open `demo.ipynb` with the Colab or Binder badge above and run all cells. The notebook runs the synthetic campaigns in quick mode (about one minute); set `QUICK = False` for the full protocol.

**From the command line:**

```bash
pip install -r requirements.txt
```

Reduced campaigns, about one minute:

```bash
python3 code/make_figures.py --quick
```

Full protocol of Section 6.3, about 10–15 minutes:

```bash
python3 code/make_figures.py
```

The figures are written to `figures/` with the file names used in the manuscript:

| Figure | File | Content |
|---|---|---|
| 1 | `timing_en.pdf` | Running times of the greedy and of the three exact implementations (regimes U, P, O) |
| 2 | `fidelity_en.pdf` | Effect of the threshold α on hierarchical fidelity and class recovery (U, n = 200) |
| 3 | `growth_en.pdf` | Number of components K and I/n² of the greedy forest (U, α = 0.7) |
| 4 | `realdata_hierarchies_en.pdf` | Exact and greedy hierarchies on the fraction-subtraction data |
| 5 | `realdata_support_en.pdf` | Bootstrap support of the implications, exact reference and IITA |

The full run also writes Tables 4–7 (LaTeX) and the raw campaign results to `results/synthetic/`. Running times depend on the machine; the other quantities are reproduced up to the sampling variability of the random matrices.

## Interactive app

The Streamlit app explores the two constructions interactively: one random matrix of regime U, P or O (hierarchies, Jaccard distances, zero-cohesion merges, intervals), the effect of the threshold α (Figure 2), running times (Figure 1), the growth of the greedy forest (Figure 3), and the three real data sets with their Q-matrices and bootstrap supports (Figures 4–5).

Online: https://oriented-hierarchies-sia.streamlit.app/. Locally:

```bash
pip install -r requirements.txt
```

```bash
streamlit run streamlit_app.py
```

## Reproducing the other results

Check that the three exact implementations agree (547 inputs, exact ties included):

```bash
python3 code/verify.py
```

Run the Section 7 analyses on the three data sets (bootstrap B = 1000, about 2 minutes):

```bash
python3 code/final_analysis.py
```

Write the Section 7 tables (Tables 8–14):

```bash
python3 code/section7_tables.py
```

Run the scaling benchmark for Table 14 (about 15 minutes):

```bash
python3 code/bench_scale.py
```

All random steps use fixed seeds.

## Contents

| Path | Content |
|---|---|
| `demo.ipynb` | Demonstration notebook: Figures 1–5 |
| `streamlit_app.py` | Interactive Streamlit app |
| `code/ohier.py` | Implication intensities (classical and entropic), pair and class cohesion, exact rule (scan, heap, tournament), two-stage greedy (Algorithm 1), Jaccard and interval diagnostics |
| `code/fast.py` | Vectorised exact implementations (Table 14) and the row-wise full scan used in Figure 1 |
| `code/synthetic.py` | Synthetic campaigns of Section 6 and Figures 1–3, Tables 4–7 |
| `code/figures_realdata.py` | Figures 4 and 5 |
| `code/make_figures.py` | Command-line reproduction of Figures 1–5 |
| `code/iita.py` | Inductive item tree analysis, minimized corrected version (Sargin & Ünlü 2009) |
| `code/qvalid.py` | Saturated G-DINA model (EM) and PVAF Q-matrix validation (de la Torre & Chiu 2016) |
| `code/analysis.py` | Helper functions (Q-matrix status of an implication, drawing of oriented hierarchies) |
| `code/final_analysis.py` | Protocol of Section 7 applied to the three data sets |
| `code/section7_tables.py` | LaTeX tables of Section 7, written from the JSON results |
| `code/verify.py` | Equivalence checks of the exact implementations |
| `code/bench_scale.py` | Running times for n up to 3200 (Table 14) |
| `code/build_notebook.py` | Builds `demo.ipynb` |
| `data/` | Response matrices, Q-matrices and TIMSS item descriptions (see `data/README.md`) |
| `figures/` | Figures 1–5 produced by the full run |
| `results/final/` | Section 7 results: JSON files, bootstrap support matrices (`.npy`), figures, tables |
| `results/synthetic/` | Section 6 campaigns: raw results (CSV) and Tables 4–7 |

## Conventions

- **Intensity.** For the real data, the entropic implication intensity (Gras et al. 2001) is used in the main analysis and the classical (Poisson) intensity as a sensitivity analysis. An implication x → y is considered only when x is solved less often than y.
- **Cohesion.** Pair cohesion is g(p) = sqrt(1 − H2(p)²) for p ≥ 1/2 and 0 otherwise. The cohesion of an ordered class is the geometric mean of its forward pair cohesions.
- **Ties.** Ties are broken on the key (−h(A,B), id(A), id(B)). Singletons have identifiers 0, …, n−1, and new classes receive n, n+1, … in order of creation.
- **Greedy selection.** The greedy uses α = 0.7 unless stated otherwise.

## Licence

- Code: MIT (see `LICENSE`).
- Data: redistributed from the R package CDM under its licence, GPL (≥ 2); see `data/README.md`.
