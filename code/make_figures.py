"""Reproduce Figures 1-5 (and Tables 4-7) of the paper.

    python3 code/make_figures.py            # full protocol of Section 6.3 (about 10-15 minutes)
    python3 code/make_figures.py --quick    # reduced campaigns for a demonstration (about 1 minute)
    python3 code/make_figures.py --bootstrap  # also rerun the Section 7 bootstrap (B = 1000, about 2 minutes)

Figures are written to figures/ with the file names used in the manuscript:
timing_en.pdf (Fig. 1), fidelity_en.pdf (Fig. 2), growth_en.pdf (Fig. 3),
realdata_hierarchies_en.pdf (Fig. 4) and realdata_support_en.pdf (Fig. 5).
Tables and the raw campaign results go to results/synthetic/ (or results/synthetic_quick/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import figures_realdata as real  # noqa: E402
import synthetic as syn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

QUICK = dict(timing=dict(ns=(20, 40, 80, 160), matrices=3, repeats=1),
             fidelity=dict(ns=(50, 200), matrices=10),
             growth=dict(ns=(50, 100, 200, 400), matrices=10))
FULL = dict(timing={}, fidelity={}, growth={})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="reduced campaigns (demonstration)")
    p.add_argument("--bootstrap", action="store_true", help="rerun the Section 7 analysis before Figure 5")
    p.add_argument("--out", default=str(ROOT / "figures"), help="output folder for the figures")
    a = p.parse_args(argv)
    cfg = QUICK if a.quick else FULL
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results" / ("synthetic_quick" if a.quick else "synthetic")
    res.mkdir(parents=True, exist_ok=True)

    timing = syn.timing_campaign(**cfg["timing"])
    fidelity = syn.fidelity_campaign(**cfg["fidelity"])
    growth = syn.growth_campaign(**cfg["growth"])
    timing.to_csv(res / "timing.csv", index=False)
    fidelity.to_csv(res / "fidelity.csv", index=False)
    growth.to_csv(res / "growth.csv", index=False)

    syn.fig_timing(timing, out / "timing_en.pdf")
    syn.fig_fidelity(fidelity, path=out / "fidelity_en.pdf")
    syn.fig_growth(growth, out / "growth_en.pdf")
    ratios, means, bias = syn.save_tables(timing, fidelity, growth, res)

    if a.bootstrap:
        import final_analysis
        final_analysis.main()
    real.fig_hierarchies(out / "realdata_hierarchies_en.pdf")
    real.fig_support(out / "realdata_support_en.pdf")

    print("\nTable 5/6 means (n = 200, alpha = 0.7):")
    print(means.round(3).to_string())
    print("\nRelative bias of n(n+1)/(K_N+1):")
    print(bias.round(3).to_string())
    print("\nMedian paired ratios at the largest n:")
    print(ratios.round(2).to_string())
    print("\nFigures written to", out)


if __name__ == "__main__":
    main()
