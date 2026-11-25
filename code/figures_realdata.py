"""Figures 4 and 5 of the paper (Section 7, fraction-subtraction data).

Figure 4: exact reference hierarchy and two-stage greedy hierarchy (entropic
intensity, alpha = 0.7). Figure 5: bootstrap support of each implication for the
exact reference and for IITA; the supports are read from
results/final/fraction_support_{exact,iita}.npy, written by final_analysis.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import draw_hierarchy, pair_status
from ohier import entropic_intensity, exact_heap, greedy_two_step, impose_frequency_order, pair_cohesion

ROOT = Path(__file__).resolve().parents[1]


def load_fraction():
    X = pd.read_csv(ROOT / "data/fraction_data.csv").values
    Q = pd.read_csv(ROOT / "data/fraction_qmatrix.csv", index_col=0).values
    return X, Q


def fig_hierarchies(path=None, alpha=0.7):
    """Figure 4."""
    X, Q = load_fraction()
    M = impose_frequency_order(entropic_intensity(X), X)
    A = pair_cohesion(M)
    HG = exact_heap(A)
    G = greedy_two_step(M, alpha=alpha, A=A)
    sub = ["".join(str(s + 1) for s in np.flatnonzero(Q[v])) for v in range(X.shape[1])]
    lab = [f"I{i + 1}" for i in range(X.shape[1])]
    with plt.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
        draw_hierarchy(axes[0], HG, lab, f"Exact reference hierarchy $\\mathcal{{H}}_G$ "
                       f"({len(HG.merges)} merges, {len(HG.roots)} roots)", sub=sub)
        draw_hierarchy(axes[1], G.hierarchy, lab, f"Two-stage greedy hierarchy $\\mathcal{{H}}_N$, "
                       f"$\\alpha={alpha}$ ({len(G.hierarchy.merges)} merges, {len(G.paths)} roots)", sub=sub)
        fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def fig_support(path=None, results=ROOT / "results/final"):
    """Figure 5."""
    X, Q = load_fraction()
    order = np.argsort(X.mean(0))
    labf = [f"I{i + 1}" for i in order]
    n = X.shape[1]
    with plt.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), constrained_layout=True)
        for ax, (name, title) in zip(axes, (("exact", "Exact reference hierarchy $\\mathcal{H}_G$"), ("iita", "IITA"))):
            S = np.load(Path(results) / f"fraction_support_{name}.npy")[np.ix_(order, order)]
            im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
            for a in range(n):
                for b in range(n):
                    if S[a, b] >= 0.5:
                        st = pair_status(Q, order[a], order[b])
                        mk = {"consistent": "+", "equal": "=", "reversed": "x", "incomparable": "o"}[st]
                        ax.text(b, a, mk, ha="center", va="center", fontsize=5.5,
                                color="white" if S[a, b] > 0.7 else "black")
            ax.set_xticks(range(n), labf, rotation=90, fontsize=5.5)
            ax.set_yticks(range(n), labf, fontsize=5.5)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("implied item $y$", fontsize=7)
        axes[0].set_ylabel("implying item $x$", fontsize=7)
        cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        cb.set_label("bootstrap support", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
