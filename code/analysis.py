"""Substantive application: oriented hierarchies of test items vs. Q-matrix.

For a binary response table (students x items) and a Q-matrix (items x skills):
  1. entropic implication intensities with the frequency convention n_a <= n_b,
     pair cohesions a = g(M);
  2. exact reference hierarchy (heap; checked against tournament and scan);
  3. two-step greedy for several thresholds alpha;
  4. hierarchy comparisons (Jaccard, zero-cohesion merges, interval recall);
  5. substantive validation: every forward pair u -> v of a class says
     "success on u implies success on v".  Under a conjunctive (DINA-type)
     reading of the Q-matrix this is *consistent* when skills(u) contains
     skills(v), *reversed* when skills(u) is strictly contained in skills(v),
     *incomparable* otherwise.  A permutation test (items relabelled at random
     with respect to the Q-matrix) gives the chance level.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

# A non-interactive backend is selected automatically when no display is available.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from ohier import (  # noqa: E402
    classical_intensity,
    compare,
    entropic_intensity,
    exact_heap,
    exact_scan,
    exact_tournament,
    greedy_two_step,
    impose_frequency_order,
    pair_cohesion,
)

ROOT = Path(__file__).resolve().parents[1]


def forward_pairs_of_hierarchy(H):
    """Forward pairs created by the merges (= forward pairs of final classes)."""
    pairs = []
    for a, b, _ in H.merges:
        for u in a:
            for v in b:
                pairs.append((u, v))
    return pairs


def pair_status(Q, u, v):
    qu, qv = Q[u].astype(bool), Q[v].astype(bool)
    sup = np.all(qu >= qv)
    sub = np.all(qu <= qv)
    if sup and sub:
        return "equal"
    if sup:
        return "consistent"
    if sub:
        return "reversed"
    return "incomparable"


def q_scores(Q, pairs):
    c = {"consistent": 0, "equal": 0, "reversed": 0, "incomparable": 0}
    for u, v in pairs:
        c[pair_status(Q, u, v)] += 1
    tot = len(pairs)
    comp = c["consistent"] + c["reversed"]
    c["total"] = tot
    c["consistency_rate"] = c["consistent"] / comp if comp else float("nan")
    c["share_consistent_or_equal"] = (c["consistent"] + c["equal"]) / tot if tot else float("nan")
    c["share_reversed"] = c["reversed"] / tot if tot else float("nan")
    return c


def permutation_test(Q, pairs, n_perm=10000, seed=1):
    """Chance distribution of the share of consistent-or-equal pairs."""
    rng = np.random.default_rng(seed)
    obs = q_scores(Q, pairs)["share_consistent_or_equal"]
    n = Q.shape[0]
    null = np.empty(n_perm)
    for t in range(n_perm):
        p = rng.permutation(n)
        null[t] = q_scores(Q[p], pairs)["share_consistent_or_equal"]
    pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, float(null.mean()), float(np.quantile(null, 0.95)), float(pval)


def closure(Q, prereq):
    """Close skill sets under prerequisites: prereq = {skill: [skills it requires]}."""
    Q = Q.copy()
    changed = True
    while changed:
        changed = False
        for s, req in prereq.items():
            for r in req:
                add = (Q[:, s] == 1) & (Q[:, r] == 0)
                if add.any():
                    Q[add, r] = 1
                    changed = True
    return Q


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def draw_hierarchy(ax, H, labels, title, A=None, sub=None):
    order = [v for r in sorted(H.roots, key=lambda s: min(s)) for v in r]
    # keep root order as returned but put larger classes first for readability
    roots = sorted(H.roots, key=lambda s: (-len(s), s[0]))
    order = [v for r in roots for v in r]
    xpos = {v: i for i, v in enumerate(order)}
    node_x = {(v,): float(xpos[v]) for v in order}
    node_y = {(v,): 0.0 for v in order}
    for step, (a, b, coh) in enumerate(H.merges, start=1):
        a, b = tuple(a), tuple(b)
        xa, xb = node_x[a], node_x[b]
        ya, yb = node_y[a], node_y[b]
        y = float(step)
        col = "#c0392b" if coh == 0.0 else "#1f4e79"
        ax.plot([xa, xa, xb, xb], [ya, y, y, yb], color=col, lw=1.6)
        ax.annotate("", xy=(xb, y), xytext=(xa, y),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ab = a + b
        node_x[ab] = (xa + xb) / 2
        node_y[ab] = y
    ax.set_xticks(range(len(order)))
    lab = [labels[v] if sub is None else f"{labels[v]}\n{sub[v]}" for v in order]
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("merge step")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------


def analyse(name, X, Q, item_labels, skill_names, alphas, prereq=None, n_perm=10000, draw_alpha=None,
            item_text=None):
    out = {"dataset": name, "N": int(X.shape[0]), "n_items": int(X.shape[1])}
    p = X.mean(axis=0)
    res_dir = ROOT / "results" / name
    res_dir.mkdir(parents=True, exist_ok=True)

    # saturation diagnostics: classical vs entropic
    sat = {}
    for iname, f in [("classical", classical_intensity), ("entropic", entropic_intensity)]:
        M = impose_frequency_order(f(X), X)
        A = pair_cohesion(M)
        off = ~np.eye(X.shape[1], dtype=bool)
        sat[iname] = {
            "positive_cohesions": int((A[off] > 0).sum()),
            "cohesions_equal_to_1": int((A[off] == 1.0).sum()),
        }
        if iname == "classical":
            Hc = exact_heap(A)
            sat[iname]["merges_at_cohesion_1"] = int(sum(1 for *_, c in Hc.merges if c == 1.0))
            sat[iname]["merges"] = len(Hc.merges)
    out["saturation"] = sat

    M = impose_frequency_order(entropic_intensity(X), X)
    A = pair_cohesion(M)
    np.save(res_dir / "intensity_entropic.npy", M)

    t0 = time.perf_counter()
    HG = exact_heap(A)
    out["time_exact_heap_s"] = time.perf_counter() - t0
    HT = exact_tournament(A)
    HS = exact_scan(A)
    assert HG.merge_sequence() == HT.merge_sequence() == HS.merge_sequence()
    out["exact_implementations_agree"] = True

    Qs = {"Q": Q}
    if prereq is not None:
        Qs["Q_closed"] = closure(Q, prereq)

    def describe(H):
        rows = []
        for a, b, c in H.merges:
            rows.append({
                "A": [item_labels[v] for v in a],
                "B": [item_labels[v] for v in b],
                "cohesion": round(float(c), 4),
            })
        return rows

    exact = {
        "K": len(HG.roots),
        "roots": [[item_labels[v] for v in r] for r in HG.roots],
        "merges": describe(HG),
    }
    fp = forward_pairs_of_hierarchy(HG)
    for qn, QQ in Qs.items():
        exact[qn] = q_scores(QQ, fp)
        obs, mu, q95, pv = permutation_test(QQ, fp, n_perm=n_perm)
        exact[qn]["perm_null_mean"] = mu
        exact[qn]["perm_null_q95"] = q95
        exact[qn]["perm_pvalue"] = pv
    out["exact"] = exact

    greedy = {}
    for alpha in alphas:
        G = greedy_two_step(M, alpha=alpha, A=A)
        cmp_ = compare(HG, G)
        fpg = forward_pairs_of_hierarchy(G.hierarchy)
        g = {"compare": cmp_, "paths": [[item_labels[v] for v in pth] for pth in G.paths],
             "merges": describe(G.hierarchy), "candidates": G.candidates}
        for qn, QQ in Qs.items():
            g[qn] = q_scores(QQ, fpg)
            obs, mu, q95, pv = permutation_test(QQ, fpg, n_perm=n_perm)
            g[qn]["perm_null_mean"] = mu
            g[qn]["perm_null_q95"] = q95
            g[qn]["perm_pvalue"] = pv
        greedy[str(alpha)] = g
    out["greedy"] = greedy

    # item table
    items = []
    for v in range(X.shape[1]):
        items.append({
            "item": item_labels[v],
            "p_correct": round(float(p[v]), 3),
            "skills": [skill_names[s] for s in np.flatnonzero(Q[v])],
            **({"text": item_text[v]} if item_text else {}),
        })
    out["items"] = items

    with open(res_dir / "results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    # figure: exact vs greedy
    if draw_alpha is not None:
        G = greedy_two_step(M, alpha=draw_alpha, A=A)
        sub = ["".join(str(s + 1) for s in np.flatnonzero(Q[v])) for v in range(X.shape[1])]
        fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.42 * X.shape[1]), 7.5))
        draw_hierarchy(axes[0], HG, item_labels, f"{name}: exact maximum-cohesion hierarchy "
                       f"({len(HG.merges)} merges, {len(HG.roots)} roots)", sub=sub)
        zero = sum(1 for *_, c in G.hierarchy.merges if c == 0.0)
        draw_hierarchy(axes[1], G.hierarchy, item_labels,
                       f"{name}: two-step greedy, alpha = {draw_alpha} "
                       f"({len(G.hierarchy.merges)} merges, {zero} with zero cohesion in red)", sub=sub)
        fig.text(0.01, 0.005, "Second label line: skills required by the item (Q-matrix). "
                 "Arrows: A -> B concatenation (success on A implies success on B).", fontsize=7)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(res_dir / "hierarchies.png", dpi=200)
        fig.savefig(res_dir / "hierarchies.pdf")
        plt.close(fig)
    return out


FRACTION_SKILLS = [
    "a1 convert whole number to fraction",
    "a2 separate whole number from fraction",
    "a3 simplify before subtracting",
    "a4 find common denominator",
    "a5 borrow from whole number part",
    "a6 column borrow",
    "a7 subtract numerators",
    "a8 reduce answer to simplest form",
]
# Item wording from the edmdata R package documentation (Tatsuoka 1984; Tatsuoka 2002).
FRACTION_TEXT = [
    "5/3 - 3/4", "3/4 - 3/8", "5/6 - 1/9", "3 1/2 - 2 3/2", "4 3/5 - 3 4/10",
    "6/7 - 4/7", "3 - 2 1/5", "2/3 - 2/3", "3 7/8 - 2", "4 4/12 - 2 7/12",
    "4 1/3 - 2 4/3", "11/8 - 1/8", "3 3/8 - 2 5/6", "3 4/5 - 3 2/5", "2 - 1/3",
    "4 5/7 - 1 4/7", "7 3/5 - 2 4/5", "4 1/10 - 2 8/10", "4 - 1 4/3", "4 1/3 - 1 5/3",
]
ECPE_SKILLS = ["s1 morphosyntactic", "s2 cohesive", "s3 lexical"]


def main():
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    Xf = pd.read_csv(ROOT / "data/fraction_data.csv").values
    Qf = pd.read_csv(ROOT / "data/fraction_qmatrix.csv", index_col=0).values
    lab_f = [f"I{i + 1}" for i in range(Xf.shape[1])]
    rf = analyse("fraction", Xf, Qf, lab_f, FRACTION_SKILLS, alphas, draw_alpha=0.7,
                 item_text=FRACTION_TEXT)

    Xe = pd.read_csv(ROOT / "data/ecpe_data.csv").values
    Qe = pd.read_csv(ROOT / "data/ecpe_qmatrix.csv", index_col=0).values
    lab_e = [f"E{i + 1}" for i in range(Xe.shape[1])]
    # Linear attribute hierarchy reported for ECPE (Templin & Bradshaw 2014):
    # morphosyntactic (s1) requires cohesive (s2), which requires lexical (s3).
    re_ = analyse("ecpe", Xe, Qe, lab_e, ECPE_SKILLS, alphas, prereq={0: [1], 1: [2]}, draw_alpha=0.7)
    return rf, re_


if __name__ == "__main__":
    main()
    print("done")
