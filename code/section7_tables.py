"""Tables and figures of Section 7 (application to real binary data) of the paper.

Reads results/final/*.json (written by final_analysis.py and bench_scale.py) and writes
results/final/section7_tables.tex and the two figures realdata_*_en.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import draw_hierarchy  # noqa: E402
from ohier import compare, entropic_intensity, exact_heap, greedy_two_step, impose_frequency_order, intervals, pair_cohesion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
FIN = OUT
DS = [("fraction", "Fractions"), ("ecpe", "ECPE"), ("timss", "TIMSS")]
R = {k: json.load(open(FIN / f"{k}.json")) for k, _ in DS}
SC = json.load(open(FIN / "scaling.json"))


def f3(x):
    return "--" if x != x else f"{x:.3f}"


def f2(x):
    return "--" if x != x else f"{x:.2f}"


def ci(v, d=2):
    fmt = f"{{:.{d}f}}"
    s = f"{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]"
    return s.replace("-", "$-$")


def table(label, caption, colspec, header, rows, small=True):
    return ("\\begin{table}[htbp]\n\t\\centering\n\t\\caption{" + caption + "}\\label{" + label + "}\n"
            + ("\t\\small\n\t\\setlength{\\tabcolsep}{4pt}\n" if small else "")
            + "\t\\begin{tabular}{" + colspec + "}\\toprule\n\t\t" + header + "\\\\\\midrule\n\t\t"
            + "\n\t\t".join(rows) + "\n\t\t\\bottomrule\n\t\\end{tabular}\n\\end{table}\n")


# ---------------------------------------------------------------- full-sample fidelity diagnostics
fid = {}
for k, _ in DS:
    X = pd.read_csv(ROOT / f"data/{k}_data.csv").values
    M = impose_frequency_order(entropic_intensity(X), X)
    A = pair_cohesion(M)
    HG = exact_heap(A)
    G = greedy_two_step(M, alpha=0.7, A=A)
    fid[k] = compare(HG, G)
    if k == "fraction":
        Q = pd.read_csv(ROOT / "data/fraction_qmatrix.csv", index_col=0).values
        sub = ["".join(str(s + 1) for s in np.flatnonzero(Q[v])) for v in range(20)]
        lab = [f"I{i + 1}" for i in range(20)]
        plt.rcParams.update({"font.size": 7})
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6))
        draw_hierarchy(axes[0], HG, lab, f"Exact reference hierarchy $\\mathcal{{H}}_G$ "
                       f"({len(HG.merges)} merges, {len(HG.roots)} roots)", sub=sub)
        draw_hierarchy(axes[1], G.hierarchy, lab, f"Two-stage greedy hierarchy $\\mathcal{{H}}_N$, "
                       f"$\\alpha=0.7$ ({len(G.hierarchy.merges)} merges, {len(G.paths)} roots)", sub=sub)
        fig.tight_layout()
        fig.savefig(OUT / "realdata_hierarchies_en.pdf")
        plt.close(fig)
# Figure: bootstrap support of implications, exact reference and IITA (readable at text width)
from analysis import pair_status  # noqa: E402
Xf = pd.read_csv(ROOT / "data/fraction_data.csv").values
Qf = pd.read_csv(ROOT / "data/fraction_qmatrix.csv", index_col=0).values
order = np.argsort(Xf.mean(0))
labf = [f"I{i + 1}" for i in order]
plt.rcParams.update({"font.size": 7})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), constrained_layout=True)
for ax, (name, title) in zip(axes, (("exact", "Exact reference hierarchy $\\mathcal{H}_G$"), ("iita", "IITA"))):
    S = np.load(FIN / f"fraction_support_{name}.npy")[np.ix_(order, order)]
    im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
    for a in range(20):
        for b in range(20):
            if S[a, b] >= 0.5:
                st = pair_status(Qf, order[a], order[b])
                mk = {"consistent": "+", "equal": "=", "reversed": "x", "incomparable": "o"}[st]
                ax.text(b, a, mk, ha="center", va="center", fontsize=5.5, color="white" if S[a, b] > 0.7 else "black")
    ax.set_xticks(range(20), labf, rotation=90, fontsize=5.5)
    ax.set_yticks(range(20), labf, fontsize=5.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("implied item $y$", fontsize=7)
axes[0].set_ylabel("implying item $x$", fontsize=7)
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label("bootstrap support", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "realdata_support_en.pdf")
plt.close(fig)

# ---------------------------------------------------------------- tables
rows = []
for k, lab in DS:
    r = R[k]
    s = r["saturation_classical"]
    K = len(pd.read_csv(ROOT / f"data/{k}_qmatrix.csv", index_col=0).columns)
    p = pd.read_csv(ROOT / f"data/{k}_data.csv").values.mean()
    rows.append(f"{lab} & {r['N']} & {r['n_items']} & {K} & {p:.2f} & {s['positive_pair_cohesions']} & "
                f"{100 * s['share_pair_cohesions_equal_1']:.1f} & {s['merges_classical']} & "
                f"{100 * s['share_merges_cohesion_ge_0999']:.1f}\\\\")
tab_data = table("tab:realdata",
                 "Real data sets: number of examinees $m$, items $n$, attributes $\\kappa$ and mean proportion of "
                 "successes $\\bar p$. Classical intensity with $m_x\\leq m_y$: number of positive pair cohesions, "
                 "percentage of them equal to one in floating point, number of merges of the exact reference "
                 "hierarchy, and percentage of these merges with a cohesion of at least $0.999$.",
                 "lrrrrrrrr", "Data set & $m$ & $n$ & $\\kappa$ & $\\bar p$ & Positive & $=1$ (\\%) & Merges & "
                 "$\\geq0.999$ (\\%)", rows)

rows = []
for k, lab in DS:
    c = fid[k]
    rows.append(f"{lab} & {c['K_exact']} & {c['K_greedy']} & {c['jaccard_all']:.3f} & {c['jaccard_internal']:.3f} & "
                f"{c['zero_cohesion_merges_greedy']} & {c['intervals']} & {c['recall_intervals']:.3f} & "
                f"{c['precision_intervals']:.3f}\\\\")
tab_fid = table("tab:realfidelity",
                "Exact reference and two-stage greedy on the real data sets (entropic intensity, $\\alpha=0.7$, "
                "full samples). $K_G$, $K_N$: numbers of roots; $D_J$ and $D_J^{\\rm int}$: Jaccard distances with "
                "and without singletons; number of greedy merges with zero cohesion; number of intervals "
                "$I(F_N)$; recall and precision of the reference classes among the intervals, singletons included.",
                "lrrrrrrrr", "Data set & $K_G$ & $K_N$ & $D_J$ & $D_J^{\\rm int}$ & Zero coh. & $I(F_N)$ & Recall & "
                "Precision", rows)

rows = []
for k, lab in DS:
    b = R[k]["bootstrap"]
    rows.append(f"{lab} & {f2(b['mean_class_support_exact'])} & {f2(b['mean_class_support_greedy'])} & "
                f"{f2(b['jaccard_to_full_exact'][0])} & {f2(b['jaccard_to_full_greedy'][0])} & "
                f"{ci(b['paired_diff_jaccard_greedy_minus_exact'])}\\\\")
tab_stab = table("tab:realstability",
                 "Bootstrap stability ($B=1000$, entropic intensity, $\\alpha=0.7$). Mean support of the ordered "
                 "classes of the full-sample hierarchies $\\mathcal H_G$ and $\\mathcal H_N$; distance $D_J^{\\rm int}$ "
                 "between each replicate hierarchy and the full-sample hierarchy (mean); paired difference of "
                 "$D_J^{\\rm int}$, greedy minus exact (mean and 95\\% percentile interval).",
                 "lccccc", "Data set & \\multicolumn{2}{c}{Class support} & \\multicolumn{2}{c}{$D_J^{\\rm int}$ to full "
                 "sample} & Paired difference\\\\ & $\\mathcal H_G$ & $\\mathcal H_N$ & $\\mathcal H_G$ & $\\mathcal H_N$ & "
                 "$\\mathcal H_N-\\mathcal H_G$", rows)

rows = []
for k, lab in DS:
    f = R[k]["full"]
    first = True
    for m, mlab in (("exact", "$\\mathcal H_G$"), ("greedy", "$\\mathcal H_N$"), ("iita", "IITA")):
        x = f[m]
        c = x["counts"]
        rows.append(f"{lab if first else ''} & {mlab} & {c['total']} & {c['consistent']} & {c['equal']} & "
                    f"{c['reversed']} & {c['incomparable']} & {f3(x['share_ok'])} & "
                    f"{f3(f['difficulty_baseline_share']) if first else ''} & {f3(x['perm_null_mean'])} & "
                    f"{('$<$0.001' if x['perm_pvalue'] < 0.001 else f3(x['perm_pvalue']))}\\\\")
        first = False
    rows.append("\\midrule")
rows = rows[:-1]
tab_val = table("tab:realvalidity",
                "Implications on the full samples (entropic intensity) and their status with respect to the "
                "Q-matrix (Definition~\\ref{def:status}): consistent, equal, reversed, incomparable. Share: "
                "proportion of consistent-or-equal implications. Diff.: the same share over all pairs with "
                "$m_x\\leq m_y$ (difficulty baseline). Perm. and $p$: mean share and one-sided $p$-value under "
                "10\\,000 random relabellings of the items. For ECPE the Q-matrix is closed under the attribute hierarchy.",
                "llrrrrrrrrr", "Data set & Method & Impl. & Cons. & Eq. & Rev. & Inc. & Share & Diff. & "
                "Perm. & $p$", rows)

def mci(v):
    return (f"{v[0]:.3f}".replace("-", "$-$"), f"[{v[1]:.3f}, {v[2]:.3f}]".replace("-", "$-$"))


rows = []
for k, lab in DS:
    b = R[k]["bootstrap"]
    first = True
    for key, clab in (("validity_gain_over_difficulty_exact", "$\\mathcal H_G$ $-$ difficulty"),
                      ("validity_gain_over_difficulty_greedy", "$\\mathcal H_N$ $-$ difficulty"),
                      ("validity_gain_over_difficulty_iita", "IITA $-$ difficulty"),
                      ("paired_validity_gain_exact_minus_greedy", "$\\mathcal H_G$ $-$ $\\mathcal H_N$ (paired)")):
        m_, c_ = mci(b[key])
        rows.append(f"{lab if first else ''} & {clab} & {m_} & {c_}\\\\")
        first = False
    rows.append("\\midrule")
rows = rows[:-1]
tab_gain = table("tab:realgain",
                 "Agreement with the Q-matrices beyond difficulty ordering: for each bootstrap replicate, share of "
                 "consistent-or-equal implications minus the difficulty baseline of the same replicate; mean and "
                 "95\\% percentile interval over $B=1000$ replicates. Last row of each block: paired difference "
                 "between the exact reference and the greedy.",
                 "llrc", "Data set & Comparison & Mean & 95\\% interval", rows)

rows = []
for k, lab in DS:
    top = R[k]["bootstrap"]["top_implications_exact"][:4]
    first = True
    for t in top:
        imp = t["implication"].replace(" => ", "$\\to$")
        rows.append(f"{lab if first else ''} & {imp} & {t['support_exact']:.2f} & {t['support_greedy']:.2f} & "
                    f"{t['support_iita']:.2f} & {t['status']}\\\\")
        first = False
    rows.append("\\midrule")
rows = rows[:-1]
tab_top = table("tab:realtop",
                "The four implications with the largest bootstrap support in the exact reference hierarchy "
                "($B=1000$), their support in the greedy hierarchy and under IITA, and their status with respect "
                "to the Q-matrix. Items are numbered as in the data sets.",
                "llrrrl", "Data set & Implication & $\\mathcal H_G$ & $\\mathcal H_N$ & IITA & Status", rows)

rows = []
for r in SC["worst_case"]:
    heap = f"{r['heap_s']:.2f}" if "heap_s" in r else "--"
    rows.append(f"{r['n']} & {r['merges']} & {r['scan_s']:.2f} & {heap} & {r['tournament_s']:.2f}\\\\")
tab_sc = table("tab:realscaling",
               "Running times in seconds of a second, vectorised implementation (one core), regime P "
               "($n-1$ merges, one matrix per size, median of three runs up to $n=800$). The heap is not run "
               "beyond $n=800$ because it stores $2(n-1)^2$ entries. All implementations returned the same merge "
               "sequence.",
               "rrrrr", "$n$ & Merges & Scanning & Heap & Tournament", rows, small=False)


(OUT / "section7_tables.tex").write_text("\n".join([tab_data, tab_fid, tab_stab, tab_val, tab_gain, tab_top, tab_sc]))
print("written:", OUT / "section7_tables.tex", OUT / "realdata_hierarchies_en.pdf", OUT / "realdata_support_en.pdf")
