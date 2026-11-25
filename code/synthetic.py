"""Synthetic campaigns of Section 6: running times, hierarchical fidelity, growth.

Protocol of Section 6.3 of the paper:
  * regimes U, P, O (zero diagonal):
      U: independent entries uniform on (0, 1);
      P: independent entries uniform on (0.51, 0.99);
      O: uniform latent permutation; forward entries uniform on (0.75, 0.99),
         reverse entries uniform on (0.01, 0.49);
  * timing campaign: n in {20, 40, 80, 160, 320}, 7 matrices per regime and size,
    each method timed 3 times (alpha = 0.7) after a warm-up, in a shuffled order;
    median over the 3 repetitions, then median and quartiles over the matrices;
    exact times start from the pair cohesions, whose computation is timed apart;
    the implementations are the reference ones of ohier.py (heap, tournament) and
    fast.scan_rows (full scan with the same additive statistics, one row at a time);
  * fidelity campaign: n in {20, 50, 100, 200}, 30 matrices per regime and size,
    alpha in {0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99};
  * growth campaign: regime U, alpha = 0.7, 30 matrices for each n in
    {50, 100, 200, 400, 800}.

Functions return pandas DataFrames; fig_timing, fig_fidelity and fig_growth draw
Figures 1-3, and the table_* functions write Tables 4-7 in LaTeX.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fast import scan_rows, tournament_vec
from ohier import compare, exact_heap, exact_tournament, greedy_two_step, intervals, pair_cohesion

REGIMES = ("U", "P", "O")
ALPHAS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
METHODS = ("Greedy", "Exact: heap", "Exact: tournament", "Exact: scan")
STYLE = {
    "Greedy": dict(color="#1f77b4", marker="o", ls=":"),
    "Exact: heap": dict(color="#d95f02", marker="s", ls="--"),
    "Exact: tournament": dict(color="#1b9e77", marker="D", ls="-."),
    "Exact: scan": dict(color="#e7298a", marker="^", ls="-"),
}


# --------------------------------------------------------------------------
# Matrix regimes
# --------------------------------------------------------------------------


def generate(regime: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Intensity matrix of regime U, P or O (zero diagonal)."""
    if regime == "U":
        M = rng.uniform(0.0, 1.0, (n, n))
    elif regime == "P":
        M = rng.uniform(0.51, 0.99, (n, n))
    elif regime == "O":
        rank = np.empty(n, dtype=int)
        rank[rng.permutation(n)] = np.arange(n)
        forward = rank[:, None] < rank[None, :]
        M = np.where(forward, rng.uniform(0.75, 0.99, (n, n)), rng.uniform(0.01, 0.49, (n, n)))
    else:
        raise ValueError(regime)
    np.fill_diagonal(M, 0.0)
    return M


def boot_ci(x, B=2000, seed=0, stat=np.mean):
    """Percentile bootstrap 95% interval of a statistic of x (NaN values dropped)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(B)])
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


# --------------------------------------------------------------------------
# Timing campaign (Table 4, Figure 1)
# --------------------------------------------------------------------------


def timing_campaign(ns=(20, 40, 80, 160, 320), matrices=7, repeats=3, alpha=0.7, seed=2026, verbose=True):
    """One row per (regime, n, matrix, method): median time over the repeats."""
    rng = np.random.default_rng(seed)
    runners = {
        "Greedy": lambda M, A: greedy_two_step(M, alpha=alpha),
        "Exact: heap": lambda M, A: exact_heap(A),
        "Exact: tournament": lambda M, A: exact_tournament(A),
        "Exact: scan": lambda M, A: scan_rows(A),
    }
    rows = []
    for regime in REGIMES:
        for n in ns:
            for k in range(matrices):
                M = generate(regime, n, rng)
                t0 = time.perf_counter()
                A = pair_cohesion(M)
                pre = time.perf_counter() - t0
                for f in runners.values():  # warm-up
                    f(M, A)
                seqs = {}
                times = {m: [] for m in runners}
                for _ in range(repeats):
                    for m in rng.permutation(list(runners)):
                        t0 = time.perf_counter()
                        out = runners[m](M, A)
                        times[m].append(time.perf_counter() - t0)
                        if m != "Greedy":
                            seqs[m] = out.merge_sequence()
                agree = len({tuple(s) for s in seqs.values()}) == 1
                for m in runners:
                    rows.append(dict(regime=regime, n=n, matrix=k, method=m, time=float(np.median(times[m])),
                                     preprocessing=pre, exact_agree=agree))
            if verbose:
                print(f"timing {regime} n={n} done", flush=True)
    df = pd.DataFrame(rows)
    assert df["exact_agree"].all(), "exact implementations disagree"
    return df


def fig_timing(df, path=None):
    """Figure 1: median times, error bars = quartiles across matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        d = df[df.regime == regime]
        for m in METHODS:
            g = d[d.method == m].groupby("n")["time"]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.errorbar(med.index, med.values, yerr=[med - q1, q3 - med], label=m, ms=4, lw=1.2,
                        capsize=2, **STYLE[m])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ns = sorted(d.n.unique())
        ax.set_xticks(ns, [str(v) for v in ns])
        ax.set_title(regime, fontsize=9)
        ax.set_xlabel("n", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper left")
    axes[0].set_ylabel("Median time (s)", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def table_timing(df, n=None):
    """Table 4: median times at the largest n, and paired scanning ratios."""
    n = n or int(df.n.max())
    d = df[df.n == n]
    med = d.pivot_table(index="regime", columns="method", values="time", aggfunc="median")
    pre = d.groupby("regime")["preprocessing"].median()
    rows = []
    for r in REGIMES:
        rows.append(f"{r} & {med.loc[r, 'Greedy']:.4f} & {med.loc[r, 'Exact: heap']:.4f} & "
                    f"{med.loc[r, 'Exact: tournament']:.4f} & {med.loc[r, 'Exact: scan']:.4f} & {pre[r]:.4f}\\\\")
    w = d.pivot_table(index=["regime", "matrix"], columns="method", values="time")
    ratios = pd.DataFrame({"scan/heap": w["Exact: scan"] / w["Exact: heap"],
                           "scan/tournament": w["Exact: scan"] / w["Exact: tournament"]}).groupby("regime").median()
    tex = ("\\begin{tabular}{lrrrrr}\\toprule\n\tRegime & Greedy & Heap & Tournament & Scanning & Preprocessing\\\\"
           "\\midrule\n\t" + "\n\t".join(rows) + "\n\t\\bottomrule\n\\end{tabular}\n")
    return tex, ratios


# --------------------------------------------------------------------------
# Fidelity campaign (Tables 5-6, Figure 2)
# --------------------------------------------------------------------------


def fidelity_campaign(ns=(20, 50, 100, 200), matrices=30, alphas=ALPHAS, seed=2027, verbose=True):
    """One row per (regime, n, matrix, alpha) with the diagnostics of Sections 6.5-6.6."""
    rng = np.random.default_rng(seed)
    rows = []
    for regime in REGIMES:
        for n in ns:
            for k in range(matrices):
                M = generate(regime, n, rng)
                A = pair_cohesion(M)
                HG = tournament_vec(A)
                cg_int = HG.classes(False)
                for a in alphas:
                    G = greedy_two_step(M, alpha=a, A=A)
                    c = compare(HG, G)
                    I = intervals(G.paths)
                    I_int = {x for x in I if len(x) > 1}
                    rows.append(dict(
                        regime=regime, n=n, matrix=k, alpha=a,
                        D_J=c["jaccard_all"], D_J_int=c["jaccard_internal"],
                        zero_fraction=c["zero_fraction_greedy"],
                        K_N=c["K_greedy"], K_G=c["K_exact"], I=c["intervals"],
                        recall=c["recall_intervals"], precision=c["precision_intervals"],
                        recall_internal=(len(cg_int & I_int) / len(cg_int)) if cg_int else np.nan,
                        I_pred=n * (n + 1) / (c["K_greedy"] + 1),
                    ))
            if verbose:
                print(f"fidelity {regime} n={n} done", flush=True)
    return pd.DataFrame(rows)


def fig_fidelity(df, regime="U", n=200, path=None):
    """Figure 2: effect of the threshold (means and 95% bootstrap intervals)."""
    d = df[(df.regime == regime) & (df.n == n)]
    alphas = sorted(d.alpha.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    panels = [
        [("D_J", "Jaccard, all nodes", "o", "-"), ("D_J_int", "Jaccard, internal nodes", "s", "--"),
         ("zero_fraction", "Zero-cohesion merge fraction", "^", ":")],
        [("recall", "Class recall", "o", "-"), ("recall_internal", "Internal-class recall", "s", "--"),
         ("precision", "Interval precision", "^", ":")],
    ]
    colors = ["#1f77b4", "#d95f02", "#1b9e77"]
    for ax, spec in zip(axes, panels):
        for (col, lab, mk, ls), colr in zip(spec, colors):
            means, lo, hi = [], [], []
            for a in alphas:
                x = d[d.alpha == a][col].to_numpy(float)
                m = np.nanmean(x)
                l, h = boot_ci(x)
                means.append(m)
                lo.append(m - l)
                hi.append(h - m)
            ax.errorbar(alphas, means, yerr=[lo, hi], label=lab, marker=mk, ls=ls, color=colr, ms=4, capsize=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"$\alpha$", fontsize=8)
        ax.set_title(f"{regime}, n = {n}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")
    axes[0].set_ylabel("Mean and 95% bootstrap interval", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def table_fidelity(df, n=200, alpha=0.7):
    """Tables 5 and 6, and the relative bias of the coalescent predictor."""
    d = df[(df.n == n) & (df.alpha == alpha)]
    m = d.groupby("regime")[["D_J", "D_J_int", "zero_fraction", "K_N", "K_G", "I", "recall", "precision",
                             "recall_internal", "I_pred"]].mean().loc[list(REGIMES)]
    t5 = "\n\t".join(f"{r} & {m.loc[r, 'D_J']:.3f} & {m.loc[r, 'D_J_int']:.3f} & {m.loc[r, 'zero_fraction']:.3f}\\\\"
                     for r in REGIMES)
    t6 = "\n\t".join(f"{r} & {m.loc[r, 'K_N']:.2f} & {m.loc[r, 'K_G']:.2f} & {m.loc[r, 'I']:.1f} & "
                     f"{m.loc[r, 'recall']:.3f} & {m.loc[r, 'precision']:.3f}\\\\" for r in REGIMES)
    tex5 = ("\\begin{tabular}{lrrr}\\toprule\n\tRegime & $D_J$ & $D_J^{\\rm int}$ & Zero-cohesion merges\\\\"
            "\\midrule\n\t" + t5 + "\n\t\\bottomrule\n\\end{tabular}\n")
    tex6 = ("\\begin{tabular}{lrrrrr}\\toprule\n\tRegime & $K_N$ & $K_G$ & $I(F_N)$ & Recall & Precision\\\\"
            "\\midrule\n\t" + t6 + "\n\t\\bottomrule\n\\end{tabular}\n")
    bias = (m["I_pred"] - m["I"]) / m["I"]
    return tex5, tex6, m, bias


# --------------------------------------------------------------------------
# Growth campaign (Table 7, Figure 3)
# --------------------------------------------------------------------------


def growth_campaign(ns=(50, 100, 200, 400, 800), matrices=30, alpha=0.7, seed=2028, verbose=True):
    """Number of components K and number of intervals I of the greedy forest, regime U."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        for k in range(matrices):
            M = generate("U", n, rng)
            G = greedy_two_step(M, alpha=alpha)
            K = len(G.paths)
            I = sum(len(p) * (len(p) + 1) // 2 for p in G.paths)
            rows.append(dict(n=n, matrix=k, K=K, I=I, I_over_n2=I / n ** 2))
        if verbose:
            print(f"growth n={n} done", flush=True)
    return pd.DataFrame(rows)


def fig_growth(df, path=None):
    """Figure 3: means of K and I/n^2 with 95% bootstrap intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, col, lab, colr in ((axes[0], "K", "K", "#1f77b4"), (axes[1], "I_over_n2", r"$I/n^2$", "#d95f02")):
        ns = sorted(df.n.unique())
        means, lo, hi = [], [], []
        for n in ns:
            x = df[df.n == n][col].to_numpy(float)
            m = x.mean()
            l, h = boot_ci(x)
            means.append(m)
            lo.append(m - l)
            hi.append(h - m)
        ax.errorbar(ns, means, yerr=[lo, hi], marker="o", color=colr, ms=4, capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(v) for v in ns])
        ax.set_xlabel("n", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(r"U, $\alpha = 0.7$", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def table_growth(df):
    """Table 7."""
    rows = []
    for n, d in df.groupby("n"):
        l, h = boot_ci(d.K)
        rows.append(f"{n} & {d.K.mean():.2f} & $[{l:.2f},{h:.2f}]$ & {d.K.max()} & {d.I_over_n2.mean():.3f}\\\\")
    return ("\\begin{tabular}{rrrrr}\\toprule\n\t$n$ & Mean of $K$ & 95\\% interval & Maximum of $K$ & "
            "Mean of $I/n^2$\\\\\\midrule\n\t" + "\n\t".join(rows) + "\n\t\\bottomrule\n\\end{tabular}\n")


def save_tables(timing, fidelity, growth, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tex4, ratios = table_timing(timing)
    tex5, tex6, means, bias = table_fidelity(fidelity)
    tex7 = table_growth(growth)
    (outdir / "table4_timing.tex").write_text(tex4)
    (outdir / "table5_fidelity.tex").write_text(tex5)
    (outdir / "table6_recovery.tex").write_text(tex6)
    (outdir / "table7_growth.tex").write_text(tex7)
    ratios.to_csv(outdir / "table4_paired_ratios.csv")
    bias.rename("relative_bias_I_pred").to_csv(outdir / "table6_predictor_bias.csv")
    return ratios, means, bias
