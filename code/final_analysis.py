"""Final analysis following PROTOCOL.md, for the three data sets.

Writes results/final/<dataset>.json, LaTeX tables in results/final/tables/,
and figures in results/final/figures/.
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
from analysis import closure, draw_hierarchy  # noqa: E402
from iita import iita, implications  # noqa: E402
from ohier import (  # noqa: E402
    classical_intensity,
    entropic_intensity,
    exact_heap,
    exact_tournament,
    greedy_two_step,
    impose_frequency_order,
    jaccard,
    pair_cohesion,
)
from qvalid import fit_gdina, pvaf_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

B_BOOT = 1000
SEED = 2026
ALPHA = 0.7
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CODES = {"consistent": 0, "equal": 1, "reversed": 2, "incomparable": 3}


def status_matrix(Q):
    Qb = Q.astype(bool)
    n = Q.shape[0]
    S = np.full((n, n), -1, dtype=np.int8)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            sup = np.all(Qb[u] >= Qb[v])
            sub = np.all(Qb[u] <= Qb[v])
            S[u, v] = 1 if (sup and sub) else 0 if sup else 2 if sub else 3
    return S


def hier_pairs(H):
    return [(u, v) for a, b, _ in H.merges for u in a for v in b]


def share_ok(ST, pairs):
    if not pairs:
        return float("nan")
    s = np.array([ST[u, v] for u, v in pairs])
    return float(np.mean((s == 0) | (s == 1)))


def counts(ST, pairs):
    s = np.array([ST[u, v] for u, v in pairs], dtype=int) if pairs else np.array([], dtype=int)
    return {k: int((s == c).sum()) for k, c in CODES.items()} | {"total": len(pairs)}


def difficulty_pairs(X):
    na = X.sum(axis=0)
    n = X.shape[1]
    return [(u, v) for u in range(n) for v in range(n) if u != v and na[u] <= na[v]]


def perm_null(ST, pairs, n_perm=10000, seed=1):
    rng = np.random.default_rng(seed)
    n = ST.shape[0]
    obs = share_ok(ST, pairs)
    null = np.empty(n_perm)
    P = np.array(pairs)
    for t in range(n_perm):
        p = rng.permutation(n)
        s = ST[np.ix_(p, p)][P[:, 0], P[:, 1]]
        null[t] = np.mean((s == 0) | (s == 1))
    return float(null.mean()), float((1 + np.sum(null >= obs)) / (n_perm + 1))


def ci(v):
    v = np.asarray([x for x in v if not np.isnan(x)])
    if v.size == 0:
        return [float("nan")] * 3
    return [float(v.mean()), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def fit_all(X, intensity="entropic", alpha=ALPHA):
    f = entropic_intensity if intensity == "entropic" else classical_intensity
    M = impose_frequency_order(f(X), X)
    A = pair_cohesion(M)
    HG = exact_heap(A)
    G = greedy_two_step(M, alpha=alpha, A=A)
    return M, A, HG, G


def saturation(X):
    M = impose_frequency_order(classical_intensity(X), X)
    A = pair_cohesion(M)
    off = ~np.eye(X.shape[1], dtype=bool)
    pos = A[off][A[off] > 0]
    H = exact_heap(A)
    coh = np.array([c for *_, c in H.merges])
    return {
        "positive_pair_cohesions": int(pos.size),
        "share_pair_cohesions_equal_1": float(np.mean(pos == 1.0)) if pos.size else float("nan"),
        "merges_classical": int(len(coh)),
        "share_merges_cohesion_ge_0999": float(np.mean(coh >= 0.999)) if coh.size else float("nan"),
        "mean_merge_cohesion_classical": float(coh.mean()) if coh.size else float("nan"),
    }


def run(name, X, Q, Qval, labels, item_text=None, prereq_note=None):
    t0 = time.perf_counter()
    n = X.shape[1]
    ST = status_matrix(Qval)
    res = {"dataset": name, "N": int(X.shape[0]), "n_items": n, "B": B_BOOT, "alpha": ALPHA}
    res["saturation_classical"] = saturation(X)

    # ---------------- full sample
    M, A, HG, G = fit_all(X)
    HT = exact_tournament(A)
    assert HG.merge_sequence() == HT.merge_sequence()
    I = iita(X)
    imp_iita = implications(I["quasi_order"])
    dp = difficulty_pairs(X)
    base = share_ok(ST, dp)
    full = {"difficulty_baseline_share": base, "difficulty_pairs": len(dp)}
    for lab, pairs in (("exact", hier_pairs(HG)), ("greedy", hier_pairs(G.hierarchy)), ("iita", imp_iita)):
        mu, pv = perm_null(ST, pairs) if pairs else (float("nan"), float("nan"))
        full[lab] = {"share_ok": share_ok(ST, pairs), "counts": counts(ST, pairs),
                     "perm_null_mean": mu, "perm_pvalue": pv}
    full["exact"]["merges"] = len(HG.merges)
    full["exact"]["roots"] = len(HG.roots)
    full["exact"]["classes"] = [[labels[v] for v in r] for r in HG.roots if len(r) > 1]
    full["exact"]["merge_list"] = [([labels[v] for v in a], [labels[v] for v in b], round(c, 4)) for a, b, c in HG.merges]
    full["greedy"]["merges"] = len(G.hierarchy.merges)
    full["greedy"]["roots"] = len(G.paths)
    full["greedy"]["zero_cohesion_merges"] = int(sum(1 for *_, c in G.hierarchy.merges if c == 0.0))
    full["greedy"]["jaccard_internal_to_exact"] = jaccard(HG.classes(False), G.hierarchy.classes(False))
    full["greedy"]["jaccard_all_to_exact"] = jaccard(HG.classes(True), G.hierarchy.classes(True))
    full["iita"]["diff"] = I["diff"]
    full["iita"]["gamma"] = I["gamma"]
    # classical sensitivity
    Mc, Ac, HGc, Gc = fit_all(X, "classical")
    full["classical_sensitivity"] = {
        "exact_merges": len(HGc.merges), "exact_share_ok": share_ok(ST, hier_pairs(HGc)),
        "greedy_share_ok": share_ok(ST, hier_pairs(Gc.hierarchy)),
        "jaccard_internal_exact_vs_greedy": jaccard(HGc.classes(False), Gc.hierarchy.classes(False)),
    }
    # alpha grid (supplement)
    grid = {}
    for a in ALPHAS:
        Ga = greedy_two_step(M, alpha=a, A=A)
        grid[str(a)] = {"merges": len(Ga.hierarchy.merges), "share_ok": share_ok(ST, hier_pairs(Ga.hierarchy)),
                        "jaccard_internal_to_exact": jaccard(HG.classes(False), Ga.hierarchy.classes(False)),
                        "zero_cohesion_merges": int(sum(1 for *_, c in Ga.hierarchy.merges if c == 0.0))}
    full["greedy_alpha_grid"] = grid
    res["full"] = full

    # ---------------- bootstrap
    rng = np.random.default_rng(SEED)
    cg0, cn0 = HG.classes(False), G.hierarchy.classes(False)
    sup_g = dict.fromkeys(cg0, 0)
    sup_n = dict.fromkeys(cn0, 0)
    Sg, Sn, Si = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    dg, dn, dx = [], [], []
    vg, vn, vi = [], [], []  # share_ok minus replicate difficulty baseline
    for _ in range(B_BOOT):
        Xb = X[rng.integers(0, X.shape[0], X.shape[0])]
        _, _, Hb, Gb = fit_all(Xb)
        Ib = iita(Xb)
        cg, cn = Hb.classes(False), Gb.hierarchy.classes(False)
        dg.append(jaccard(cg0, cg))
        dn.append(jaccard(cn0, cn))
        dx.append(jaccard(cg, cn))
        for c in cg0:
            sup_g[c] += c in cg
        for c in cn0:
            sup_n[c] += c in cn
        pg, pn, pi = hier_pairs(Hb), hier_pairs(Gb.hierarchy), implications(Ib["quasi_order"])
        for S, pairs in ((Sg, pg), (Sn, pn), (Si, pi)):
            for u, v in pairs:
                S[u, v] += 1
        bb = share_ok(ST, difficulty_pairs(Xb))
        vg.append(share_ok(ST, pg) - bb)
        vn.append(share_ok(ST, pn) - bb)
        vi.append(share_ok(ST, pi) - bb)
    Sg, Sn, Si = Sg / B_BOOT, Sn / B_BOOT, Si / B_BOOT
    boot = {
        "jaccard_to_full_exact": ci(dg), "jaccard_to_full_greedy": ci(dn),
        "paired_diff_jaccard_greedy_minus_exact": ci(np.array(dn) - np.array(dg)),
        "jaccard_exact_vs_greedy_per_replicate": ci(dx),
        "mean_class_support_exact": float(np.mean(list(sup_g.values())) / B_BOOT) if sup_g else float("nan"),
        "mean_class_support_greedy": float(np.mean(list(sup_n.values())) / B_BOOT) if sup_n else float("nan"),
        "class_support_exact": {"-".join(labels[v] for v in c): s / B_BOOT for c, s in sorted(sup_g.items(), key=lambda t: -t[1])},
        "class_support_greedy": {"-".join(labels[v] for v in c): s / B_BOOT for c, s in sorted(sup_n.items(), key=lambda t: -t[1])},
        "validity_gain_over_difficulty_exact": ci(vg),
        "validity_gain_over_difficulty_greedy": ci(vn),
        "validity_gain_over_difficulty_iita": ci(vi),
        "paired_validity_gain_exact_minus_greedy": ci(np.array(vg) - np.array(vn)),
    }
    for lab, S in (("exact", Sg), ("greedy", Sn), ("iita", Si)):
        for thr in (0.5, 0.75):
            idx = [(u, v) for u, v in np.argwhere(S >= thr)]
            boot[f"pairs_support_ge_{thr}_{lab}"] = counts(ST, idx) | {"share_ok": share_ok(ST, idx)}
    top = sorted(((Sg[u, v], u, v) for u in range(n) for v in range(n) if Sg[u, v] > 0), reverse=True)[:15]
    boot["top_implications_exact"] = [{"implication": f"{labels[u]} => {labels[v]}", "support_exact": float(s),
                                       "support_greedy": float(Sn[u, v]), "support_iita": float(Si[u, v]),
                                       "status": [k for k, c in CODES.items() if c == ST[u, v]][0]} for s, u, v in top]
    res["bootstrap"] = boot

    # ---------------- flags
    flags = []
    for u, v in np.argwhere(Sg >= 0.5):
        if ST[u, v] == CODES["reversed"]:
            flags.append((int(u), int(v)))
    res["flags"] = []
    fit = None
    for u, v in flags:
        if fit is None:
            fit = fit_gdina(X, Q)
        diff_attr = [k for k in range(Q.shape[1]) if Q[u, k] != Q[v, k]]
        pv = {}
        for j in (u, v):
            pv[labels[j]] = {"original": pvaf_of(fit, j, Q[j])}
            for k in diff_attr:
                q = Q[j].copy()
                q[k] = 1 - q[k]
                pv[labels[j]][f"toggle_attr{k + 1}"] = pvaf_of(fit, j, q)
        res["flags"].append({
            "implication": f"{labels[u]} => {labels[v]}",
            "support_exact": float(Sg[u, v]), "support_greedy": float(Sn[u, v]), "support_iita": float(Si[u, v]),
            "q_u": Q[u].astype(int).tolist(), "q_v": Q[v].astype(int).tolist(),
            "text_u": item_text[u] if item_text else None, "text_v": item_text[v] if item_text else None,
            "p_u": float(X[:, u].mean()), "p_v": float(X[:, v].mean()),
            "pvaf": pv,
        })
    res["seconds"] = time.perf_counter() - t0
    with open(OUT / f"{name}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.save(OUT / f"{name}_support_exact.npy", Sg)
    np.save(OUT / f"{name}_support_greedy.npy", Sn)
    np.save(OUT / f"{name}_support_iita.npy", Si)

    # ---------------- figures
    sub = ["".join(str(s + 1) if s < 9 else chr(ord('a') + s - 9) for s in np.flatnonzero(Q[v])) for v in range(n)]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.42 * n), 7.5))
    draw_hierarchy(axes[0], HG, labels, f"{name}: exact maximum-cohesion hierarchy "
                   f"({len(HG.merges)} merges, {len(HG.roots)} roots)", sub=sub)
    draw_hierarchy(axes[1], G.hierarchy, labels, f"{name}: two-step greedy, alpha = {ALPHA} "
                   f"({len(G.hierarchy.merges)} merges)", sub=sub)
    fig.text(0.01, 0.005, "Second label line: skills required by the item (Q-matrix). "
             "Arrows: A -> B concatenation (success on A implies success on B).", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "figures" / f"{name}_hierarchies.pdf")
    fig.savefig(OUT / "figures" / f"{name}_hierarchies.png", dpi=200)
    plt.close(fig)

    order = np.argsort(X.mean(0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (lab, S) in zip(axes, (("exact hierarchy", Sg), ("greedy (alpha = 0.7)", Sn), ("IITA", Si))):
        SS = S[np.ix_(order, order)]
        im = ax.imshow(SS, cmap="Blues", vmin=0, vmax=1)
        for a in range(n):
            for b in range(n):
                if SS[a, b] >= 0.5:
                    st = ST[order[a], order[b]]
                    mk = {0: "+", 1: "=", 2: "x", 3: "o"}[int(st)]
                    ax.text(b, a, mk, ha="center", va="center", fontsize=6.5,
                            color="white" if SS[a, b] > 0.7 else "black")
        ax.set_xticks(range(n), [labels[i] for i in order], rotation=90, fontsize=5.5)
        ax.set_yticks(range(n), [labels[i] for i in order], fontsize=5.5)
        ax.set_title(f"{name}: support of u => v, {lab}", fontsize=9)
    axes[0].set_ylabel("u (implying item); items sorted from hardest to easiest")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="bootstrap support (B = 1000)")
    fig.text(0.01, 0.01, "Marks for support >= 0.5: + consistent with Q, = equal skills, x reversed, o incomparable.",
             fontsize=7)
    fig.savefig(OUT / "figures" / f"{name}_support.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}_support.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return res


FRACTION_TEXT = [
    "5/3 - 3/4", "3/4 - 3/8", "5/6 - 1/9", "3 1/2 - 2 3/2", "4 3/5 - 3 4/10",
    "6/7 - 4/7", "3 - 2 1/5", "2/3 - 2/3", "3 7/8 - 2", "4 4/12 - 2 7/12",
    "4 1/3 - 2 4/3", "11/8 - 1/8", "3 3/8 - 2 5/6", "3 4/5 - 3 2/5", "2 - 1/3",
    "4 5/7 - 1 4/7", "7 3/5 - 2 4/5", "4 1/10 - 2 8/10", "4 - 1 4/3", "4 1/3 - 1 5/3",
]


def main(which=("fraction", "ecpe", "timss")):
    out = {}
    if "fraction" in which:
        X = pd.read_csv(ROOT / "data/fraction_data.csv").values
        Q = pd.read_csv(ROOT / "data/fraction_qmatrix.csv", index_col=0).values
        out["fraction"] = run("fraction", X, Q, Q, [f"I{i + 1}" for i in range(20)], FRACTION_TEXT)
    if "ecpe" in which:
        X = pd.read_csv(ROOT / "data/ecpe_data.csv").values
        Q = pd.read_csv(ROOT / "data/ecpe_qmatrix.csv", index_col=0).values
        out["ecpe"] = run("ecpe", X, Q, closure(Q, {0: [1], 1: [2]}), [f"E{i + 1}" for i in range(28)])
    if "timss" in which:
        X = pd.read_csv(ROOT / "data/timss_data.csv").values
        Q = pd.read_csv(ROOT / "data/timss_qmatrix.csv", index_col=0).values
        info = pd.read_csv(ROOT / "data/timss_iteminfo.csv")
        out["timss"] = run("timss", X, Q, Q, [f"T{i + 1}" for i in range(23)], info["description"].tolist())
    for k, r in out.items():
        f, b = r["full"], r["bootstrap"]
        print(f"== {k} ({r['seconds']:.0f}s)  baseline={f['difficulty_baseline_share']:.3f}  "
              f"exact={f['exact']['share_ok']:.3f} ({f['exact']['counts']['total']} impl.)  "
              f"greedy={f['greedy']['share_ok']:.3f}  iita={f['iita']['share_ok']:.3f} ({f['iita']['counts']['total']})")
        print(f"   gain over difficulty [95% CI]: exact {b['validity_gain_over_difficulty_exact']}, "
              f"greedy {b['validity_gain_over_difficulty_greedy']}, iita {b['validity_gain_over_difficulty_iita']}")
        print(f"   class support exact {b['mean_class_support_exact']:.3f} greedy {b['mean_class_support_greedy']:.3f}; "
              f"D_J to full exact {b['jaccard_to_full_exact'][0]:.3f} greedy {b['jaccard_to_full_greedy'][0]:.3f}; "
              f"paired diff {b['paired_diff_jaccard_greedy_minus_exact']}")
        print("   flags:", [(fl['implication'], round(fl['support_exact'], 2), round(fl['support_iita'], 2)) for fl in r["flags"]])


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("fraction", "ecpe", "timss"))
