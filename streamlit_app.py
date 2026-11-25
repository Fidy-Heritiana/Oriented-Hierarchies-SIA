"""Interactive demonstration of the paper
"Geometrically cohesive oriented hierarchies: exact quadratic computation and limits of a marginal selection".

Run locally with:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import synthetic as syn  # noqa: E402
from analysis import closure, draw_hierarchy  # noqa: E402
from fast import tournament_vec  # noqa: E402
from final_analysis import counts, difficulty_pairs, hier_pairs, share_ok, status_matrix  # noqa: E402
from ohier import (  # noqa: E402
    classical_intensity,
    compare,
    entropic_intensity,
    exact_heap,
    greedy_two_step,
    impose_frequency_order,
    pair_cohesion,
)

REPO = "https://github.com/Fidy-Heritiana/Oriented-Hierarchies-SIA"
DATASETS = {
    "Fraction subtraction (536 × 20)": ("fraction", "I"),
    "ECPE (2922 × 28)": ("ecpe", "E"),
    "TIMSS 2003 (757 × 23)": ("timss", "T"),
}
STATUS_MARK = {0: "+", 1: "=", 2: "x", 3: "o"}
STATUS_NAME = {0: "consistent", 1: "equal", 2: "reversed", 3: "incomparable"}

st.set_page_config(page_title="Oriented hierarchies (SIA)", layout="wide")
st.title("Geometrically cohesive oriented hierarchies")
st.markdown(
    "Interactive companion to the paper *Geometrically cohesive oriented hierarchies: exact quadratic computation "
    "and limits of a marginal selection*. It compares the **exact reference rule** (maximum geometric cohesion, "
    "Algorithm 3 / matrix tournament) with the **two-stage greedy selection** (Algorithm 1). "
    f"Code, data and the full reproduction of the figures: [{REPO}]({REPO})."
)

tabs = st.tabs([
    "One matrix",
    "Threshold α (Fig. 2)",
    "Running times (Fig. 1)",
    "Growth (Fig. 3)",
    "Real data (Fig. 4–5)",
    "About",
])


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. One matrix
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Exact reference versus greedy selection on one random matrix")
    c1, c2, c3, c4 = st.columns(4)
    regime = c1.selectbox("Regime", ["U", "P", "O"], help="U: uniform (0,1); P: uniform (0.51,0.99); "
                          "O: latent order, forward (0.75,0.99), reverse (0.01,0.49)")
    n = c2.slider("Number of variables n", 4, 300, 20)
    alpha = c3.slider("Greedy threshold α", 0.0, 0.99, 0.7, 0.01)
    seed = c4.number_input("Seed", 0, 10_000, 1)

    @st.cache_data(show_spinner=False)
    def one_matrix(regime, n, alpha, seed):
        M = syn.generate(regime, n, np.random.default_rng(seed))
        A = pair_cohesion(M)
        HG = tournament_vec(A)
        G = greedy_two_step(M, alpha=alpha, A=A)
        return M, HG, G, compare(HG, G)

    M, HG, G, c = one_matrix(regime, n, alpha, int(seed))
    m = st.columns(6)
    m[0].metric("Exact merges / roots", f"{len(HG.merges)} / {c['K_exact']}")
    m[1].metric("Greedy merges / paths", f"{len(G.hierarchy.merges)} / {c['K_greedy']}")
    m[2].metric("D_J (all classes)", f"{c['jaccard_all']:.3f}")
    m[3].metric("D_J (internal classes)", f"{c['jaccard_internal']:.3f}")
    zf = c["zero_fraction_greedy"]
    m[4].metric("Greedy merges with zero cohesion", "–" if zf != zf else f"{zf:.0%}")
    m[5].metric("Intervals I(F_N)", f"{c['intervals']}")
    st.caption(f"Recall of the reference classes among the intervals: {c['recall_intervals']:.3f}; "
               f"precision of the intervals: {c['precision_intervals']:.3f} (singletons included).")
    if n <= 40:
        lab = [f"x{i + 1}" for i in range(n)]
        with plt.rc_context({"font.size": 7}):
            fig, axes = plt.subplots(2, 1, figsize=(max(6, 0.32 * n), 5.6))
            draw_hierarchy(axes[0], HG, lab, f"Exact reference hierarchy ({len(HG.merges)} merges)")
            draw_hierarchy(axes[1], G.hierarchy, lab,
                           f"Two-stage greedy, α = {alpha:g} ({len(G.hierarchy.merges)} merges; red: zero cohesion)")
            fig.tight_layout()
        show(fig)
    else:
        st.info("The hierarchies are drawn for n ≤ 40.")

# --------------------------------------------------------------------------
# 2. Threshold (Figure 2)
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Effect of the threshold α on fidelity and class recovery")
    st.markdown("The paper uses regime U, n = 200 and 30 matrices (Figure 2, Tables 5–6). "
                "Smaller settings run faster here.")
    c1, c2, c3, c4 = st.columns(4)
    regime2 = c1.selectbox("Regime ", ["U", "P", "O"])
    n2 = c2.selectbox("n", [20, 50, 100, 200], index=2)
    m2 = c3.slider("Matrices", 3, 30, 10)
    seed2 = c4.number_input("Seed ", 0, 10_000, 2027)
    if st.button("Run", key="run_fid"):
        with st.spinner("Running the fidelity campaign..."):
            df = syn.fidelity_campaign(ns=(n2,), matrices=m2, seed=int(seed2), verbose=False, regimes=(regime2,))
        show(syn.fig_fidelity(df, regime=regime2, n=n2))
        tab = (df.groupby("alpha")[["D_J", "D_J_int", "zero_fraction", "K_N", "K_G", "I", "recall",
                                    "recall_internal", "precision"]].mean().round(3))
        st.dataframe(tab)
    else:
        st.info("Choose the settings and press **Run**.")

# --------------------------------------------------------------------------
# 3. Running times (Figure 1)
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Running times of the greedy and of the three exact implementations")
    st.markdown("The exact implementations (full scan, heap, matrix tournament) must return the same merge "
                "sequence; the campaign checks it on every matrix. Times depend on the machine.")
    c1, c2, c3 = st.columns(3)
    regimes3 = c1.multiselect("Regimes", ["U", "P", "O"], default=["U", "P", "O"])
    ns3 = c2.multiselect("Sizes n", [20, 40, 80, 160, 320], default=[20, 40, 80, 160])
    m3 = c3.slider("Matrices per size", 1, 7, 2)
    if st.button("Run", key="run_time"):
        if not regimes3 or not ns3:
            st.warning("Select at least one regime and one size.")
        else:
            with st.spinner("Timing..."):
                df = syn.timing_campaign(ns=tuple(sorted(ns3)), matrices=m3, repeats=1, verbose=False,
                                         regimes=tuple(regimes3))
            show(syn.fig_timing(df))
            med = df.pivot_table(index=["regime", "n"], columns="method", values="time", aggfunc="median")
            st.dataframe(med.round(4))
    else:
        st.info("Choose the settings and press **Run**.")

# --------------------------------------------------------------------------
# 4. Growth (Figure 3)
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Number of paths K and intervals I/n² of the greedy forest (regime U)")
    c1, c2, c3 = st.columns(3)
    ns4 = c1.multiselect("Sizes n ", [50, 100, 200, 400, 800], default=[50, 100, 200, 400])
    m4 = c2.slider("Matrices per size ", 3, 30, 10)
    a4 = c3.slider("Threshold α ", 0.0, 0.99, 0.7, 0.01)
    if st.button("Run", key="run_growth"):
        if not ns4:
            st.warning("Select at least one size.")
        else:
            with st.spinner("Running the growth campaign..."):
                df = syn.growth_campaign(ns=tuple(sorted(ns4)), matrices=m4, alpha=a4, verbose=False)
            show(syn.fig_growth(df))
            st.dataframe(df.groupby("n")[["K", "I_over_n2"]].agg(["mean", "max"]).round(3))
    else:
        st.info("Choose the settings and press **Run**.")

# --------------------------------------------------------------------------
# 5. Real data (Figures 4-5)
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Test responses and expert Q-matrices (Section 7)")
    c1, c2, c3 = st.columns(3)
    ds_label = c1.selectbox("Data set", list(DATASETS))
    intensity = c2.radio("Implication intensity", ["entropic", "classical"], horizontal=True)
    a5 = c3.slider("Greedy threshold α  ", 0.5, 0.99, 0.7, 0.01)
    key, prefix = DATASETS[ds_label]

    @st.cache_data(show_spinner=False)
    def real_data(key, intensity, alpha):
        X = pd.read_csv(ROOT / f"data/{key}_data.csv").values
        Q = pd.read_csv(ROOT / f"data/{key}_qmatrix.csv", index_col=0).values
        Qval = closure(Q, {0: [1], 1: [2]}) if key == "ecpe" else Q
        f = entropic_intensity if intensity == "entropic" else classical_intensity
        M = impose_frequency_order(f(X), X)
        A = pair_cohesion(M)
        HG = exact_heap(A)
        G = greedy_two_step(M, alpha=alpha, A=A)
        return X, Q, Qval, HG, G

    X, Q, Qval, HG, G = real_data(key, intensity, a5)
    n5 = X.shape[1]
    lab = [f"{prefix}{i + 1}" for i in range(n5)]
    ST = status_matrix(Qval)
    sub = ["".join(str(s + 1) if s < 9 else chr(ord("a") + s - 9) for s in np.flatnonzero(Q[v])) for v in range(n5)]
    coh = [c for *_, c in HG.merges]
    if intensity == "classical" and coh:
        st.warning(f"Classical intensity: {np.mean(np.array(coh) >= 0.999):.0%} of the exact merges occur at a "
                   "cohesion of at least 0.999, so the tie-breaking rule decides them (Section 7.2).")
    with plt.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(2, 1, figsize=(max(6.4, 0.32 * n5), 5.6))
        draw_hierarchy(axes[0], HG, lab, f"Exact reference hierarchy ({len(HG.merges)} merges, "
                       f"{len(HG.roots)} roots)", sub=sub)
        draw_hierarchy(axes[1], G.hierarchy, lab, f"Two-stage greedy, α = {a5:g} "
                       f"({len(G.hierarchy.merges)} merges, {len(G.paths)} roots)", sub=sub)
        fig.tight_layout()
    show(fig)
    st.caption("Second label line: attributes required by the item in the Q-matrix. "
               "Arrows: A → B concatenation (success on A implies success on B).")

    base = share_ok(ST, difficulty_pairs(X))
    rows = []
    for name, pairs in (("Exact reference", hier_pairs(HG)), ("Greedy", hier_pairs(G.hierarchy))):
        cnt = counts(ST, pairs)
        rows.append({"Hierarchy": name, "Implications": cnt["total"], "Consistent": cnt["consistent"],
                     "Equal": cnt["equal"], "Reversed": cnt["reversed"], "Incomparable": cnt["incomparable"],
                     "Share consistent or equal": round(share_ok(ST, pairs), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Difficulty baseline (all pairs ordered by difficulty): {base:.3f}. "
               + ("For ECPE the Q-matrix is closed under the attribute hierarchy." if key == "ecpe" else ""))

    st.markdown("#### Bootstrap support of the implications (B = 1000, entropic intensity, α = 0.7)")
    method = st.radio("Construction", ["exact", "greedy", "iita"], horizontal=True,
                      format_func={"exact": "Exact reference", "greedy": "Greedy", "iita": "IITA"}.get)
    S = np.load(ROOT / f"results/final/{key}_support_{method}.npy")
    order = np.argsort(X.mean(0))
    SS = S[np.ix_(order, order)]
    with plt.rc_context({"font.size": 7}):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(SS, cmap="Blues", vmin=0, vmax=1)
        for a in range(n5):
            for b in range(n5):
                if SS[a, b] >= 0.5:
                    ax.text(b, a, STATUS_MARK[int(ST[order[a], order[b]])], ha="center", va="center", fontsize=5.5,
                            color="white" if SS[a, b] > 0.7 else "black")
        ax.set_xticks(range(n5), [lab[i] for i in order], rotation=90, fontsize=5.5)
        ax.set_yticks(range(n5), [lab[i] for i in order], fontsize=5.5)
        ax.set_xlabel("implied item y")
        ax.set_ylabel("implying item x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="bootstrap support")
        fig.tight_layout()
    left, right = st.columns([3, 2])
    with left:
        show(fig)
    with right:
        st.markdown("Items sorted from the hardest to the easiest. Marks for a support of at least 0.5: "
                    "**+** consistent with the Q-matrix, **=** equal attribute sets, **x** reversed, "
                    "**o** incomparable.")
        res = json.loads((ROOT / f"results/final/{key}.json").read_text())
        top = pd.DataFrame(res["bootstrap"]["top_implications_exact"][:8])
        top.columns = ["Implication", "Exact", "Greedy", "IITA", "Status"]
        st.dataframe(top.round(2), hide_index=True)
        for fl in res["flags"]:
            st.warning(f"Flagged: **{fl['implication']}** (support {fl['support_exact']:.2f} in the exact hierarchy, "
                       f"{fl['support_iita']:.2f} under IITA) is reversed with respect to the Q-matrix. "
                       + (f"Items: {fl['text_u']} and {fl['text_v']}." if fl.get("text_u") else ""))

# --------------------------------------------------------------------------
# 6. About
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"""
**Pair cohesion.** $g(p)=\\sqrt{{1-H_2(p)^2}}$ for $p\\ge 1/2$ and $0$ otherwise. The cohesion of an ordered class
is the geometric mean of its forward pair cohesions.

**Exact reference rule.** At each step, merge the ordered pair of active classes $(A,B)$ whose concatenation $AB$
has maximal cohesion; stop when it is zero. Ties are broken on $(-h(A,B),\\mathrm{{id}}(A),\\mathrm{{id}}(B))$.
Additive statistics give $O(n^2\\log n)$ with a heap and $O(n^2)$ with a matrix tournament.

**Two-stage greedy (Algorithm 1).** Keep for each unordered pair the orientation of larger intensity if it is at
least $\\alpha$; accept arcs by decreasing weight under in- and out-degree constraints; remove cycles with union–find.

**Synthetic regimes.** U: entries uniform on (0,1). P: uniform on (0.51,0.99). O: latent order, forward entries on
(0.75,0.99), reverse entries on (0.01,0.49).

**Real data.** Three data sets of the R package CDM (fraction subtraction, ECPE, TIMSS 2003), entropic implication
intensity with the constraint $m_x\\le m_y$; bootstrap over examinees ($B=1000$).

The figures of the paper are reproduced exactly (up to the sampling of random matrices and machine-dependent
running times) by `python3 code/make_figures.py` or by the notebook `demo.ipynb`: [{REPO}]({REPO}).
""")
