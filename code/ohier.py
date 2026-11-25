"""Oriented hierarchies under geometric implicative cohesion.

Implements, following the definitions of the manuscript
"Geometrically cohesive oriented hierarchies":

* implication intensities from a binary table (classical Poisson intensity
  and Gras' entropic intensity);
* pair cohesion a_xy = g(M_xy), g(p) = sqrt(1 - H2(p)^2) for p >= 1/2, else 0;
* the reference rule (merge the ordered pair of active classes (A, B) whose
  concatenation AB has maximal positive geometric cohesion), in three exact
  implementations that must return the same merge sequence:
    - exact_scan       : O(n^3) full scan over active pairs (verification),
    - exact_heap       : O(n^2 log n) lazy max-heap (Algorithm 3),
    - exact_tournament : O(n^2) matrix tournament (Section 4.4);
* the two-step marginal greedy selection (Algorithm 1), returning the path
  forest and its merge hierarchy;
* comparison utilities (Jaccard distances, interval recall / precision).

Tie-breaking: candidates are compared on the key (-h(A,B), id(A), id(B)),
lexicographically minimised; singletons have ids 0..n-1 and new classes get
n, n+1, ... in order of creation.  Zero-cohesion candidates have first
coordinate +inf, invalid entries (+inf, BIG, BIG).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

# --------------------------------------------------------------------------
# Intensities and pair cohesion
# --------------------------------------------------------------------------


def _h2(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, -p * np.log2(p), 0.0)
        b = np.where(p < 1, -(1 - p) * np.log2(1 - p), 0.0)
    return a + b


def classical_intensity(X):
    """Gras' classical implication intensity phi(a -> b) (Poisson model).

    phi(a, b) = P(Poisson(lambda) > n_{a and not b}),
    lambda = n_a * n_{not b} / N.
    Pairs with lambda = 0 (b always true or a never true) get 0.
    """
    X = np.asarray(X, dtype=np.int64)
    N = X.shape[0]
    na = X.sum(axis=0)
    nab = X.T @ X
    counter = na[:, None] - nab  # n_{a, not b}
    lam = na[:, None] * (N - na[None, :]) / N
    phi = np.where(lam > 0, poisson.sf(counter, np.where(lam > 0, lam, 1.0)), 0.0)
    np.fill_diagonal(phi, 0.0)
    return phi


def entropic_intensity(X, alpha=2.0):
    """Gras et al. (2001) entropic intensity sqrt(i(a,b) * phi(a,b)).

    i(a,b) = [(1 - h1^alpha)(1 - h2^alpha)]^(1/(2 alpha)) with
    h1 = H2(n_{a not b} / n_a) if that ratio < 1/2 else 1,
    h2 = H2(n_{a not b} / n_{not b}) if that ratio < 1/2 else 1.
    """
    X = np.asarray(X, dtype=np.int64)
    N = X.shape[0]
    na = X.sum(axis=0).astype(float)
    nab = (X.T @ X).astype(float)
    counter = na[:, None] - nab
    nnotb = (N - na)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(na[:, None] > 0, counter / na[:, None], 1.0)
        r2 = np.where(nnotb > 0, counter / nnotb, 1.0)
    h1 = np.where(r1 < 0.5, _h2(r1), 1.0)
    h2 = np.where(r2 < 0.5, _h2(r2), 1.0)
    inc = ((1 - h1 ** alpha) * (1 - h2 ** alpha)) ** (1.0 / (2 * alpha))
    phi = classical_intensity(X)
    out = np.sqrt(inc * phi)
    np.fill_diagonal(out, 0.0)
    return out


def impose_frequency_order(M, X):
    """Keep a -> b only when n_a <= n_b (standard SIA convention)."""
    na = np.asarray(X).sum(axis=0)
    M = M.copy()
    M[na[:, None] > na[None, :]] = 0.0
    return M


def pair_cohesion(M):
    """a_xy = g(M_xy) with g(p) = sqrt(1 - H2(p)^2) for p >= 1/2, 0 otherwise."""
    M = np.asarray(M, dtype=float)
    g = np.where(M >= 0.5, np.sqrt(np.clip(1 - _h2(M) ** 2, 0.0, 1.0)), 0.0)
    g = np.where(M >= 1.0, 1.0, g)
    np.fill_diagonal(g, 0.0)
    return g


def class_cohesion(seq, A):
    """Geometric cohesion q(C) of an ordered class C = seq (Definition 2)."""
    r = len(seq)
    if r < 2:
        return 1.0
    vals = [A[seq[i], seq[j]] for i in range(r) for j in range(i + 1, r)]
    if min(vals) <= 0:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


# --------------------------------------------------------------------------
# Exact reference rule
# --------------------------------------------------------------------------

INF = float("inf")


@dataclass
class Hierarchy:
    """Result of an agglomerative procedure on n vertices."""

    n: int
    merges: list = field(default_factory=list)  # (seqA, seqB, cohesion of AB)
    roots: list = field(default_factory=list)  # ordered sequences of final classes

    def classes(self, singletons=True):
        out = set()
        if singletons:
            out.update((i,) for i in range(self.n))
        for a, b, _ in self.merges:
            out.add(tuple(a) + tuple(b))
        return out

    def merge_sequence(self):
        return [(tuple(a), tuple(b)) for a, b, _ in self.merges]


class _State:
    """Shared additive statistics (Lemma 1) for the exact implementations."""

    def __init__(self, A):
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        self.n = n
        pos = A > 0
        logA = np.zeros_like(A)
        logA[pos] = np.log(A[pos])
        Z = (~pos).astype(np.int64)
        np.fill_diagonal(Z, 0)
        np.fill_diagonal(logA, 0.0)
        # cross statistics between slots (ordered: slot i before slot j)
        self.Zc = Z.copy()
        self.Lc = logA.copy()
        self.Zi = np.zeros(n, dtype=np.int64)  # internal zero counts
        self.Li = np.zeros(n)  # internal log sums
        self.size = np.ones(n, dtype=np.int64)
        self.ident = np.arange(n, dtype=np.int64)
        self.active = np.ones(n, dtype=bool)
        self.seq = [(i,) for i in range(n)]
        self.next_id = n

    def h(self, i, j):
        """log cohesion of the concatenation slot_i . slot_j (or -inf)."""
        if self.Zi[i] + self.Zi[j] + self.Zc[i, j] > 0:
            return -INF
        s = self.size[i] + self.size[j]
        return (self.Li[i] + self.Li[j] + self.Lc[i, j]) / (s * (s - 1) / 2)

    def key(self, i, j):
        hv = self.h(i, j)
        return (INF if hv == -INF else -hv, int(self.ident[i]), int(self.ident[j]))

    def merge(self, i, j):
        """Concatenate slot i then slot j into slot i; deactivate j."""
        hv = self.h(i, j)
        coh = 0.0 if hv == -INF else math.exp(hv)
        rec = (self.seq[i], self.seq[j], coh)
        self.Zi[i] = self.Zi[i] + self.Zi[j] + self.Zc[i, j]
        self.Li[i] = self.Li[i] + self.Li[j] + self.Lc[i, j]
        self.size[i] += self.size[j]
        self.seq[i] = self.seq[i] + self.seq[j]
        # cross statistics with every other slot (Lemma 1)
        self.Zc[i, :] += self.Zc[j, :]
        self.Lc[i, :] += self.Lc[j, :]
        self.Zc[:, i] += self.Zc[:, j]
        self.Lc[:, i] += self.Lc[:, j]
        self.Zc[i, i] = 0
        self.Lc[i, i] = 0.0
        self.active[j] = False
        self.ident[i] = self.next_id
        self.next_id += 1
        return rec


def exact_scan(A):
    """O(n^3) exhaustive selection with the same keys (verification)."""
    st = _State(A)
    H = Hierarchy(st.n)
    while st.active.sum() > 1:
        act = np.flatnonzero(st.active)
        best, bi, bj = None, -1, -1
        for i in act:
            for j in act:
                if i == j:
                    continue
                k = st.key(i, j)
                if best is None or k < best:
                    best, bi, bj = k, i, j
        if best[0] == INF:
            break
        H.merges.append(st.merge(bi, bj))
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    return H


def exact_heap(A, return_stats=False):
    """Algorithm 3: lazy max-heap on exact keys, O(n^2 log n)."""
    st = _State(A)
    n = st.n
    H = Hierarchy(n)
    heap = []
    for i in range(n):
        for j in range(n):
            if i != j:
                heap.append(st.key(i, j) + (i, j))
    heapq.heapify(heap)
    inserted = len(heap)
    while heap:
        k0, ida, idb, i, j = heapq.heappop(heap)
        if not (st.active[i] and st.active[j] and st.ident[i] == ida and st.ident[j] == idb):
            continue
        if k0 == INF:
            break
        H.merges.append(st.merge(i, j))
        for d in np.flatnonzero(st.active):
            if d == i:
                continue
            heapq.heappush(heap, st.key(i, d) + (i, d))
            heapq.heappush(heap, st.key(d, i) + (d, i))
            inserted += 2
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    if return_stats:
        return H, {"heap_insertions": inserted}
    return H


def exact_tournament(A, return_stats=False):
    """Matrix tournament (quaternary tree over an N x N grid), O(n^2).

    Leaves hold the key triple (k0, id_i, id_j) of slot pair (i, j); level r
    stores, for each 2^r x 2^r square, the flat index of its best leaf.
    After a merge only rows/columns i and j change; each affected square is
    recomputed once per level (vectorised).
    """
    st = _State(A)
    n = st.n
    N = 1
    while N < n:
        N *= 2
    N = max(N, 2)
    BIG = np.int64(10 ** 12)
    K0 = np.full((N, N), INF)
    K1 = np.full((N, N), BIG, dtype=np.int64)
    K2 = np.full((N, N), BIG, dtype=np.int64)

    def set_leaf(i, j):
        if i == j or i >= n or j >= n or not (st.active[i] and st.active[j]):
            K0[i, j], K1[i, j], K2[i, j] = INF, BIG, BIG
        else:
            k = st.key(i, j)
            K0[i, j], K1[i, j], K2[i, j] = k

    for i in range(n):
        for j in range(n):
            set_leaf(i, j)
    k0f, k1f, k2f = K0.reshape(-1), K1.reshape(-1), K2.reshape(-1)

    def best_of(cands):
        # cands: array (m, 4) of flat leaf indices -> best per row
        b = cands[:, 0].copy()
        for c in range(1, cands.shape[1]):
            x = cands[:, c]
            a0, x0 = k0f[b], k0f[x]
            a1, x1 = k1f[b], k1f[x]
            a2, x2 = k2f[b], k2f[x]
            better = (x0 < a0) | ((x0 == a0) & ((x1 < a1) | ((x1 == a1) & (x2 < a2))))
            b = np.where(better, x, b)
        return b

    levels = [np.arange(N * N, dtype=np.int64).reshape(N, N)]
    while levels[-1].shape[0] > 1:
        prev = levels[-1]
        m = prev.shape[0] // 2
        c = np.stack(
            [prev[0::2, 0::2], prev[0::2, 1::2], prev[1::2, 0::2], prev[1::2, 1::2]], axis=-1
        ).reshape(-1, 4)
        levels.append(best_of(c).reshape(m, m))
    recomputed = 0

    def update(lines):
        nonlocal recomputed
        for r in range(1, len(levels)):
            prev, cur = levels[r - 1], levels[r]
            m = cur.shape[0]
            blocks = sorted({ln >> r for ln in lines})
            rows = set()
            for bidx in blocks:
                rows.update((bidx, c) for c in range(m))
                rows.update((c, bidx) for c in range(m))
            cells = np.array(sorted(rows), dtype=np.int64)
            a, b = cells[:, 0], cells[:, 1]
            c = np.stack(
                [prev[2 * a, 2 * b], prev[2 * a, 2 * b + 1], prev[2 * a + 1, 2 * b], prev[2 * a + 1, 2 * b + 1]],
                axis=-1,
            )
            cur[a, b] = best_of(c)
            recomputed += len(cells)

    H = Hierarchy(n)
    while True:
        root = levels[-1][0, 0]
        if k0f[root] == INF:
            break
        i, j = divmod(int(root), N)
        H.merges.append(st.merge(i, j))
        for d in range(n):
            set_leaf(i, d)
            set_leaf(d, i)
            set_leaf(j, d)
            set_leaf(d, j)
        update([i, j])
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    if return_stats:
        return H, {"recomputed_nodes": recomputed, "N": N}
    return H


# --------------------------------------------------------------------------
# Two-step marginal greedy selection (Algorithm 1)
# --------------------------------------------------------------------------


@dataclass
class GreedyResult:
    n: int
    arcs: list  # accepted arcs of the path forest, in acceptance order
    hierarchy: Hierarchy
    paths: list  # final paths (ordered sequences)
    candidates: int


def greedy_two_step(M, alpha=0.5, A=None):
    """Degree-constrained selection followed by acyclic reconstruction.

    M: oriented intensities (n x n). For each unordered pair the orientation
    with the higher intensity is kept (ties: i -> j for i < j), and retained
    if its weight >= alpha.  If A (pair cohesions) is given, the cohesion of
    each created class is recorded.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] >= M[j, i]:
                w, x, y = M[i, j], i, j
            else:
                w, x, y = M[j, i], j, i
            if w >= alpha and w > 0:
                cand.append((-w, x, y))
    cand.sort()
    out_used = np.zeros(n, dtype=bool)
    in_used = np.zeros(n, dtype=bool)
    L = []
    for negw, x, y in cand:
        if not out_used[x] and not in_used[y]:
            L.append((x, y))
            out_used[x] = True
            in_used[y] = True
    parent = list(range(n))
    seq = {i: (i,) for i in range(n)}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    F = []
    H = Hierarchy(n)
    for x, y in L:
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        F.append((x, y))
        a, b = seq[rx], seq[ry]
        assert a[-1] == x and b[0] == y
        coh = class_cohesion(a + b, A) if A is not None else float("nan")
        H.merges.append((a, b, coh))
        parent[ry] = rx
        seq[rx] = a + b
        del seq[ry]
    roots = sorted(seq.values(), key=lambda s: s[0])
    H.roots = roots
    return GreedyResult(n, F, H, roots, len(cand))


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------


def jaccard(S, T):
    S, T = set(S), set(T)
    U = S | T
    if not U:
        return 0.0
    return 1 - len(S & T) / len(U)


def intervals(paths):
    out = set()
    for p in paths:
        p = tuple(p)
        for s in range(len(p)):
            for t in range(s + 1, len(p) + 1):
                out.add(p[s:t])
    return out


def compare(HG: Hierarchy, G: GreedyResult):
    cg = HG.classes(True)
    cn = G.hierarchy.classes(True)
    I = intervals(G.paths)
    zero = sum(1 for _, _, c in G.hierarchy.merges if c == 0.0)
    nm = len(G.hierarchy.merges)
    return {
        "K_exact": len(HG.roots),
        "K_greedy": len(G.paths),
        "merges_exact": len(HG.merges),
        "merges_greedy": nm,
        "jaccard_all": jaccard(cg, cn),
        "jaccard_internal": jaccard(HG.classes(False), G.hierarchy.classes(False)),
        "zero_cohesion_merges_greedy": zero,
        "zero_fraction_greedy": zero / nm if nm else float("nan"),
        "intervals": len(I),
        "recall_intervals": len(cg & I) / len(cg),
        "precision_intervals": len(cg & I) / len(I),
    }
