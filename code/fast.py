"""Vectorised versions of the exact implementations, for timing.

They use the same additive statistics, the same floating-point expressions and
the same lexicographic tie-breaking as ohier.py, and are checked against it in
bench_scale.py (identical merge sequences).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from ohier import INF, Hierarchy, _State

BIG = np.int64(10 ** 12)


def _row_keys(st, i, others):
    """Vectorised keys of (i, d) and (d, i) for d in `others`."""
    s = st.size[i] + st.size[others]
    den = s * (s - 1) / 2
    z_id = st.Zi[i] + st.Zi[others] + st.Zc[i, others]
    z_di = st.Zi[others] + st.Zi[i] + st.Zc[others, i]
    h_id = (st.Li[i] + st.Li[others] + st.Lc[i, others]) / den
    h_di = (st.Li[others] + st.Li[i] + st.Lc[others, i]) / den
    k_id = np.where(z_id > 0, INF, -h_id)
    k_di = np.where(z_di > 0, INF, -h_di)
    return k_id, k_di


def scan_vec(A):
    """O(n^3) exhaustive selection, vectorised per step."""
    st = _State(A)
    H = Hierarchy(st.n)
    while st.active.sum() > 1:
        act = np.flatnonzero(st.active)
        s = st.size[act][:, None] + st.size[act][None, :]
        den = s * (s - 1) / 2
        Z = st.Zi[act][:, None] + st.Zi[act][None, :] + st.Zc[np.ix_(act, act)]
        Hm = (st.Li[act][:, None] + st.Li[act][None, :] + st.Lc[np.ix_(act, act)]) / np.where(den > 0, den, 1)
        K0 = np.where(Z > 0, INF, -Hm)
        np.fill_diagonal(K0, np.nan)
        kmin = np.nanmin(K0)
        if kmin == INF:
            break
        cand = np.argwhere(K0 == kmin)
        ids = st.ident[act]
        best = min(cand, key=lambda c: (ids[c[0]], ids[c[1]]))
        H.merges.append(st.merge(act[best[0]], act[best[1]]))
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    return H


def heap_vec(A):
    """Lazy heap (Algorithm 3) with vectorised key computation."""
    st = _State(A)
    n = st.n
    H = Hierarchy(n)
    idx = np.arange(n)
    s = 2.0
    Z = st.Zc
    K0 = np.where(Z > 0, INF, -(st.Lc / (s * (s - 1) / 2)))
    heap = [(float(K0[i, j]), i, j, i, j) for i in range(n) for j in range(n) if i != j]
    heapq.heapify(heap)
    while heap:
        k0, ida, idb, i, j = heapq.heappop(heap)
        if not (st.active[i] and st.active[j] and st.ident[i] == ida and st.ident[j] == idb):
            continue
        if k0 == INF:
            break
        H.merges.append(st.merge(i, j))
        others = idx[st.active & (idx != i)]
        k_id, k_di = _row_keys(st, i, others)
        u = int(st.ident[i])
        for d, a, b in zip(others.tolist(), k_id.tolist(), k_di.tolist()):
            idd = int(st.ident[d])
            heapq.heappush(heap, (a, u, idd, i, d))
            heapq.heappush(heap, (b, idd, u, d, i))
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    return H


def tournament_vec(A):
    """Matrix tournament with vectorised leaf and level updates, O(n^2)."""
    st = _State(A)
    n = st.n
    N = 2
    while N < n:
        N *= 2
    K0 = np.full((N, N), INF)
    K1 = np.full((N, N), BIG, dtype=np.int64)
    K2 = np.full((N, N), BIG, dtype=np.int64)
    s = 2.0
    k0 = np.where(st.Zc > 0, INF, -(st.Lc / (s * (s - 1) / 2)))
    K0[:n, :n] = k0
    K1[:n, :n] = np.arange(n)[:, None]
    K2[:n, :n] = np.arange(n)[None, :]
    d = np.arange(n)
    K0[d, d], K1[d, d], K2[d, d] = INF, BIG, BIG
    k0f, k1f, k2f = K0.reshape(-1), K1.reshape(-1), K2.reshape(-1)

    def best_of(c):
        b = c[..., 0].copy()
        for t in range(1, 4):
            x = c[..., t]
            a0, x0 = k0f[b], k0f[x]
            a1, x1 = k1f[b], k1f[x]
            better = (x0 < a0) | ((x0 == a0) & ((x1 < a1) | ((x1 == a1) & (k2f[x] < k2f[b]))))
            b = np.where(better, x, b)
        return b

    levels = [np.arange(N * N, dtype=np.int64).reshape(N, N)]
    while levels[-1].shape[0] > 1:
        p = levels[-1]
        c = np.stack([p[0::2, 0::2], p[0::2, 1::2], p[1::2, 0::2], p[1::2, 1::2]], axis=-1)
        levels.append(best_of(c))
    idx = np.arange(n)
    H = Hierarchy(n)
    while True:
        root = int(levels[-1][0, 0])
        if k0f[root] == INF:
            break
        i, j = divmod(root, N)
        H.merges.append(st.merge(i, j))
        # invalidate row/col j
        K0[j, :], K1[j, :], K2[j, :] = INF, BIG, BIG
        K0[:, j], K1[:, j], K2[:, j] = INF, BIG, BIG
        others = idx[st.active & (idx != i)]
        k_id, k_di = _row_keys(st, i, others)
        u = st.ident[i]
        K0[i, :], K1[i, :], K2[i, :] = INF, BIG, BIG
        K0[:, i], K1[:, i], K2[:, i] = INF, BIG, BIG
        K0[i, others], K1[i, others], K2[i, others] = k_id, u, st.ident[others]
        K0[others, i], K1[others, i], K2[others, i] = k_di, st.ident[others], u
        for r in range(1, len(levels)):
            prev, cur = levels[r - 1], levels[r]
            for ln in {i >> r, j >> r}:
                # row block ln and column block ln
                cols = np.arange(cur.shape[0])
                for a, b in ((np.full_like(cols, ln), cols), (cols, np.full_like(cols, ln))):
                    c = np.stack([prev[2 * a, 2 * b], prev[2 * a, 2 * b + 1],
                                  prev[2 * a + 1, 2 * b], prev[2 * a + 1, 2 * b + 1]], axis=-1)
                    cur[a, b] = best_of(c)
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    return H


def greedy_vec(M, alpha=0.7):
    """Candidate generation and sorting vectorised; same output as greedy_two_step."""
    from ohier import greedy_two_step
    return greedy_two_step(M, alpha=alpha)


def scan_rows(A):
    """Full scan with additive statistics, one vectorised row of candidates at a time.

    At each step every active class i evaluates the keys of (i, j) for all other
    active j; the best key over all rows is merged. O(n^3) work, O(n^2) row
    evaluations. This is the scanning reference of the synthetic campaign.
    """
    st = _State(A)
    H = Hierarchy(st.n)
    idx = np.arange(st.n)
    while st.active.sum() > 1:
        act = idx[st.active]
        best = None
        for i in act:
            others = act[act != i]
            k_id, _ = _row_keys(st, i, others)
            kmin = k_id.min()
            js = others[k_id == kmin]
            j = js[np.argmin(st.ident[js])]
            cand = (float(kmin), int(st.ident[i]), int(st.ident[j]), int(i), int(j))
            if best is None or cand[:3] < best[:3]:
                best = cand
        if best[0] == INF:
            break
        H.merges.append(st.merge(best[3], best[4]))
    H.roots = [st.seq[i] for i in np.flatnonzero(st.active)]
    return H
