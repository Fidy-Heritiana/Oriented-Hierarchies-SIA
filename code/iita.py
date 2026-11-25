"""Inductive item tree analysis (IITA), minimized corrected version.

Sargin, A. & Ünlü, A. (2009). Inductive item tree analysis: corrections,
improvements, and comparisons. Mathematical Social Sciences, 58, 376-392.
Schrepp, M. (2003). A method for the analysis of hierarchical dependencies
between items of a questionnaire. Methods of Psychological Research Online.

Convention: i <= j means "solving j implies solving i" (i is a prerequisite
of j).  Counterexamples to i <= j: b_ij = #{x_i = 0, x_j = 1}.

1. Inductive generation: A_0 is built from pairs with b_ij = 0; for
   L = 1, 2, ..., candidate pairs with b_ij <= L are added to A_{L-1}, and
   candidates that would break transitivity are removed iteratively.  Every
   A_L is a quasi-order.
2. Fit: diff(R) = sum_{i != j} (b_ij - b*_ij)^2 / (m (m - 1)), with expected
   counterexamples (corrected estimators)
     i <= j                 : b*_ij = gamma p_j N
     not i <= j, j <= i     : b*_ij = (p_j - p_i + gamma p_i) N
     neither                : b*_ij = (1 - p_i) p_j N
   and gamma chosen to minimise diff (closed form, clipped to [0, 1]).
3. The quasi-order with the smallest diff is selected.
"""
from __future__ import annotations

import numpy as np


def counterexamples(X):
    X = np.asarray(X, dtype=np.int64)
    # b[i, j] = #{x_i = 0 and x_j = 1}
    return (1 - X).T @ X


def inductive_quasi_orders(B):
    m = B.shape[0]
    off = ~np.eye(m, dtype=bool)
    A = np.eye(m, dtype=bool)
    orders = []
    # S only changes at the observed values of b, so iterating over them is
    # equivalent to iterating over every integer L = 0, 1, ..., max b.
    levels = sorted(set(int(v) for v in B[off]) | {0}) if off.any() else [0]
    for L in levels:
        S = (B <= L) & off & ~A
        if not S.any():
            if L == 0:
                orders.append(A.copy())
            continue
        changed = True
        while changed:
            R = A | S
            # (i, j) in S is removed if exists k: (j,k) in R and (i,k) not in R,
            # or exists k: (k,i) in R and (k,j) not in R
            Ri = R.astype(np.int64)
            notR = (~R).astype(np.int64)
            viol1 = (notR @ Ri.T) > 0  # [i,j] = sum_k ~R[i,k] R[j,k]
            viol2 = (Ri.T @ notR) > 0  # [i,j] = sum_k R[k,i] ~R[k,j]
            bad = S & (viol1 | viol2)
            changed = bool(bad.any())
            S = S & ~bad
        A = A | S
        if not orders or not np.array_equal(orders[-1], A):
            orders.append(A.copy())
    return orders


def diff_mini(R, B, p, N):
    m = B.shape[0]
    off = ~np.eye(m, dtype=bool)
    Rt = R.T
    pi = p[:, None]
    pj = p[None, :]
    case1 = R & off
    case2 = ~R & Rt & off
    case3 = ~R & ~Rt & off
    a = np.where(case1, 0.0, np.where(case2, (pj - pi) * N, (1 - pi) * pj * N))
    c = np.where(case1, pj * N, np.where(case2, pi * N, 0.0))
    a, c, b = a[off], c[off], B[off].astype(float)
    den = (c * c).sum()
    gamma = float(np.clip((c * (b - a)).sum() / den, 0.0, 1.0)) if den > 0 else 0.0
    bstar = a + gamma * c
    return float(((b - bstar) ** 2).sum() / (m * (m - 1))), gamma


def iita(X):
    X = np.asarray(X, dtype=np.int64)
    N, m = X.shape
    B = counterexamples(X)
    p = X.mean(axis=0)
    orders = inductive_quasi_orders(B)
    best = None
    for R in orders:
        d, g = diff_mini(R, B, p, N)
        if best is None or d < best[0]:
            best = (d, g, R)
    d, g, R = best
    return {"quasi_order": R, "diff": d, "gamma": g, "n_candidates": len(orders)}


def implications(R):
    """List of implications u => v (success on u implies success on v), i.e. v <= u."""
    m = R.shape[0]
    return [(j, i) for i in range(m) for j in range(m) if i != j and R[i, j]]
