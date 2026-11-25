"""Model-based Q-matrix validation: saturated G-DINA (EM) and the PVAF method.

de la Torre, J. & Chiu, C.-Y. (2016). A general method of empirical Q-matrix
validation. Psychometrika, 81, 253-273.

* Saturated G-DINA, identity link, saturated attribute distribution over the
  2^K attribute profiles, fitted by marginal maximum likelihood (EM).
* For item j and a candidate q-vector q, latent profiles are grouped by their
  restriction to the attributes of q; the success probability of each group is
  estimated from the expected counts of the fitted model (fitted under the
  provisional Q), and
      sigma2_j(q) = sum_g w_g (P_g - Pbar_j)^2,   PVAF_j(q) = sigma2_j(q) / sigma2_j(1_K),
  where w_g is the estimated proportion of profiles in group g.
* Suggested q-vector: among the q-vectors with PVAF >= eps (eps = 0.95 by
  default), the one with the fewest attributes; ties broken by larger PVAF.
  All 2^K - 1 non-zero q-vectors are examined.
"""
from __future__ import annotations

import numpy as np


def profiles(K):
    L = 2 ** K
    return ((np.arange(L)[:, None] >> np.arange(K)[None, :]) & 1).astype(np.int8)


def _group_index(alpha, q):
    """Index of the reduced group of each profile for q-vector q (0/1 array)."""
    idx = np.flatnonzero(q)
    if idx.size == 0:
        return np.zeros(alpha.shape[0], dtype=np.int64)
    w = (1 << np.arange(idx.size)).astype(np.int64)
    return alpha[:, idx].astype(np.int64) @ w


def fit_gdina(X, Q, max_iter=2000, tol=1e-7, eps_p=1e-4, seed=0):
    """Saturated G-DINA by EM. Returns dict with posterior, prior, P (J x L)."""
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=np.int8)
    N, J = X.shape
    K = Q.shape[1]
    alpha = profiles(K)
    L = alpha.shape[0]
    groups = [_group_index(alpha, Q[j]) for j in range(J)]
    ngroups = [int(2 ** Q[j].sum()) for j in range(J)]
    rng = np.random.default_rng(seed)
    # start: monotone-ish success probabilities (0.2 for no mastery -> 0.8 full)
    P = np.empty((J, L))
    for j in range(J):
        kj = max(int(Q[j].sum()), 1)
        frac = alpha[:, np.flatnonzero(Q[j])].sum(axis=1) / kj if Q[j].sum() else np.zeros(L)
        P[j] = 0.2 + 0.6 * frac + rng.uniform(-0.01, 0.01, L)
    prior = np.full(L, 1.0 / L)
    ll_old = -np.inf
    for it in range(max_iter):
        lp = X @ np.log(P) + (1 - X) @ np.log(1 - P) + np.log(prior)[None, :]
        m = lp.max(axis=1, keepdims=True)
        w = np.exp(lp - m)
        s = w.sum(axis=1, keepdims=True)
        post = w / s
        ll = float((m[:, 0] + np.log(s[:, 0])).sum())
        Nl = post.sum(axis=0)  # expected count per profile
        R = X.T @ post  # J x L expected correct per profile
        for j in range(J):
            g = groups[j]
            num = np.bincount(g, weights=R[j], minlength=ngroups[j])
            den = np.bincount(g, weights=Nl, minlength=ngroups[j])
            pg = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.5)
            P[j] = np.clip(pg[g], eps_p, 1 - eps_p)
        prior = np.maximum(Nl / N, 1e-12)
        prior /= prior.sum()
        if ll - ll_old < tol:
            break
        ll_old = ll
    return {"alpha": alpha, "post": post, "prior": prior, "P": P, "R": R, "Nl": Nl,
            "loglik": ll, "iterations": it + 1}


def pvaf(fit, j, eps=0.95):
    """PVAF of every non-zero q-vector for item j; returns sorted candidates."""
    alpha, R, Nl = fit["alpha"], fit["R"], fit["Nl"]
    K = alpha.shape[1]
    w_tot = Nl.sum()

    def sigma2(q):
        g = _group_index(alpha, q)
        ng = int(2 ** int(q.sum()))
        num = np.bincount(g, weights=R[j], minlength=ng)
        den = np.bincount(g, weights=Nl, minlength=ng)
        pg = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        wg = den / w_tot
        pbar = float((wg * pg).sum())
        return float((wg * (pg - pbar) ** 2).sum())

    full = sigma2(np.ones(K, dtype=np.int8))
    cands = []
    for code in range(1, 2 ** K):
        q = ((code >> np.arange(K)) & 1).astype(np.int8)
        cands.append((q, sigma2(q) / full if full > 0 else 0.0))
    ok = [(q, v) for q, v in cands if v >= eps]
    best = min(ok, key=lambda t: (int(t[0].sum()), -t[1]))
    return best, cands


def validate_q(X, Q, eps=0.95):
    """Fit G-DINA under Q, then PVAF suggestion for every item."""
    fit = fit_gdina(X, Q)
    out = []
    for j in range(Q.shape[0]):
        (qs, vs), cands = pvaf(fit, j, eps)
        d = {tuple(q.tolist()): v for q, v in cands}
        out.append({
            "item": j,
            "q_original": Q[j].astype(int).tolist(),
            "pvaf_original": float(d[tuple(np.asarray(Q[j], dtype=np.int8).tolist())]) if Q[j].sum() else 0.0,
            "q_suggested": qs.astype(int).tolist(),
            "pvaf_suggested": float(vs),
            "changed": bool(not np.array_equal(qs, Q[j])),
        })
    return fit, out


def pvaf_of(fit, j, q):
    q = np.asarray(q, dtype=np.int8)
    _, cands = pvaf(fit, j, eps=0.0)
    for qq, v in cands:
        if np.array_equal(qq, q):
            return float(v)
    return 0.0
