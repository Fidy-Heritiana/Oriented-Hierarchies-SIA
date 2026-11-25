"""Scaling benchmark of the exact implementations (vectorised Python/NumPy).

Two input families:
  * worst case: regime P (all pair cohesions positive), so the hierarchy has
    n - 1 merges;
  * simulated item banks: DINA responses of N = 2000 examinees to n items,
    with K = 10 attributes in a linear hierarchy and 1 to 3 attributes per item,
    entropic intensities (the realistic input of the application).
Times are medians over repetitions; output in results/final/scaling.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fast import heap_vec, scan_vec, tournament_vec  # noqa: E402
from ohier import entropic_intensity, greedy_two_step, impose_frequency_order, pair_cohesion  # noqa: E402
from verify import regime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def timeit(f, *a, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = f(*a)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def dina_bank(rng, n, N=2000, K=10):
    # linear attribute hierarchy: mastery of attribute k requires k-1
    level = rng.integers(0, K + 1, N)
    alpha = (np.arange(K)[None, :] < level[:, None]).astype(int)
    # add some non-hierarchical noise in profiles
    flip = rng.random(alpha.shape) < 0.05
    alpha = np.where(flip, 1 - alpha, alpha)
    Q = np.zeros((n, K), dtype=int)
    for j in range(n):
        Q[j, rng.choice(K, rng.integers(1, 4), replace=False)] = 1
    eta = (alpha @ Q.T) == Q.sum(1)[None, :]
    g = rng.uniform(0.05, 0.25, n)
    s = rng.uniform(0.05, 0.25, n)
    return (rng.random((N, n)) < np.where(eta, 1 - s, g)).astype(int)


def main():
    rng = np.random.default_rng(2026)
    res = {"worst_case": [], "item_bank": []}
    for n in [100, 200, 400, 800, 1600, 3200]:
        A = pair_cohesion(regime(rng, n, "P"))
        reps = 3 if n <= 800 else 1
        row = {"n": n}
        t, H = timeit(tournament_vec, A, reps=reps)
        row["tournament_s"], row["merges"] = t, len(H.merges)
        ref = H.merge_sequence()
        t, H2 = timeit(scan_vec, A, reps=reps)
        row["scan_s"] = t
        assert H2.merge_sequence() == ref
        if n <= 800:
            t, H3 = timeit(heap_vec, A, reps=reps)
            row["heap_s"] = t
            assert H3.merge_sequence() == ref
        res["worst_case"].append(row)
        print("worst", row, flush=True)
    for n in [25, 50, 100, 200, 400, 800]:
        X = dina_bank(rng, n)
        reps = 3
        row = {"n": n, "N": X.shape[0]}
        t, M = timeit(lambda X: impose_frequency_order(entropic_intensity(X), X), X, reps=reps)
        row["intensity_s"] = t
        A = pair_cohesion(M)
        t, H = timeit(tournament_vec, A, reps=reps)
        row["tournament_s"], row["merges"] = t, len(H.merges)
        t, H2 = timeit(scan_vec, A, reps=reps)
        row["scan_s"] = t
        assert H2.merge_sequence() == H.merge_sequence()
        t, H3 = timeit(heap_vec, A, reps=reps)
        row["heap_s"] = t
        t, G = timeit(lambda M: greedy_two_step(M, alpha=0.7), M, reps=reps)
        row["greedy_s"] = t
        best = min(row["tournament_s"], row["scan_s"], row["heap_s"])
        row["bootstrap_B1000_minutes_best_exact"] = 1000 * (row["intensity_s"] + best) / 60
        row["bootstrap_B1000_minutes_scan"] = 1000 * (row["intensity_s"] + row["scan_s"]) / 60
        res["item_bank"].append(row)
        print("bank", row, flush=True)
    with open(ROOT / "results/final/scaling.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
