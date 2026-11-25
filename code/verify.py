"""Check that scan, heap and tournament give identical merge sequences."""
import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ohier import class_cohesion, exact_heap, exact_scan, exact_tournament, greedy_two_step, pair_cohesion  # noqa: E402


def regime(rng, n, kind):
    if kind == "U":
        M = rng.uniform(0, 1, (n, n))
    elif kind == "P":
        M = rng.uniform(0.51, 0.99, (n, n))
    else:  # O
        perm = rng.permutation(n)
        rank = np.empty(n, int)
        rank[perm] = np.arange(n)
        fwd = rank[:, None] < rank[None, :]
        M = np.where(fwd, rng.uniform(0.75, 0.99, (n, n)), rng.uniform(0.01, 0.49, (n, n)))
    np.fill_diagonal(M, 0)
    return M


def main():
    rng = np.random.default_rng(20261125)
    checked = 0
    max_ins = {}
    for n in [3, 5, 8, 12, 20, 40]:
        for kind in "UPO":
            reps = 30 if n <= 20 else 5
            for _ in range(reps):
                A = pair_cohesion(regime(rng, n, kind))
                hs = exact_scan(A)
                hh, sh = exact_heap(A, return_stats=True)
                ht, stt = exact_tournament(A, return_stats=True)
                assert hs.merge_sequence() == hh.merge_sequence() == ht.merge_sequence(), (n, kind)
                assert sorted(hs.roots) == sorted(hh.roots) == sorted(ht.roots)
                for a, b, c in hh.merges:
                    assert abs(c - class_cohesion(a + b, A)) < 1e-9
                assert sh["heap_insertions"] <= 2 * (n - 1) ** 2
                N = stt["N"]
                assert stt["recomputed_nodes"] <= 4 * (N - 1) * (n - 1) * 2  # two lines per merge
                max_ins[n] = max(max_ins.get(n, 0), sh["heap_insertions"])
                G = greedy_two_step(regime(rng, n, kind), alpha=0.7, A=A)
                for p in G.paths:
                    assert len(set(p)) == len(p)
                assert sum(len(p) for p in G.paths) == n
                checked += 1
    # exact ties: all-zero and all-one off-diagonal scores
    ties = 0
    for n in range(2, 13):
        for val in (0.0, 1.0):
            A = np.full((n, n), val)
            np.fill_diagonal(A, 0)
            hs, hh, ht = exact_scan(A), exact_heap(A), exact_tournament(A)
            assert hs.merge_sequence() == hh.merge_sequence() == ht.merge_sequence()
            ties += 1
    # mixed ties (scores in {0, 0.5, 1})
    for n in [4, 6, 9]:
        for _ in range(20):
            A = rng.choice([0.0, 0.8, 1.0], size=(n, n))
            np.fill_diagonal(A, 0)
            hs, hh, ht = exact_scan(A), exact_heap(A), exact_tournament(A)
            assert hs.merge_sequence() == hh.merge_sequence() == ht.merge_sequence()
            ties += 1
    print(f"random inputs checked: {checked}; tie inputs checked: {ties}")
    print("max heap insertions by n:", max_ins, "(bound 2(n-1)^2)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
