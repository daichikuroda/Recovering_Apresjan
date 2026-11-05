"""Numerical checks of the worked examples in the paper.

  1. Example ex:running (4 points): T* = {{x1,x2}}; single linkage forms
     {x1,x2,x3}, complete and average linkage form {x1,x2,x4}; all three
     trimmed trees equal T*.
  2. Figure fig:ward_fails (5 points): d is a metric, T* = {{x1,x2}, {x3,x4},
     {x1,x2,x3,x4}}, and Ward linkage (Lance-Williams update on d, and SciPy's
     Ward on Euclidean distances) forms {x1,x2,x5}, so T* is not contained in
     the Ward tree.
  3. Figure fig:dasgupta: Dasgupta costs of the five trees on 4 leaves for the
     unit-weight clique.

Linkages are run with a direct implementation of Algorithm 1 (general update
rule f), which also checks that every merge is a strict minimum (no ties).

Usage:  python examples.py
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from apresjan import apresjan_hierarchy, internal_clusters, is_valid, trimmed_linkage

RUNNING_EXAMPLE = np.array([
    [0.0, 1.1, 2.0, 2.2],
    [1.1, 0.0, 3.0, 2.2],
    [2.0, 3.0, 0.0, 2.5],
    [2.2, 2.2, 2.5, 0.0],
])

WARD_COUNTEREXAMPLE = np.array([
    [0, 4, 15, 15, 16],
    [4, 0, 15, 15, 16],
    [15, 15, 0, 3, 21],
    [15, 15, 3, 0, 21],
    [16, 16, 21, 21, 0],
], dtype=float)

# Update rules f(d(t1,t3), d(t2,t3), d(t1,t2), |t1|, |t2|, |t3|) = d(t1 u t2, t3).
UPDATE_RULES = {
    "single": lambda d13, d23, d12, n1, n2, n3: min(d13, d23),
    "complete": lambda d13, d23, d12, n1, n2, n3: max(d13, d23),
    "average": lambda d13, d23, d12, n1, n2, n3: (n1 * d13 + n2 * d23) / (n1 + n2),
    # Lance-Williams coefficients of the appendix table, applied to d.
    "ward": lambda d13, d23, d12, n1, n2, n3: (
        (n1 + n3) * d13 + (n2 + n3) * d23 - n3 * d12) / (n1 + n2 + n3),
    # Same update on squared distances (what SciPy's "ward" does).
    "ward_squared": lambda d13, d23, d12, n1, n2, n3: np.sqrt((
        (n1 + n3) * d13**2 + (n2 + n3) * d23**2 - n3 * d12**2) / (n1 + n2 + n3)),
}


def linkage_clusters(d: np.ndarray, rule: str, *, strict: bool = True) -> list[frozenset]:
    """Algorithm 1: repeatedly merge the closest pair and update with f.

    Returns the clusters in the order they are formed (the last is the root).
    With ``strict=True``, raises if the closest pair is not unique.
    """
    f = UPDATE_RULES[rule]
    clusters = [frozenset([i]) for i in range(len(d))]
    dist = {(a, b): d[min(a), min(b)] for a, b in combinations(clusters, 2)}
    dist.update({(b, a): v for (a, b), v in list(dist.items())})
    formed = []
    while len(clusters) > 1:
        pairs = sorted(combinations(clusters, 2), key=lambda p: dist[p])
        t1, t2 = pairs[0]
        if strict and len(pairs) > 1 and dist[pairs[1]] == dist[pairs[0]]:
            raise ValueError(f"tie between {pairs[0]} and {pairs[1]}")
        merged = t1 | t2
        others = [t for t in clusters if t not in (t1, t2)]
        for t3 in others:
            value = f(dist[t1, t3], dist[t2, t3], dist[t1, t2], len(t1), len(t2), len(t3))
            dist[merged, t3] = dist[t3, merged] = value
        clusters = others + [merged]
        formed.append(merged)
    return formed


def brute_force_apresjan(d: np.ndarray) -> set:
    """All valid clusters C with 2 <= |C| < k, by enumeration (small k only)."""
    k = len(d)
    return {
        frozenset(c)
        for size in range(2, k)
        for c in combinations(range(k), size)
        if is_valid(d, c)
    }


def is_metric(d: np.ndarray) -> bool:
    k = len(d)
    return all(d[i, j] <= d[i, l] + d[l, j] for i in range(k) for j in range(k) for l in range(k))


def dasgupta_cost(tree, weights: np.ndarray) -> float:
    """sum_{i<j} w_ij |leaves(lca(i, j))| for a tree given as nested tuples."""
    def leaves(node):
        return frozenset([node]) if isinstance(node, int) else frozenset().union(*map(leaves, node))

    def clusters(node):
        if isinstance(node, int):
            return [leaves(node)]
        return [leaves(node)] + [c for child in node for c in clusters(child)]

    all_clusters = clusters(tree)
    cost = 0.0
    for i, j in combinations(sorted(leaves(tree)), 2):
        lca = min((c for c in all_clusters if i in c and j in c), key=len)
        cost += weights[i, j] * len(lca)
    return cost


DASGUPTA_TREES = {  # Figure fig:dasgupta (b)-(f), leaves 0..3
    "b": (0, 1, 2, 3),
    "c": ((0, 1), (2, 3)),
    "d": (((0, 1), 2), 3),
    "e": ((0, 1, 2), 3),
    "f": ((0, 1), 2, 3),
}


def show(clusters) -> str:
    """Clusters in the paper's 1-based notation."""
    return ", ".join("{" + ",".join(f"x{i + 1}" for i in sorted(c)) + "}"
                     for c in sorted(clusters, key=lambda c: (len(c), sorted(c))))


def main() -> None:
    d = RUNNING_EXAMPLE
    print("Example ex:running")
    print(f"  metric: {is_metric(d)}")
    print(f"  T* (brute force):        {show(brute_force_apresjan(d))}")
    for rule in ("single", "complete", "average"):
        tree = linkage_clusters(d, rule)[:-1]
        print(f"  {rule:8s} linkage tree:  {show(tree)}  -> pruned: "
              f"{show(apresjan_hierarchy(d, rule))}")

    d = WARD_COUNTEREXAMPLE
    print("Figure fig:ward_fails")
    print(f"  metric: {is_metric(d)}")
    print(f"  T* (brute force):        {show(brute_force_apresjan(d))}")
    for rule in ("ward", "ward_squared"):
        print(f"  {rule:12s} merges:   {show(linkage_clusters(d, rule)[:-1])}")
    print(f"  SciPy ward tree:         {show(internal_clusters(d, 'ward'))}")
    _, pruned = trimmed_linkage(d, "ward")
    print(f"  Ward tree after pruning: {show(pruned)}  (misses {{x1,x2,x3,x4}})")

    print("Figure fig:dasgupta (unit-weight clique on 4 nodes)")
    weights = np.ones((4, 4))
    for name, tree in DASGUPTA_TREES.items():
        print(f"  ({name}) {tree}: Dasgupta cost {dasgupta_cost(tree, weights):g}")


if __name__ == "__main__":
    main()
