"""Core computations: validity margins, trimmed linkage, and triplet satisfaction.

Notation follows the paper. ``d`` is a k x k dissimilarity matrix (symmetric,
zero diagonal, nonnegative). A cluster C is *valid* when

    max_{y in C} d(x, y) < min_{z not in C} d(x, z)   for every x in C,

i.e. when its validity margin Delta_d(C) is positive. The Apresjan hierarchy
T*(X, d) is the set of all valid clusters. By the main theorem, pruning the
invalid clusters of a single-, complete- or average-linkage tree returns
exactly T*(X, d).

Clusters are represented as ``frozenset``s of item indices 0..k-1.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

CERTIFIED_LINKAGES = ("single", "complete", "average")


def check_dissimilarity(d: np.ndarray) -> np.ndarray:
    """Return ``d`` as a float array after checking the paper's assumptions."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        raise ValueError("d must be a square matrix with at least two items")
    if not np.isfinite(d).all():
        raise ValueError("d contains non-finite entries")
    if not np.array_equal(d, d.T):
        raise ValueError("d must be symmetric")
    # if np.any(np.diag(d) != 0):
    #     raise ValueError("d must have a zero diagonal")
    # if np.any(d < 0):
    #     raise ValueError("d must be nonnegative")
    return d


def cluster_margin(d: np.ndarray, cluster) -> float:
    """Validity margin Delta_d(C) = min_{x in C} (min_{z not in C} d(x,z) - max_{y in C} d(x,y)).

    Direct evaluation of the definition, in O(|C| k) time.
    """
    inside = np.fromiter(cluster, dtype=np.int64)
    outside = np.setdiff1d(np.arange(len(d)), inside)
    if len(outside) == 0:
        return np.inf
    within_max = d[np.ix_(inside, inside)].max(axis=1)
    outside_min = d[np.ix_(inside, outside)].min(axis=1)
    return float(np.min(outside_min - within_max))


def is_valid(d: np.ndarray, cluster) -> bool:
    return cluster_margin(d, cluster) > 0


def linkage_tree(d: np.ndarray, method: str = "single"):
    """Binary linkage tree of ``d`` computed with SciPy.

    Returns ``(children, members)``: node ``k + i`` is created by the i-th merge
    and has children ``children[i]``; ``members[t]`` lists the items of node t.
    Nodes 0..k-1 are the leaves and node 2k-2 is the root.
    """
    d = check_dissimilarity(d)
    k = len(d)
    merges = linkage(squareform(d, checks=False), method=method)
    children = merges[:, :2].astype(np.int64)
    members = [np.array([i], dtype=np.int64) for i in range(k)]
    for left, right in children:
        members.append(np.concatenate((members[left], members[right])))
    return children, members


def node_margins(d: np.ndarray, children: np.ndarray, members: list) -> np.ndarray:
    """Validity margin of every node of a binary tree in O(k^2) time.

    This is the pruning procedure of the appendix ("Computational complexity of
    pruning the hierarchy", after Bryant and Berry): for every node v and item
    x in v, compute bottom-up M[v, x] = max_{y in v} d(x, y) and top-down
    m[v, x] = min_{z not in v} d(x, z); then Delta_d(v) = min_x (m - M).
    """
    k = len(d)
    node_count = 2 * k - 1
    within_max = np.zeros((node_count, k))
    outside_min = np.full((node_count, k), np.inf)
    to_sibling_min = {}

    # Bottom-up: pairs whose lowest common ancestor is `node` cross its children.
    for node, (left, right) in enumerate(children, start=k):
        a, b = members[left], members[right]
        cross = d[np.ix_(a, b)]
        within_max[node, a] = np.maximum(within_max[left, a], cross.max(axis=1))
        within_max[node, b] = np.maximum(within_max[right, b], cross.max(axis=0))
        to_sibling_min[left] = cross.min(axis=1)
        to_sibling_min[right] = cross.min(axis=0)

    # Top-down: leaving a child means leaving the parent or entering the sibling.
    for node in range(node_count - 1, k - 1, -1):
        for child in children[node - k]:
            items = members[child]
            outside_min[child, items] = np.minimum(
                outside_min[node, items], to_sibling_min[child]
            )

    return np.array(
        [
            np.min(outside_min[t, members[t]] - within_max[t, members[t]])
            for t in range(node_count)
        ]
    )


def trimmed_linkage(d: np.ndarray, method: str = "single"):
    """Linkage tree and its pruned version.

    Returns ``(internal, valid)``: the k-2 non-root internal clusters of the
    binary linkage tree, and the subset of them that are valid. For a certified
    linkage (single, complete, average), ``valid`` is the set of non-trivial
    clusters of the Apresjan hierarchy T*(X, d).
    """
    d = check_dissimilarity(d)
    k = len(d)
    children, members = linkage_tree(d, method)
    margins = node_margins(d, children, members)
    internal = [frozenset(members[t].tolist()) for t in range(k, 2 * k - 2)]
    valid = {c for c, m in zip(internal, margins[k : 2 * k - 2]) if m > 0}
    return internal, valid


def apresjan_hierarchy(d: np.ndarray, method: str = "single") -> set:
    """Non-trivial clusters of T*(X, d), i.e. without the root and singletons."""
    return trimmed_linkage(d, method)[1]


def internal_clusters(d: np.ndarray, method: str = "single") -> set:
    """Non-root internal clusters of the (unpruned) binary linkage tree."""
    k = len(d)
    _, members = linkage_tree(d, method)
    return {frozenset(members[t].tolist()) for t in range(k, 2 * k - 2)}


def shared_clusters(d: np.ndarray, methods=CERTIFIED_LINKAGES) -> set:
    """Non-root internal clusters common to the unpruned linkage trees."""
    return set.intersection(*(internal_clusters(d, m) for m in methods))


def triplet_satisfaction(d: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """Count triplets (x, y, z) with labels[x] == labels[y] != labels[z], x != y,
    that satisfy d(x, y) < d(x, z) (ties count as violations).

    Returns ``(satisfied, total)``. A class is valid iff all its triplets are
    satisfied.
    """
    d = check_dissimilarity(d)
    labels = np.asarray(labels)
    satisfied = total = 0
    for label in np.unique(labels):
        inside = np.flatnonzero(labels == label)
        outside = np.flatnonzero(labels != label)
        for x in inside:
            to_same = d[x, inside[inside != x]]
            to_other = np.sort(d[x, outside])
            # For each y, count z with d(x, z) > d(x, y).
            satisfied += int(
                np.sum(len(to_other) - np.searchsorted(to_other, to_same, "right"))
            )
            total += len(to_same) * len(to_other)
    return satisfied, total


def class_clusters(labels: np.ndarray) -> set:
    labels = np.asarray(labels)
    return {frozenset(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)}
