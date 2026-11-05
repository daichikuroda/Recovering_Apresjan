"""Table 1: Iris, Wine, Breast Cancer and Digits with Euclidean and cosine dissimilarities.

For each data set and dissimilarity, report
  * triplet satisfaction of the ground-truth classes (%),
  * number of ground-truth classes that are valid clusters,
  * retained: number (and %) of the k-2 non-root internal clusters of the
    linkage tree that are valid, i.e. the size of T*(X, d) without root/leaves,
  * shared: % of non-root internal clusters common to the unpruned single-,
    complete- and average-linkage trees.

Features are used without rescaling; exact duplicate feature vectors are
removed so that d(x, y) > 0 for x != y.

Usage:  python benchmarks.py [--output results/table1_benchmarks.tsv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from apresjan import (
    apresjan_hierarchy,
    class_clusters,
    is_valid,
    shared_clusters,
    triplet_satisfaction,
)

DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
    "Digits": load_digits,
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a scikit-learn data set and drop exact duplicate rows (keeping the first)."""
    features, labels = DATASETS[name](return_X_y=True)
    _, first = np.unique(features, axis=0, return_index=True)
    first.sort()
    return features[first], labels[first]


def euclidean_dissimilarity(features: np.ndarray) -> np.ndarray:
    return squareform(pdist(features, "euclidean"))


def cosine_dissimilarity(features: np.ndarray) -> np.ndarray:
    """1 - cos(x, y): strictly decreasing in the cosine similarity, so it has
    the same valid clusters."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    d = 1.0 - similarity
    np.fill_diagonal(d, 0.0)
    return d


DISSIMILARITIES = {"Euclidean": euclidean_dissimilarity, "Cosine": cosine_dissimilarity}


def evaluate(d: np.ndarray, labels: np.ndarray) -> dict:
    """All quantities of one row of Table 1 for a dissimilarity matrix."""
    k = len(d)
    valid = apresjan_hierarchy(d)
    shared = shared_clusters(d)
    satisfied, total = triplet_satisfaction(d, labels)
    classes = class_clusters(labels)
    return {
        "k": k,
        "triplets_%": 100 * satisfied / total,
        "valid_classes": sum(is_valid(d, c) for c in classes),
        "classes": len(classes),
        "retained": len(valid),
        "retained_%": 100 * len(valid) / (k - 2),
        "shared": len(shared),
        "shared_%": 100 * len(shared) / (k - 2),
    }


def run() -> list[dict]:
    rows = []
    for name in DATASETS:
        features, labels = load_dataset(name)
        for dissimilarity, make in DISSIMILARITIES.items():
            row = {"dataset": name, "dissimilarity": dissimilarity}
            row.update(evaluate(make(features), labels))
            rows.append(row)
    return rows


def format_row(row: dict) -> dict:
    return {
        "dataset": row["dataset"],
        "k": row["k"],
        "dissimilarity": row["dissimilarity"],
        "triplets_%": f"{row['triplets_%']:.2f}",
        "valid_classes": f"{row['valid_classes']}/{row['classes']}",
        "retained": row["retained"],
        "retained_%": f"{row['retained_%']:.1f}",
        "shared": row["shared"],
        "shared_%": f"{row['shared_%']:.1f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("results/table1_benchmarks.tsv"))
    args = parser.parse_args()

    rows = [format_row(row) for row in run()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        for stream in (handle, sys.stdout):
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
