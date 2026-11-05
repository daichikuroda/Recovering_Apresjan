"""Planted hierarchy under noise (appendix, "Robustness to perturbations", Figure fig:synthetic).

k = 36 items: 9 fine clusters of 4 items, nested in 3 coarse clusters of 3
fine clusters. The ultrametric d_0 equals 2 within a fine cluster, 6 between
fine clusters of the same coarse cluster and 12 between coarse clusters, so
T*(X, d_0) is the planted hierarchy (12 non-root internal clusters, minimum
margin 4). For a noise level rho we set d_rho = d_0 + 4 rho U, where U is
symmetric with zero diagonal and i.i.d. Uniform[-1, 1] entries above the
diagonal; entries are floored at machine epsilon to keep d_rho positive (only
active for rho >= 1/2). Each draw U^(r) is reused across the rho grid.

Proposition (robustness) guarantees that every planted cluster is recovered
for rho < 1/2. For each rho we report, over 50 draws (mean and std):
  * planted clusters still valid for d_rho,
  * planted clusters present in the (unpruned) single-linkage tree,
  * planted clusters retained after pruning (= recall of T*(X, d_rho)).

Usage:  python synthetic.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from apresjan import cluster_margin, trimmed_linkage

SEED = 20260801
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REPEATS = 50


def planted_hierarchy(coarse=3, fine_per_coarse=3, leaves_per_fine=4):
    """Fine and coarse labels, and the set of planted non-root clusters."""
    fine = np.repeat(np.arange(coarse * fine_per_coarse), leaves_per_fine)
    coarse_labels = fine // fine_per_coarse
    clusters = {
        frozenset(np.flatnonzero(labels == c).tolist())
        for labels in (fine, coarse_labels)
        for c in np.unique(labels)
    }
    return fine, coarse_labels, clusters


def planted_ultrametric(fine, coarse, distances=(2.0, 6.0, 12.0)) -> np.ndarray:
    within_fine, within_coarse, between_coarse = distances
    d = np.full((len(fine), len(fine)), between_coarse)
    d[coarse[:, None] == coarse[None, :]] = within_coarse
    d[fine[:, None] == fine[None, :]] = within_fine
    np.fill_diagonal(d, 0.0)
    return d


def perturb(d: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    """d + epsilon * U with U symmetric, U_ij ~ Unif[-1, 1] for i < j, floored at eps."""
    rows, cols = np.triu_indices(len(d), 1)
    noise = np.random.default_rng(seed).uniform(-epsilon, epsilon, size=len(rows))
    upper = np.maximum(d[rows, cols] + noise, np.finfo(np.float64).eps)
    result = np.zeros_like(d)
    result[rows, cols] = upper
    result[cols, rows] = upper
    return result


def trial(d: np.ndarray, planted: set) -> dict:
    """Percentages of planted clusters that are valid / in the tree / retained."""
    internal, valid = trimmed_linkage(d, "single")
    n = len(planted)
    return {
        "planted_valid_%": 100 * sum(cluster_margin(d, c) > 0 for c in planted) / n,
        "in_single_linkage_%": 100 * len(planted & set(internal)) / n,
        "retained_%": 100 * len(planted & valid) / n,
        "exact_recovery": float(valid == planted),
    }


def run(rhos=RHOS, repeats=REPEATS, seed=SEED) -> list[dict]:
    fine, coarse, planted = planted_hierarchy()
    d0 = planted_ultrametric(fine, coarse)
    min_margin = min(cluster_margin(d0, c) for c in planted)  # = 4
    draw_seeds = np.random.default_rng(seed).integers(
        0, np.iinfo(np.int32).max, size=repeats, dtype=np.int64
    )
    summary = []
    for rho in rhos:
        epsilon = rho * min_margin
        trials = [trial(perturb(d0, epsilon, int(s)), planted) for s in draw_seeds]
        row = {"rho": rho, "epsilon": epsilon, "draws": repeats}
        for key in ("planted_valid_%", "in_single_linkage_%", "retained_%"):
            values = np.array([t[key] for t in trials])
            row[f"{key}_mean"] = values.mean()
            row[f"{key}_std"] = values.std()
        row["exact_recovery_%"] = 100 * np.mean([t["exact_recovery"] for t in trials])
        summary.append(row)
    return summary


def plot_recall(summary: list[dict], path: Path) -> None:
    """Right panel of Figure fig:synthetic (saved as PDF and PNG)."""
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts
    plt.rcParams["ps.fonttype"] = 42
    rho = np.array([row["rho"] for row in summary])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    curves = [
        ("planted_valid_%", "o", "-", "Planted clusters still valid"),
        ("in_single_linkage_%", "s", "dashed", "Present in single linkage"),
        ("retained_%", "^", "--", "Retained after pruning"),
    ]
    for key, marker, linestyle, label in curves:
        mean = np.array([row[f"{key}_mean"] for row in summary])
        std = np.array([row[f"{key}_std"] for row in summary])
        line = ax.plot(rho, mean, marker=marker, linestyle=linestyle, label=label)[0]
        ax.errorbar(rho, mean, yerr=std, fmt="none", ecolor=line.get_color(),
                    elinewidth=1.0, capsize=3)
    ax.axvline(0.5, color="0.3", linestyle=":", linewidth=1.2)  # rho = 1/2
    ax.set_ylim(-2, 102)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Recall (%)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_summary.tsv"))
    parser.add_argument("--figure", type=Path, default=Path("results/synthetic_hierarchy_recall.pdf"))
    args = parser.parse_args()

    fine, coarse, planted = planted_hierarchy()
    d0 = planted_ultrametric(fine, coarse)
    internal, valid = trimmed_linkage(d0, "single")
    removed = len(internal) - len(valid)
    print(f"rho = 0: {len(internal)} non-root internal clusters in the binary tree, "
          f"{len(valid)} valid (planted: {len(planted)}, exact: {valid == planted}), "
          f"{removed} removed ({100 * removed / len(internal):.1f}%)")

    summary = run(repeats=args.repeats, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    plot_recall(summary, args.figure)
    for row in summary:
        print(f"rho={row['rho']:.1f}  recall={row['retained_%_mean']:5.1f} "
              f"± {row['retained_%_std']:4.1f}%  still valid={row['planted_valid_%_mean']:5.1f}%  "
              f"in single linkage={row['in_single_linkage_%_mean']:5.1f}%")


if __name__ == "__main__":
    main()
