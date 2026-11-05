# Recovering the Apresjan Hierarchy Using Linkage-Based Clustering

Code for the numerical results of the paper

> **Recovering the Apresjan Hierarchy Using Linkage-Based Clustering**<br>
> Maximilien Dreveton\*, Daichi Kuroda\*, Matthias Grossglauser, Patrick Thiran<br>
> *Advances in Neural Information Processing Systems (NeurIPS), 2026*<br>
> \* Equal contribution.

The code computes the Apresjan hierarchy T*(X, d) by pruning the invalid
clusters of a linkage tree, using the O(k²) procedure described in the
appendix of the paper.

| Script | Reproduces | Runtime |
|---|---|---|
| `benchmarks.py` | Table 1 (`tab:experiments`): Iris, Wine, Breast Cancer, Digits | ~2 s |
| `cifar100.py` | Table 2 (`tab:cifar`): CIFAR-100 with raw pixels and DINOv2 | minutes (feature extraction) |
| `synthetic.py` | Figure `fig:synthetic` (right panel) and the planted-hierarchy numbers in the appendix | ~2 s |
| `examples.py` | Example `ex:running`, Figure `fig:ward_fails` (Ward), Figure `fig:dasgupta` costs | <1 s |
| `apresjan.py` | Core library: validity margins, trimmed linkage, triplet satisfaction | — |

## Setup

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, matplotlib
pip install torch torchvision pillow     # only for cifar100.py
```

## Reproduce

Run from this folder:

```bash
python benchmarks.py     # -> results/table1_benchmarks.tsv
python synthetic.py      # -> results/synthetic_summary.tsv, results/synthetic_hierarchy_recall.{pdf,png}
python examples.py       # prints the checks of the worked examples
python cifar100.py --data-root data --download   # -> results/table2_cifar100.tsv
```

`cifar100.py` caches the DINOv2 features in
`results/cifar100_dinov2_vits14_test_10_per_fine.npz` (keys `features`,
`indices`). By default DINOv2 is loaded with `torch.hub` (official weights);
use `--dinov2-repo PATH` for a local clone. Features computed on a different
device (CPU / CUDA / MPS) can differ in the last digits, so to reproduce
Table 2 exactly, pass the feature cache from the original run with
`--features`.

## Protocol details

- **Benchmarks.** Data from scikit-learn, features not rescaled, exact duplicate rows removed (one point of Iris, so k = 149). Cosine dissimilarity is 1 − cos(x, y).
- **Linkage.** `scipy.cluster.hierarchy.linkage`. The pruned hierarchy does not depend on tie-breaking for single, complete and average linkage; the *shared* statistic does.
- **Retained / shared.** Percentages of the k − 2 non-root internal clusters of the binary tree.
- **Triplet satisfaction.** Percentage of triplets (x, y, z), x ≠ y in the same class and z in another class, with d(x, y) < d(x, z); ties count as violations.
- **Planted hierarchy.** ρ ∈ {0, 0.1, …, 1}, 50 draws per level, seed 20260801. Each draw U⁽ʳ⁾ is reused across the ρ grid (only the amplitude 4ρ changes). Perturbed entries are floored at machine epsilon.
- **CIFAR-100.** Ten test images per fine class drawn without replacement with seed 20260801 (k = 1000).

## Environment

The results in `results/` were produced with Python 3.13, NumPy 2.5, SciPy 1.18,
scikit-learn 1.9 and matplotlib 3.11. The original results (Python 3.11.7,
Anaconda base environment) are identical.

## Citation

```bibtex
@inproceedings{dreveton2026apresjan,
  title     = {Recovering the Apresjan Hierarchy Using Linkage-Based Clustering},
  author    = {Dreveton, Maximilien and Kuroda, Daichi and Grossglauser, Matthias and Thiran, Patrick},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
