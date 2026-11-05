"""Table 2: CIFAR-100 (two-level label hierarchy) with raw pixels and DINOv2 features.

Ten test images per fine class are drawn without replacement (seed 20260801,
k = 1000). Geometries:
  * raw RGB Euclidean: pixels scaled to [0, 1] and flattened (3072 dims),
  * DINOv2 cosine:     cosine distance between l2-normalised DINOv2 features,
  * DINOv2 Euclidean:  Euclidean distance between unnormalised features.
DINOv2 ViT-S/14 features are the output of the model's forward pass (final
class token after layer norm) on images resized to 224x224 (bicubic,
antialiased) with ImageNet mean/std normalisation.

For each geometry we report triplet satisfaction of the fine and coarse
classes, the number of valid fine/coarse classes, and the retained and shared
numbers (and %) of the k-2 non-root internal clusters.

Requires torch and torchvision (only for feature extraction). Features are
cached in an .npz file (keys `features`, `indices`); an existing cache from the
original experiment script can be passed with --features.

Usage:  python cifar100.py --data-root data [--download] [--features cache.npz]
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from apresjan import (
    apresjan_hierarchy,
    class_clusters,
    is_valid,
    shared_clusters,
    triplet_satisfaction,
)

SEED = 20260801
PER_CLASS = 10
MODEL = "dinov2_vits14"


def load_cifar100_test(root: Path, download: bool = False):
    """Test images (N, 32, 32, 3) uint8 and fine/coarse labels, from the official archive."""
    archive = root / "cifar-100-python" / "test"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"{archive} not found; use --download")
        from torchvision.datasets import CIFAR100

        CIFAR100(root=root, train=False, download=True)
    with archive.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.array(batch["fine_labels"]), np.array(batch["coarse_labels"])


def balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """`per_class` indices per label drawn without replacement, sorted."""
    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
        for c in np.unique(labels)
    ]
    return np.sort(np.concatenate(chosen))


def dinov2_features(images: np.ndarray, model_name: str, repo: str | None,
                    batch_size: int = 32) -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision.transforms import (
        Compose, InterpolationMode, Normalize, Resize, ToTensor,
    )

    transform = Compose([
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    if repo is None:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    else:  # local clone of the DINOv2 repository
        model = torch.hub.load(repo, model_name, source="local")
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model.eval().to(device)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([transform(Image.fromarray(image))
                                 for image in images[start:start + batch_size]])
            batches.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(batches)


def load_or_compute_features(path: Path, images, indices, model_name, repo) -> np.ndarray:
    if path.exists():
        with np.load(path) as cache:
            if not np.array_equal(cache["indices"], indices):
                raise ValueError(f"{path} was computed for a different sample")
            return cache["features"]
    features = dinov2_features(images[indices], model_name, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, indices=indices, model=model_name)
    return features


def geometries(raw_pixels: np.ndarray, features: np.ndarray) -> dict:
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return {
        "Raw RGB Euclidean": squareform(pdist(raw_pixels, "euclidean")),
        "DINOv2 cosine": squareform(pdist(unit, "cosine")),
        "DINOv2 Euclidean": squareform(pdist(features, "euclidean")),
    }


def evaluate(d: np.ndarray, fine: np.ndarray, coarse: np.ndarray) -> dict:
    k = len(d)
    valid = apresjan_hierarchy(d)
    shared = shared_clusters(d)
    row = {}
    for level, labels in (("fine", fine), ("coarse", coarse)):
        satisfied, total = triplet_satisfaction(d, labels)
        classes = class_clusters(labels)
        row[f"{level}_triplets_%"] = f"{100 * satisfied / total:.2f}"
        row[f"{level}_valid_classes"] = f"{sum(is_valid(d, c) for c in classes)}/{len(classes)}"
    row.update({
        "retained": len(valid),
        "retained_%": f"{100 * len(valid) / (k - 2):.1f}",
        "shared": len(shared),
        "shared_%": f"{100 * len(shared) / (k - 2):.1f}",
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--features", type=Path,
                        default=Path("results/cifar100_dinov2_vits14_test_10_per_fine.npz"))
    parser.add_argument("--dinov2-repo", help="local DINOv2 clone (default: download via torch.hub)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--output", type=Path, default=Path("results/table2_cifar100.tsv"))
    args = parser.parse_args()

    images, fine_all, coarse_all = load_cifar100_test(args.data_root, args.download)
    indices = balanced_indices(fine_all, args.per_class, args.seed)
    fine, coarse = fine_all[indices], coarse_all[indices]
    features = load_or_compute_features(args.features, images, indices,
                                        args.model, args.dinov2_repo)
    raw_pixels = images[indices].reshape(len(indices), -1) / 255.0

    rows = [{"geometry": name, **evaluate(d, fine, coarse)}
            for name, d in geometries(raw_pixels, features).items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        for stream in (handle, sys.stdout):
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
