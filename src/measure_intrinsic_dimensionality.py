"""Measure intrinsic dimensionality of cached deep embeddings via PCA.

Given an HDF5 embedding cache, this script:
1) loads all embeddings,
2) flattens to [total_features, feature_dim],
3) standardizes features,
4) runs PCA,
5) reports the number of principal components required for 90% and 95%
   explained variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_embeddings_2d(h5_path: str | Path) -> np.ndarray:
    """Load per-key HDF5 embeddings and return a 2D matrix [N_total, D]."""

    cache_path = Path(h5_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {cache_path}")

    flattened_chunks: list[np.ndarray] = []
    feature_dim: int | None = None

    with h5py.File(cache_path, "r") as handle:
        if len(handle.keys()) == 0:
            raise ValueError("HDF5 file has no datasets to analyze")

        for key in sorted(handle.keys()):
            node = handle[key]
            if not isinstance(node, h5py.Dataset):
                continue

            if not np.issubdtype(node.dtype, np.number):
                continue

            array = np.asarray(node, dtype=np.float32)
            if array.ndim < 2:
                continue

            current_dim = int(array.shape[-1])
            if feature_dim is None:
                feature_dim = current_dim
            elif current_dim != feature_dim:
                raise ValueError(
                    f"Inconsistent feature dimension for key '{key}': expected {feature_dim}, got {current_dim}"
                )

            flattened_chunks.append(array.reshape(-1, current_dim))

    if not flattened_chunks:
        raise ValueError("No numeric embedding datasets with ndim >= 2 were found in the HDF5 file")

    flattened = np.vstack(flattened_chunks)
    if flattened.shape[0] < 2:
        raise ValueError("Need at least 2 feature vectors to run PCA")
    return flattened


def standardize_features(features_2d: np.ndarray) -> np.ndarray:
    """Standardize feature matrix to zero mean and unit variance."""

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    return scaler.fit_transform(features_2d)


def fit_pca(features_2d: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Fit PCA and return fitted model plus cumulative explained variance."""

    pca = PCA(svd_solver="full", random_state=42)
    pca.fit(features_2d)
    cumulative_explained = np.cumsum(pca.explained_variance_ratio_)
    return pca, cumulative_explained


def components_for_variance(cumulative_explained: np.ndarray, target: float) -> int:
    """Return the minimum 1-based component count reaching target variance."""

    if not (0.0 < target <= 1.0):
        raise ValueError(f"target must be in (0, 1], got {target}")
    return int(np.searchsorted(cumulative_explained, target, side="left") + 1)


def analyze_intrinsic_dimensionality(h5_path: str | Path) -> dict[str, float | int]:
    """Run the full intrinsic dimensionality analysis pipeline."""

    features = load_embeddings_2d(h5_path)
    standardized = standardize_features(features)
    pca, cumulative_explained = fit_pca(standardized)

    pcs_90 = components_for_variance(cumulative_explained, 0.90)
    pcs_95 = components_for_variance(cumulative_explained, 0.95)

    return {
        "total_features": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "total_components": int(pca.n_components_),
        "components_for_90pct": int(pcs_90),
        "components_for_95pct": int(pcs_95),
        "variance_at_90_components": float(cumulative_explained[pcs_90 - 1]),
        "variance_at_95_components": float(cumulative_explained[pcs_95 - 1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure embedding intrinsic dimensionality using PCA.")
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Path to HDF5 cache file (for example cache_mvtec_wresnet50_screw.h5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_intrinsic_dimensionality(args.h5_path)

    print(f"HDF5 file: {Path(args.h5_path)}")
    print(f"Total feature vectors: {result['total_features']}")
    print(f"Feature dimension: {result['feature_dim']}")
    print(f"Total PCA components: {result['total_components']}")
    print("Intrinsic Dimensionality Results")
    print(f"Components for 90% variance: {result['components_for_90pct']}")
    print(f"Components for 95% variance: {result['components_for_95pct']}")
    print(f"Cumulative variance at 90% point: {result['variance_at_90_components']:.6f}")
    print(f"Cumulative variance at 95% point: {result['variance_at_95_components']:.6f}")


if __name__ == "__main__":
    main()