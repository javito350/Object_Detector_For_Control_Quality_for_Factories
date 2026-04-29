from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
from torchvision import transforms as T


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from utils.image_loader import MVTecStyleDataset


CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

FEATURE_DIM = 1536
DEFAULT_IMG_SIZE = 256
DEFAULT_N_SHOT = 10
DEFAULT_SUPPORT_SEED = 111


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure 95%% intrinsic dimensionality and NNDV for each MVTec AD category."
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT_DIR / "data" / "mvtec")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT_DIR / "results" / "intrinsic_dimensionality.csv",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--n-shot", type=int, default=DEFAULT_N_SHOT)
    parser.add_argument("--support-seed", type=int, default=DEFAULT_SUPPORT_SEED)
    parser.add_argument(
        "--apply-p4m-support",
        action="store_true",
        help="If set, compute metrics from the 8x augmented support orbit.",
    )
    return parser.parse_args()


def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def sample_seeded_support_samples(
    samples: list[tuple[str, int]],
    n_shot: int,
    seed: int,
) -> list[tuple[str, int]]:
    if n_shot > len(samples):
        raise ValueError(
            f"Requested n_shot={n_shot}, but only {len(samples)} train/good samples are available."
        )

    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    selected_idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in selected_idx]


def build_support_dataloader(
    dataset_root: Path,
    category: str,
    img_size: int,
    n_shot: int,
    support_seed: int,
) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(dataset_root),
        category=category,
        is_train=True,
        img_size=img_size,
    )
    dataset.samples = sample_seeded_support_samples(dataset.samples, n_shot=n_shot, seed=support_seed)
    dataset.transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def extract_support_features(
    dataloader: DataLoader,
    extractor: SymmetryAwareFeatureExtractor,
    apply_p4m_support: bool,
) -> np.ndarray:
    feature_batches: list[np.ndarray] = []

    for images, _, _ in dataloader:
        images = images.to(extractor.device)
        features = extractor.extract_patch_features(images, apply_p4m=apply_p4m_support)
        feature_batches.append(np.asarray(features, dtype=np.float32))

    if not feature_batches:
        raise ValueError("No support features were extracted.")

    stacked = np.vstack(feature_batches).astype(np.float32, copy=False)
    if stacked.ndim != 2 or stacked.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature matrix [N, {FEATURE_DIM}], got {stacked.shape}")
    return stacked


def components_for_95pct_variance(features: np.ndarray) -> int:
    if features.shape[0] < 2:
        raise ValueError("Need at least 2 feature vectors to fit PCA.")

    pca = PCA(svd_solver="full", random_state=42)
    pca.fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, 0.95, side="left") + 1)


def compute_nndv(features: np.ndarray) -> float:
    """Nearest-neighbor distance variance — proxy for anomaly signal magnitude."""
    dists = pairwise_distances(features, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    return float(np.var(nn_dists))


def append_result(output_csv: Path, category: str, intrinsic_dim: int, nndv: float) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["Category", "Intrinsic Dimensionality", "NNDV"])
        writer.writerow([category, intrinsic_dim, f"{nndv:.6f}"])


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SymmetryAwareFeatureExtractor(backbone="wide_resnet50_2", device=device)

    for category in CATEGORIES:
        print(f"[IntrinsicDim] Processing {category}...")
        support_loader = build_support_dataloader(
            dataset_root=args.dataset_root,
            category=category,
            img_size=args.img_size,
            n_shot=args.n_shot,
            support_seed=args.support_seed,
        )
        features = extract_support_features(
            dataloader=support_loader,
            extractor=extractor,
            apply_p4m_support=args.apply_p4m_support,
        )
        
        # Calculate BOTH metrics
        intrinsic_dim = components_for_95pct_variance(features)
        nndv = compute_nndv(features)
        
        append_result(args.output_csv, category, intrinsic_dim, nndv)
        print(f"[IntrinsicDim] {category}: 95% variance at {intrinsic_dim} PCs | NNDV: {nndv:.6f}")

    print(f"Saved intrinsic dimensionality and NNDV results to {args.output_csv}")


if __name__ == "__main__":
    main()