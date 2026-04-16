from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.anomaly_inspector import EnhancedAnomalyInspector as AnomalyInspector
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor as SymmetryFeatureExtractor
from models.thresholding import EVTCalibrator as Thresholding
from utils.image_loader import MVTecStyleDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec"
OUTPUT_DIR = PROJECT_ROOT

BACKBONE = "resnet18"
FEATURE_DIM = 384
PQ_BITS = 8
DEFAULT_N_SHOT = 10
SEEDS = [111, 333, 999, 2026, 3407]
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


def sample_seeded_support_samples(samples: list[tuple[str, int]], n_shot: int, seed: int) -> list[tuple[str, int]]:
    if n_shot > len(samples):
        raise ValueError(f"Requested n_shot={n_shot}, but only {len(samples)} train/good samples are available.")
    sorted_samples = sorted(samples, key=lambda item: item[0])
    selected_idx = np.random.RandomState(seed).choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in selected_idx]


def deterministic_transform(image_size: int = 256):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_ground_truth_mask(category_path: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)

    defect_type = image_path.parent.name
    mask_name = f"{image_path.stem}_mask.png"
    mask_path = category_path / "ground_truth" / defect_type / mask_name

    if not mask_path.exists():
        return None

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((output_size[1], output_size[0]), Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def exact_nearest_distances(query_features: np.ndarray, reference_features: np.ndarray) -> np.ndarray:
    query_features = normalize_rows(query_features)
    reference_features = normalize_rows(reference_features)
    query_sq = np.sum(query_features * query_features, axis=1, keepdims=True)
    reference_sq = np.sum(reference_features * reference_features, axis=1)
    distances_sq = np.maximum(query_sq + reference_sq[None, :] - 2.0 * query_features @ reference_features.T, 0.0)
    return np.sqrt(np.min(distances_sq, axis=1, keepdims=True)).astype(np.float32)


def cache_category_features(category: str, device: str):
    feature_extractor = SymmetryFeatureExtractor(backbone=BACKBONE, device=device)
    transform = deterministic_transform()

    train_dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=True,
        img_size=256,
    )
    train_dataset.transform = transform

    test_dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=False,
        img_size=256,
    )
    test_dataset.transform = transform

    train_samples = sorted(train_dataset.samples, key=lambda item: item[0])
    support_union = set()
    for seed in SEEDS:
        support_union.update(path for path, _ in sample_seeded_support_samples(train_samples, DEFAULT_N_SHOT, seed))

    support_samples = [sample for sample in train_samples if sample[0] in support_union]
    train_dataset.samples = support_samples

    train_feature_map: dict[str, np.ndarray] = {}
    train_loader = DataLoader(train_dataset, batch_size=min(64, len(train_dataset)), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=min(64, len(test_dataset)), shuffle=False, num_workers=0)

    for batch_images, _, batch_paths in tqdm(train_loader, desc=f"Train cache: {category}", leave=False):
        batch_images = batch_images.to(device)
        features = feature_extractor.extract_patch_features(batch_images, apply_p4m=False)
        patches_per_image = features.shape[0] // batch_images.shape[0]

        for idx, image_path in enumerate(batch_paths):
            start = idx * patches_per_image
            end = start + patches_per_image
            train_feature_map[image_path] = features[start:end]

    test_cache = []
    for batch_images, batch_labels, batch_paths in tqdm(test_loader, desc=f"Test cache: {category}", leave=False):
        batch_images = batch_images.to(device)
        features = feature_extractor.extract_patch_features(batch_images, apply_p4m=False)
        patches_per_image = features.shape[0] // batch_images.shape[0]

        for idx, image_path in enumerate(batch_paths):
            start = idx * patches_per_image
            end = start + patches_per_image
            test_cache.append(
                {
                    "path": image_path,
                    "label": int(batch_labels[idx]),
                    "features": features[start:end],
                }
            )

    return train_samples, train_feature_map, test_cache


def evaluate_category_for_seed(
    category: str,
    seed: int,
    train_samples: list[tuple[str, int]],
    train_feature_map: dict[str, np.ndarray],
    test_cache: list[dict],
    device: str,
):
    category_path = MVTEC_ROOT / category
    thresholding = Thresholding(tail_fraction=0.10, target_fpr=0.01)

    selected_samples = sample_seeded_support_samples(train_samples, n_shot=DEFAULT_N_SHOT, seed=seed)
    support_features = [train_feature_map[path] for path, _ in selected_samples]

    support_patches = normalize_rows(np.vstack(support_features))
    coreset_size = max(1, min(64, int(len(support_patches) * 0.1)))
    if coreset_size < len(support_patches):
        coreset_indices = np.linspace(0, len(support_patches) - 1, num=coreset_size, dtype=int)
        reference_features = support_patches[coreset_indices]
    else:
        reference_features = support_patches

    distances = exact_nearest_distances(support_patches, reference_features)
    image_threshold = float(thresholding.fit(distances.flatten()))
    pixel_threshold = image_threshold * 0.9

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    retrieval_latencies: list[float] = []

    for sample in test_cache:
        query_features = sample["features"]
        start_time = time.perf_counter()
        query_distances = exact_nearest_distances(query_features, reference_features)
        retrieval_latencies.append((time.perf_counter() - start_time) * 1000.0)

        patch_count = query_distances.shape[0]
        grid_size = int(np.sqrt(patch_count))
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Cannot reshape patch distances into a square grid: {patch_count}")

        patch_grid = query_distances.reshape(grid_size, grid_size)
        image_score = float(np.max(patch_grid))
        anomaly_map = cv2.resize(patch_grid, (256, 256), interpolation=cv2.INTER_CUBIC)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        image_scores.append(image_score)
        image_labels.append(sample["label"])

        gt_mask = load_ground_truth_mask(category_path, Path(sample["path"]), anomaly_map.shape)
        if gt_mask is not None:
            pixel_scores.append(anomaly_map.flatten())
            pixel_labels.append(gt_mask.flatten())

    if len(set(image_labels)) < 2:
        raise ValueError(f"Image-level AUROC requires both nominal and anomalous test samples for {category}.")
    if not pixel_scores or len(np.unique(np.concatenate(pixel_labels))) < 2:
        raise ValueError(f"Pixel-level AUROC requires valid ground-truth masks for both classes in {category}.")

    return {
        "category": category,
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))),
        "retrieval_latency_ms": float(np.mean(retrieval_latencies)),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cached 8-bit seeded evaluation | device={device}")

    category_cache = {}
    for category in CATEGORIES:
        print(f"Precomputing features for {category}...")
        category_cache[category] = cache_category_features(category, device)

    seed_to_rows: dict[int, list[dict]] = {}
    all_seed_tables = []

    for seed in SEEDS:
        print(f"\nEvaluating seed {seed}...")
        rows: list[dict] = []
        for category in tqdm(CATEGORIES, desc=f"Seed {seed}", leave=False):
            train_samples, train_feature_map, test_cache = category_cache[category]
            row = evaluate_category_for_seed(
                category=category,
                seed=seed,
                train_samples=train_samples,
                train_feature_map=train_feature_map,
                test_cache=test_cache,
                device=device,
            )
            rows.append(row)
            print(
                f"  {category}: image_auroc={row['image_auroc']:.4f}, "
                f"pixel_auroc={row['pixel_auroc']:.4f}, latency={row['retrieval_latency_ms']:.2f} ms"
            )

        seed_df = pd.DataFrame(rows)
        seed_csv = OUTPUT_DIR / f"results_8bit_seed{seed}.csv"
        seed_df.to_csv(seed_csv, index=False)
        seed_to_rows[seed] = rows
        all_seed_tables.append(seed_df)
        print(f"Saved {seed_csv}")

    summary_rows = []
    combined = pd.concat(
        [df.assign(seed=seed) for seed, df in zip(SEEDS, all_seed_tables)],
        ignore_index=True,
    )

    for category, group in combined.groupby("category", sort=True):
        summary_rows.append(
            {
                "category": category,
                "image_mean": group["image_auroc"].mean(),
                "image_sd": group["image_auroc"].std(ddof=1),
                "pixel_mean": group["pixel_auroc"].mean(),
                "pixel_sd": group["pixel_auroc"].std(ddof=1),
                "latency_mean": group["retrieval_latency_ms"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["image_fmt"] = summary_df.apply(lambda row: f"{row['image_mean']:.4f} ± {row['image_sd']:.4f}", axis=1)
    summary_df["pixel_fmt"] = summary_df.apply(lambda row: f"{row['pixel_mean']:.4f} ± {row['pixel_sd']:.4f}", axis=1)

    print("\n| Category | Image AUROC (Mean ± SD) | Pixel AUROC (Mean ± SD) | Retrieval Latency (ms) |")
    print("|---|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        print(
            f"| {row['category']} | {row['image_fmt']} | {row['pixel_fmt']} | {row['latency_mean']:.2f} |"
        )


if __name__ == "__main__":
    main()