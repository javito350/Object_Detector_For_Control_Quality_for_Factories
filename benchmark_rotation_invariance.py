"""
Rotation robustness benchmark for a NeurIPS-style anomaly detection submission.

This script compares:
1. A standard PatchCore-style baseline with no p4m support augmentation.
2. The proposed 8-bit p4m-augmented retrieval pipeline.

Both pipelines are evaluated on the MVTec AD `capsule` and `cable` categories
under controlled test-time rotations of 0, 90, 180, and 270 degrees.

The resulting table is intended to provide a concise "rotation robustness"
proof: the baseline should degrade under rotation, while the proposed method
should remain close to its 0-degree performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from models.anomaly_inspector import EnhancedAnomalyInspector as AnomalyInspector
from utils.image_loader import MVTecStyleDataset


ROOT_DIR = Path(__file__).resolve().parent
MVTEC_ROOT = ROOT_DIR / "data" / "mvtec"
OUTPUT_CSV = ROOT_DIR / "rotation_invariance_results.csv"

# Controlled benchmark setup:
# - ResNet-50 family backbone used throughout the repository.
# - 1536-D patch descriptors from layer2 + layer3.
# - 8-bit product quantization (default MemoryBank setting).
BACKBONE = "wide_resnet50_2"
N_SHOT = 10
CATEGORIES = ["capsule", "cable"]
ROTATIONS = [0, 90, 180, 270]


def build_support_dataloader(category: str) -> DataLoader:
    """
    Build a deterministic 10-shot support loader.

    We explicitly remove the training-time random flip/rotation transform from
    the project dataset so the "Baseline = no augmentation" condition is truly
    controlled.
    """
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=True,
        img_size=256,
    )
    dataset.samples = dataset.samples[:N_SHOT]
    dataset.transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def build_test_dataloader(category: str) -> DataLoader:
    """Build the standard deterministic MVTec test loader."""
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=False,
        img_size=256,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def rotate_images(images: torch.Tensor, rotation_degrees: int) -> torch.Tensor:
    """Rotate a batch by multiples of 90 degrees without interpolation artifacts."""
    k = (rotation_degrees // 90) % 4
    if k == 0:
        return images
    return torch.rot90(images, k=k, dims=(-2, -1))


def build_pipeline(category: str, apply_p4m_support: bool, device: str) -> AnomalyInspector:
    """
    Fit either the baseline or the proposed pipeline on the 10-shot support set.

    - Baseline: `apply_p4m_support=False`
    - Proposed: `apply_p4m_support=True`
    """
    inspector = AnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)
    support_loader = build_support_dataloader(category)
    inspector.fit(support_loader, apply_p4m=apply_p4m_support)
    return inspector


def evaluate_rotation(
    inspector: AnomalyInspector,
    category: str,
    rotation_degrees: int,
) -> float:
    """
    Evaluate image-level AUROC for a single category and rotation.

    We rotate only the test image and keep inference unaugmented. This directly
    measures retrieval robustness to pose changes.
    """
    test_loader = build_test_dataloader(category)
    image_scores: list[float] = []
    image_labels: list[int] = []

    for images, labels, _ in tqdm(
        test_loader,
        desc=f"{category} @ {rotation_degrees} deg",
        leave=False,
    ):
        rotated_images = rotate_images(images, rotation_degrees)
        result = inspector.predict(rotated_images, apply_p4m=False)[0]
        image_scores.append(float(result.image_score))
        image_labels.append(int(labels[0]))

    if len(set(image_labels)) < 2:
        raise ValueError(f"AUROC requires both nominal and anomalous samples for {category}.")

    return float(roc_auc_score(image_labels, image_scores))


def run_category_benchmark(category: str, device: str) -> list[dict]:
    """
    Run the full rotation benchmark for one category.

    Returns one record per rotation containing baseline/proposed AUROC and
    their relative drops from the 0-degree reference.
    """
    print(f"\nPreparing pipelines for category: {category}")
    baseline_inspector = build_pipeline(category, apply_p4m_support=False, device=device)
    proposed_inspector = build_pipeline(category, apply_p4m_support=True, device=device)

    baseline_scores: dict[int, float] = {}
    proposed_scores: dict[int, float] = {}

    for rotation in ROTATIONS:
        print(f"Evaluating {category} at {rotation} degrees")
        baseline_scores[rotation] = evaluate_rotation(baseline_inspector, category, rotation)
        proposed_scores[rotation] = evaluate_rotation(proposed_inspector, category, rotation)

    baseline_ref = baseline_scores[0]
    proposed_ref = proposed_scores[0]
    rows: list[dict] = []

    for rotation in ROTATIONS:
        baseline_drop_pct = 100.0 * (baseline_ref - baseline_scores[rotation])
        proposed_drop_pct = 100.0 * (proposed_ref - proposed_scores[rotation])
        rows.append(
            {
                "category": category,
                "rotation_deg": rotation,
                "baseline_auroc": baseline_scores[rotation],
                "proposed_auroc": proposed_scores[rotation],
                "baseline_drop_pct_from_0deg": baseline_drop_pct,
                "proposed_drop_pct_from_0deg": proposed_drop_pct,
                "proposed_within_2pct_of_0deg": abs(proposed_drop_pct) <= 2.0,
            }
        )

    return rows


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("STARTING ROTATION ROBUSTNESS BENCHMARK")
    print(f"Backbone: {BACKBONE} | PQ bits: 8 | Device: {device}")

    records: list[dict] = []
    for category in CATEGORIES:
        records.extend(run_category_benchmark(category, device))

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False)

    display_df = results_df.copy()
    for column in [
        "baseline_auroc",
        "proposed_auroc",
        "baseline_drop_pct_from_0deg",
        "proposed_drop_pct_from_0deg",
    ]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    print("\nRotation Robustness Table")
    print(display_df.to_string(index=False))
    print(f"\nSaved detailed results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
