"""
wacv_exp_b_2x2_ablation.py
===========================
WACV Experiment B — Symmetry vs Retrieval 2×2 Factorial Ablation.

Reviewer concern:
    "It is unclear whether gains come from symmetry augmentation
     or FAISS approximation."

This script evaluates all four conditions of a controlled 2×2 ablation:

    Condition   | p4m Symmetry | Retrieval Method
    ──────────────────────────────────────────────
    (a) Baseline |     ❌       | Exact k-NN
    (b) Symmetry |     ✅       | Exact k-NN
    (c) Proposed |     ✅       | FAISS IVF-PQ (8-bit)
    (d) PQ-only  |     ❌       | FAISS IVF-PQ (8-bit)

All conditions use:
    - Wide ResNet-50 backbone (1536-dim features)
    - N=10 support images, seed=111
    - MVTec AD 15 categories
    - Identical preprocessing pipeline

Output:
    results/wacv/exp_b_2x2_ablation.csv

Run from the project root:
    python src/wacv_exp_b_2x2_ablation.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import faiss
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from models.memory_bank import MemoryBank
from models.thresholding import EVTCalibrator
from utils.image_loader import MVTecStyleDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "wacv" / "exp_b_2x2_ablation.csv"
BACKBONE = "wide_resnet50_2"
FEATURE_DIM = 1536
N_SHOT = 10
PQ_BITS = 8
SUPPORT_SEED = 111
IMG_SIZE = 256

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

FIELDNAMES = [
    "condition", "condition_label", "p4m", "retrieval",
    "category", "image_auroc", "pixel_auroc", "retrieval_latency_ms",
]


@dataclass
class AblationCondition:
    name: str
    label: str
    apply_p4m: bool
    use_pq: bool  # True = FAISS IVF-PQ, False = exact k-NN


CONDITIONS = [
    AblationCondition(name="a", label="No-p4m + Exact", apply_p4m=False, use_pq=False),
    AblationCondition(name="b", label="p4m + Exact",    apply_p4m=True,  use_pq=False),
    AblationCondition(name="c", label="p4m + FAISS-PQ", apply_p4m=True,  use_pq=True),
    AblationCondition(name="d", label="No-p4m + FAISS-PQ", apply_p4m=False, use_pq=True),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    faiss.omp_set_num_threads(1)


def sample_seeded_support(
    samples: list[tuple[str, int]], n_shot: int, seed: int
) -> list[tuple[str, int]]:
    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in idx]


def build_loaders(category: str):
    train_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    train_ds.samples = sample_seeded_support(train_ds.samples, N_SHOT, SUPPORT_SEED)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    test_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    return train_loader, test_loader


def enable_exact_search(memory_bank: MemoryBank) -> None:
    """Replace IVF-PQ index with a flat exact-search index."""
    if memory_bank.features is None:
        raise ValueError("Memory bank features are not available for exact search.")
    exact_index = faiss.IndexFlatL2(memory_bank.dimension)
    exact_index.add(memory_bank.features)
    memory_bank.index = exact_index
    memory_bank.is_trained = True


def load_ground_truth_mask(category_path, image_path, output_size):
    from PIL import Image as PILImage
    image_path = Path(image_path)
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)
    defect_type = image_path.parent.name
    mask_name = f"{image_path.stem}_mask.png"
    mask_path = category_path / "ground_truth" / defect_type / mask_name
    if not mask_path.exists():
        return None
    mask = PILImage.open(mask_path).convert("L")
    mask = mask.resize((output_size[1], output_size[0]), PILImage.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def evaluate_condition(
    condition: AblationCondition,
    category: str,
    feature_extractor: SymmetryAwareFeatureExtractor,
    device: str,
) -> dict:
    """Evaluate a single condition on a single category."""
    category_path = MVTEC_ROOT / category

    # Build memory bank with appropriate settings
    use_pq = condition.use_pq
    memory_bank = MemoryBank(
        dimension=FEATURE_DIM, use_gpu=(device == "cuda"), use_pq=use_pq
    )

    train_loader, test_loader = build_loaders(category)

    # Extract support features with/without p4m
    support_features = []
    for images, _, _ in train_loader:
        images = images.to(device)
        feats = feature_extractor.extract_patch_features(
            images, apply_p4m=condition.apply_p4m
        )
        support_features.append(feats)

    # Build the bank
    build_kwargs = {"coreset_percentage": 0.1}
    if use_pq:
        build_kwargs["pq_bits"] = PQ_BITS
    memory_bank.build(support_features, **build_kwargs)

    # If exact search was requested but bank was built with PQ for coreset/orbit reduction,
    # replace with flat index
    if not use_pq:
        enable_exact_search(memory_bank)

    # EVT threshold
    calibrator = EVTCalibrator(tail_fraction=0.10, target_fpr=0.01)
    cal_feats = np.vstack(support_features).astype(np.float32)
    cal_dists, _ = memory_bank.query(cal_feats, k=1)
    image_threshold = calibrator.fit(cal_dists.flatten())
    pixel_threshold = image_threshold * 0.9

    # Inference
    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    retrieval_latencies = []

    for images, labels, paths in test_loader:
        images = images.to(device)
        B, _, H, W = images.shape

        with torch.no_grad():
            patch_feats = feature_extractor.extract_patch_features(images, apply_p4m=False)
            t0 = time.perf_counter()
            distances, _ = memory_bank.query(patch_feats, k=1)
            t1 = time.perf_counter()
            retrieval_latencies.append((t1 - t0) * 1000.0)

        patch_grid = distances.reshape(B, 28, 28)
        for idx in range(B):
            img_score = float(np.max(patch_grid[idx]))
            image_scores.append(img_score)
            image_labels.append(int(labels[idx].item()))

            amap = cv2.resize(patch_grid[idx], (W, H), interpolation=cv2.INTER_CUBIC)
            amap = gaussian_filter(amap, sigma=4)

            img_path = paths[idx] if isinstance(paths, (list, tuple)) else paths
            gt_mask = load_ground_truth_mask(category_path, img_path, amap.shape)
            if gt_mask is not None:
                pixel_scores.append(amap.flatten())
                pixel_labels.append(gt_mask.flatten())

    # Metrics
    if len(set(image_labels)) < 2:
        raise ValueError(f"[{category}] Need both classes for AUROC")

    img_auroc = float(roc_auc_score(image_labels, image_scores))
    pix_auroc = float("nan")
    if pixel_scores and len(np.unique(np.concatenate(pixel_labels))) >= 2:
        pix_auroc = float(roc_auc_score(
            np.concatenate(pixel_labels), np.concatenate(pixel_scores)
        ))

    return {
        "condition": condition.name,
        "condition_label": condition.label,
        "p4m": condition.apply_p4m,
        "retrieval": "FAISS-PQ" if condition.use_pq else "Exact",
        "category": category,
        "image_auroc": round(img_auroc, 6),
        "pixel_auroc": round(pix_auroc, 6),
        "retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 4),
    }


# ── CLI + main ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WACV EXP-B: 2×2 Symmetry vs Retrieval ablation."
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["a", "b", "c", "d"],
        help="Which conditions to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())

    # Single backbone shared across all conditions
    log.info("Loading frozen %s backbone...", BACKBONE)
    feature_extractor = SymmetryAwareFeatureExtractor(backbone=BACKBONE, device=device)

    selected_conditions = [c for c in CONDITIONS if c.name in args.conditions]
    all_results = []

    for cond in selected_conditions:
        log.info("=" * 70)
        log.info("CONDITION (%s): %s", cond.name, cond.label)
        log.info("  p4m=%s | retrieval=%s", cond.apply_p4m,
                 "FAISS-PQ" if cond.use_pq else "Exact")
        log.info("=" * 70)

        for category in CATEGORIES:
            log.info("[%s] Evaluating: %s", cond.name, category)
            try:
                result = evaluate_condition(cond, category, feature_extractor, device)
                all_results.append(result)
                log.info(
                    "[%s] %s → I-AUROC=%.4f  P-AUROC=%.4f  Lat=%.2fms",
                    cond.name, category,
                    result["image_auroc"], result["pixel_auroc"],
                    result["retrieval_latency_ms"],
                )
            except Exception as exc:
                log.error("[%s] %s FAILED: %s", cond.name, category, exc)

    # Save
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Saved → %s", args.output_csv)

    # Console summary table
    print()
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║            EXP-B: 2×2 Ablation Summary (Mean AUROC)                 ║")
    print("╠═══════╦══════════════════════╦══════════════╦═══════════════╦════════╣")
    print(f"║ {'Cond':>5} ║ {'Label':<20} ║ {'Img AUROC':>12} ║ {'Pix AUROC':>13} ║ {'Lat':>6} ║")
    print("╠═══════╬══════════════════════╬══════════════╬═══════════════╬════════╣")
    for cond in selected_conditions:
        cond_results = [r for r in all_results if r["condition"] == cond.name]
        if not cond_results:
            continue
        mean_img = np.mean([r["image_auroc"] for r in cond_results])
        pix_vals = [r["pixel_auroc"] for r in cond_results if not np.isnan(r["pixel_auroc"])]
        mean_pix = np.mean(pix_vals) if pix_vals else float("nan")
        mean_lat = np.mean([r["retrieval_latency_ms"] for r in cond_results])
        print(
            f"║ ({cond.name:>3}) ║ {cond.label:<20} ║ {mean_img:>12.4f} ║ "
            f"{mean_pix:>13.4f} ║ {mean_lat:>5.1f}ms ║"
        )
    print("╚═══════╩══════════════════════╩══════════════╩═══════════════╩════════╝")
    print("\nEXP-B COMPLETE.")


if __name__ == "__main__":
    main()
