"""
wacv_fix3_visa_expansion.py
===========================
WACV FIX-3 — VisA Evaluation Expansion.

Reviewer concern:
    "Cross-domain results are underpowered and inconclusive."

For N ∈ {1, 3, 5, 10} × 5 seeds × 12 VisA categories:
    Evaluate Image AUROC and Pixel AUROC (PRO-AUC where masks available)
    using the full INT8 pipeline (p4m + 8-bit FAISS IVF-PQ + EVT).

Output:
    results/wacv/fix3_visa_nshot.csv

Run from the project root:
    python src/wacv_fix3_visa_expansion.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
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
from PIL import Image as PILImage

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
# VisA in MVTec-style directory format
VISA_ROOT = PROJECT_ROOT / "data" / "visa_mvtec_format"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "wacv" / "fix3_visa_nshot.csv"
BACKBONE = "wide_resnet50_2"
FEATURE_DIM = 1536
PQ_BITS = 8
IMG_SIZE = 256

NSHOTS = [1, 3, 5, 10]
SEEDS = [111, 333, 999, 2026, 3407]

VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2", "pcb1",
    "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

FIELDNAMES = [
    "n_shot", "seed", "category",
    "image_auroc", "pixel_auroc",
    "retrieval_latency_ms",
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
    if n_shot > len(samples):
        raise ValueError(
            f"Requested n_shot={n_shot}, but only {len(samples)} "
            "train/good samples are available."
        )
    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in idx]


def load_ground_truth_mask(category_path, image_path, output_size):
    image_path = Path(image_path)
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)
    defect_type = image_path.parent.name
    mask_name = f"{image_path.stem}_mask.png"
    mask_path = category_path / "ground_truth" / defect_type / mask_name
    if not mask_path.exists():
        # Try alternative naming conventions
        mask_name_alt = f"{image_path.stem}.png"
        mask_path = category_path / "ground_truth" / defect_type / mask_name_alt
        if not mask_path.exists():
            return None
    mask = PILImage.open(mask_path).convert("L")
    mask = mask.resize((output_size[1], output_size[0]), PILImage.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def evaluate_visa_category(
    category: str,
    n_shot: int,
    seed: int,
    feature_extractor: SymmetryAwareFeatureExtractor,
    device: str,
) -> dict:
    """Evaluate one VisA category at one n_shot and seed."""
    category_path = VISA_ROOT / category

    # Build support loader
    train_ds = MVTecStyleDataset(
        root_dir=str(VISA_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    train_ds.samples = sample_seeded_support(train_ds.samples, n_shot, seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    # Extract support features (p4m ON)
    support_features = []
    for images, _, _ in train_loader:
        images = images.to(device)
        feats = feature_extractor.extract_patch_features(images, apply_p4m=True)
        support_features.append(feats)

    # Build memory bank
    memory_bank = MemoryBank(
        dimension=FEATURE_DIM, use_gpu=(device == "cuda"), use_pq=True
    )
    memory_bank.build(support_features, coreset_percentage=0.1, pq_bits=PQ_BITS)

    # EVT threshold
    calibrator = EVTCalibrator(tail_fraction=0.10, target_fpr=0.01)
    cal_feats = np.vstack(support_features).astype(np.float32)
    cal_dists, _ = memory_bank.query(cal_feats, k=1)
    image_threshold = calibrator.fit(cal_dists.flatten())
    pixel_threshold = image_threshold * 0.9

    # Build test loader
    test_ds = MVTecStyleDataset(
        root_dir=str(VISA_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

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
    img_auroc = float("nan")
    if len(set(image_labels)) >= 2:
        img_auroc = float(roc_auc_score(image_labels, image_scores))

    pix_auroc = float("nan")
    if pixel_scores and len(np.unique(np.concatenate(pixel_labels))) >= 2:
        pix_auroc = float(roc_auc_score(
            np.concatenate(pixel_labels), np.concatenate(pixel_scores)
        ))

    return {
        "n_shot": n_shot,
        "seed": seed,
        "category": category,
        "image_auroc": round(img_auroc, 6),
        "pixel_auroc": round(pix_auroc, 6),
        "retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 4),
    }


# ── CLI + main ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WACV FIX-3: VisA evaluation expansion."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--nshots", type=int, nargs="+", default=NSHOTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    if not VISA_ROOT.is_dir():
        log.error(
            "VisA dataset not found at: %s\n"
            "Please extract the VisA dataset in MVTec-style format to this path.",
            VISA_ROOT,
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    log.info("Loading frozen %s backbone...", BACKBONE)
    feature_extractor = SymmetryAwareFeatureExtractor(backbone=BACKBONE, device=device)

    total = len(args.nshots) * len(args.seeds) * len(VISA_CATEGORIES)
    log.info(
        "Running %d evaluations: %d n-shots × %d seeds × %d categories",
        total, len(args.nshots), len(args.seeds), len(VISA_CATEGORIES),
    )

    all_results = []
    counter = 0

    for n_shot in args.nshots:
        for seed in args.seeds:
            for category in VISA_CATEGORIES:
                counter += 1
                log.info("[%d/%d] N=%d seed=%d cat=%s", counter, total, n_shot, seed, category)
                try:
                    result = evaluate_visa_category(
                        category, n_shot, seed, feature_extractor, device
                    )
                    all_results.append(result)
                    log.info(
                        "  → I-AUROC=%.4f  P-AUROC=%.4f  Lat=%.2fms",
                        result["image_auroc"], result["pixel_auroc"],
                        result["retrieval_latency_ms"],
                    )
                except Exception as exc:
                    log.error("  FAILED: %s", exc)

    # Save CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Saved → %s", args.output_csv)

    # Console summary
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║    FIX-3: VisA N-Shot Sensitivity (Mean ± Std, 5 seeds)  ║")
    print("╠════════╦═══════════════════════╦═════════════════════════╣")
    print(f"║ {'N':>6} ║ {'Image AUROC':>21} ║ {'Pixel AUROC':>23} ║")
    print("╠════════╬═══════════════════════╬═════════════════════════╣")
    for ns in args.nshots:
        ns_results = [r for r in all_results if r["n_shot"] == ns]
        img_vals = [r["image_auroc"] for r in ns_results if not np.isnan(r["image_auroc"])]
        pix_vals = [r["pixel_auroc"] for r in ns_results if not np.isnan(r["pixel_auroc"])]
        img_str = f"{np.mean(img_vals):.4f} ± {np.std(img_vals):.4f}" if img_vals else "N/A"
        pix_str = f"{np.mean(pix_vals):.4f} ± {np.std(pix_vals):.4f}" if pix_vals else "N/A"
        print(f"║ {ns:>6} ║ {img_str:>21} ║ {pix_str:>23} ║")
    print("╚════════╩═══════════════════════╩═════════════════════════╝")
    print("\nFIX-3 COMPLETE.")


if __name__ == "__main__":
    main()
