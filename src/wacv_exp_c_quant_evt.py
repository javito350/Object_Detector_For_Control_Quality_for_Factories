"""
wacv_exp_c_quant_evt.py
=======================
WACV Experiment C — Quantization–EVT Interaction Analysis.

Reviewer concern:
    "The quantization–EVT interaction is claimed but not rigorously demonstrated."

For PQ bit-depths {4, 6, 8, 10, 12} × seeds {111, 333, 999} × all 15 MVTec categories:
    1. Build memory bank (p4m=ON, pq_bits=b)
    2. Evaluate Image AUROC on test set
    3. Fit GPD (Peaks-over-Threshold) on nominal test distances
    4. Record GPD shape (xi), scale (sigma), threshold (u), AUROC

Output:
    results/wacv/exp_c_quant_evt_full.csv

Run from the project root:
    python src/wacv_exp_c_quant_evt.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from scipy.stats import genpareto
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
from utils.image_loader import MVTecStyleDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "wacv" / "exp_c_quant_evt_full.csv"
BACKBONE = "wide_resnet50_2"
FEATURE_DIM = 1536
N_SHOT = 10
IMG_SIZE = 256
TAIL_FRACTION = 0.05  # top-5% for POT

BIT_DEPTHS = [4, 6, 8, 10, 12]
SEEDS = [111, 333, 999]

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

FIELDNAMES = [
    "bit_depth", "seed", "category",
    "image_auroc",
    "gpd_shape_xi", "gpd_scale_sigma", "gpd_threshold_u",
    "gpd_n_tail", "gpd_n_nominal",
    "evt_decision_boundary",
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


def fit_gpd_pot(nominal_distances: np.ndarray, tail_fraction: float) -> dict:
    """
    Peaks-over-Threshold GPD fit on nominal distances.
    Returns GPD parameters and decision boundary for FPR=0.01.
    """
    sorted_d = np.sort(nominal_distances)
    tail_idx = int(len(sorted_d) * (1.0 - tail_fraction))
    tail_idx = min(tail_idx, len(sorted_d) - 10)
    u = float(sorted_d[tail_idx])
    exceedances = sorted_d[tail_idx:] - u

    if len(exceedances) < 5 or float(np.max(exceedances)) < 1e-12:
        return {
            "gpd_shape_xi": float("nan"),
            "gpd_scale_sigma": float("nan"),
            "gpd_threshold_u": u,
            "gpd_n_tail": len(exceedances),
            "gpd_n_nominal": len(nominal_distances),
            "evt_decision_boundary": u * 1.05,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape, _loc, scale = genpareto.fit(exceedances, floc=0)

    # Decision boundary for target FPR = 0.01
    target_fpr = 0.01
    if abs(shape) < 1e-5:
        margin = -scale * np.log(target_fpr / tail_fraction)
    else:
        margin = (scale / shape) * (((target_fpr / tail_fraction) ** -shape) - 1)

    return {
        "gpd_shape_xi": float(shape),
        "gpd_scale_sigma": float(scale),
        "gpd_threshold_u": u,
        "gpd_n_tail": len(exceedances),
        "gpd_n_nominal": len(nominal_distances),
        "evt_decision_boundary": u + margin,
    }


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def evaluate_category_at_bitdepth(
    category: str,
    bit_depth: int,
    seed: int,
    feature_extractor: SymmetryAwareFeatureExtractor,
    device: str,
) -> dict:
    """Evaluate one category at one bit depth with one seed."""
    # Build support loader
    train_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    train_ds.samples = sample_seeded_support(train_ds.samples, N_SHOT, seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    # Extract support features (p4m ON)
    support_features = []
    for images, _, _ in train_loader:
        images = images.to(device)
        feats = feature_extractor.extract_patch_features(images, apply_p4m=True)
        support_features.append(feats)

    # Build PQ memory bank at specified bit depth
    memory_bank = MemoryBank(
        dimension=FEATURE_DIM, use_gpu=(device == "cuda"), use_pq=True
    )
    memory_bank.build(support_features, coreset_percentage=0.1, pq_bits=bit_depth)

    # Build test loader
    test_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Run test inference
    image_scores = []
    image_labels = []
    nominal_patch_distances = []

    for images, labels, _ in test_loader:
        images = images.to(device)
        with torch.no_grad():
            patch_feats = feature_extractor.extract_patch_features(images, apply_p4m=False)
            distances, _ = memory_bank.query(patch_feats, k=1)

        B = images.shape[0]
        patch_grid = distances.reshape(B, 28, 28)
        for idx in range(B):
            img_score = float(np.max(patch_grid[idx]))
            image_scores.append(img_score)
            lbl = int(labels[idx].item())
            image_labels.append(lbl)

            # Collect patch distances from nominal images for GPD fit
            if lbl == 0:
                nominal_patch_distances.append(distances[idx * 784:(idx + 1) * 784].flatten())

    # Image AUROC
    if len(set(image_labels)) < 2:
        raise ValueError(f"Need both classes for AUROC in {category}")
    img_auroc = float(roc_auc_score(image_labels, image_scores))

    # GPD fit on nominal patch distances
    if nominal_patch_distances:
        all_nominal = np.concatenate(nominal_patch_distances)
        gpd_result = fit_gpd_pot(all_nominal, TAIL_FRACTION)
    else:
        gpd_result = {
            "gpd_shape_xi": float("nan"),
            "gpd_scale_sigma": float("nan"),
            "gpd_threshold_u": float("nan"),
            "gpd_n_tail": 0,
            "gpd_n_nominal": 0,
            "evt_decision_boundary": float("nan"),
        }

    return {
        "bit_depth": bit_depth,
        "seed": seed,
        "category": category,
        "image_auroc": round(img_auroc, 6),
        **{k: round(v, 8) if isinstance(v, float) else v for k, v in gpd_result.items()},
    }


# ── CLI + main ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WACV EXP-C: Quantization–EVT interaction analysis."
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--bit-depths", type=int, nargs="+", default=BIT_DEPTHS,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=CATEGORIES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    log.info("Loading frozen %s backbone...", BACKBONE)
    feature_extractor = SymmetryAwareFeatureExtractor(backbone=BACKBONE, device=device)

    total = len(args.bit_depths) * len(args.seeds) * len(args.categories)
    log.info(
        "Running %d evaluations: %d bits × %d seeds × %d categories",
        total, len(args.bit_depths), len(args.seeds), len(args.categories),
    )

    all_results = []
    completed_configs = set()
    
    # Load existing results to support resuming
    if args.output_csv.exists():
        log.info("Found existing CSV, loading completed configurations to resume...")
        with args.output_csv.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    bd = int(row["bit_depth"])
                    s = int(row["seed"])
                    c = row["category"]
                    completed_configs.add((bd, s, c))
                    all_results.append({
                        "bit_depth": bd,
                        "seed": s,
                        "category": c,
                        "image_auroc": float(row["image_auroc"]) if row.get("image_auroc") else float('nan'),
                        "gpd_shape_xi": float(row["gpd_shape_xi"]) if row.get("gpd_shape_xi") else float('nan'),
                        "gpd_scale_sigma": float(row["gpd_scale_sigma"]) if row.get("gpd_scale_sigma") else float('nan'),
                    })
                except ValueError:
                    pass
        log.info("Loaded %d previously completed configurations.", len(completed_configs))
    else:
        # Create file and write header
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()

    counter = 0

    for bit_depth in args.bit_depths:
        for seed in args.seeds:
            for category in args.categories:
                counter += 1
                
                if (bit_depth, seed, category) in completed_configs:
                    log.info(
                        "[%d/%d] bits=%d seed=%d cat=%s (SKIPPING - ALREADY DONE)",
                        counter, total, bit_depth, seed, category,
                    )
                    continue

                log.info(
                    "[%d/%d] bits=%d seed=%d cat=%s",
                    counter, total, bit_depth, seed, category,
                )
                try:
                    result = evaluate_category_at_bitdepth(
                        category, bit_depth, seed, feature_extractor, device
                    )
                    all_results.append(result)
                    
                    # Save progressively
                    with args.output_csv.open("a", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                        writer.writerow(result)
                        
                    log.info(
                        "  → AUROC=%.4f  ξ=%.6f  σ=%.6f",
                        result["image_auroc"],
                        result["gpd_shape_xi"],
                        result["gpd_scale_sigma"],
                    )
                except Exception as exc:
                    log.error("  FAILED: %s", exc)

    log.info("Saved → %s", args.output_csv)

    # Console summary: mean ± std per bit depth
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║        EXP-C: Quantization–EVT Interaction Summary (Mean ± Std)        ║")
    print("╠══════════╦══════════════════╦═════════════════════╦═════════════════════╣")
    print(f"║ {'Bits':>8} ║ {'Image AUROC':>16} ║ {'GPD ξ (shape)':>19} ║ {'GPD σ (scale)':>19} ║")
    print("╠══════════╬══════════════════╬═════════════════════╬═════════════════════╣")
    for bd in args.bit_depths:
        bd_results = [r for r in all_results if r["bit_depth"] == bd]
        if not bd_results:
            continue
        aurocs = [r["image_auroc"] for r in bd_results]
        xis = [r["gpd_shape_xi"] for r in bd_results if not np.isnan(r["gpd_shape_xi"])]
        sigmas = [r["gpd_scale_sigma"] for r in bd_results if not np.isnan(r["gpd_scale_sigma"])]
        print(
            f"║ {bd:>8} ║ {np.mean(aurocs):>7.4f} ± {np.std(aurocs):>5.4f} ║ "
            f"{np.mean(xis):>8.6f} ± {np.std(xis):>6.4f} ║ "
            f"{np.mean(sigmas):>8.6f} ± {np.std(sigmas):>6.4f} ║"
        )
    print("╚══════════╩══════════════════╩═════════════════════╩═════════════════════╝")
    print("\nEXP-C COMPLETE.")


if __name__ == "__main__":
    main()
