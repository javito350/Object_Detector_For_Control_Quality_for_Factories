"""
wacv_fix4_evt_calibration.py
============================
WACV FIX-4 — EVT Calibration Verification.

Reviewer concern:
    "Thresholding claims are qualitative and not statistically validated."

For each MVTec category:
    1. Build memory bank (proposed 8-bit pipeline, seed=111)
    2. Fit EVT threshold on support distances
    3. Apply threshold to test set (image-level)
    4. Compute empirical FPR on nominal test images
    5. Bootstrap (B=1000): resample test nominals, recompute FPR each time
    6. Report: empirical_fpr, target_fpr (0.01), calibration_error, 95% CI

Output:
    results/wacv/fix4_evt_calibration.csv

Run from the project root:
    python src/wacv_fix4_evt_calibration.py
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
from torch.utils.data import DataLoader

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
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "wacv" / "fix4_evt_calibration.csv"
BACKBONE = "wide_resnet50_2"
FEATURE_DIM = 1536
N_SHOT = 10
PQ_BITS = 8
SUPPORT_SEED = 111
IMG_SIZE = 256
TARGET_FPR = 0.01
TAIL_FRACTION = 0.10
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

FIELDNAMES = [
    "category",
    "n_test_nominal", "n_test_anomalous",
    "evt_threshold",
    "empirical_fpr", "target_fpr", "calibration_error",
    "empirical_fnr",
    "ci_lower_95", "ci_upper_95",
    "ci_lower_99", "ci_upper_99",
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


def bootstrap_fpr(
    nominal_scores: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Bootstrap the empirical FPR to estimate confidence intervals.
    
    For each bootstrap iteration:
        - Resample nominal test scores with replacement
        - Compute FPR = fraction that exceed the threshold
    
    Returns percentile-based confidence intervals.
    """
    rng = np.random.RandomState(seed)
    n = len(nominal_scores)
    bootstrap_fprs = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        resample = nominal_scores[rng.choice(n, size=n, replace=True)]
        bootstrap_fprs[b] = float(np.mean(resample > threshold))

    return {
        "ci_lower_95": float(np.percentile(bootstrap_fprs, 2.5)),
        "ci_upper_95": float(np.percentile(bootstrap_fprs, 97.5)),
        "ci_lower_99": float(np.percentile(bootstrap_fprs, 0.5)),
        "ci_upper_99": float(np.percentile(bootstrap_fprs, 99.5)),
        "bootstrap_mean": float(np.mean(bootstrap_fprs)),
        "bootstrap_std": float(np.std(bootstrap_fprs)),
    }


def evaluate_category_calibration(
    category: str,
    feature_extractor: SymmetryAwareFeatureExtractor,
    device: str,
) -> dict:
    """Evaluate EVT calibration for one MVTec category."""
    # Build support loader
    train_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    train_ds.samples = sample_seeded_support(train_ds.samples, N_SHOT, SUPPORT_SEED)
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

    # EVT threshold calibration
    calibrator = EVTCalibrator(tail_fraction=TAIL_FRACTION, target_fpr=TARGET_FPR)
    cal_feats = np.vstack(support_features).astype(np.float32)
    cal_dists, _ = memory_bank.query(cal_feats, k=1)
    evt_threshold = calibrator.fit(cal_dists.flatten())

    # Test inference — compute image-level anomaly scores
    test_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    image_scores = []
    image_labels = []

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
            image_labels.append(int(labels[idx].item()))

    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)

    # Split into nominal and anomalous
    nominal_mask = image_labels == 0
    anomalous_mask = image_labels == 1
    nominal_scores = image_scores[nominal_mask]
    anomalous_scores = image_scores[anomalous_mask]

    n_nominal = int(nominal_mask.sum())
    n_anomalous = int(anomalous_mask.sum())

    # Empirical FPR and FNR
    if n_nominal > 0:
        empirical_fpr = float(np.mean(nominal_scores > evt_threshold))
    else:
        empirical_fpr = float("nan")

    if n_anomalous > 0:
        empirical_fnr = float(np.mean(anomalous_scores <= evt_threshold))
    else:
        empirical_fnr = float("nan")

    calibration_error = abs(empirical_fpr - TARGET_FPR)

    # Bootstrap CI
    if n_nominal >= 5:
        bs = bootstrap_fpr(nominal_scores, evt_threshold, N_BOOTSTRAP, BOOTSTRAP_SEED)
    else:
        bs = {
            "ci_lower_95": float("nan"),
            "ci_upper_95": float("nan"),
            "ci_lower_99": float("nan"),
            "ci_upper_99": float("nan"),
        }

    return {
        "category": category,
        "n_test_nominal": n_nominal,
        "n_test_anomalous": n_anomalous,
        "evt_threshold": round(float(evt_threshold), 6),
        "empirical_fpr": round(empirical_fpr, 6),
        "target_fpr": TARGET_FPR,
        "calibration_error": round(calibration_error, 6),
        "empirical_fnr": round(empirical_fnr, 6),
        "ci_lower_95": round(bs["ci_lower_95"], 6),
        "ci_upper_95": round(bs["ci_upper_95"], 6),
        "ci_lower_99": round(bs["ci_lower_99"], 6),
        "ci_upper_99": round(bs["ci_upper_99"], 6),
    }


# ── CLI + main ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WACV FIX-4: EVT calibration verification with bootstrap CI."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    log.info("Loading frozen %s backbone...", BACKBONE)
    feature_extractor = SymmetryAwareFeatureExtractor(backbone=BACKBONE, device=device)

    log.info(
        "WACV FIX-4: EVT Calibration Verification (target FPR=%.2f, B=%d bootstrap)",
        TARGET_FPR, args.n_bootstrap,
    )

    all_results = []
    for i, category in enumerate(CATEGORIES, 1):
        log.info("[%d/%d] %s", i, len(CATEGORIES), category)
        try:
            result = evaluate_category_calibration(category, feature_extractor, device)
            all_results.append(result)
            log.info(
                "  → FPR=%.4f (target=%.2f, err=%.4f) FNR=%.4f  CI95=[%.4f, %.4f]",
                result["empirical_fpr"], TARGET_FPR, result["calibration_error"],
                result["empirical_fnr"], result["ci_lower_95"], result["ci_upper_95"],
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
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                 FIX-4: EVT Calibration Verification                        ║")
    print("║                 Target FPR = 0.01 | Bootstrap B=1000                       ║")
    print("╠═══════════════╦════════════╦════════════╦══════════════╦════════════════════╣")
    print(f"║ {'Category':<13} ║ {'Emp. FPR':>10} ║ {'Cal. Error':>10} ║ {'Emp. FNR':>12} ║ {'95% CI':>18} ║")
    print("╠═══════════════╬════════════╬════════════╬══════════════╬════════════════════╣")
    cal_errors = []
    for r in all_results:
        ci_str = f"[{r['ci_lower_95']:.4f}, {r['ci_upper_95']:.4f}]"
        print(
            f"║ {r['category']:<13} ║ {r['empirical_fpr']:>10.4f} ║ "
            f"{r['calibration_error']:>10.4f} ║ {r['empirical_fnr']:>12.4f} ║ "
            f"{ci_str:>18} ║"
        )
        cal_errors.append(r["calibration_error"])
    print("╠═══════════════╬════════════╬════════════╬══════════════╬════════════════════╣")
    if cal_errors:
        mean_err = np.mean(cal_errors)
        print(f"║ {'MEAN':>13} ║ {'':>10} ║ {mean_err:>10.4f} ║ {'':>12} ║ {'':>18} ║")
    print("╚═══════════════╩════════════╩════════════╩══════════════╩════════════════════╝")

    # Pass/Fail verdict
    if cal_errors:
        within_tol = sum(1 for e in cal_errors if e <= 0.05)
        print(f"\n★ {within_tol}/{len(cal_errors)} categories within 5% calibration tolerance")
        if within_tol >= len(cal_errors) * 0.8:
            print("★ VERDICT: EVT calibration is CONSISTENT across categories.")
        else:
            print("★ VERDICT: EVT calibration shows INSTABILITY in some categories.")

    print("\nFIX-4 COMPLETE.")


if __name__ == "__main__":
    main()
