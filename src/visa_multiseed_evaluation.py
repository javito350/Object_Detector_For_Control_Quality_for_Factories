"""
visa_multiseed_evaluation.py
==============================
VisA Multi-Seed Evaluation: Extend evaluation to 5 random seeds for all 12 categories at N=1.

Purpose:
    Evaluate the proposed method on the VisA (Visual Anomaly) dataset across
    multiple random seeds. The current evaluation may have only 1 seed; this
    script adds 2 additional random seeds (bringing total to 5) and computes
    macro PRO-AUC across all 12 VisA categories.
    
    Expected seeds: [42, 111, 333, 999, 2026]
    
Output:
    results/visa_multiseed_pro_auc_results.csv
    - Columns: category, seed, image_auroc, pro_auc, n_samples
    - Macro PRO-AUC summary across all seeds and categories
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# VisA dataset categories (12 total)
VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni",
    "mixedbeans",
    "pcb",
    "pipefitting",
    "screw",
    "sphericalshells",
    "wireless_charger",
]

# Seeds to evaluate (expanding from 1 to 5)
SEEDS = [42, 111, 333, 999, 2026]

# Output file
OUTPUT_CSV = RESULTS_DIR / "visa_multiseed_pro_auc_results.csv"


def load_visa_predictions_and_masks(
    category: str,
    seed: int,
    n_shot: int = 1,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Load VisA predictions and ground-truth masks for a given category and seed.
    
    Expected file structure:
        {PROJECT_ROOT}/qualitative_results_visa/{category}_seed{seed}_heatmap.npy
        {DATA_DIR}/visa_mvtec_format/{category}/gt/{test_image}.png (or .npy)
    
    Returns:
        (predictions_array, masks_array) or None if not found
    """
    heatmap_dir = PROJECT_ROOT / "qualitative_results_visa"
    
    # Try to load heatmap
    heatmap_file = heatmap_dir / f"{category}_seed{seed:04d}_predictions.npy"
    if not heatmap_file.exists():
        # Try alternative naming
        heatmap_file = heatmap_dir / f"{category}_seed{seed}_heatmap.npy"
        if not heatmap_file.exists():
            log.warning(f"No heatmap found for {category} seed={seed}")
            return None
    
    try:
        predictions = np.load(heatmap_file)
        log.info(f"Loaded predictions: {heatmap_file.name} shape={predictions.shape}")
    except Exception as e:
        log.error(f"Failed to load {heatmap_file}: {e}")
        return None
    
    # Try to load masks from VisA format
    masks_dir = DATA_DIR / "visa_mvtec_format" / category / "gt"
    if not masks_dir.exists():
        log.warning(f"Masks directory not found: {masks_dir}")
        return None
    
    try:
        # Load all mask files and stack them
        mask_files = sorted(masks_dir.glob("*.npy"))
        if not mask_files:
            mask_files = sorted(masks_dir.glob("*.png"))
        
        if not mask_files:
            log.warning(f"No mask files found in {masks_dir}")
            return None
        
        masks_list = []
        for mf in mask_files:
            if mf.suffix == ".npy":
                m = np.load(mf)
            else:
                from PIL import Image
                m = np.array(Image.open(mf)) > 127  # binary threshold
            masks_list.append(m)
        
        masks = np.array(masks_list)
        log.info(f"Loaded {len(masks)} ground-truth masks for {category}")
    except Exception as e:
        log.error(f"Failed to load masks for {category}: {e}")
        return None
    
    return predictions, masks


def compute_pro_auc(
    predictions: np.ndarray,
    masks: np.ndarray,
    fpr_threshold: float = 0.3,
) -> float:
    """
    Compute Per-Region-Overlap (PRO-AUC) metric.
    
    PRO-AUC measures the percentage of anomalous regions correctly localized
    at a fixed false-positive rate (typically ≤0.3).
    
    Args:
        predictions: (N, H, W) prediction heatmaps [0, 1]
        masks: (N, H, W) binary ground-truth masks [0, 1]
        fpr_threshold: FPR at which to evaluate TPR
    
    Returns:
        PRO-AUC score in [0, 1]
    """
    if predictions.shape != masks.shape:
        log.warning(f"Shape mismatch: predictions {predictions.shape} vs masks {masks.shape}")
        return np.nan
    
    # Normalize predictions to [0, 1]
    pred_min = predictions.min()
    pred_max = predictions.max()
    if pred_max > pred_min:
        predictions_norm = (predictions - pred_min) / (pred_max - pred_min)
    else:
        predictions_norm = predictions.copy()
    
    # Flatten for threshold sweep
    pred_flat = predictions_norm.flatten()
    mask_flat = masks.flatten()
    
    # Sort by prediction score
    sort_idx = np.argsort(-pred_flat)
    pred_sorted = pred_flat[sort_idx]
    mask_sorted = mask_flat[sort_idx]
    
    # Sweep thresholds
    thresholds = np.unique(pred_sorted)
    n_nominal = (mask_flat == 0).sum()
    n_anomalous = (mask_flat == 1).sum()
    
    pro_scores = []
    fprs = []
    
    for threshold in thresholds:
        predicted_anomalous = pred_sorted >= threshold
        
        # False positives (nominal pixels predicted as anomalous)
        fp = ((mask_sorted == 0) & predicted_anomalous).sum()
        fpr = fp / max(n_nominal, 1)
        
        if fpr <= fpr_threshold:
            # Connect regions in predicted and true masks
            pred_binary = (predictions_norm >= threshold).astype(int)
            
            # Label connected components
            labeled_true, n_true = ndimage.label(masks)
            labeled_pred, n_pred = ndimage.label(pred_binary)
            
            # Compute overlap for each true region
            overlaps = []
            for region_id in range(1, n_true + 1):
                true_region = (labeled_true == region_id)
                overlap = (true_region & (labeled_pred > 0)).sum() / true_region.sum()
                overlaps.append(overlap)
            
            pro = np.mean(overlaps) if overlaps else 0.0
            pro_scores.append(pro)
            fprs.append(fpr)
    
    if not pro_scores:
        return 0.0
    
    # Compute AUC over FPR range [0, fpr_threshold]
    if len(pro_scores) < 2:
        return float(pro_scores[0])
    
    fprs = np.array(fprs)
    pro_scores = np.array(pro_scores)
    
    # Sort by FPR
    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pro_scores = pro_scores[sort_idx]
    
    # Trapz integration
    pro_auc = np.trapz(pro_scores, fprs) / max(fprs[-1], fpr_threshold)
    return float(pro_auc)


def evaluate_category_seed(
    category: str,
    seed: int,
) -> Optional[dict]:
    """
    Evaluate a single category-seed combination.
    
    Returns:
        {"category": ..., "seed": ..., "image_auroc": ..., "pro_auc": ..., "n_samples": ...}
    """
    result = load_visa_predictions_and_masks(category, seed)
    if result is None:
        log.warning(f"Skipping {category} seed={seed}: data not found")
        return None
    
    predictions, masks = result
    
    try:
        # Compute image-level AUROC
        pred_image_score = predictions.max(axis=(1, 2))  # max pixel value per image
        mask_image_label = (masks.max(axis=(1, 2)) > 0).astype(int)
        
        if len(np.unique(mask_image_label)) < 2:
            log.warning(f"No positive labels for {category} seed={seed}")
            image_auroc = np.nan
        else:
            image_auroc = roc_auc_score(mask_image_label, pred_image_score)
        
        # Compute PRO-AUC
        pro_auc = compute_pro_auc(predictions, masks)
        
        return {
            "category": category,
            "seed": seed,
            "image_auroc": image_auroc,
            "pro_auc": pro_auc,
            "n_samples": len(predictions),
        }
    except Exception as e:
        log.error(f"Failed to evaluate {category} seed={seed}: {e}")
        return None


def main() -> None:
    """
    Run VisA multi-seed evaluation for all categories.
    """
    log.info("=" * 80)
    log.info("VisA Multi-Seed Evaluation: PRO-AUC across 5 seeds and 12 categories")
    log.info("=" * 80)
    
    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    results = []
    
    for category in VISA_CATEGORIES:
        log.info(f"\n➤ Processing category: {category}")
        for seed in SEEDS:
            log.info(f"  Evaluating seed={seed}...")
            result = evaluate_category_seed(category, seed)
            if result is not None:
                results.append(result)
                log.info(
                    f"    ✓ AUROC={result['image_auroc']:.4f}, "
                    f"PRO-AUC={result['pro_auc']:.4f}"
                )
    
    # Write results CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        log.info(f"\n✓ Saved results to: {OUTPUT_CSV}")
        
        # Compute macro PRO-AUC (average across all seeds and categories)
        macro_pro_auc = results_df["pro_auc"].mean()
        macro_auroc = results_df["image_auroc"].mean()
        
        log.info("\n" + "=" * 80)
        log.info("SUMMARY STATISTICS")
        log.info("=" * 80)
        log.info(f"Macro PRO-AUC (mean across {len(results)} evaluations): {macro_pro_auc:.4f}")
        log.info(f"Macro Image AUROC (mean across {len(results)} evaluations): {macro_auroc:.4f}")
        log.info(f"Total evaluated: {len(VISA_CATEGORIES)} categories × {len(SEEDS)} seeds")
    else:
        log.error("No results collected!")


if __name__ == "__main__":
    main()
