"""
extract_real_evt_calibration_example.py
=========================================
Extract Real EVT Calibration Data: Print actual tail distances and threshold (τ) values.

Purpose:
    Extract concrete numbers from a successful MVTec calibration run to replace
    fabricated examples in the thesis text. Shows:
    - Top 4 tail distance values (X_tail) from nominal test set
    - Fitted GPD parameters (shape ξ, scale σ)
    - Resulting threshold τ for target FPR
    
Output:
    Printed to console; can be copy-pasted into thesis text with proper formatting.
    
    Example output:
    =====================================
    Category: bottle | Seed: 111
    Tail Distances (top 4): [2.456, 2.234, 2.101, 1.987]
    GPD Fit: ξ = +0.0234, σ = 0.3456
    Threshold τ (FPR=0.01): 1.234
    =====================================
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from scipy import ndimage
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

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
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Example configuration: select a category and seed
EXAMPLE_CATEGORY = "bottle"
EXAMPLE_SEED = 111
EXAMPLE_N_SHOT = 10
TAIL_FRACTION = 0.05  # top 5% of nominal distances


def set_low_resource_mode() -> None:
    """Limit CPU threading for FAISS."""
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def build_mvtec_support_set(
    category: str,
    n_shot: int,
    seed: int,
    img_size: int = 256,
) -> Optional[torch.Tensor]:
    """
    Load and extract features from MVTec support set (normal images).
    
    Returns:
        (N, D) tensor of patch-level features, or None if not found
    """
    try:
        from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
        from utils.image_loader import MVTecStyleDataset
        
        device = torch.device("cpu")
        extractor = SymmetryAwareFeatureExtractor("wide_resnet50_2", device=device)
        
        dataset = MVTecStyleDataset(
            root_dir=DATA_DIR / "mvtec",
            category=category,
            is_train=True,
            img_size=img_size,
        )
        
        # Seeded sampling
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=min(n_shot, len(dataset)), replace=False)
        
        features_list = []
        with torch.no_grad():
            for idx in indices:
                img, _, _ = dataset[idx]
                img = img.unsqueeze(0).to(device)
                patch_features = extractor.extract_patch_features(img, apply_p4m=True)
                features_list.append(patch_features)
        
        features = np.concatenate(features_list, axis=0)  # (N_patches, D)
        log.info(f"Built support set: {features.shape[0]} patches")
        return torch.from_numpy(features).float()
    
    except Exception as e:
        log.error(f"Failed to build support set: {e}")
        return None


def compute_distances_to_support(
    support_features: torch.Tensor,
    test_features: torch.Tensor,
) -> np.ndarray:
    """
    Compute L2 distances from test patches to nearest neighbor in support set.
    
    Args:
        support_features: (N_support, D)
        test_features: (N_test, D)
    
    Returns:
        (N_test,) array of nearest-neighbor L2 distances
    """
    try:
        import faiss
        
        support_np = support_features.numpy().astype(np.float32)
        test_np = test_features.numpy().astype(np.float32)
        
        # Build exact index
        index = faiss.IndexFlatL2(support_np.shape[1])
        index.add(support_np)
        
        # Query test set
        distances, _ = index.search(test_np, k=1)
        distances = distances.squeeze(axis=1)  # (N_test,)
        
        log.info(f"Computed distances: shape={distances.shape}")
        return distances
    
    except Exception as e:
        log.error(f"Failed to compute distances: {e}")
        return None


def fit_gpd_to_tail(
    distances: np.ndarray,
    tail_fraction: float = 0.05,
) -> Optional[tuple[float, float, float]]:
    """
    Fit Generalized Pareto Distribution (GPD) to the tail of distances.
    
    Args:
        distances: (N,) array of distances
        tail_fraction: fraction of largest distances to use as tail
    
    Returns:
        (shape_xi, scale_sigma, threshold_u) or None
    """
    try:
        # Define tail threshold as (1 - tail_fraction) percentile
        u = np.percentile(distances, (1 - tail_fraction) * 100)
        tail_distances = distances[distances > u]
        
        if len(tail_distances) < 2:
            log.error("Not enough tail samples")
            return None
        
        # Fit GPD: x_exceedance = distance - u
        x_exceedance = tail_distances - u
        
        # scipy.stats.genpareto.fit returns (shape, loc, scale)
        # We fix loc=0 for standard POT
        params = genpareto.fit(x_exceedance, floc=0)
        shape_xi = params[0]
        scale_sigma = params[2]
        
        log.info(f"GPD Fit: ξ = {shape_xi:+.6f}, σ = {scale_sigma:.6f}")
        log.info(f"Tail Threshold u = {u:.6f}")
        log.info(f"Tail Samples: {len(tail_distances)} / {len(distances)}")
        
        return shape_xi, scale_sigma, u
    
    except Exception as e:
        log.error(f"Failed to fit GPD: {e}")
        return None


def compute_evt_threshold(
    shape_xi: float,
    scale_sigma: float,
    threshold_u: float,
    target_fpr: float = 0.01,
    n_nominal: int = 1000,
) -> float:
    """
    Compute EVT threshold (τ) for a target false-positive rate.
    
    Uses inverse survival function of GPD:
    FPR = P(X > τ) = (1 + shape * (τ - u) / scale)^(-1/shape)
    
    Args:
        shape_xi, scale_sigma, threshold_u: GPD parameters
        target_fpr: target false-positive rate
        n_nominal: reference number of nominal samples (for FPR definition)
    
    Returns:
        Threshold τ
    """
    try:
        if abs(shape_xi) < 1e-6:
            # Exponential case
            tau = threshold_u - scale_sigma * np.log(target_fpr)
        else:
            # General GPD case
            tau = threshold_u + (scale_sigma / shape_xi) * (
                (target_fpr ** (-shape_xi)) - 1
            )
        
        return float(tau)
    
    except Exception as e:
        log.error(f"Failed to compute threshold: {e}")
        return None


def main() -> None:
    """
    Extract real EVT calibration example from one MVTec category-seed.
    """
    log.info("=" * 80)
    log.info(f"Extracting Real EVT Calibration Example")
    log.info(f"Category: {EXAMPLE_CATEGORY} | Seed: {EXAMPLE_SEED} | N={EXAMPLE_N_SHOT}")
    log.info("=" * 80)
    
    set_low_resource_mode()
    
    # Build support set
    log.info("\n[1/4] Building support set from normal (train) images...")
    support_features = build_mvtec_support_set(
        EXAMPLE_CATEGORY,
        n_shot=EXAMPLE_N_SHOT,
        seed=EXAMPLE_SEED,
    )
    if support_features is None:
        log.error("Could not build support set")
        return
    
    # Load test set
    log.info("\n[2/4] Loading test set (nominal images only)...")
    try:
        from utils.image_loader import MVTecStyleDataset
        from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
        
        device = torch.device("cpu")
        extractor = SymmetryAwareFeatureExtractor("wide_resnet50_2", device=device)
        
        dataset = MVTecStyleDataset(
            root_dir=DATA_DIR / "mvtec",
            category=EXAMPLE_CATEGORY,
            is_train=False,
            img_size=256,
        )
        
        # Extract features from nominal test images only
        test_features_list = []
        with torch.no_grad():
            for i, (img, label, path) in enumerate(dataset):
                if label != 0:  # Skip anomalous images
                    continue
                img = img.unsqueeze(0).to(device)
                patch_features = extractor.extract_patch_features(img, apply_p4m=False)
                test_features_list.append(patch_features)
                
                if len(test_features_list) >= 50:  # Limit to reasonable number
                    break
        
        test_features = np.concatenate(test_features_list, axis=0)
        test_features = torch.from_numpy(test_features).float()
        log.info(f"Loaded test set: {test_features.shape[0]} patches from nominal images")
    
    except Exception as e:
        log.error(f"Failed to load test set: {e}")
        return
    
    # Compute distances
    log.info("\n[3/4] Computing nearest-neighbor distances...")
    distances = compute_distances_to_support(support_features, test_features)
    if distances is None:
        return
    
    # Fit GPD to tail
    log.info("\n[4/4] Fitting GPD to tail distances...")
    gpd_result = fit_gpd_to_tail(distances, tail_fraction=TAIL_FRACTION)
    if gpd_result is None:
        return
    
    shape_xi, scale_sigma, threshold_u = gpd_result
    
    # Extract top 4 tail distances
    sorted_distances = np.sort(distances)[::-1]  # descending
    top_4_distances = sorted_distances[:4]
    
    # Compute EVT threshold
    evt_threshold = compute_evt_threshold(shape_xi, scale_sigma, threshold_u, target_fpr=0.01)
    
    # Print formatted output for thesis
    log.info("\n" + "=" * 80)
    log.info("REAL EVT CALIBRATION EXAMPLE (for thesis text)")
    log.info("=" * 80)
    print("\n" + "~" * 80)
    print(f"Category: {EXAMPLE_CATEGORY.capitalize()} | Seed: {EXAMPLE_SEED}")
    print("~" * 80)
    print(f"\nTop 4 Tail Distances (X_tail):  {[f'{x:.3f}' for x in top_4_distances]}")
    print(f"GPD Shape Parameter:            ξ = {shape_xi:+.6f}")
    print(f"GPD Scale Parameter:            σ = {scale_sigma:.6f}")
    print(f"Tail Threshold:                 u = {threshold_u:.6f}")
    print(f"EVT-derived Threshold (τ):      τ = {evt_threshold:.6f}")
    print(f"Target FPR:                     0.01 (1%)")
    print("~" * 80 + "\n")
    
    log.info("✓ Example ready for thesis insertion")


if __name__ == "__main__":
    main()
