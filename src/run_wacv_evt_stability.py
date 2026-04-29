"""
run_wacv_evt_stability.py
=========================
WACV overnight experiment: EVT (GPD) stability under quantization.

For the MVTec *screw* category with N=10 support images, this script:

  1. Loads the support set (N=10, seeded) and the full test set.
  2. Extracts patch-level embeddings using the frozen Wide ResNet-50
     (SymmetryAwareFeatureExtractor, 1536-dim features).
  3. Builds FOUR independent FAISS memory banks from the support features:
       • Exact  — IndexFlatL2          (lossless baseline)
       • 4-bit  — IndexIVFPQ (nbits=4)
       • 8-bit  — IndexIVFPQ (nbits=8)
       • 12-bit — IndexIVFPQ (nbits=12)
  4. Queries every test-set patch against each bank and collects the
     nearest-neighbor L2 distances.
  5. Applies Peaks-over-Threshold (POT) EVT:
       • Takes the top-5% of nominal (label=0) test distances as the tail.
       • Fits a Generalized Pareto Distribution (GPD) via
         scipy.stats.genpareto.fit (MLE, loc forced to 0).
       • Records the GPD shape (ξ / xi) and scale (σ / sigma) parameters.
  6. Saves (Index_Type, Bit_Depth, GPD_Shape_xi, GPD_Scale_sigma,
            Tail_Threshold_u, N_Tail_Samples) to evt_quantization_stability.csv.

NOTE on scipy function:
  The user specification mentions scipy.stats.genextreme (GEV), but the
  project's existing EVTCalibrator (thresholding.py) correctly uses
  scipy.stats.genpareto for Peaks-over-Threshold.  This script follows the
  established project convention (GPD / POT) which is the statistically
  appropriate model for tail exceedances.

Run from the project root:
    python run_wacv_evt_stability.py

Optional overrides:
    python run_wacv_evt_stability.py --n-shot 10 --support-seed 111 \\
        --tail-fraction 0.05 --dataset-root data/mvtec
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

# ── Thread caps — must be set BEFORE numpy / faiss import ─────────────────────
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from scipy.stats import genpareto
from torch.utils.data import DataLoader
from torchvision import transforms as T

# ── Project-local imports ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from utils.image_loader import MVTecStyleDataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORY            = "screw"
DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "mvtec"
DEFAULT_OUTPUT_CSV   = ROOT_DIR / "evt_quantization_stability.csv"
DEFAULT_IMG_SIZE     = 256
DEFAULT_N_SHOT       = 10
DEFAULT_SUPPORT_SEED = 111
DEFAULT_TAIL_FRAC    = 0.05       # top-5 % of nominal distances → POT tail
FEATURE_DIM          = 1536       # layer2 (512) + layer3 (1024) from WRN-50

# ── Index configuration ───────────────────────────────────────────────────────
@dataclass
class IndexConfig:
    label: str        # human-readable name
    bit_depth: int    # 0 = exact / lossless
    use_pq: bool


INDEX_CONFIGS: List[IndexConfig] = [
    IndexConfig(label="Exact",  bit_depth=0,  use_pq=False),
    IndexConfig(label="4-bit",  bit_depth=4,  use_pq=True),
    IndexConfig(label="8-bit",  bit_depth=8,  use_pq=True),
    IndexConfig(label="12-bit", bit_depth=12, use_pq=True),
]

# ── CSV fieldnames ─────────────────────────────────────────────────────────────
FIELDNAMES = [
    "Index_Type",
    "Bit_Depth",
    "GPD_Shape_xi",
    "GPD_Scale_sigma",
    "Tail_Threshold_u",
    "N_Tail_Samples",
    "N_Total_Nominal_Patches",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WACV EVT stability: fit GPD to nominal distances of Exact vs "
            "4/8/12-bit PQ banks on MVTec screw (N=10)."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help=f"Path to MVTec dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help=f"Destination CSV file (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--img-size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"Image resize resolution (default: {DEFAULT_IMG_SIZE})",
    )
    parser.add_argument(
        "--n-shot", type=int, default=DEFAULT_N_SHOT,
        help=f"Support-set size N (default: {DEFAULT_N_SHOT})",
    )
    parser.add_argument(
        "--support-seed", type=int, default=DEFAULT_SUPPORT_SEED,
        help=f"NumPy seed for reproducible support sampling (default: {DEFAULT_SUPPORT_SEED})",
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=DEFAULT_TAIL_FRAC,
        help=f"Fraction of top nominal distances used as GPD tail (default: {DEFAULT_TAIL_FRAC})",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Resource throttle
# ─────────────────────────────────────────────────────────────────────────────
def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(1)


# ─────────────────────────────────────────────────────────────────────────────
# Seeded support sampling  (identical contract to evaluate_pca_256.py)
# ─────────────────────────────────────────────────────────────────────────────
def sample_seeded_support(
    samples: list[tuple[str, int]],
    n_shot: int,
    seed: int,
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


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
def _eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_support_loader(
    dataset_root: Path,
    category: str,
    img_size: int,
    n_shot: int,
    support_seed: int,
) -> DataLoader:
    ds = MVTecStyleDataset(
        root_dir=str(dataset_root),
        category=category,
        is_train=True,
        img_size=img_size,
    )
    ds.samples   = sample_seeded_support(ds.samples, n_shot, support_seed)
    ds.transform = _eval_transform(img_size)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


def build_test_loader(
    dataset_root: Path,
    category: str,
    img_size: int,
) -> DataLoader:
    ds = MVTecStyleDataset(
        root_dir=str(dataset_root),
        category=category,
        is_train=False,
        img_size=img_size,
    )
    ds.transform = _eval_transform(img_size)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_embeddings(
    loader: DataLoader,
    extractor: SymmetryAwareFeatureExtractor,
    apply_p4m: bool,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract patch embeddings from *loader*.

    Returns
    -------
    patches : np.ndarray, shape [N_patches, FEATURE_DIM]
    labels  : np.ndarray, shape [N_patches]  (image-level label, broadcast)
    """
    all_patches: list[np.ndarray] = []
    all_labels:  list[np.ndarray] = []

    for images, labels, _paths in loader:
        images = images.to(extractor.device)
        patches = extractor.extract_patch_features(images, apply_p4m=apply_p4m)
        patches = patches.astype(np.float32)
        n_patches = patches.shape[0]
        img_label = int(labels[0].item())
        all_patches.append(patches)
        all_labels.append(np.full(n_patches, img_label, dtype=np.int32))

    if not all_patches:
        raise RuntimeError(f"No features extracted from {desc} loader.")

    return np.vstack(all_patches), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index builders
# ─────────────────────────────────────────────────────────────────────────────
def _choose_subquantizers(dim: int) -> int:
    """Largest M that divides *dim* and is <= dim (FAISS PQ requirement)."""
    for m in [64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]:
        if m <= dim and dim % m == 0:
            return m
    return 1


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def build_exact_index(features: np.ndarray) -> faiss.Index:
    """Lossless L2 flat index — the oracle baseline."""
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index


def build_pq_index(features: np.ndarray, pq_bits: int) -> faiss.Index:
    """
    IVF-PQ index with *pq_bits* bits per sub-quantizer code.
    nlist is capped so training is stable even for small support sets.
    """
    dim   = features.shape[1]
    n     = features.shape[0]
    nlist = max(1, min(100, n // 10))
    M     = _choose_subquantizers(dim)

    quantizer = faiss.IndexFlatL2(dim)
    index     = faiss.IndexIVFPQ(quantizer, dim, nlist, M, pq_bits)
    index.train(features)
    index.add(features)
    return index


def build_index(features: np.ndarray, cfg: IndexConfig) -> faiss.Index:
    """Dispatch to the right builder based on *cfg*."""
    norm_features = _l2_normalize(features)
    if cfg.use_pq:
        log.info("  Building %s index (IVF-PQ, nbits=%d) …",
                 cfg.label, cfg.bit_depth)
        return build_pq_index(norm_features, cfg.bit_depth)
    else:
        log.info("  Building Exact index (IndexFlatL2) …")
        return build_exact_index(norm_features)


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-neighbour distances
# ─────────────────────────────────────────────────────────────────────────────
def query_distances(
    index: faiss.Index,
    query_patches: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """Return the k=1 NN L2 distance for every row of *query_patches*."""
    norm_q = _l2_normalize(query_patches.astype(np.float32))
    distances, _ = index.search(norm_q, k)
    return distances.flatten()  # shape [N_patches]


# ─────────────────────────────────────────────────────────────────────────────
# EVT — GPD fit via Peaks-over-Threshold
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GPDResult:
    shape_xi:    float
    scale_sigma: float
    threshold_u: float
    n_tail:      int
    n_total:     int


def fit_gpd_pot(
    distances: np.ndarray,
    labels: np.ndarray,
    tail_fraction: float,
) -> GPDResult:
    """
    Peaks-over-Threshold GPD fit on nominal (label==0) patch distances.

    Steps
    -----
    1. Isolate nominal distances (images labelled 0 = good/normal).
    2. Determine threshold u = (1 - tail_fraction) quantile.
    3. Extract exceedances: x_i - u  for x_i > u.
    4. Fit GPD via MLE (scipy genpareto, loc forced to 0).
    5. Return shape ξ, scale σ, and diagnostic metadata.
    """
    nominal_dist = distances[labels == 0]
    if len(nominal_dist) == 0:
        raise RuntimeError(
            "No nominal (label=0) patches found in test set — "
            "cannot fit GPD tail."
        )

    sorted_d = np.sort(nominal_dist)
    tail_idx  = int(len(sorted_d) * (1.0 - tail_fraction))
    # Guard: ensure at least a handful of tail samples
    tail_idx  = min(tail_idx, len(sorted_d) - 10)
    u         = float(sorted_d[tail_idx])
    exceedances = sorted_d[tail_idx:] - u

    if len(exceedances) < 5 or float(np.max(exceedances)) < 1e-12:
        warnings.warn(
            "Tail has near-zero variance; GPD fit may be degenerate. "
            "Consider reducing --tail-fraction.",
            RuntimeWarning,
        )
        return GPDResult(
            shape_xi=float("nan"),
            scale_sigma=float("nan"),
            threshold_u=u,
            n_tail=len(exceedances),
            n_total=len(nominal_dist),
        )

    # MLE fit — genpareto.fit returns (shape, loc, scale); loc is fixed to 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape, _loc, scale = genpareto.fit(exceedances, floc=0)

    return GPDResult(
        shape_xi=float(shape),
        scale_sigma=float(scale),
        threshold_u=u,
        n_tail=len(exceedances),
        n_total=len(nominal_dist),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(output_csv: Path, rows: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Results saved → %s", output_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    # ── Validate paths ────────────────────────────────────────────────────────
    category_path = args.dataset_root / CATEGORY
    if not category_path.is_dir():
        log.error(
            "Category directory not found: %s\n"
            "Check --dataset-root or ensure MVTec is extracted.",
            category_path,
        )
        sys.exit(1)

    # ── Backbone ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    log.info("Loading frozen Wide ResNet-50 backbone …")
    extractor = SymmetryAwareFeatureExtractor(
        backbone="wide_resnet50_2", device=device
    )

    # ── Support features (p4m ON — matches training regime) ──────────────────
    log.info("=" * 60)
    log.info("Extracting SUPPORT features (N=%d, seed=%d, p4m=ON) …",
             args.n_shot, args.support_seed)
    support_loader = build_support_loader(
        args.dataset_root, CATEGORY, args.img_size,
        args.n_shot, args.support_seed,
    )
    support_patches, _ = extract_embeddings(
        support_loader, extractor, apply_p4m=True, desc="support"
    )
    log.info("Support matrix: %s patches × %d dims",
             f"{support_patches.shape[0]:,}", support_patches.shape[1])

    # ── Test features (p4m OFF — inference mode) ──────────────────────────────
    log.info("=" * 60)
    log.info("Extracting TEST features (p4m=OFF) …")
    test_loader = build_test_loader(args.dataset_root, CATEGORY, args.img_size)
    test_patches, test_labels = extract_embeddings(
        test_loader, extractor, apply_p4m=False, desc="test"
    )
    log.info(
        "Test matrix: %s patches × %d dims  |  unique labels: %s",
        f"{test_patches.shape[0]:,}", test_patches.shape[1],
        np.unique(test_labels).tolist(),
    )

    n_nominal = int((test_labels == 0).sum())
    log.info("Nominal test patches: %s  |  Anomalous: %s",
             f"{n_nominal:,}",
             f"{int((test_labels == 1).sum()):,}")

    # ── Loop over 4 index configs ─────────────────────────────────────────────
    results: list[dict] = []

    for cfg in INDEX_CONFIGS:
        log.info("=" * 60)
        log.info("[%s] Building FAISS index …", cfg.label)

        try:
            index = build_index(support_patches, cfg)
        except Exception as exc:
            log.error("[%s] Index build FAILED: %s", cfg.label, exc, exc_info=True)
            results.append({
                "Index_Type":              cfg.label,
                "Bit_Depth":               cfg.bit_depth if cfg.use_pq else "Exact",
                "GPD_Shape_xi":            "ERROR",
                "GPD_Scale_sigma":         "ERROR",
                "Tail_Threshold_u":        "ERROR",
                "N_Tail_Samples":          "ERROR",
                "N_Total_Nominal_Patches": "ERROR",
            })
            continue

        log.info("[%s] Querying test patches …", cfg.label)
        try:
            distances = query_distances(index, test_patches)
        except Exception as exc:
            log.error("[%s] Query FAILED: %s", cfg.label, exc, exc_info=True)
            results.append({
                "Index_Type":              cfg.label,
                "Bit_Depth":               cfg.bit_depth if cfg.use_pq else "Exact",
                "GPD_Shape_xi":            "ERROR",
                "GPD_Scale_sigma":         "ERROR",
                "Tail_Threshold_u":        "ERROR",
                "N_Tail_Samples":          "ERROR",
                "N_Total_Nominal_Patches": "ERROR",
            })
            continue

        log.info(
            "[%s] Distance stats — min=%.4f, max=%.4f, mean=%.4f",
            cfg.label,
            float(distances.min()),
            float(distances.max()),
            float(distances.mean()),
        )

        # ── GPD / POT fit ─────────────────────────────────────────────────────
        log.info("[%s] Fitting GPD (POT, tail=%.0f%%) …",
                 cfg.label, args.tail_fraction * 100)
        try:
            gpd = fit_gpd_pot(distances, test_labels, args.tail_fraction)
        except Exception as exc:
            log.error("[%s] GPD fit FAILED: %s", cfg.label, exc, exc_info=True)
            results.append({
                "Index_Type":              cfg.label,
                "Bit_Depth":               cfg.bit_depth if cfg.use_pq else "Exact",
                "GPD_Shape_xi":            "ERROR",
                "GPD_Scale_sigma":         "ERROR",
                "Tail_Threshold_u":        "ERROR",
                "N_Tail_Samples":          "ERROR",
                "N_Total_Nominal_Patches": "ERROR",
            })
            continue

        log.info(
            "[%s] GPD fit → ξ (shape)=%.6f  σ (scale)=%.6f  "
            "u=%.6f  n_tail=%d / %d",
            cfg.label,
            gpd.shape_xi, gpd.scale_sigma,
            gpd.threshold_u, gpd.n_tail, gpd.n_total,
        )

        results.append({
            "Index_Type":              cfg.label,
            "Bit_Depth":               cfg.bit_depth if cfg.use_pq else "Exact",
            "GPD_Shape_xi":            round(gpd.shape_xi,    8),
            "GPD_Scale_sigma":         round(gpd.scale_sigma, 8),
            "Tail_Threshold_u":        round(gpd.threshold_u, 8),
            "N_Tail_Samples":          gpd.n_tail,
            "N_Total_Nominal_Patches": gpd.n_total,
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(args.output_csv, results)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         EVT Quantization Stability — GPD Results (MVTec Screw)      ║")
    print("╠══════════╦═══════════╦══════════════════╦═══════════════════════════╣")
    print(f"║ {'Index':<8} ║ {'Bit-Depth':<9} ║ {'ξ  (shape)':<16} ║ {'σ  (scale)':<25} ║")
    print("╠══════════╬═══════════╬══════════════════╬═══════════════════════════╣")
    for row in results:
        xi    = row["GPD_Shape_xi"]
        sigma = row["GPD_Scale_sigma"]
        xi_s    = f"{xi:.6f}"    if isinstance(xi,    float) else str(xi)
        sigma_s = f"{sigma:.6f}" if isinstance(sigma, float) else str(sigma)
        print(
            f"║ {str(row['Index_Type']):<8} ║ "
            f"{str(row['Bit_Depth']):<9} ║ "
            f"{xi_s:<16} ║ "
            f"{sigma_s:<25} ║"
        )
    print("╚══════════╩═══════════╩══════════════════╩═══════════════════════════╝")
    print(f"\nFull results saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
