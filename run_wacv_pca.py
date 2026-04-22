"""
run_wacv_pca.py
===============
Overnight standalone script for the WACV PCA analysis.

For each of the 12 VisA categories:
  1. Load N=10 support images (seeded for reproducibility).
  2. Extract patch-level embeddings via the frozen Wide ResNet-50
     backbone (SymmetryAwareFeatureExtractor, p4m augmentation ON,
     producing shape [N*8*784, 1536]).
  3. Flatten all patch embeddings into one matrix and run
     sklearn PCA (svd_solver='full') to capture cumulative variance.
  4. Determine the minimum number of principal components needed
     to explain >= 95% of the total variance.
  5. Write (Category, N_Components_95pct) to visa_pca_variance.csv.

Run from the project root:
    python run_wacv_pca.py

Optional overrides:
    python run_wacv_pca.py --n-shot 5 --support-seed 42
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# ── Thread-count caps (must be set BEFORE numpy / torch import) ───────────────
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import transforms as T

# ── Project-local imports ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from utils.image_loader import MVTecStyleDataset

# ── Constants ─────────────────────────────────────────────────────────────────
VISA_CATEGORIES: list[str] = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "visa_mvtec_format"
DEFAULT_OUTPUT_CSV   = ROOT_DIR / "visa_pca_variance.csv"
DEFAULT_IMG_SIZE     = 256
DEFAULT_N_SHOT       = 10
DEFAULT_SUPPORT_SEED = 111
VARIANCE_THRESHOLD   = 0.95          # 95 % explained variance target


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WACV PCA analysis: number of components for 95 %% variance "
            "per VisA category using a Wide ResNet-50 feature extractor."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the MVTec-format VisA dataset root "
             f"(default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Destination CSV file (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Image resize resolution (default: {DEFAULT_IMG_SIZE})",
    )
    parser.add_argument(
        "--n-shot",
        type=int,
        default=DEFAULT_N_SHOT,
        help=f"Support-set size N (default: {DEFAULT_N_SHOT})",
    )
    parser.add_argument(
        "--support-seed",
        type=int,
        default=DEFAULT_SUPPORT_SEED,
        help=f"NumPy seed for reproducible support sampling (default: {DEFAULT_SUPPORT_SEED})",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD,
        help=f"Cumulative explained-variance target (default: {VARIANCE_THRESHOLD})",
    )
    return parser.parse_args()


# ── Resource throttle (mirrors evaluate_pca_256.py) ──────────────────────────
def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


# ── Seeded support sampling (identical to evaluate_pca_256.py) ───────────────
def sample_seeded_support(
    samples: list[tuple[str, int]],
    n_shot: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Deterministically sample *n_shot* items from *samples* using *seed*."""
    if n_shot > len(samples):
        raise ValueError(
            f"Requested n_shot={n_shot}, but only {len(samples)} "
            "train/good samples are available for this category."
        )
    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in idx]


# ── DataLoader builder ────────────────────────────────────────────────────────
def build_support_dataloader(
    dataset_root: Path,
    category: str,
    img_size: int,
    n_shot: int,
    support_seed: int,
) -> DataLoader:
    """Return a DataLoader for the N=*n_shot* support images of *category*."""
    dataset = MVTecStyleDataset(
        root_dir=str(dataset_root),
        category=category,
        is_train=True,
        img_size=img_size,
    )
    dataset.samples = sample_seeded_support(dataset.samples, n_shot, support_seed)
    # Override any augmenting transform with a deterministic eval transform
    dataset.transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_support_embeddings(
    dataloader: DataLoader,
    extractor: SymmetryAwareFeatureExtractor,
) -> np.ndarray:
    """
    Run the support set through the frozen backbone with p4m augmentation.

    Each image → 8 augmented views → 784 patches per view → 1536-dim vector.
    Returns a flat matrix of shape [N * 8 * 784, 1536].
    """
    all_patches: list[np.ndarray] = []
    for images, _labels, _paths in dataloader:
        images = images.to(extractor.device)
        # apply_p4m=True: 8× symmetry expansion (matches training regime)
        patches = extractor.extract_patch_features(images, apply_p4m=True)
        all_patches.append(patches.astype(np.float32))

    if not all_patches:
        raise RuntimeError("No embeddings were extracted — dataloader is empty.")

    return np.vstack(all_patches)


# ── PCA helper ────────────────────────────────────────────────────────────────
def components_for_variance(
    embedding_matrix: np.ndarray,
    variance_threshold: float,
) -> tuple[int, PCA]:
    """
    Fit a full-SVD PCA on *embedding_matrix* and return:
      (n_components, fitted_pca)
    where n_components is the smallest k such that the first k components
    explain >= *variance_threshold* of total variance.

    Note: n_components is capped at min(n_samples, n_features).
    """
    n_samples, n_features = embedding_matrix.shape
    max_components = min(n_samples, n_features)

    pca = PCA(n_components=max_components, svd_solver="full", random_state=42)
    pca.fit(embedding_matrix)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # np.searchsorted returns the first index where cumvar >= threshold
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1
    # Clamp to valid range
    k = max(1, min(k, max_components))

    return k, pca


# ── Per-category pipeline ─────────────────────────────────────────────────────
def analyse_category(
    dataset_root: Path,
    category: str,
    img_size: int,
    n_shot: int,
    support_seed: int,
    variance_threshold: float,
    extractor: SymmetryAwareFeatureExtractor,
) -> dict[str, object]:
    """
    Full pipeline for one VisA category.
    Returns a dict with keys: Category, N_Components_95pct,
    Variance_Explained, Total_Patches, Feature_Dim.
    """
    log.info("── %s: building support dataloader (N=%d, seed=%d) …",
             category, n_shot, support_seed)
    loader = build_support_dataloader(dataset_root, category, img_size,
                                      n_shot, support_seed)

    log.info("── %s: extracting patch embeddings …", category)
    t0 = time.perf_counter()
    embeddings = extract_support_embeddings(loader, extractor)
    elapsed = time.perf_counter() - t0
    log.info("── %s: extracted %s patches of dim %d in %.1f s",
             category, f"{embeddings.shape[0]:,}", embeddings.shape[1], elapsed)

    log.info("── %s: fitting PCA (full SVD) …", category)
    t1 = time.perf_counter()
    k, pca = components_for_variance(embeddings, variance_threshold)
    elapsed = time.perf_counter() - t1

    # Actual explained variance of the chosen k components
    actual_var = float(np.sum(pca.explained_variance_ratio_[:k]))

    log.info(
        "── %s: %d components explain %.4f variance "
        "(target >= %.2f) — PCA fit: %.1f s",
        category, k, actual_var, variance_threshold, elapsed,
    )

    return {
        "Category":          category,
        "N_Components_95pct": k,
        "Variance_Explained": round(actual_var, 6),
        "Total_Patches":     embeddings.shape[0],
        "Feature_Dim":       embeddings.shape[1],
    }


# ── CSV writer ────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "Category",
    "N_Components_95pct",
    "Variance_Explained",
    "Total_Patches",
    "Feature_Dim",
]


def save_results(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Results saved → %s", output_csv)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    # ── Validate dataset root ─────────────────────────────────────────────────
    if not args.dataset_root.is_dir():
        log.error(
            "Dataset root not found: %s\n"
            "Run  python src/convert_visa.py  first to create it.",
            args.dataset_root,
        )
        sys.exit(1)

    missing = [c for c in VISA_CATEGORIES
               if not (args.dataset_root / c).is_dir()]
    if missing:
        log.warning("Missing category directories (will skip): %s", missing)

    # ── Backbone ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    log.info("Loading Wide ResNet-50 backbone …")
    extractor = SymmetryAwareFeatureExtractor(backbone="wide_resnet50_2",
                                              device=device)

    # ── Category loop ─────────────────────────────────────────────────────────
    results: list[dict[str, object]] = []
    t_global = time.perf_counter()

    categories_to_run = [c for c in VISA_CATEGORIES
                         if (args.dataset_root / c).is_dir()]
    log.info("Running PCA analysis on %d categories …", len(categories_to_run))

    for i, category in enumerate(categories_to_run, 1):
        log.info("=" * 60)
        log.info("[%d / %d]  %s", i, len(categories_to_run), category)
        try:
            row = analyse_category(
                dataset_root=args.dataset_root,
                category=category,
                img_size=args.img_size,
                n_shot=args.n_shot,
                support_seed=args.support_seed,
                variance_threshold=args.variance_threshold,
                extractor=extractor,
            )
            results.append(row)
        except Exception as exc:
            log.error("FAILED for '%s': %s", category, exc, exc_info=True)
            # Record failure so we know which category had the problem
            results.append(
                {
                    "Category":           category,
                    "N_Components_95pct": "ERROR",
                    "Variance_Explained": "ERROR",
                    "Total_Patches":      "ERROR",
                    "Feature_Dim":        "ERROR",
                }
            )

    total_elapsed = time.perf_counter() - t_global
    log.info("=" * 60)
    log.info("All categories done in %.1f s (%.1f min).",
             total_elapsed, total_elapsed / 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(args.output_csv, results)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          VisA PCA 95% Variance — Summary             ║")
    print("╠══════════════════════════════════════╦═══════════════╣")
    print(f"║  {'Category':<36} ║  {'Components':>11} ║")
    print("╠══════════════════════════════════════╬═══════════════╣")
    for row in results:
        print(
            f"║  {str(row['Category']):<36} "
            f"║  {str(row['N_Components_95pct']):>11} ║"
        )
    print("╚══════════════════════════════════════╩═══════════════╝")
    print(f"\nFull results saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
