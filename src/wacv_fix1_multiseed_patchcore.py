"""
wacv_fix1_multiseed_patchcore.py
================================
WACV FIX-1 — Multi-Seed PatchCore Baseline.

Reviewer concern:
    "Baseline comparison is statistically unreliable (single seed)."

This script runs the anomalib PatchCore baseline with 5 seeds across all
15 MVTec categories, reporting per-category Image AUROC (mean ± std).

Seeds: [0, 1, 42, 123, 999]
Protocol: N=10, WRN-50 backbone, 256×256, identical preprocessing, CPU

Output:
    results/wacv/fix1_patchcore_multiseed.csv

Run from the project root:
    python src/wacv_fix1_multiseed_patchcore.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

# os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "wacv" / "fix1_patchcore_multiseed.csv"
SEEDS = [0, 1, 42, 123, 999]
N_SHOT = 10
IMAGE_SIZE = 256

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

FIELDNAMES = [
    "category", "seed", "image_auroc", "inference_latency_ms",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_single(
    category: str, seed: int, dataset_root: Path, accelerator: str
) -> dict:
    """Run a single PatchCore evaluation for one category and seed."""
    # Import anomalib lazily to avoid import errors if not installed
    try:
        from anomalib import models as anomalib_models
        from anomalib.data.datasets.image.mvtecad import MVTecADDataset
        from anomalib.engine import Engine
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "anomalib is required for the PatchCore baseline. "
            "Install it with: pip install anomalib"
        ) from exc

    seed_everything(seed)

    # Build datasets
    train_dataset = MVTecADDataset(root=dataset_root, category=category, split="train")
    test_dataset = MVTecADDataset(root=dataset_root, category=category, split="test")

    # Sample N support images
    if N_SHOT > len(train_dataset):
        raise ValueError(
            f"N_SHOT={N_SHOT} > available train images ({len(train_dataset)})"
        )
    indices = random.Random(seed).sample(range(len(train_dataset)), N_SHOT)
    support_dataset = train_dataset.subsample(sorted(indices))

    train_loader = DataLoader(
        dataset=support_dataset,
        batch_size=min(16, len(support_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=support_dataset.collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=min(16, len(test_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
    )

    # Build PatchCore model (WRN-50 backbone, same as our pipeline)
    model = anomalib_models.Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        precision="float32",
        pre_processor=anomalib_models.Patchcore.configure_pre_processor(
            image_size=(IMAGE_SIZE, IMAGE_SIZE)
        ),
    )

    engine = Engine(
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=PROJECT_ROOT / "results" / "tmp" / f"patchcore_{seed}_{category}",
    )
    engine.fit(model=model, train_dataloaders=train_loader)

    start_time = time.perf_counter()
    predictions = engine.predict(model=model, dataloaders=test_loader)
    elapsed = time.perf_counter() - start_time

    # Compute AUROC
    scores, labels = [], []
    for batch in predictions:
        if batch.pred_score is not None and batch.gt_label is not None:
            scores.extend(batch.pred_score.detach().cpu().numpy().flatten().tolist())
            labels.extend(batch.gt_label.detach().cpu().numpy().astype(int).flatten().tolist())

    if len(set(labels)) < 2:
        raise ValueError(f"AUROC requires both classes for {category}")

    img_auroc = float(roc_auc_score(labels, scores))
    avg_latency_ms = (elapsed / len(test_dataset)) * 1000.0

    return {
        "category": category,
        "seed": seed,
        "image_auroc": round(img_auroc, 6),
        "inference_latency_ms": round(avg_latency_ms, 4),
    }


# ── CLI + main ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WACV FIX-1: Multi-seed PatchCore baseline."
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=MVTEC_ROOT,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
    )
    parser.add_argument(
        "--accelerator", type=str, default="cpu", choices=["cpu", "auto", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = len(args.seeds) * len(CATEGORIES)
    log.info("WACV FIX-1: Multi-Seed PatchCore Baseline")
    log.info("Seeds: %s | Categories: %d | Total evals: %d", args.seeds, len(CATEGORIES), total)

    all_results = []
    counter = 0

    for seed in args.seeds:
        for category in CATEGORIES:
            counter += 1
            log.info("[%d/%d] seed=%d category=%s", counter, total, seed, category)
            try:
                result = evaluate_single(
                    category, seed, args.dataset_path, args.accelerator
                )
                all_results.append(result)
                log.info("  → AUROC=%.4f  Lat=%.2fms", result["image_auroc"], result["inference_latency_ms"])
            except Exception as exc:
                log.error("  FAILED: %s", exc)

    # Save CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Saved → %s", args.output_csv)

    # Console summary: per-category mean ± std across seeds
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║    FIX-1: PatchCore Baseline — Mean ± Std (5 seeds)   ║")
    print("╠═══════════════╦════════════════════════════════════════╣")
    print(f"║ {'Category':<13} ║ {'Image AUROC (mean ± std)':>38} ║")
    print("╠═══════════════╬════════════════════════════════════════╣")
    overall_means = []
    for cat in CATEGORIES:
        cat_results = [r["image_auroc"] for r in all_results if r["category"] == cat]
        if cat_results:
            m, s = np.mean(cat_results), np.std(cat_results)
            overall_means.append(m)
            print(f"║ {cat:<13} ║ {m:>18.4f} ± {s:<17.4f} ║")
    print("╠═══════════════╬════════════════════════════════════════╣")
    if overall_means:
        om, os_ = np.mean(overall_means), np.std(overall_means)
        print(f"║ {'OVERALL':<13} ║ {om:>18.4f} ± {os_:<17.4f} ║")
    print("╚═══════════════╩════════════════════════════════════════╝")
    print("\nFIX-1 COMPLETE.")


if __name__ == "__main__":
    main()
