"""Evaluate anomalib PatchCore on MVTec under a strict few-shot constraint."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from anomalib import models as anomalib_models
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.engine import Engine
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "mvtec"
DEFAULT_IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict few-shot PatchCore baseline on MVTec.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_ROOT, help="Path to data/mvtec")
    parser.add_argument("--category", type=str, required=True, help="MVTec category, e.g. screw")
    parser.add_argument("--n-shot", type=int, default=10, help="Number of normal support images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for support sampling")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="PatchCore input size")
    parser.add_argument("--batch-size", type=int, default=16, help="Train/test batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "auto", "cuda"], help="Trainer accelerator")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_support_indices(dataset: MVTecADDataset, n_shot: int, seed: int) -> list[int]:
    if n_shot > len(dataset):
        raise ValueError(f"Requested n-shot={n_shot}, but only {len(dataset)} train images are available")

    indices = random.Random(seed).sample(range(len(dataset)), n_shot)
    if len(indices) != n_shot:
        raise RuntimeError(f"Few-shot constraint violated: expected {n_shot}, got {len(indices)}")
    return sorted(indices)


def build_dataloaders(
    dataset_root: Path,
    category: str,
    n_shot: int,
    seed: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[MVTecADDataset, DataLoader, MVTecADDataset, DataLoader]:
    train_dataset = MVTecADDataset(root=dataset_root, category=category, split="train")
    test_dataset = MVTecADDataset(root=dataset_root, category=category, split="test")

    support_indices = sample_support_indices(train_dataset, n_shot=n_shot, seed=seed)
    support_dataset = train_dataset.subsample(support_indices)

    train_loader = DataLoader(
        dataset=support_dataset,
        batch_size=min(batch_size, len(support_dataset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=support_dataset.collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    return support_dataset, train_loader, test_dataset, test_loader


def build_patchcore(image_size: int) -> object:
    return anomalib_models.Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        precision="float32",
        pre_processor=anomalib_models.Patchcore.configure_pre_processor(image_size=(image_size, image_size)),
    )


def compute_image_auroc(predictions: list) -> float:
    scores: list[float] = []
    labels: list[int] = []

    for batch in predictions:
        batch_scores = batch.pred_score
        batch_labels = batch.gt_label
        if batch_scores is None or batch_labels is None:
            raise RuntimeError("PatchCore predictions did not include score or label fields")

        scores.extend(batch_scores.detach().cpu().numpy().reshape(-1).tolist())
        labels.extend(batch_labels.detach().cpu().numpy().astype(int).reshape(-1).tolist())

    if len(set(labels)) < 2:
        raise RuntimeError("Image AUROC requires both nominal and anomalous test samples")

    return float(roc_auc_score(labels, scores))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    _, train_loader, test_dataset, test_loader = build_dataloaders(
        dataset_root=args.dataset_path,
        category=args.category,
        n_shot=args.n_shot,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_patchcore(image_size=args.image_size)
    engine = Engine(accelerator=args.accelerator, devices=1, max_epochs=1, logger=False)

    engine.fit(model=model, train_dataloaders=train_loader)

    start_time = time.perf_counter()
    predictions = engine.predict(model=model, dataloaders=test_loader)
    elapsed_seconds = time.perf_counter() - start_time

    image_auroc = compute_image_auroc(predictions)
    avg_latency_ms = (elapsed_seconds / len(test_dataset)) * 1000.0

    print("PatchCore Baseline")
    print(f"Category: {args.category}")
    print(f"Support images: {args.n_shot}")
    print(f"Image AUROC: {image_auroc:.6f}")
    print(f"Average inference latency (ms/image): {avg_latency_ms:.6f}")


if __name__ == "__main__":
    main()