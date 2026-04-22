"""Evaluate a recent anomalib baseline under strict few-shot constraints.

This script builds a temporary MVTec category view where the train/good split is
reduced to exactly N support images (default N=10), trains an anomalib model,
evaluates on the full standard test split, and reports image AUROC and average
inference latency per image.
"""

from __future__ import annotations

import argparse
import inspect
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomalib baseline on MVTec with strict few-shot support.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to MVTec root, e.g. data/mvtec")
    parser.add_argument("--category", type=str, required=True, help="MVTec category, e.g. screw, cable")
    parser.add_argument("--n-shot", type=int, default=10, help="Number of normal support images to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for support sampling")
    parser.add_argument(
        "--model",
        type=str,
        default="fastflow",
        choices=["fastflow", "simplenet", "cfa"],
        help="Contemporaneous anomalib baseline to evaluate",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Train/eval batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def _list_images(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(directory.glob(pattern))
    return sorted(paths)


def sample_support_images(category_dir: Path, n_shot: int, seed: int) -> list[Path]:
    train_good_dir = category_dir / "train" / "good"
    if not train_good_dir.exists():
        raise FileNotFoundError(f"Missing train/good directory: {train_good_dir}")

    all_train_images = _list_images(train_good_dir)
    if len(all_train_images) < n_shot:
        raise ValueError(f"Requested n-shot={n_shot}, but only found {len(all_train_images)} images in {train_good_dir}")

    rng = random.Random(seed)
    selected = rng.sample(all_train_images, n_shot)
    if len(selected) != n_shot:
        raise RuntimeError(f"Few-shot constraint violated: expected {n_shot}, got {len(selected)}")
    return sorted(selected)


def materialize_few_shot_category(category_dir: Path, selected_support: list[Path], temp_root: Path) -> Path:
    category = category_dir.name
    dest_category_dir = temp_root / category
    dest_train_good = dest_category_dir / "train" / "good"
    dest_train_good.mkdir(parents=True, exist_ok=True)

    # Copy exactly N support images and discard the rest.
    for path in selected_support:
        shutil.copy2(path, dest_train_good / path.name)

    # Keep the original full MVTec test split for evaluation.
    shutil.copytree(category_dir / "test", dest_category_dir / "test", dirs_exist_ok=True)

    ground_truth_src = category_dir / "ground_truth"
    if ground_truth_src.exists():
        shutil.copytree(ground_truth_src, dest_category_dir / "ground_truth", dirs_exist_ok=True)

    return dest_category_dir


def _instantiate_with_supported_kwargs(cls: type, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(cls)
    accepted = {name for name in signature.parameters.keys()}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


def build_model(model_name: str):
    from anomalib import models as anomalib_models

    candidates: dict[str, tuple[str, ...]] = {
        "fastflow": ("Fastflow", "FastFlow"),
        "simplenet": ("SimpleNet",),
        "cfa": ("Cfa", "CFA"),
    }

    class_name_candidates = candidates[model_name]
    model_cls = None
    for class_name in class_name_candidates:
        if hasattr(anomalib_models, class_name):
            model_cls = getattr(anomalib_models, class_name)
            break

    if model_cls is None:
        raise ImportError(f"Could not find anomalib model class for '{model_name}' in anomalib.models")

    # Keep defaults close to common anomalib configs, with edge-friendly image size.
    model_kwargs = {
        "input_size": (256, 256),
        "backbone": "resnet18",
    }
    return _instantiate_with_supported_kwargs(model_cls, model_kwargs)


def build_dataloaders(mvtec_root: Path, category: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    from anomalib.data.datasets.image.mvtecad import MVTecADDataset

    train_dataset = MVTecADDataset(root=mvtec_root, category=category, split="train")
    test_dataset = MVTecADDataset(root=mvtec_root, category=category, split="test")

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after few-shot materialization")
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty for the target category")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn,
    )
    return train_loader, test_loader


def _extract_image_auroc(test_results: Any) -> float:
    candidate_keys = {
        "image_auroc",
        "image_AUROC",
        "image-auroc",
        "test_image_auroc",
        "test_image_AUROC",
    }

    def _walk(obj: Any) -> float | None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in candidate_keys and isinstance(value, (float, int)):
                    return float(value)
            for value in obj.values():
                found = _walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    result = _walk(test_results)
    if result is None:
        raise RuntimeError(f"Could not find Image AUROC in anomalib test output: {test_results}")
    return result


def count_test_images(category_dir: Path) -> int:
    test_dir = category_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    total = 0
    for subdir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        total += len(_list_images(subdir))
    if total == 0:
        raise RuntimeError(f"No test images found in {test_dir}")
    return total


def evaluate_baseline(
    dataset_root: Path,
    category: str,
    n_shot: int,
    seed: int,
    model_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, float | int | str]:
    from anomalib.engine import Engine

    category_dir = dataset_root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    selected_support = sample_support_images(category_dir, n_shot=n_shot, seed=seed)

    with tempfile.TemporaryDirectory(prefix=f"fewshot_{category}_") as temp_dir_str:
        temp_root = Path(temp_dir_str)
        temp_category_dir = materialize_few_shot_category(category_dir, selected_support, temp_root)
        train_loader, test_loader = build_dataloaders(
            mvtec_root=temp_root,
            category=category,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        model = build_model(model_name)

        engine = Engine(
            accelerator="cpu",
            devices=1,
            max_epochs=20,
            logger=False,
        )
        engine.fit(model=model, train_dataloaders=train_loader)

        test_results = engine.test(model=model, dataloaders=test_loader)
        image_auroc = _extract_image_auroc(test_results)

        num_test_images = count_test_images(temp_category_dir)
        start = time.perf_counter()
        _ = engine.predict(model=model, dataloaders=test_loader)
        elapsed = time.perf_counter() - start
        latency_ms_per_image = (elapsed / num_test_images) * 1000.0

    return {
        "category": category,
        "model": model_name,
        "n_shot": n_shot,
        "num_test_images": num_test_images,
        "image_auroc": float(image_auroc),
        "avg_latency_ms_per_image": float(latency_ms_per_image),
    }


def main() -> None:
    args = parse_args()

    metrics = evaluate_baseline(
        dataset_root=args.dataset_path,
        category=args.category,
        n_shot=args.n_shot,
        seed=args.seed,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Few-Shot Baseline Evaluation")
    print(f"Category: {metrics['category']}")
    print(f"Model: {metrics['model']}")
    print(f"N-shot support: {metrics['n_shot']}")
    print(f"Test images: {metrics['num_test_images']}")
    print(f"Image AUROC: {metrics['image_auroc']:.6f}")
    print(f"Average inference latency per image (ms): {metrics['avg_latency_ms_per_image']:.6f}")


if __name__ == "__main__":
    main()