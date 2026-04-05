"""Generate qualitative heatmap evidence for the MVTec hero categories.

This script builds the winning inspection pipeline used in the experiments:
Wide ResNet-50 backbone, 8-bit FAISS PQ, and p4m support-set augmentation
with N=1 support image per category (1-shot).

For each target category, it scores the full test split, selects the top 3
images with the highest image-level anomaly scores, and writes a 1x3 figure
per image to qualitative_results/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "mvtec"
OUTPUT_DIR = ROOT_DIR / "qualitative_results"

TARGET_CATEGORIES = ("cable", "capsule", "screw")
N_SHOT = 1
BACKBONE = "wide_resnet50_2"
PQ_BITS = 8
IMAGE_SIZE = 256

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SquarePad


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, 0, str(image_path)


def build_transform() -> T.Compose:
    return T.Compose(
        [
            SquarePad(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def list_image_files(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(directory.glob(pattern))
    return sorted(paths)


def build_support_loader(category_root: Path, transform) -> DataLoader:
    support_dir = category_root / "train" / "good"
    support_images = list_image_files(support_dir)
    if len(support_images) < N_SHOT:
        raise ValueError(f"Need at least {N_SHOT} support images in {support_dir}, found {len(support_images)}")

    dataset = ImagePathDataset(support_images[:N_SHOT], transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def build_test_loader(category_root: Path, transform) -> DataLoader:
    test_dir = category_root / "test"
    test_images: list[Path] = []
    for defect_dir in sorted([path for path in test_dir.iterdir() if path.is_dir()]):
        test_images.extend(list_image_files(defect_dir))

    if not test_images:
        raise ValueError(f"No test images found in {test_dir}")

    dataset = ImagePathDataset(test_images, transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def load_ground_truth_mask(category_root: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)

    defect_type = image_path.parent.name
    mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing ground-truth mask for {image_path.name}: {mask_path}")

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((output_size[1], output_size[0]), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def make_figure(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    anomaly_map: np.ndarray,
    image_score: float,
    image_path: Path,
    category: str,
):
    height, width = image_rgb.shape[:2]
    heatmap = cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_min = float(np.min(heatmap))
    heatmap_max = float(np.max(heatmap))
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{category.upper()} | {image_path.name} | Score: {image_score:.4f}", fontsize=14)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original RGB Test Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Binary Mask")
    axes[1].axis("off")

    axes[2].imshow(image_rgb)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[2].set_title("Predicted Anomaly Heatmap")
    axes[2].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def build_inspector(transform, category_root: Path) -> EnhancedAnomalyInspector:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inspector = EnhancedAnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)
    support_loader = build_support_loader(category_root, transform)
    inspector.fit(support_loader, apply_p4m=True)
    return inspector


def process_category(category: str, transform) -> None:
    category_root = DATA_DIR / category
    if not category_root.exists():
        raise FileNotFoundError(f"Category path not found: {category_root}")

    print(f"Processing category: {category}")
    inspector = build_inspector(transform, category_root)
    test_loader = build_test_loader(category_root, transform)

    scored_samples: list[dict] = []

    for images, _, paths in test_loader:
        image_path = Path(paths[0])
        result = inspector.predict(images, apply_p4m=False)[0]

        original_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        gt_mask = load_ground_truth_mask(category_root, image_path, original_rgb.shape[:2])

        scored_samples.append(
            {
                "image_path": image_path,
                "image_rgb": original_rgb,
                "gt_mask": gt_mask,
                "result": result,
            }
        )

    top_samples = sorted(scored_samples, key=lambda item: item["result"].image_score, reverse=True)[:3]
    if len(top_samples) < 3:
        raise ValueError(f"Expected at least 3 test images for {category}, found {len(top_samples)}")

    for rank, sample in enumerate(top_samples, start=1):
        figure = make_figure(
            image_rgb=sample["image_rgb"],
            gt_mask=sample["gt_mask"],
            anomaly_map=sample["result"].anomaly_map,
            image_score=sample["result"].image_score,
            image_path=sample["image_path"],
            category=category,
        )
        output_path = OUTPUT_DIR / f"{category}_heatmap_rank{rank}.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"  Saved {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top anomaly heatmaps for the hero MVTec categories.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(TARGET_CATEGORIES),
        choices=list(TARGET_CATEGORIES),
        help="Categories to process (default: cable capsule screw)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = build_transform()

    for category in args.categories:
        process_category(category, transform)


if __name__ == "__main__":
    main()