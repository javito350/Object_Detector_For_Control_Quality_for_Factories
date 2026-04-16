"""Generate qualitative heatmap evidence and raw data for the MVTec hero categories.

This script builds the winning inspection pipeline used in the experiments:
Wide ResNet-50 backbone, 8-bit FAISS PQ, and p4m support-set augmentation.
It saves both .png visualizations and .npy raw anomaly maps for PRO-score calculation.
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

# Project Path Setup
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "mvtec"
OUTPUT_DIR = ROOT_DIR / "qualitative_results"

# Experiment Configuration
TARGET_CATEGORIES = ("cable", "capsule", "screw")
N_SHOT = 1
BACKBONE = "wide_resnet50_2"
PQ_BITS = 8
IMAGE_SIZE = 256

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SquarePad

# --- Dataset Utilities ---

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
    return T.Compose([
        SquarePad(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def list_image_files(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(directory.glob(pattern))
    return sorted(paths)

def build_support_loader(category_root: Path, transform) -> DataLoader:
    support_dir = category_root / "train" / "good"
    support_images = list_image_files(support_dir)
    if len(support_images) < N_SHOT:
        raise ValueError(f"Need {N_SHOT} images in {support_dir}, found {len(support_images)}")
    dataset = ImagePathDataset(support_images[:N_SHOT], transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

def build_test_loader(category_root: Path, transform) -> DataLoader:
    test_dir = category_root / "test"
    test_images: list[Path] = []
    for defect_dir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
        test_images.extend(list_image_files(defect_dir))
    dataset = ImagePathDataset(test_images, transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

def load_ground_truth_mask(category_root: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)
    defect_type = image_path.parent.name
    mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    mask = Image.open(mask_path).convert("L").resize((output_size[1], output_size[0]), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)

# --- Visualization Logic ---

def make_figure(image_rgb, gt_mask, anomaly_map, image_score, image_path, category):
    height, width = image_rgb.shape[:2]
    heatmap = cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Normalization for display only
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(heatmap)

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

# --- Core Execution Logic ---

def build_inspector(transform, category_root: Path) -> EnhancedAnomalyInspector:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inspector = EnhancedAnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)
    support_loader = build_support_loader(category_root, transform)
    inspector.fit(support_loader, apply_p4m=True)
    return inspector

def process_category(category: str, transform) -> None:
    category_root = DATA_DIR / category
    print(f"Processing category: {category}")
    inspector = build_inspector(transform, category_root)
    test_loader = build_test_loader(category_root, transform)

    scored_samples = []
    for images, _, paths in test_loader:
        image_path = Path(paths[0])
        result = inspector.predict(images, apply_p4m=False)[0]
        original_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        gt_mask = load_ground_truth_mask(category_root, image_path, original_rgb.shape[:2])

        scored_samples.append({
            "image_path": image_path,
            "image_rgb": original_rgb,
            "gt_mask": gt_mask,
            "result": result,
        })

    # Select Top-3 samples by anomaly score
    top_samples = sorted(scored_samples, key=lambda x: x["result"].image_score, reverse=True)[:3]

    for rank, sample in enumerate(top_samples, start=1):
        # Generate and save visual plot
        figure = make_figure(
            image_rgb=sample["image_rgb"],
            gt_mask=sample["gt_mask"],
            anomaly_map=sample["result"].anomaly_map,
            image_score=sample["result"].image_score,
            image_path=sample["image_path"],
            category=category,
        )
        img_out = OUTPUT_DIR / f"{category}_heatmap_rank{rank}.png"
        figure.savefig(img_out, dpi=300, bbox_inches="tight")
        plt.close(figure)

        # SAVE RAW DATA FOR PRO-SCORE RIGOR
        npy_out = OUTPUT_DIR / f"{category}_heatmap_rank{rank}.npy"
        np.save(npy_out, sample["result"].anomaly_map.astype(np.float32))
        
        # ---> THIS IS THE MAGIC LINE <---
        print(f"   Rank {rank} Inference Time: {sample['result'].inference_time_ms:.2f} ms")
        
        print(f"   Saved {img_out.name} and {npy_out.name}")

def recover_topk_paths_for_category(category: str, k: int) -> list[Path]:
    """Rebuild the top-k scored image paths for a category."""
    transform = build_transform()
    category_root = DATA_DIR / category
    inspector = build_inspector(transform, category_root)
    test_loader = build_test_loader(category_root, transform)

    scored: list[tuple[Path, float]] = []
    for images, _, paths in test_loader:
        image_path = Path(paths[0])
        result = inspector.predict(images, apply_p4m=False)[0]
        scored.append((image_path, float(result.image_score)))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [path for path, _ in scored[:k]]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heatmaps and raw data for MVTec hero categories.")
    parser.add_argument("--categories", nargs="+", default=list(TARGET_CATEGORIES))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = build_transform()

    for category in args.categories:
        process_category(category, transform)

if __name__ == "__main__":
    main()