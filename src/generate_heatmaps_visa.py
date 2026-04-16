"""
Generate qualitative heatmap evidence and raw data for VisA.
'Skeleton Key' Edition: Uses recursive rglob to bypass all VisA naming inconsistencies.
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
DATA_DIR = ROOT_DIR / "data" / "visa_mvtec_format"
OUTPUT_DIR = ROOT_DIR / "qualitative_results_visa"

# Experiment Configuration
TARGET_CATEGORIES = ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
N_SHOT = 1
BACKBONE = "wide_resnet50_2"
IMAGE_SIZE = 256

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SquarePad

class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self) -> int: return len(self.image_paths)
    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), 0, str(image_path)

def build_transform() -> T.Compose:
    return T.Compose([
        SquarePad(), 
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def list_image_files(directory: Path) -> list[Path]:
    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(directory.rglob(pattern))
    return sorted(paths)

def build_support_loader(category_root: Path, transform) -> DataLoader:
    support_dir = category_root / "train" / "good"
    support_images = list_image_files(support_dir)
    if not support_images: raise ValueError(f"No images in {support_dir}")
    return DataLoader(ImagePathDataset(support_images[:N_SHOT], transform), batch_size=1, shuffle=False)

def build_test_loader(category_root: Path, transform) -> DataLoader:
    test_dir = category_root / "test"
    test_images = list_image_files(test_dir)
    return DataLoader(ImagePathDataset(test_images, transform), batch_size=1, shuffle=False)

def load_ground_truth_mask(category_root: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    """SKELETON KEY MASK LOADER: Searches recursively for any file matching the image ID."""
    if "good" in image_path.parts:
        return np.zeros(output_size, dtype=np.uint8)
    
    gt_root = category_root / "ground_truth"
    image_id = image_path.stem  # e.g., "000"
    
    # Search for anything starting with the ID in the ground_truth tree
    # This handles 'anomaly/000.png', 'bad/000_mask.png', etc.
    patterns = [f"{image_id}.png", f"{image_id}_mask.png", f"{image_id}*"]
    
    for pattern in patterns:
        matches = list(gt_root.rglob(pattern))
        for match in matches:
            if match.is_file():
                try:
                    mask = Image.open(match).convert("L").resize((output_size[1], output_size[0]), Image.NEAREST)
                    return (np.asarray(mask) > 0).astype(np.uint8)
                except Exception:
                    continue
    
    return np.zeros(output_size, dtype=np.uint8)

def process_category(category: str, transform) -> None:
    category_root = DATA_DIR / category
    print(f"Processing category: {category}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inspector = EnhancedAnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)
    inspector.fit(build_support_loader(category_root, transform), apply_p4m=True)
    
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
            "result": result
        })

# Save Top 3 Heatmaps and raw .npy data for PRO calculation
    top_samples = sorted(scored_samples, key=lambda x: x["result"].image_score, reverse=True)[:3]
    for rank, sample in enumerate(top_samples, start=1):
        plt.figure()
        plt.imshow(sample["image_rgb"])
        overlay = cv2.resize(sample["result"].anomaly_map, (sample["image_rgb"].shape[1], sample["image_rgb"].shape[0]))
        plt.imshow(overlay, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(OUTPUT_DIR / f"{category}_heatmap_rank{rank}.png")
        plt.close()
        np.save(OUTPUT_DIR / f"{category}_heatmap_rank{rank}.npy", sample["result"].anomaly_map.astype(np.float32))
        
        # ---> ADD THIS EXACT LINE <---
        print(f"   Rank {rank} Inference Time: {sample['result'].inference_time_ms:.2f} ms")

def recover_topk_paths_for_category(category: str, k: int) -> list[Path]:
    transform = build_transform()
    category_root = DATA_DIR / category
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inspector = EnhancedAnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)
    inspector.fit(build_support_loader(category_root, transform), apply_p4m=True)

    test_loader = build_test_loader(category_root, transform)
    scored = []
    for images, _, paths in test_loader:
        image_path = Path(paths[0])
        result = inspector.predict(images, apply_p4m=False)[0]
        scored.append((image_path, float(result.image_score)))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [path for path, _ in scored[:k]]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=list(TARGET_CATEGORIES))
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = build_transform()
    
    for category in args.categories:
        process_category(category, transform)

if __name__ == "__main__":
    main()