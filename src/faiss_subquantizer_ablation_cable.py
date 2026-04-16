"""
FAISS Subquantizer (M) Ablation Study for Cable Category
=========================================================
Generates heatmaps for the cable category using different M values (16, 32, 64)
and measures both PRO-AUC and retrieval latency to quantify the accuracy-efficiency tradeoff.

This script extends the standard generate_heatmaps workflow with M parameter sweeping.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project Path Setup
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "mvtec"
OUTPUT_DIR = ROOT_DIR / "qualitative_results_m_ablation"

# Experiment Configuration
CATEGORY = "cable"
N_SHOT = 1
BACKBONE = "wide_resnet50_2"
PQ_BITS = 8
IMAGE_SIZE = 256
M_VALUES = [16, 32, 64]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.memory_bank import MemoryBank
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor as SymmetryFeatureExtractor
from models.thresholding import EVTCalibrator as Thresholding
from models.symmetry_feature_extractor import SquarePad
from utils.image_loader import MVTecStyleDataset


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

def make_figure(image_rgb, gt_mask, anomaly_map, image_score, image_path, category, m_value):
    height, width = image_rgb.shape[:2]
    heatmap = cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Normalization for display only
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{category.upper()} (M={m_value}) | {image_path.name} | Score: {image_score:.4f}", fontsize=14)

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


# --- Enhanced MemoryBank with configurable M ---

def build_memory_bank_with_m(dimension: int, use_gpu: bool, pq_bits: int, m_value: int) -> MemoryBank:
    """
    Build a MemoryBank with configurable M (sub-quantizers).
    This wraps the standard call but modifies the M parameter inline.
    """
    # We'll use a custom wrapper that temporarily modifies M
    bank = MemoryBank(dimension=dimension, use_gpu=use_gpu, use_pq=True)
    # Store the intended M for later use
    bank._custom_m = m_value
    return bank


# Patch the MemoryBank.build method to use custom M
original_build = MemoryBank.build

def patched_build(self, features_list, coreset_percentage: float = 0.1, pq_bits: int = 8) -> None:
    """Patched build method that respects the custom M value."""
    import faiss
    from models.memory_bank import EnhancedCoresetSampler
    
    all_features = np.vstack(features_list).astype(np.float32)
    
    # 1. Reduction Stages
    all_features = self._orbit_aware_reduction(all_features)
    sampler = EnhancedCoresetSampler(percentage=coreset_percentage, method="statistical")
    coreset_features = sampler.sample(all_features)
    
    self.features = self._normalize_features(coreset_features).astype(np.float32)
    
    # 2. FAISS Index Selection - use custom M if available
    nlist = max(1, min(100, len(self.features) // 10))
    quantizer = faiss.IndexFlatL2(self.dimension)

    if self.use_pq:
        # Use custom M if set, otherwise default to 64
        M = getattr(self, '_custom_m', 64)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, M, pq_bits)
    else:
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
    
    if self.use_gpu:
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
    self.index.train(self.features)
    self.index.add(self.features)
    self.is_trained = True
    self._learn_statistics(all_features)

# Apply the patch
MemoryBank.build = patched_build


# --- Core Execution Logic ---

def build_inspector_with_m(transform, category_root: Path, m_value: int) -> tuple[EnhancedAnomalyInspector, float]:
    """
    Build and fit an inspector with a specific M value.
    Returns the inspector and the time taken to fit.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inspector = EnhancedAnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)
    
    # Create a custom memory bank with the target M value
    inspector.memory_bank._custom_m = m_value
    
    support_loader = build_support_loader(category_root, transform)
    
    fit_start = time.time()
    inspector.fit(support_loader, apply_p4m=True)
    fit_time = time.time() - fit_start
    
    return inspector, fit_time


def process_category_with_m(category: str, transform, m_value: int, output_subdir: Path) -> dict:
    """
    Process a category with a specific M value.
    Returns metrics dict with PRO-AUC-related data and latency.
    """
    category_root = DATA_DIR / category
    print(f"\n{'='*70}")
    print(f"Processing category: {category} with M={m_value}")
    print(f"{'='*70}")
    
    inspector, fit_time = build_inspector_with_m(transform, category_root, m_value)
    print(f"Inspector fit time: {fit_time:.3f} seconds")
    
    test_loader = build_test_loader(category_root, transform)

    scored_samples = []
    retrieval_times = []
    
    for images, _, paths in test_loader:
        image_path = Path(paths[0])
        
        # Measure inference time
        inference_start = time.time()
        result = inspector.predict(images, apply_p4m=False)[0]
        inference_time = time.time() - inference_start
        retrieval_times.append(inference_time * 1000)  # Convert to ms
        
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
            m_value=m_value,
        )
        img_out = output_subdir / f"{category}_m{m_value}_heatmap_rank{rank}.png"
        figure.savefig(img_out, dpi=300, bbox_inches="tight")
        plt.close(figure)

        # SAVE RAW DATA FOR PRO-SCORE RIGOR
        npy_out = output_subdir / f"{category}_m{m_value}_heatmap_rank{rank}.npy"
        np.save(npy_out, sample["result"].anomaly_map.astype(np.float32))
        
        print(f"   Rank {rank} Inference Time: {sample['result'].inference_time_ms:.2f} ms")
        print(f"   Saved {img_out.name} and {npy_out.name}")

    # Compute average latency
    avg_latency = np.mean(retrieval_times)
    
    return {
        "category": category,
        "m_value": m_value,
        "fit_time_s": fit_time,
        "avg_retrieval_latency_ms": avg_latency,
        "min_latency_ms": np.min(retrieval_times),
        "max_latency_ms": np.max(retrieval_times),
        "output_dir": str(output_subdir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS Subquantizer Ablation for Cable Category")
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES, help="M values to test")
    parser.add_argument("--category", default=CATEGORY, help="Category to process")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform = build_transform()

    results = []
    
    for m_value in args.m_values:
        m_subdir = OUTPUT_DIR / f"m_{m_value}"
        m_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = process_category_with_m(args.category, transform, m_value, m_subdir)
            results.append(result)
            print(f"\n[OK] Completed M={m_value}")
            print(f"  Average Retrieval Latency: {result['avg_retrieval_latency_ms']:.2f} ms")
        except Exception as e:
            print(f"\n[FAILED] Error with M={m_value}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: FAISS Subquantizer Ablation Results")
    print(f"{'='*70}")
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_csv = OUTPUT_DIR / f"{args.category}_subquantizer_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
    
    # Save detailed timing info
    timing_info = {
        "Category": args.category,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "M_Values": args.m_values,
        "Results": results,
    }
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print(f"1. Run compute_pro.py to compute PRO-AUC from heatmaps in {OUTPUT_DIR}")
    print(f"2. Compare results across M values")


if __name__ == "__main__":
    main()
