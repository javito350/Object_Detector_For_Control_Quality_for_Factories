"""
NeurIPS-style ablation study: remove p4m symmetry augmentation from the support set.

This script runs a controlled MVTec AD evaluation using the same 8-bit FAISS PQ
retrieval pipeline as the main model, but disables p4m rotations/reflections
during support-set feature extraction. The goal is to isolate the contribution
of geometric symmetry expansion to anomaly-detection performance.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.anomaly_inspector import EnhancedAnomalyInspector as AnomalyInspector
from models.memory_bank import MemoryBank
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor as SymmetryFeatureExtractor
from models.thresholding import EVTCalibrator as Thresholding
from utils.image_loader import MVTecStyleDataset


ROOT_DIR = Path(__file__).resolve().parent
MVTEC_ROOT = ROOT_DIR / "data" / "mvtec"
OUTPUT_CSV = ROOT_DIR / "results_ablation_no_p4m.csv"

# Controlled pipeline settings:
# - ResNet-50 family backbone used by this repository (`wide_resnet50_2`)
# - layer2 (512) + layer3 (1024) patch descriptors = 1536-D
# - 8-bit product quantization for the FAISS retrieval backend
BACKBONE = "wide_resnet50_2"
FEATURE_DIM = 1536
PQ_BITS = 8
N_SHOT = 10

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def build_train_dataloader(category: str) -> DataLoader:
    """Load the 10 normal support images used in the few-shot memory bank."""
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=True,
        img_size=256,
    )
    dataset.samples = dataset.samples[:N_SHOT]
    return DataLoader(dataset, batch_size=1, shuffle=False)


def build_test_dataloader(category: str) -> DataLoader:
    """Load the full MVTec test split for a given category."""
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=False,
        img_size=256,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def extract_support_features_without_p4m(
    dataloader: DataLoader,
    feature_extractor: SymmetryFeatureExtractor,
    device: str,
) -> list[np.ndarray]:
    """
    Extract support features without any p4m expansion.

    This is the ablation itself: the support memory bank sees only the original
    10 images, with no rotations and no reflections.
    """
    feature_batches: list[np.ndarray] = []

    for images, _, _ in tqdm(dataloader, desc="Extracting support features", leave=False):
        images = images.to(device)
        features = feature_extractor.extract_patch_features(images, apply_p4m=False)
        feature_batches.append(features)

    if not feature_batches:
        raise ValueError("No normal training images were found for this category.")

    return feature_batches


def fit_evt_thresholds(
    memory_bank: MemoryBank,
    thresholding: Thresholding,
    feature_batches: list[np.ndarray],
) -> tuple[float, float]:
    """
    Fit EVT thresholds on support distances, matching the main evaluation logic.
    """
    training_features = np.vstack(feature_batches).astype(np.float32)
    distances, _ = memory_bank.query(training_features, k=1)
    image_threshold = float(thresholding.fit(distances.flatten()))
    pixel_threshold = image_threshold * 0.9
    return image_threshold, pixel_threshold


def load_ground_truth_mask(category_path: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray | None:
    """Load the binary defect mask aligned to the predicted anomaly-map size."""
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)

    defect_type = image_path.parent.name
    mask_name = f"{image_path.stem}_mask.png"
    mask_path = category_path / "ground_truth" / defect_type / mask_name

    if not mask_path.exists():
        return None

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((output_size[1], output_size[0]), Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def run_inference(
    inspector: AnomalyInspector,
    images: torch.Tensor,
    pixel_threshold: float,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], float]:
    """
    Run test-time inference.

    Inference remains unaugmented, matching the standard retrieval pipeline.
    Retrieval latency measures only the FAISS `query()` call.
    """
    images = images.to(inspector.device)
    batch_size, _, height, width = images.shape

    with torch.no_grad():
        patch_features = inspector.feature_extractor.extract_patch_features(images, apply_p4m=False)

        retrieval_start = time.perf_counter()
        distances, _ = inspector.memory_bank.query(patch_features, k=1)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

    patch_grid = distances.reshape(batch_size, 28, 28)
    image_scores: list[float] = []
    anomaly_maps: list[np.ndarray] = []
    binary_masks: list[np.ndarray] = []

    for idx in range(batch_size):
        image_score = float(np.max(patch_grid[idx]))
        anomaly_map = cv2.resize(patch_grid[idx], (width, height), interpolation=cv2.INTER_CUBIC)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        binary_mask = (anomaly_map > pixel_threshold).astype(np.uint8)

        image_scores.append(image_score)
        anomaly_maps.append(anomaly_map)
        binary_masks.append(binary_mask)

    return image_scores, anomaly_maps, binary_masks, retrieval_latency_ms


def evaluate_category(category: str) -> dict:
    """
    Evaluate one MVTec category under the no-p4m ablation condition.
    """
    category_path = MVTEC_ROOT / category
    if not category_path.exists():
        raise FileNotFoundError(f"Category path not found: {category_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = SymmetryFeatureExtractor(backbone=BACKBONE, device=device)
    memory_bank = MemoryBank(dimension=FEATURE_DIM, use_gpu=(device == "cuda"), use_pq=True)
    thresholding = Thresholding(tail_fraction=0.10, target_fpr=0.01)
    inspector = AnomalyInspector(backbone=BACKBONE, device=device, use_pq=True)

    # Keep the pipeline components synchronized with our controlled ablation setup.
    inspector.feature_extractor = feature_extractor
    inspector.memory_bank = memory_bank

    train_loader = build_train_dataloader(category)
    test_loader = build_test_dataloader(category)

    # Core ablation: build the support memory bank with NO p4m augmentation.
    support_features = extract_support_features_without_p4m(train_loader, feature_extractor, device)
    memory_bank.build(
        support_features,
        coreset_percentage=inspector.coreset_percentage,
        pq_bits=PQ_BITS,
    )

    image_threshold, pixel_threshold = fit_evt_thresholds(memory_bank, thresholding, support_features)
    inspector.image_threshold = image_threshold
    inspector.pixel_threshold = pixel_threshold

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    retrieval_latencies: list[float] = []

    for images, labels, paths in tqdm(test_loader, desc="Running test inference", leave=False):
        batch_scores, batch_maps, _, retrieval_latency_ms = run_inference(
            inspector,
            images,
            pixel_threshold=pixel_threshold,
        )
        retrieval_latencies.append(retrieval_latency_ms)

        for idx, image_path_str in enumerate(paths):
            image_path = Path(image_path_str)
            image_scores.append(batch_scores[idx])
            image_labels.append(int(labels[idx].item()))

            gt_mask = load_ground_truth_mask(category_path, image_path, batch_maps[idx].shape)
            if gt_mask is not None:
                pixel_scores.append(batch_maps[idx].flatten())
                pixel_labels.append(gt_mask.flatten())

    if len(set(image_labels)) < 2:
        raise ValueError("Image-level AUROC requires both nominal and anomalous test samples.")

    if not pixel_scores or len(np.unique(np.concatenate(pixel_labels))) < 2:
        raise ValueError("Pixel-level AUROC requires valid ground-truth masks for both classes.")

    return {
        "category": category,
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))),
        "retrieval_latency_ms": float(np.mean(retrieval_latencies)),
    }


def main() -> None:
    results: list[dict] = []

    print("STARTING P4M ABLATION: no p4m support augmentation")
    print(f"Backbone: {BACKBONE} | Feature dim: {FEATURE_DIM} | PQ bits: {PQ_BITS} | Shots: {N_SHOT}")

    for category in CATEGORIES:
        print(f"\nEvaluating category: {category}")
        print(f"Data path: data/mvtec/{category}")

        try:
            category_result = evaluate_category(category)
            results.append(category_result)
            print(
                f"Completed {category} | "
                f"Image AUROC: {category_result['image_auroc']:.4f} | "
                f"Pixel AUROC: {category_result['pixel_auroc']:.4f} | "
                f"Retrieval Latency: {category_result['retrieval_latency_ms']:.4f} ms"
            )
        except Exception as exc:
            print(f"Error while evaluating {category}: {exc}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved ablation results to {OUTPUT_CSV}")
    if not results_df.empty:
        print(results_df)


if __name__ == "__main__":
    main()
