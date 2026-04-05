"""CPU profiling utility for offline commissioning vs online inference latency.

This script is intentionally split into two strict phases:
1) offline_commissioning_phase: heavy one-time setup
2) online_inference_phase: per-image live latency
"""

import argparse
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor


ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data" / "water_bottles"


def apply_p4m_symmetry(images: torch.Tensor) -> torch.Tensor:
    """Generate 8 orthogonal p4m transforms for each image in a batch.

    Returns a tensor with shape (8N, C, H, W) for input shape (N, C, H, W).
    """
    if images.ndim != 4:
        raise ValueError("Expected shape (N, C, H, W) for p4m augmentation")

    rotations = [
        images,
        torch.rot90(images, k=1, dims=(2, 3)),
        torch.rot90(images, k=2, dims=(2, 3)),
        torch.rot90(images, k=3, dims=(2, 3)),
    ]

    reflected = torch.flip(images, dims=(3,))
    reflected_rotations = [
        reflected,
        torch.rot90(reflected, k=1, dims=(2, 3)),
        torch.rot90(reflected, k=2, dims=(2, 3)),
        torch.rot90(reflected, k=3, dims=(2, 3)),
    ]

    return torch.cat(rotations + reflected_rotations, dim=0)


class ProfileFeatureModel:
    """Adapter that exposes extract_features(images) for profiling."""

    def __init__(self, extractor: SymmetryAwareFeatureExtractor):
        self.extractor = extractor
        self.extractor.eval()
        self.device = getattr(extractor, "device", "cpu")

    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        with torch.no_grad():
            symmetry_features = self.extractor.extract_symmetry_features(images)

            # Use deterministic key ordering to keep feature vectors stable.
            keys = sorted(symmetry_features.keys())
            pooled: List[torch.Tensor] = []
            for key in keys:
                pooled.append(F.adaptive_avg_pool2d(symmetry_features[key], output_size=(1, 1)).flatten(1))

            deep_feature = torch.cat(pooled, dim=1)
            symmetry_score = self.extractor.compute_symmetry_consistency(symmetry_features).unsqueeze(1)
            embedding = torch.cat((deep_feature, symmetry_score), dim=1)

        return embedding.detach().cpu().numpy().astype(np.float32)


class FaissIndexAdapter:
    """Small FAISS wrapper with add/search/ntotal interface."""

    def __init__(self, dimension: int, index_type: str = "ivf_flat"):
        self.dimension = dimension
        self.index_type = index_type
        self._index = None

    @property
    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def add(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("FAISS expects features with shape (N, D)")
        if features.shape[1] != self.dimension:
            raise ValueError(f"Feature dim mismatch: expected {self.dimension}, got {features.shape[1]}")

        if self.index_type == "flat":
            if self._index is None:
                self._index = faiss.IndexFlatL2(self.dimension)
            self._index.add(features)
            return

        if self._index is None:
            # IVF needs enough training vectors; fallback to Flat for tiny commissioning sets.
            if features.shape[0] < 64:
                self._index = faiss.IndexFlatL2(self.dimension)
            else:
                nlist = max(1, min(16, features.shape[0] // 32))
                quantizer = faiss.IndexFlatL2(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)

        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            self._index.train(features)

        self._index.add(features)

    def search(self, query_feature: np.ndarray, k: int = 1):
        if self._index is None:
            raise ValueError("Index is empty. Run offline commissioning first.")

        query_feature = np.asarray(query_feature, dtype=np.float32)
        if query_feature.ndim == 1:
            query_feature = query_feature.reshape(1, -1)

        return self._index.search(query_feature, k)


def offline_commissioning_phase(normal_images, model, index):
    """The heavy lifting. Done once per product line."""
    print("Starting Offline Commissioning...")
    start_time = time.perf_counter()

    augmented_set = apply_p4m_symmetry(normal_images)
    features = model.extract_features(augmented_set)
    index.add(features)

    end_time = time.perf_counter()
    total_time_s = end_time - start_time
    print(f"Offline Setup Complete. Total Time: {total_time_s:.4f} seconds")
    print(f"Memory Bank Size: {index.ntotal} vectors")
    return index, total_time_s


def online_inference_phase(test_image, model, index):
    """The live assembly line. Must be lightning fast."""
    # --- TIMING BLOCK 1: Feature Extraction ---
    t0 = time.perf_counter()
    query_feature = model.extract_features(test_image)
    t1 = time.perf_counter()
    extraction_time = (t1 - t0) * 1000.0

    # --- TIMING BLOCK 2: Memory Retrieval ---
    t2 = time.perf_counter()
    distances, indices = index.search(query_feature, k=1)
    t3 = time.perf_counter()
    retrieval_time = (t3 - t2) * 1000.0

    anomaly_score = float(distances[0][0])
    return anomaly_score, extraction_time, retrieval_time


def load_image_tensor(image_path: Path, transform, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def collect_image_paths(folder: Path, limit: int) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths: List[Path] = []
    for ext in exts:
        paths.extend(folder.glob(ext))
    # Windows file systems are case-insensitive, so de-duplicate by normalized path.
    dedup = {}
    for path in paths:
        dedup[str(path).lower()] = path

    unique_paths = sorted(dedup.values())
    return unique_paths[:limit]


def load_feature_extractor(device: str):
    torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

    preferred_weights = [
        WEIGHTS_DIR / "calibrated_inspector.pth",
        WEIGHTS_DIR / "sensitive_inspector.pth",
    ]

    for weight_path in preferred_weights:
        if weight_path.exists():
            inspector = torch.load(str(weight_path), map_location=device, weights_only=False)
            print(f"Loaded extractor from: {weight_path.name}")
            extractor = inspector.feature_extractor
            extractor.to(device)
            extractor.eval()
            return extractor

    print("No serialized inspector found. Falling back to pretrained backbone initialization.")
    return SymmetryAwareFeatureExtractor(backbone="wide_resnet50_2", symmetry_type="both", device=device)


def print_money_table(offline_ms: float, avg_extract_ms: float, avg_retrieve_ms: float, index_type: str) -> None:
    retrieval_label = "FAISS IVF-Flat" if index_type == "ivf_flat" else "FAISS Flat L2"

    print("\n" + "=" * 78)
    print("Processing Stage | Computational Task | Avg CPU Latency")
    print("-" * 78)
    print(
        "Offline Commissioning | 8x p4m Augmentation + ResNet Forward Passes "
        f"| {offline_ms:.2f} ms (One-time)"
    )
    print(f"Online Inference: Stage 1 | Single ResNet Forward Pass | {avg_extract_ms:.2f} ms")
    print(f"Online Inference: Stage 2 | {retrieval_label} Memory Retrieval | {avg_retrieve_ms:.2f} ms")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Profile offline commissioning and online retrieval latency.")
    parser.add_argument("--normal-count", type=int, default=10, help="Number of train/good images used for offline phase.")
    parser.add_argument("--pilot-count", type=int, default=10, help="Number of pilot images used for online loop.")
    parser.add_argument("--index-type", choices=["ivf_flat", "flat"], default="ivf_flat", help="FAISS index type.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Run profiling on CPU or CUDA.")
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Profiling device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    normal_dir = DATA_DIR / "train" / "good"
    test_dir = DATA_DIR / "test"

    normal_paths = collect_image_paths(normal_dir, args.normal_count)
    pilot_paths = collect_image_paths(test_dir, args.pilot_count)

    if len(normal_paths) == 0:
        raise FileNotFoundError(f"No training images found in {normal_dir}")
    if len(pilot_paths) == 0:
        raise FileNotFoundError(f"No pilot images found in {test_dir}")

    print(f"Normal images (offline): {len(normal_paths)}")
    print(f"Pilot images (online): {len(pilot_paths)}")

    extractor = load_feature_extractor(device)
    model = ProfileFeatureModel(extractor)

    dummy = torch.zeros((1, 3, 224, 224), device=device)
    feature_dim = model.extract_features(dummy).shape[1]
    index = FaissIndexAdapter(dimension=feature_dim, index_type=args.index_type)

    normal_tensors = [load_image_tensor(path, transform, device) for path in normal_paths]
    normal_batch = torch.cat(normal_tensors, dim=0)

    index, offline_seconds = offline_commissioning_phase(normal_batch, model, index)

    # Warm-up one inference to reduce first-call overhead in averages.
    _ = online_inference_phase(load_image_tensor(pilot_paths[0], transform, device), model, index)

    extraction_times = []
    retrieval_times = []
    anomaly_scores = []

    print("\nStarting Online Loop...")
    for image_path in pilot_paths:
        test_tensor = load_image_tensor(image_path, transform, device)
        anomaly_score, extraction_ms, retrieval_ms = online_inference_phase(test_tensor, model, index)
        extraction_times.append(extraction_ms)
        retrieval_times.append(retrieval_ms)
        anomaly_scores.append(anomaly_score)
        print(
            f"{image_path.name:<35} score={anomaly_score:10.6f} "
            f"extract={extraction_ms:8.3f} ms retrieve={retrieval_ms:8.3f} ms"
        )

    avg_extract = float(np.mean(extraction_times))
    avg_retrieve = float(np.mean(retrieval_times))
    avg_score = float(np.mean(anomaly_scores))

    print_money_table(
        offline_ms=offline_seconds * 1000.0,
        avg_extract_ms=avg_extract,
        avg_retrieve_ms=avg_retrieve,
        index_type=args.index_type,
    )

    print(f"Average anomaly score across pilot set: {avg_score:.6f}")


if __name__ == "__main__":
    main()