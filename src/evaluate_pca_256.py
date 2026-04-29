from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.memory_bank import EnhancedCoresetSampler
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from models.thresholding import EVTCalibrator
from utils.image_loader import MVTecStyleDataset


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

DEFAULT_IMG_SIZE = 256
DEFAULT_N_SHOT = 10
DEFAULT_SUPPORT_SEED = 111
FEATURE_DIM = 1536
PCA_DIM = 256


class PCAMemoryBank:
    def __init__(self, dimension: int, use_gpu: bool, use_pq: bool = True) -> None:
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_pq = use_pq
        self.index: faiss.Index | None = None
        self.features: np.ndarray | None = None
        self.is_trained = False

    def _choose_subquantizers(self) -> int:
        candidate_ms = [64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]
        for m_value in candidate_ms:
            if m_value <= self.dimension and self.dimension % m_value == 0:
                return m_value
        return 1

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms

    def _orbit_aware_reduction(self, features: np.ndarray, similarity_threshold: float = 0.98) -> np.ndarray:
        if len(features) < 2:
            return features
        normalized = self._normalize_features(features)
        keep_indices = [0]
        for idx in range(1, len(features)):
            similarity = float(np.dot(normalized[idx], normalized[keep_indices[-1]]))
            if similarity < similarity_threshold:
                keep_indices.append(idx)
        return features[keep_indices]

    def build(self, features_list: list[np.ndarray], coreset_percentage: float, pq_bits: int) -> None:
        if not features_list:
            raise ValueError("features_list is empty.")

        all_features = np.vstack([np.asarray(batch, dtype=np.float32) for batch in features_list]).astype(np.float32)
        all_features = self._orbit_aware_reduction(all_features)

        sampler = EnhancedCoresetSampler(percentage=coreset_percentage, method="statistical")
        coreset_features = sampler.sample(all_features)

        self.features = self._normalize_features(coreset_features).astype(np.float32, copy=False)
        nlist = max(1, min(100, len(self.features) // 10))
        quantizer = faiss.IndexFlatL2(self.dimension)
        M = self._choose_subquantizers()
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, M, pq_bits)

        if self.use_gpu:
            try:
                resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(resources, 0, self.index)
            except Exception:
                pass

        self.index.train(self.features)
        self.index.add(self.features)
        self.is_trained = True

    def query(self, query_features: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_trained or self.index is None:
            raise RuntimeError("Memory bank is not built.")
        normalized = self._normalize_features(np.asarray(query_features, dtype=np.float32)).astype(np.float32)
        return self.index.search(normalized, k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 8-bit FAISS MVTec pipeline with a 1536->256 PCA layer."
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT_DIR / "data" / "mvtec")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT_DIR / "results" / "pca_256_ablation.csv",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--n-shot", type=int, default=DEFAULT_N_SHOT)
    parser.add_argument("--support-seed", type=int, default=DEFAULT_SUPPORT_SEED)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument("--n-components", type=int, default=PCA_DIM)
    return parser.parse_args()


def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(1)


def sample_seeded_support_samples(
    samples: list[tuple[str, int]],
    n_shot: int,
    seed: int,
) -> list[tuple[str, int]]:
    if n_shot > len(samples):
        raise ValueError(
            f"Requested n_shot={n_shot}, but only {len(samples)} train/good samples are available."
        )

    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    selected_idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in selected_idx]


def build_support_dataloader(
    dataset_root: Path,
    category: str,
    img_size: int,
    n_shot: int,
    support_seed: int,
) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(dataset_root),
        category=category,
        is_train=True,
        img_size=img_size,
    )
    dataset.samples = sample_seeded_support_samples(dataset.samples, n_shot=n_shot, seed=support_seed)
    dataset.transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def build_test_dataloader(dataset_root: Path, category: str, img_size: int) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(dataset_root),
        category=category,
        is_train=False,
        img_size=img_size,
    )
    dataset.transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def load_ground_truth_mask(category_path: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray | None:
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


def extract_feature_batches(
    dataloader: DataLoader,
    extractor: SymmetryAwareFeatureExtractor,
    apply_p4m_support: bool,
) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    for images, _, _ in dataloader:
        images = images.to(extractor.device)
        features = extractor.extract_patch_features(images, apply_p4m=apply_p4m_support)
        batches.append(np.asarray(features, dtype=np.float32))
    if not batches:
        raise ValueError("No feature batches were extracted.")
    return batches


def fit_pca_from_support(feature_batches: list[np.ndarray], n_components: int) -> PCA:
    support_matrix = np.vstack(feature_batches).astype(np.float32, copy=False)
    if support_matrix.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected support features with dim={FEATURE_DIM}, got {support_matrix.shape}")
    effective_components = min(n_components, support_matrix.shape[0], support_matrix.shape[1])
    pca = PCA(n_components=effective_components, svd_solver="full", random_state=42)
    pca.fit(support_matrix)
    return pca


def transform_batches(feature_batches: list[np.ndarray], pca: PCA) -> list[np.ndarray]:
    transformed: list[np.ndarray] = []
    for batch in feature_batches:
        transformed_batch = pca.transform(np.asarray(batch, dtype=np.float32)).astype(np.float32, copy=False)
        transformed.append(transformed_batch)
    return transformed


def fit_evt_threshold(memory_bank: PCAMemoryBank, transformed_support_batches: list[np.ndarray]) -> float:
    support_matrix = np.vstack(transformed_support_batches).astype(np.float32, copy=False)
    distances, _ = memory_bank.query(support_matrix, k=1)
    calibrator = EVTCalibrator(tail_fraction=0.10, target_fpr=0.01)
    return float(calibrator.fit(distances.flatten()))


def append_csv_row(output_csv: Path, row: dict[str, float | int | str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    fieldnames = ["Category", "Image AUROC", "Retrieval Latency (ms)", "PCA Components", "PQ Bits"]

    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_category(
    dataset_root: Path,
    category: str,
    img_size: int,
    n_shot: int,
    support_seed: int,
    pq_bits: int,
    coreset_percentage: float,
    n_components: int,
    extractor: SymmetryAwareFeatureExtractor,
) -> dict[str, float | int | str]:
    category_path = dataset_root / category
    support_loader = build_support_dataloader(
        dataset_root=dataset_root,
        category=category,
        img_size=img_size,
        n_shot=n_shot,
        support_seed=support_seed,
    )
    test_loader = build_test_dataloader(dataset_root=dataset_root, category=category, img_size=img_size)

    support_feature_batches = extract_feature_batches(
        dataloader=support_loader,
        extractor=extractor,
        apply_p4m_support=True,
    )
    pca = fit_pca_from_support(support_feature_batches, n_components=n_components)
    transformed_support_batches = transform_batches(support_feature_batches, pca)

    memory_bank = PCAMemoryBank(
        dimension=int(pca.n_components_),
        use_gpu=(extractor.device == "cuda"),
        use_pq=True,
    )
    memory_bank.build(
        transformed_support_batches,
        coreset_percentage=coreset_percentage,
        pq_bits=pq_bits,
    )
    _ = fit_evt_threshold(memory_bank, transformed_support_batches)

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    retrieval_latencies: list[float] = []

    for images, labels, paths in test_loader:
        images = images.to(extractor.device)
        batch_size, _, height, width = images.shape

        with torch.no_grad():
            test_features = extractor.extract_patch_features(images, apply_p4m=False).astype(np.float32, copy=False)
            reduced_test_features = pca.transform(test_features).astype(np.float32, copy=False)
            retrieval_start = time.perf_counter()
            distances, _ = memory_bank.query(reduced_test_features, k=1)
            retrieval_latencies.append((time.perf_counter() - retrieval_start) * 1000.0)

        patch_grid = distances.reshape(batch_size, 28, 28)
        for idx, image_path_str in enumerate(paths):
            image_score = float(np.max(patch_grid[idx]))
            image_scores.append(image_score)
            image_labels.append(int(labels[idx].item()))

            anomaly_map = cv2.resize(patch_grid[idx], (width, height), interpolation=cv2.INTER_CUBIC)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt_mask = load_ground_truth_mask(category_path, Path(image_path_str), anomaly_map.shape)
            if gt_mask is not None:
                pixel_scores.append(anomaly_map.flatten())
                pixel_labels.append(gt_mask.flatten())

    if len(set(image_labels)) < 2:
        raise ValueError(f"Image AUROC could not be computed for {category}: labels lack both classes.")

    image_auroc = float(roc_auc_score(image_labels, image_scores))
    mean_latency_ms = float(np.mean(retrieval_latencies))

    if pixel_scores and len(np.unique(np.concatenate(pixel_labels))) >= 2:
        pixel_auroc = float(roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores)))
        print(f"[PCA256] {category}: Image AUROC={image_auroc:.4f}, Pixel AUROC={pixel_auroc:.4f}, Latency={mean_latency_ms:.4f} ms")
    else:
        print(f"[PCA256] {category}: Image AUROC={image_auroc:.4f}, Latency={mean_latency_ms:.4f} ms")

    return {
        "Category": category,
        "Image AUROC": image_auroc,
        "Retrieval Latency (ms)": mean_latency_ms,
        "PCA Components": int(pca.n_components_),
        "PQ Bits": int(pq_bits),
    }


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SymmetryAwareFeatureExtractor(backbone="wide_resnet50_2", device=device)

    for category in CATEGORIES:
        row = evaluate_category(
            dataset_root=args.dataset_root,
            category=category,
            img_size=args.img_size,
            n_shot=args.n_shot,
            support_seed=args.support_seed,
            pq_bits=args.pq_bits,
            coreset_percentage=args.coreset_percentage,
            n_components=args.n_components,
            extractor=extractor,
        )
        append_csv_row(args.output_csv, row)

    print(f"Saved PCA-256 ablation results to {args.output_csv}")


if __name__ == "__main__":
    main()
