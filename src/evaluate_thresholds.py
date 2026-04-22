"""Compare EVT and percentile thresholding on top of a cached MVTec memory bank."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from build_embedding_cache import WideResNetPatchExtractor, load_cached_embeddings
from models.symmetry_feature_extractor import SquarePad
from models.thresholding import EVTCalibrator


DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "mvtec"
DEFAULT_CACHE_PATH = ROOT_DIR / "cache_screw.h5"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class MVTecTestImageDataset(Dataset):
    def __init__(self, category_root: Path, image_size: int) -> None:
        self.category_root = category_root
        self.samples: list[tuple[Path, int]] = []
        self.transform = T.Compose(
            [
                SquarePad(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        test_root = category_root / "test"
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test split: {test_root}")

        good_dir = test_root / "good"
        if good_dir.exists():
            for path in self._collect_image_paths(good_dir):
                self.samples.append((path, 0))

        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name != "good"):
            for path in self._collect_image_paths(defect_dir):
                self.samples.append((path, 1))

        if not self.samples:
            raise RuntimeError(f"No test images found in {test_root}")

    @staticmethod
    def _collect_image_paths(directory: Path) -> list[Path]:
        paths: list[Path] = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(directory.glob(f"*{extension}"))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EVT and percentile thresholds on a cached MVTec bank.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help="Path to cache_screw.h5")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Path to data/mvtec")
    parser.add_argument("--category", type=str, default=None, help="MVTec category; defaults to cache metadata")
    parser.add_argument("--batch-size", type=int, default=16, help="Test batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--tail-fraction", type=float, default=0.10, help="Tail fraction for EVT calibration")
    parser.add_argument("--target-fpr", type=float, default=0.01, help="Target FPR for EVT calibration")
    parser.add_argument("--nlist", type=int, default=256, help="Number of IVF coarse clusters")
    parser.add_argument("--nprobe", type=int, default=32, help="Number of IVF clusters to probe at search time")
    return parser.parse_args()


def build_faiss_index(flat_embeddings: np.ndarray, dimension: int, nlist: int, nprobe: int) -> faiss.Index:
    vectors = np.asarray(flat_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError(f"Expected flat embeddings shape [N, {dimension}], got {vectors.shape}")

    # Flat search over the full patch cache can be prohibitively slow. IVF keeps
    # the run practical while still querying the full memory bank.
    n_vectors = vectors.shape[0]
    if n_vectors < 50_000:
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        return index

    nlist = max(1, min(nlist, n_vectors // 64))
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

    train_count = min(8_192, n_vectors)
    if train_count < n_vectors:
        rng = np.random.default_rng(42)
        train_ids = rng.choice(n_vectors, size=train_count, replace=False)
        train_vectors = vectors[train_ids]
    else:
        train_vectors = vectors

    index.train(train_vectors)
    index.add(vectors)
    index.nprobe = max(1, min(nprobe, nlist))
    return index


def compute_image_scores_from_embeddings(embeddings: np.ndarray, index: faiss.IndexFlatL2) -> np.ndarray:
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings with shape [N, P, D], got {embeddings.shape}")

    num_images, patch_count, feature_dim = embeddings.shape
    flat_embeddings = embeddings.reshape(-1, feature_dim).astype(np.float32, copy=False)
    distances, indices = index.search(flat_embeddings, 2)

    query_image_ids = np.repeat(np.arange(num_images, dtype=np.int32), patch_count)
    nearest_image_ids = query_image_ids[indices[:, 0]]
    use_second_neighbor = nearest_image_ids == query_image_ids
    selected_distances = np.where(use_second_neighbor, distances[:, 1], distances[:, 0])

    return selected_distances.reshape(num_images, patch_count).max(axis=1)


def compute_test_scores(
    loader: DataLoader,
    extractor: WideResNetPatchExtractor,
    index: faiss.IndexFlatL2,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []

    extractor.eval()
    for images, batch_labels, _ in loader:
        with torch.inference_mode():
            patch_embeddings = extractor(images)

        batch_scores = index.search(
            patch_embeddings.reshape(-1, patch_embeddings.shape[-1]).cpu().numpy().astype(np.float32, copy=False),
            1,
        )[0].reshape(patch_embeddings.shape[0], patch_embeddings.shape[1]).max(axis=1)

        scores.extend(batch_scores.tolist())
        labels.extend(np.asarray(batch_labels, dtype=np.int32).reshape(-1).tolist())

    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def compute_rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    predictions = scores > threshold
    negatives = labels == 0
    positives = labels == 1

    false_positives = np.logical_and(predictions, negatives).sum()
    false_negatives = np.logical_and(~predictions, positives).sum()

    fpr = float(false_positives / negatives.sum()) if negatives.any() else 0.0
    fnr = float(false_negatives / positives.sum()) if positives.any() else 0.0
    return fpr, fnr


def main() -> None:
    args = parse_args()
    cache = load_cached_embeddings(args.cache_path)
    category = args.category or cache.category
    if not category or category == "unknown":
        category = args.cache_path.stem.replace("cache_", "")

    category_root = args.dataset_root / category
    if not category_root.exists():
        raise FileNotFoundError(f"Category directory not found: {category_root}")

    index = build_faiss_index(
        flat_embeddings=cache.flat_embeddings,
        dimension=cache.feature_dim,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )
    nominal_scores = compute_image_scores_from_embeddings(cache.embeddings, index)

    evt_calibrator = EVTCalibrator(tail_fraction=args.tail_fraction, target_fpr=args.target_fpr)
    evt_threshold = float(evt_calibrator.fit(nominal_scores))
    percentile_threshold = float(np.percentile(nominal_scores, 95))

    extractor = WideResNetPatchExtractor(device="cpu", pool_size=cache.patch_grid_size)
    test_dataset = MVTecTestImageDataset(category_root=category_root, image_size=cache.image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(args.batch_size, len(test_dataset)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_scores, test_labels = compute_test_scores(test_loader, extractor, index)

    evt_fpr, evt_fnr = compute_rates(test_scores, test_labels, evt_threshold)
    pct_fpr, pct_fnr = compute_rates(test_scores, test_labels, percentile_threshold)

    print("Threshold Comparison")
    print(f"Category: {category}")
    print(f"Nominal scores used for calibration: {len(nominal_scores)}")
    print(f"Test images scored: {len(test_scores)}")
    print(f"EVT threshold: {evt_threshold:.6f}")
    print(f"EVT FPR: {evt_fpr:.6f}")
    print(f"EVT FNR: {evt_fnr:.6f}")
    print(f"95th percentile threshold: {percentile_threshold:.6f}")
    print(f"95th percentile FPR: {pct_fpr:.6f}")
    print(f"95th percentile FNR: {pct_fnr:.6f}")


if __name__ == "__main__":
    main()