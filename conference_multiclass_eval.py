from pathlib import Path
import time

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from models.anomaly_inspector import EnhancedAnomalyInspector as AnomalyInspector
from models.memory_bank import MemoryBank
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor as SymmetryFeatureExtractor
from models.thresholding import EVTCalibrator as Thresholding
from utils.image_loader import MVTecStyleDataset


ROOT_DIR = Path(__file__).resolve().parent
MVTEC_ROOT = ROOT_DIR / "data" / "mvtec"
BACKBONE = "resnet18"
# ResNet18 layer2 (128 channels) + layer3 (256 channels) = 384-D patch descriptors.
FEATURE_DIM = 384
BIT_RATES = [8]

CATEGORIES = [
    "bottle",
    "screw",
]


def build_train_dataloader(category: str) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=True,
        img_size=256,
    )
    dataset.samples = dataset.samples[:10]
    return DataLoader(dataset, batch_size=1, shuffle=False)


def build_test_dataloader(category: str) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=False,
        img_size=256,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def extract_support_features(dataloader: DataLoader, feature_extractor: SymmetryFeatureExtractor, device: str):
    feature_batches = []

    for images, _, _ in tqdm(dataloader, desc="Extracting support features", leave=False):
        images = images.to(device)
        features = feature_extractor.extract_patch_features(images, apply_p4m=True)
        feature_batches.append(features)

    if not feature_batches:
        raise ValueError("No normal training images were found for this category.")

    return feature_batches


def fit_evt_thresholds(memory_bank: MemoryBank, thresholding: Thresholding, feature_batches):
    training_features = np.vstack(feature_batches).astype(np.float32)
    distances, _ = memory_bank.query(training_features, k=1)
    image_threshold = float(thresholding.fit(distances.flatten()))
    pixel_threshold = image_threshold * 0.9
    return image_threshold, pixel_threshold


def load_ground_truth_mask(category_path: Path, image_path: Path, output_size):
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
    image_threshold: float,
    pixel_threshold: float,
):
    images = images.to(inspector.device)
    batch_size, _, height, width = images.shape

    with torch.no_grad():
        patch_features = inspector.feature_extractor.extract_patch_features(images, apply_p4m=False)

        retrieval_start = time.perf_counter()
        distances, _ = inspector.memory_bank.query(patch_features, k=1)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

    patch_grid = distances.reshape(batch_size, 28, 28)
    image_scores = []
    anomaly_maps = []
    binary_masks = []

    for idx in range(batch_size):
        image_score = float(np.max(patch_grid[idx]))
        anomaly_map = cv2.resize(patch_grid[idx], (width, height), interpolation=cv2.INTER_CUBIC)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        binary_mask = (anomaly_map > pixel_threshold).astype(np.uint8)

        image_scores.append(image_score)
        anomaly_maps.append(anomaly_map)
        binary_masks.append(binary_mask)

    return image_scores, anomaly_maps, binary_masks, retrieval_latency_ms


def enable_exact_search(memory_bank: MemoryBank) -> None:
    if memory_bank.features is None:
        raise ValueError("Memory bank features are not available for exact search.")

    exact_index = faiss.IndexFlatL2(memory_bank.dimension)
    exact_index.add(memory_bank.features)

    if memory_bank.use_gpu:
        res = faiss.StandardGpuResources()
        exact_index = faiss.index_cpu_to_gpu(res, 0, exact_index)

    memory_bank.index = exact_index
    memory_bank.is_trained = True


def estimate_quantization_error(memory_bank: MemoryBank):
    """
    Estimate PQ distortion from reconstructed vectors in the trained FAISS index.

    We reconstruct the full set of vectors currently stored in the populated
    memory bank and measure the L2 error against the original normalized vectors
    in `memory_bank.features`.
    """
    if memory_bank.features is None or len(memory_bank.features) == 0:
        raise ValueError("Memory bank features are not available for quantization analysis.")

    if not memory_bank.use_pq:
        raise ValueError("Quantization-error analysis requires a PQ memory bank.")

    index = memory_bank.index
    if index is None:
        raise ValueError("Memory bank index is not initialized.")

    if memory_bank.use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    reconstructed = index.reconstruct_n(0, index.ntotal)
    original = memory_bank.features[: index.ntotal]
    errors = np.linalg.norm(original - reconstructed, axis=1)

    return {
        "r_max": float(np.max(errors)),
        "avg_error": float(np.mean(errors)),
        "vector_count": int(index.ntotal),
    }


def evaluate_category(category: str, pq_bits: int | None = None, use_exact_search: bool = False):
    category_path = MVTEC_ROOT / category
    if not category_path.exists():
        raise FileNotFoundError(f"Category path not found: {category_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_pq = not use_exact_search

    feature_extractor = SymmetryFeatureExtractor(backbone=BACKBONE, device=device)
    memory_bank = MemoryBank(dimension=FEATURE_DIM, use_gpu=(device == "cuda"), use_pq=use_pq)
    thresholding = Thresholding(tail_fraction=0.10, target_fpr=0.01)
    inspector = AnomalyInspector(backbone=BACKBONE, device=device, use_pq=use_pq)

    inspector.feature_extractor = feature_extractor
    inspector.memory_bank = memory_bank

    train_loader = build_train_dataloader(category)
    test_loader = build_test_dataloader(category)

    support_features = extract_support_features(train_loader, feature_extractor, device)
    build_kwargs = {"coreset_percentage": inspector.coreset_percentage}
    if pq_bits is not None:
        build_kwargs["pq_bits"] = pq_bits
    memory_bank.build(support_features, **build_kwargs)

    quantization_stats = None
    if use_pq and pq_bits == 8:
        quantization_stats = estimate_quantization_error(memory_bank)

    if use_exact_search:
        enable_exact_search(memory_bank)

    image_threshold, pixel_threshold = fit_evt_thresholds(memory_bank, thresholding, support_features)

    inspector.image_threshold = image_threshold
    inspector.pixel_threshold = pixel_threshold

    image_scores = []
    image_labels = []
    pixel_scores = []
    pixel_labels = []
    retrieval_latencies = []

    for images, labels, paths in tqdm(test_loader, desc="Running test inference", leave=False):
        batch_scores, batch_maps, _, retrieval_latency_ms = run_inference(
            inspector,
            images,
            image_threshold=image_threshold,
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

    result = {
        "category": category,
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))),
        "retrieval_latency_ms": float(np.mean(retrieval_latencies)),
    }
    if quantization_stats is not None:
        result["r_max"] = quantization_stats["r_max"]
        result["r_avg"] = quantization_stats["avg_error"]
    return result


def run_evaluation(output_csv: Path, pq_bits: int | None = None, use_exact_search: bool = False):
    results = []

    for category in CATEGORIES:
        print(f"\nEvaluating category: {category}")
        print(f"Data path: data/mvtec/{category}")

        try:
            category_result = evaluate_category(
                category,
                pq_bits=pq_bits,
                use_exact_search=use_exact_search,
            )
            results.append(category_result)
            print(
                f"Completed {category} | "
                f"Image AUROC: {category_result['image_auroc']:.4f} | "
                f"Pixel AUROC: {category_result['pixel_auroc']:.4f} | "
                f"Retrieval Latency: {category_result['retrieval_latency_ms']:.4f} ms"
            )
            if "r_max" in category_result:
                print(
                    f"Quantization Error [{category}] | "
                    f"R_max: {category_result['r_max']:.6f} | "
                    f"R_avg: {category_result['r_avg']:.6f}"
                )
        except Exception as exc:
            print(f"Error while evaluating {category}: {exc}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\nSaved results to {output_csv}")
    if not results_df.empty:
        print(results_df)


def main():
    for bit_rate in BIT_RATES:
        print(f"\nSTARTING BIT-RATE: {bit_rate}")
        run_evaluation(
            output_csv=ROOT_DIR / f"results_bits_{bit_rate}.csv",
            pq_bits=bit_rate,
            use_exact_search=False,
        )
    print("SMOKE TEST COMPLETE: Results saved to CSV.")


if __name__ == "__main__":
    main()
