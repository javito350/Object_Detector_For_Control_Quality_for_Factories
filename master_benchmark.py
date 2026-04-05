"""
Master benchmark runner for the Edge AI Trilemma thesis experiments.

This script orchestrates four benchmark suites:
1. Trilemma sweep across bit-rates and exact search.
2. Score-range diagnostics with empirical R_max analysis.
3. Rotation robustness against a standard PatchCore-style baseline.
4. N-shot sensitivity analysis.

Design goals:
- Reuse the project's existing dataloaders, feature extractor, and retrieval stack.
- Skip all heatmap/image saving for overnight runs.
- Keep each category isolated with try/except so one failure does not stop the run.
- Log all failures to `errors.log`.
"""

from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import datetime
from pathlib import Path
import platform
import time

import cv2
import faiss
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from models.anomaly_inspector import EnhancedAnomalyInspector as AnomalyInspector
from models.memory_bank import MemoryBank
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor as SymmetryFeatureExtractor
from models.thresholding import EVTCalibrator as Thresholding
from utils.image_loader import MVTecStyleDataset


ROOT_DIR = Path(__file__).resolve().parent
MVTEC_ROOT = ROOT_DIR / "data" / "mvtec"
ERROR_LOG = ROOT_DIR / "errors.log"
RUN_METADATA = ROOT_DIR / "run_metadata.txt"

TRILEMMA_OUTPUT = ROOT_DIR / "trilemma_results.csv"
RMAX_OUTPUT = ROOT_DIR / "rmax_analysis.csv"
ROTATION_OUTPUT = ROOT_DIR / "rotation_robustness.csv"
NSHOT_OUTPUT = ROOT_DIR / "nshot_sensitivity.csv"

# We use the project's ResNet-50 family backbone so the "standard PatchCore"
# comparison and the proposed p4m system share the same feature extractor family.
BACKBONE = "wide_resnet50_2"
FEATURE_DIM = 1536
DEFAULT_N_SHOT = 10
IMG_SIZE = 256
BIT_OPTIONS = [4, 8, 12, "Oracle"]
ROTATIONS = [0, 90, 180, 270]
NSHOT_VALUES = [1, 3, 5, 10]

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


def log_failure(benchmark_name: str, category: str, exc: Exception) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {benchmark_name} | {category} | {type(exc).__name__}: {exc}\n")


def print_progress(benchmark_name: str, category: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"[{benchmark_name}] {category} complete | elapsed: {elapsed:.2f}s")


def write_run_metadata() -> None:
    with RUN_METADATA.open("w", encoding="utf-8") as handle:
        handle.write(f"Run started: {datetime.datetime.now()}\n")
        handle.write(f"Platform: {platform.platform()}\n")
        handle.write(f"CPU: {platform.processor()}\n")
        handle.write(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB\n")
        handle.write(f"Python: {platform.python_version()}\n")


def build_train_dataloader(category: str, n_shot: int) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=True,
        img_size=IMG_SIZE,
    )
    dataset.samples = dataset.samples[:n_shot]
    dataset.transform = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def build_test_dataloader(category: str) -> DataLoader:
    dataset = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT),
        category=category,
        is_train=False,
        img_size=IMG_SIZE,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def extract_support_features(
    dataloader: DataLoader,
    feature_extractor: SymmetryFeatureExtractor,
    device: str,
    apply_p4m_support: bool,
) -> list[np.ndarray]:
    feature_batches: list[np.ndarray] = []

    for images, _, _ in tqdm(dataloader, desc="Extracting support features", leave=False):
        images = images.to(device)
        features = feature_extractor.extract_patch_features(images, apply_p4m=apply_p4m_support)
        feature_batches.append(features)

    if not feature_batches:
        raise ValueError("No normal training images were found for this category.")

    return feature_batches


def fit_evt_thresholds(
    memory_bank: MemoryBank,
    thresholding: Thresholding,
    feature_batches: list[np.ndarray],
) -> tuple[float, float]:
    training_features = np.vstack(feature_batches).astype(np.float32)
    distances, _ = memory_bank.query(training_features, k=1)
    image_threshold = float(thresholding.fit(distances.flatten()))
    pixel_threshold = image_threshold * 0.9
    return image_threshold, pixel_threshold


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


def rotate_images(images: torch.Tensor, rotation_degrees: int) -> torch.Tensor:
    k = (rotation_degrees // 90) % 4
    if k == 0:
        return images
    return torch.rot90(images, k=k, dims=(-2, -1))


def enable_exact_search(memory_bank: MemoryBank) -> None:
    if memory_bank.features is None:
        raise ValueError("Memory bank features are not available for exact search.")

    exact_index = faiss.IndexFlatL2(memory_bank.dimension)
    exact_index.add(memory_bank.features)

    if memory_bank.use_gpu:
        resources = faiss.StandardGpuResources()
        exact_index = faiss.index_cpu_to_gpu(resources, 0, exact_index)

    memory_bank.index = exact_index
    memory_bank.is_trained = True


def compute_quantization_error(memory_bank: MemoryBank) -> tuple[float, float]:
    if memory_bank.features is None or memory_bank.index is None:
        raise ValueError("Memory bank must be populated before computing quantization error.")
    if not memory_bank.use_pq:
        raise ValueError("Quantization error is only defined for PQ banks.")

    index = memory_bank.index
    if memory_bank.use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    reconstructed = index.reconstruct_n(0, index.ntotal)
    original = memory_bank.features[: index.ntotal]
    distances = np.linalg.norm(original - reconstructed, axis=1)
    return float(np.max(distances)), float(np.mean(distances))


def build_inspector(
    category: str,
    n_shot: int,
    apply_p4m_support: bool,
    pq_bits: int | None,
    exact_search: bool,
    compute_pq_diagnostics: bool = False,
) -> tuple[AnomalyInspector, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = SymmetryFeatureExtractor(backbone=BACKBONE, device=device)
    memory_bank = MemoryBank(
        dimension=FEATURE_DIM,
        use_gpu=(device == "cuda"),
        use_pq=not exact_search,
    )
    thresholding = Thresholding(tail_fraction=0.10, target_fpr=0.01)
    inspector = AnomalyInspector(backbone=BACKBONE, device=device, use_pq=not exact_search)

    inspector.feature_extractor = feature_extractor
    inspector.memory_bank = memory_bank

    train_loader = build_train_dataloader(category, n_shot=n_shot)
    support_features = extract_support_features(
        train_loader,
        feature_extractor,
        device=device,
        apply_p4m_support=apply_p4m_support,
    )

    build_kwargs = {"coreset_percentage": inspector.coreset_percentage}
    if pq_bits is not None:
        build_kwargs["pq_bits"] = pq_bits
    memory_bank.build(support_features, **build_kwargs)

    diagnostics: dict[str, float] = {}
    if compute_pq_diagnostics:
        r_max, r_avg = compute_quantization_error(memory_bank)
        diagnostics["R_max"] = r_max
        diagnostics["R_avg"] = r_avg

    if exact_search:
        enable_exact_search(memory_bank)

    image_threshold, pixel_threshold = fit_evt_thresholds(memory_bank, thresholding, support_features)
    inspector.image_threshold = image_threshold
    inspector.pixel_threshold = pixel_threshold

    return inspector, diagnostics


def evaluate_category(
    category: str,
    n_shot: int,
    apply_p4m_support: bool,
    pq_bits: int | None,
    exact_search: bool,
    rotation_degrees: int = 0,
    collect_score_ranges: bool = False,
    compute_pq_diagnostics: bool = False,
) -> dict:
    category_path = MVTEC_ROOT / category
    if not category_path.exists():
        raise FileNotFoundError(f"Category path not found: {category_path}")

    inspector, diagnostics = build_inspector(
        category=category,
        n_shot=n_shot,
        apply_p4m_support=apply_p4m_support,
        pq_bits=pq_bits,
        exact_search=exact_search,
        compute_pq_diagnostics=compute_pq_diagnostics,
    )
    test_loader = build_test_dataloader(category)

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    retrieval_latencies: list[float] = []

    for images, labels, paths in tqdm(test_loader, desc="Running test inference", leave=False):
        images = rotate_images(images, rotation_degrees).to(inspector.device)
        batch_size, _, height, width = images.shape

        with torch.no_grad():
            patch_features = inspector.feature_extractor.extract_patch_features(images, apply_p4m=False)
            retrieval_start = time.perf_counter()
            distances, _ = inspector.memory_bank.query(patch_features, k=1)
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
        raise ValueError("Image-level AUROC requires both nominal and anomalous test samples.")

    result = {
        "Category": category,
        "AUROC": float(roc_auc_score(image_labels, image_scores)),
        "Latency_ms": float(np.mean(retrieval_latencies)),
    }

    if pixel_scores and len(np.unique(np.concatenate(pixel_labels))) >= 2:
        result["PixelAUROC"] = float(
            roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))
        )
    else:
        result["PixelAUROC"] = np.nan

    if collect_score_ranges:
        scores = np.asarray(image_scores, dtype=np.float32)
        labels_array = np.asarray(image_labels, dtype=np.int32)
        normal_scores = scores[labels_array == 0]
        anomaly_scores = scores[labels_array == 1]

        result["normal_min"] = float(np.min(normal_scores))
        result["normal_max"] = float(np.max(normal_scores))
        result["anomaly_min"] = float(np.min(anomaly_scores))
        result["anomaly_max"] = float(np.max(anomaly_scores))

    result.update(diagnostics)
    return result


def benchmark_a_trilemma_sweep() -> None:
    rows: list[dict] = []
    benchmark_name = "BENCHMARK A"

    for bit_setting in BIT_OPTIONS:
        for category in CATEGORIES:
            start_time = time.perf_counter()
            try:
                exact_search = bit_setting == "Oracle"
                pq_bits = None if exact_search else int(bit_setting)
                metrics = evaluate_category(
                    category=category,
                    n_shot=DEFAULT_N_SHOT,
                    apply_p4m_support=True,
                    pq_bits=pq_bits,
                    exact_search=exact_search,
                )
                rows.append(
                    {
                        "Category": category,
                        "Bits": bit_setting,
                        "AUROC": metrics["AUROC"],
                        "PixelAUROC": metrics["PixelAUROC"],
                        "Latency_ms": metrics["Latency_ms"],
                    }
                )
            except Exception as exc:
                log_failure(benchmark_name, f"{category} | bits={bit_setting}", exc)
            finally:
                print_progress(benchmark_name, f"{category} | bits={bit_setting}", start_time)

    pd.DataFrame(rows).to_csv(TRILEMMA_OUTPUT, index=False)


def benchmark_b_rmax_analysis() -> None:
    rows: list[dict] = []
    benchmark_name = "BENCHMARK B"

    for category in CATEGORIES:
        start_time = time.perf_counter()
        try:
            metrics = evaluate_category(
                category=category,
                n_shot=DEFAULT_N_SHOT,
                apply_p4m_support=True,
                pq_bits=8,
                exact_search=False,
                collect_score_ranges=True,
                compute_pq_diagnostics=True,
            )
            denominator = metrics["anomaly_max"] - metrics["normal_min"]
            r_max_pct = (metrics["R_max"] / denominator * 100.0) if denominator > 0 else np.nan
            rows.append(
                {
                    "Category": category,
                    "R_max": metrics["R_max"],
                    "R_avg": metrics["R_avg"],
                    "ScoreRangeMin": metrics["normal_min"],
                    "ScoreRangeMax": metrics["anomaly_max"],
                    "R_max_pct": r_max_pct,
                }
            )
        except Exception as exc:
            log_failure(benchmark_name, category, exc)
        finally:
            print_progress(benchmark_name, category, start_time)

    pd.DataFrame(rows).to_csv(RMAX_OUTPUT, index=False)


def benchmark_c_rotation_robustness() -> None:
    rows: list[dict] = []
    benchmark_name = "BENCHMARK C"

    for category in CATEGORIES:
        start_time = time.perf_counter()
        try:
            your_system_inspector, _ = build_inspector(
                category=category,
                n_shot=DEFAULT_N_SHOT,
                apply_p4m_support=True,
                pq_bits=8,
                exact_search=False,
            )
            patchcore_inspector, _ = build_inspector(
                category=category,
                n_shot=DEFAULT_N_SHOT,
                apply_p4m_support=False,
                pq_bits=None,
                exact_search=True,
            )

            category_path = MVTEC_ROOT / category
            test_loader = build_test_dataloader(category)

            for rotation_degrees in ROTATIONS:
                your_scores: list[float] = []
                patchcore_scores: list[float] = []
                labels_all: list[int] = []

                for images, labels, _ in tqdm(
                    test_loader,
                    desc=f"{category} rotation {rotation_degrees}",
                    leave=False,
                ):
                    rotated = rotate_images(images, rotation_degrees)
                    labels_all.append(int(labels[0]))

                    your_result = your_system_inspector.predict(rotated, apply_p4m=False)[0]
                    patchcore_result = patchcore_inspector.predict(rotated, apply_p4m=False)[0]
                    your_scores.append(float(your_result.image_score))
                    patchcore_scores.append(float(patchcore_result.image_score))

                your_auroc = float(roc_auc_score(labels_all, your_scores))
                patchcore_auroc = float(roc_auc_score(labels_all, patchcore_scores))
                rows.append(
                    {
                        "Category": category,
                        "Rotation": rotation_degrees,
                        "YourAUROC": your_auroc,
                        "PatchCoreAUROC": patchcore_auroc,
                        "Delta": your_auroc - patchcore_auroc,
                    }
                )
        except Exception as exc:
            log_failure(benchmark_name, category, exc)
        finally:
            print_progress(benchmark_name, category, start_time)

    pd.DataFrame(rows).to_csv(ROTATION_OUTPUT, index=False)


def benchmark_d_nshot_sweep() -> None:
    rows: list[dict] = []
    benchmark_name = "BENCHMARK D"

    for n_shot in NSHOT_VALUES:
        for category in CATEGORIES:
            start_time = time.perf_counter()
            try:
                metrics = evaluate_category(
                    category=category,
                    n_shot=n_shot,
                    apply_p4m_support=True,
                    pq_bits=8,
                    exact_search=False,
                )
                rows.append(
                    {
                        "Category": category,
                        "N": n_shot,
                        "AUROC": metrics["AUROC"],
                    }
                )
            except Exception as exc:
                log_failure(benchmark_name, f"{category} | N={n_shot}", exc)
            finally:
                print_progress(benchmark_name, f"{category} | N={n_shot}", start_time)

    pd.DataFrame(rows).to_csv(NSHOT_OUTPUT, index=False)


def main() -> None:
    ERROR_LOG.write_text("", encoding="utf-8")
    write_run_metadata()

    print("Starting master benchmark suite...")
    benchmark_a_trilemma_sweep()
    benchmark_b_rmax_analysis()
    benchmark_c_rotation_robustness()
    benchmark_d_nshot_sweep()
    print("Master benchmark suite complete.")


if __name__ == "__main__":
    main()
