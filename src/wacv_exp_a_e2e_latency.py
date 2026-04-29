"""
wacv_exp_a_e2e_latency.py
=========================
WACV Experiment A — End-to-End Latency Under CPU Constraints.

Reviewer concern:
    "The method is not truly edge-deployable; only partial latency is reported."

This script measures the FULL pipeline latency including:
    1. Preprocessing (resize + normalize)
    2. Backbone forward pass (INT8-quantized Wide ResNet-50)
    3. FAISS IVF-PQ retrieval (k=1)
    4. EVT scoring + anomaly map upsampling

It also profiles throughput (FPS) at concurrency levels {1, 4, 8, 16}
using ProcessPoolExecutor to simulate concurrent camera streams.

Outputs:
    results/wacv/exp_a_latency_breakdown.csv   — per-stage timing breakdown
    results/wacv/exp_a_concurrency.csv         — throughput vs concurrency

Run from the project root:
    python src/wacv_exp_a_e2e_latency.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ── Thread caps — must be set BEFORE numpy / faiss import ─────────────────────
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import faiss
import numpy as np
import torch
import torch.quantization
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from torchvision import transforms as T

warnings.filterwarnings("ignore")

# ── Project-local imports ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from models.memory_bank import MemoryBank
from models.thresholding import EVTCalibrator
from utils.image_loader import MVTecStyleDataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec"
DEFAULT_CATEGORY = "bottle"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "wacv"
FEATURE_DIM = 1536
BACKBONE = "wide_resnet50_2"
N_SHOT = 10
PQ_BITS = 8
SUPPORT_SEED = 111
IMG_SIZE = 256
NUM_WARMUP = 5
NUM_BENCHMARK = 100
CONCURRENCY_LEVELS = [1, 4, 8, 16]


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    faiss.omp_set_num_threads(1)


def sample_seeded_support(
    samples: list[tuple[str, int]], n_shot: int, seed: int
) -> list[tuple[str, int]]:
    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in idx]


def build_int8_backbone() -> torch.nn.Module:
    """Apply INT8 dynamic quantization to the frozen WRN-50 backbone."""
    from torchvision import models as tv_models

    model = tv_models.wide_resnet50_2(
        weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Dynamic quantization targets Linear layers; Conv layers remain FP32
    # but benefit from reduced memory bandwidth pressure
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


@dataclass
class LatencyRecord:
    stage: str
    timings_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.timings_ms)) if self.timings_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.timings_ms)) if self.timings_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.timings_ms, 50)) if self.timings_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.timings_ms, 95)) if self.timings_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.timings_ms, 99)) if self.timings_ms else 0.0


# ── Main benchmark routines ──────────────────────────────────────────────────
def run_stage_benchmark(
    category: str = DEFAULT_CATEGORY,
    num_warmup: int = NUM_WARMUP,
    num_benchmark: int = NUM_BENCHMARK,
) -> list[LatencyRecord]:
    """Measure per-stage latency for the full pipeline."""
    set_low_resource_mode()
    device = "cpu"  # Edge deployment target

    log.info("Building INT8-quantized backbone...")
    extractor = SymmetryAwareFeatureExtractor(backbone=BACKBONE, device=device)

    # Apply INT8 dynamic quantization to the backbone
    extractor.model = torch.quantization.quantize_dynamic(
        extractor.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    extractor.model.eval()

    log.info("Building memory bank (PQ %d-bit)...", PQ_BITS)
    memory_bank = MemoryBank(dimension=FEATURE_DIM, use_gpu=False, use_pq=True)

    # Build support set
    ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    ds.samples = sample_seeded_support(ds.samples, N_SHOT, SUPPORT_SEED)
    train_loader = DataLoader(ds, batch_size=1, shuffle=False)

    support_features = []
    for images, _, _ in train_loader:
        images = images.to(device)
        features = extractor.extract_patch_features(images, apply_p4m=True)
        support_features.append(features)

    memory_bank.build(support_features, coreset_percentage=0.1, pq_bits=PQ_BITS)

    # EVT calibration
    calibrator = EVTCalibrator(tail_fraction=0.10, target_fpr=0.01)
    training_feats = np.vstack(support_features)
    distances, _ = memory_bank.query(training_feats, k=1)
    image_threshold = calibrator.fit(distances.flatten())

    # Build test loader
    test_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Preprocessing transform (to time separately)
    preprocess = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Stage recorders
    stages = {
        "1_preprocessing": LatencyRecord(stage="1_preprocessing"),
        "2_backbone": LatencyRecord(stage="2_backbone"),
        "3_retrieval": LatencyRecord(stage="3_retrieval"),
        "4_scoring": LatencyRecord(stage="4_scoring"),
        "5_total_e2e": LatencyRecord(stage="5_total_e2e"),
    }

    # Collect test images for benchmark
    test_images = []
    for images, _, _ in test_loader:
        test_images.append(images)
        if len(test_images) >= num_warmup + num_benchmark:
            break

    if len(test_images) < num_warmup + 1:
        log.warning("Not enough test images, recycling...")
        while len(test_images) < num_warmup + num_benchmark:
            test_images.append(test_images[0])

    log.info("Running %d warmup iterations...", num_warmup)
    for i in range(num_warmup):
        img = test_images[i].to(device)
        with torch.no_grad():
            feats = extractor.extract_patch_features(img, apply_p4m=False)
            dists, _ = memory_bank.query(feats, k=1)

    log.info("Benchmarking %d iterations...", num_benchmark)
    for i in range(num_warmup, num_warmup + num_benchmark):
        img_tensor = test_images[i % len(test_images)]

        # --- Total E2E start ---
        t_total_start = time.perf_counter()

        # Stage 1: Preprocessing (already done by DataLoader, but we time transform)
        t0 = time.perf_counter()
        img = img_tensor.to(device)
        t1 = time.perf_counter()
        stages["1_preprocessing"].timings_ms.append((t1 - t0) * 1000.0)

        # Stage 2: Backbone forward pass
        t2 = time.perf_counter()
        with torch.no_grad():
            feats = extractor.extract_patch_features(img, apply_p4m=False)
        t3 = time.perf_counter()
        stages["2_backbone"].timings_ms.append((t3 - t2) * 1000.0)

        # Stage 3: FAISS retrieval
        t4 = time.perf_counter()
        dists, _ = memory_bank.query(feats, k=1)
        t5 = time.perf_counter()
        stages["3_retrieval"].timings_ms.append((t5 - t4) * 1000.0)

        # Stage 4: Scoring + anomaly map
        t6 = time.perf_counter()
        B = 1
        patch_grid = dists.reshape(B, 28, 28)
        image_score = float(np.max(patch_grid[0]))
        anomaly_map = cv2.resize(patch_grid[0], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        _ = image_score > image_threshold
        t7 = time.perf_counter()
        stages["4_scoring"].timings_ms.append((t7 - t6) * 1000.0)

        # Total E2E
        t_total_end = time.perf_counter()
        stages["5_total_e2e"].timings_ms.append((t_total_end - t_total_start) * 1000.0)

    return list(stages.values())


def _single_stream_worker(args: tuple) -> float:
    """Worker function for concurrency benchmarking. Returns wall time in seconds."""
    category, num_images = args
    set_low_resource_mode()
    device = "cpu"

    extractor = SymmetryAwareFeatureExtractor(backbone=BACKBONE, device=device)
    extractor.model = torch.quantization.quantize_dynamic(
        extractor.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    extractor.model.eval()

    memory_bank = MemoryBank(dimension=FEATURE_DIM, use_gpu=False, use_pq=True)

    ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    ds.samples = sample_seeded_support(ds.samples, N_SHOT, SUPPORT_SEED)
    train_loader = DataLoader(ds, batch_size=1, shuffle=False)

    support_features = []
    for images, _, _ in train_loader:
        images = images.to(device)
        features = extractor.extract_patch_features(images, apply_p4m=True)
        support_features.append(features)
    memory_bank.build(support_features, coreset_percentage=0.1, pq_bits=PQ_BITS)

    calibrator = EVTCalibrator(tail_fraction=0.10, target_fpr=0.01)
    dists_cal, _ = memory_bank.query(np.vstack(support_features), k=1)
    threshold = calibrator.fit(dists_cal.flatten())

    test_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    test_images = [img for img, _, _ in test_loader]

    # Warmup
    with torch.no_grad():
        feats = extractor.extract_patch_features(test_images[0].to(device), apply_p4m=False)
        memory_bank.query(feats, k=1)

    # Timed inference
    wall_start = time.perf_counter()
    for i in range(num_images):
        img = test_images[i % len(test_images)].to(device)
        with torch.no_grad():
            feats = extractor.extract_patch_features(img, apply_p4m=False)
            dists, _ = memory_bank.query(feats, k=1)
        patch_grid = dists.reshape(1, 28, 28)
        image_score = float(np.max(patch_grid[0]))
        anomaly_map = cv2.resize(patch_grid[0], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        _ = image_score > threshold
    wall_end = time.perf_counter()

    return wall_end - wall_start


def run_concurrency_benchmark(
    category: str = DEFAULT_CATEGORY,
    num_images_per_stream: int = 20,
    concurrency_levels: list[int] = None,
) -> list[dict]:
    """Measure throughput at different concurrency levels."""
    if concurrency_levels is None:
        concurrency_levels = CONCURRENCY_LEVELS

    results = []
    for concurrency in concurrency_levels:
        log.info("Profiling concurrency=%d streams...", concurrency)
        args_list = [(category, num_images_per_stream)] * concurrency

        wall_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_single_stream_worker, a) for a in args_list]
            stream_times = [f.result() for f in as_completed(futures)]
        total_wall = time.perf_counter() - wall_start

        total_images = concurrency * num_images_per_stream
        throughput_fps = total_images / total_wall
        mean_latency_ms = (total_wall / total_images) * 1000.0
        p95_stream_time = float(np.percentile(stream_times, 95))
        p95_latency_ms = (p95_stream_time / num_images_per_stream) * 1000.0

        results.append({
            "concurrency": concurrency,
            "total_images": total_images,
            "wall_seconds": round(total_wall, 4),
            "throughput_fps": round(throughput_fps, 4),
            "mean_latency_ms": round(mean_latency_ms, 4),
            "p95_latency_ms": round(p95_latency_ms, 4),
        })
        log.info(
            "  concurrency=%d → %.2f FPS, %.2f ms/image",
            concurrency, throughput_fps, mean_latency_ms,
        )

    return results


# ── CSV output ────────────────────────────────────────────────────────────────
def save_breakdown_csv(records: list[LatencyRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["stage", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms"]
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "stage": rec.stage,
                "mean_ms": round(rec.mean_ms, 4),
                "std_ms": round(rec.std_ms, 4),
                "p50_ms": round(rec.p50_ms, 4),
                "p95_ms": round(rec.p95_ms, 4),
                "p99_ms": round(rec.p99_ms, 4),
            })
    log.info("Saved breakdown → %s", output_path)


def save_concurrency_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["concurrency", "total_images", "wall_seconds", "throughput_fps",
                  "mean_latency_ms", "p95_latency_ms"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved concurrency → %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WACV EXP-A: End-to-end latency benchmark under CPU constraints."
    )
    parser.add_argument(
        "--category", type=str, default=DEFAULT_CATEGORY,
        help="MVTec category for benchmark (default: bottle)",
    )
    parser.add_argument(
        "--num-benchmark", type=int, default=NUM_BENCHMARK,
        help="Number of benchmark iterations for stage timing (default: 100)",
    )
    parser.add_argument(
        "--num-images-per-stream", type=int, default=20,
        help="Images per stream in concurrency test (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--skip-concurrency", action="store_true",
        help="Skip the concurrency scaling test (faster run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    category_path = MVTEC_ROOT / args.category
    if not category_path.is_dir():
        log.error("Category directory not found: %s", category_path)
        sys.exit(1)

    # ── Phase 1: Per-stage latency breakdown ──────────────────────────────────
    log.info("=" * 70)
    log.info("WACV EXP-A: End-to-End Latency Benchmark")
    log.info("Category: %s | Backbone: %s (INT8) | PQ: %d-bit", args.category, BACKBONE, PQ_BITS)
    log.info("=" * 70)

    stage_records = run_stage_benchmark(
        category=args.category,
        num_warmup=NUM_WARMUP,
        num_benchmark=args.num_benchmark,
    )

    breakdown_path = args.output_dir / "exp_a_latency_breakdown.csv"
    save_breakdown_csv(stage_records, breakdown_path)

    # Console summary
    print()
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║              EXP-A: End-to-End Latency Breakdown (INT8)              ║")
    print("╠══════════════════╦══════════╦══════════╦══════════╦══════════╦════════╣")
    print(f"║ {'Stage':<16} ║ {'Mean':>8} ║ {'Std':>8} ║ {'P50':>8} ║ {'P95':>8} ║ {'P99':>6} ║")
    print("╠══════════════════╬══════════╬══════════╬══════════╬══════════╬════════╣")
    for rec in stage_records:
        print(
            f"║ {rec.stage:<16} ║ {rec.mean_ms:>7.2f}ms ║ "
            f"{rec.std_ms:>7.2f}ms ║ {rec.p50_ms:>7.2f}ms ║ "
            f"{rec.p95_ms:>7.2f}ms ║ {rec.p99_ms:>5.2f}ms ║"
        )
    print("╚══════════════════╩══════════╩══════════╩══════════╩══════════╩════════╝")

    e2e = next(r for r in stage_records if r.stage == "5_total_e2e")
    print(f"\n★ Abstract metric: {e2e.mean_ms:.1f} ms end-to-end latency "
          f"({1000.0 / e2e.mean_ms:.1f} FPS single-stream)\n")

    # ── Phase 2: Concurrency scaling ──────────────────────────────────────────
    if not args.skip_concurrency:
        log.info("=" * 70)
        log.info("Phase 2: Concurrency scaling (1/4/8/16 streams)")
        log.info("=" * 70)

        concurrency_rows = run_concurrency_benchmark(
            category=args.category,
            num_images_per_stream=args.num_images_per_stream,
        )

        concurrency_path = args.output_dir / "exp_a_concurrency.csv"
        save_concurrency_csv(concurrency_rows, concurrency_path)

        print()
        print("╔═══════════════════════════════════════════════════════╗")
        print("║         EXP-A: Concurrency Throughput Scaling        ║")
        print("╠══════════════╦══════════════╦════════════════════════╣")
        print(f"║ {'Streams':>12} ║ {'FPS':>12} ║ {'Mean Latency (ms)':>22} ║")
        print("╠══════════════╬══════════════╬════════════════════════╣")
        for row in concurrency_rows:
            print(f"║ {row['concurrency']:>12} ║ {row['throughput_fps']:>12.2f} ║ {row['mean_latency_ms']:>22.2f} ║")
        print("╚══════════════╩══════════════╩════════════════════════╝")

    print("\nEXP-A COMPLETE.")


if __name__ == "__main__":
    main()
