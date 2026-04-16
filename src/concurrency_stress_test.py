"""
Industrial multi-tenancy stress test for the Edge AI Trilemma.

This script validates that an 8-bit PQ memory bank can sustain concurrent
multi-camera retrieval workloads while maintaining bounded latency and strong
throughput scaling in an industrial deployment setting.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT_DIR / "concurrency_stress_results.csv"
OUTPUT_FIGURE = ROOT_DIR / "concurrency_stress_plot.png"

DIMENSION = 1536
SUB_QUANTIZERS = 192
PQ_BITS = 8
MEMORY_BANK_SIZE = 50_000
QUERY_BATCH_SIZE = 256
K_NEIGHBORS = 1
CAMERA_COUNTS = [1, 2, 4, 8, 16, 32, 64]
REQUESTS_PER_CAMERA = 8
TRAINING_SET_SIZE = 20_000
BASE_SEED = 42

_FAISS_INDEX: faiss.IndexPQ | None = None


def _build_index(seed: int) -> faiss.IndexPQ:
    rng = np.random.default_rng(seed)

    train_vectors = rng.standard_normal((TRAINING_SET_SIZE, DIMENSION), dtype=np.float32)
    memory_vectors = rng.standard_normal((MEMORY_BANK_SIZE, DIMENSION), dtype=np.float32)

    index = faiss.IndexPQ(DIMENSION, SUB_QUANTIZERS, PQ_BITS)
    index.train(train_vectors)
    index.add(memory_vectors)
    return index


def _init_worker() -> None:
    global _FAISS_INDEX
    _FAISS_INDEX = _build_index(BASE_SEED)


def camera_stream_worker(task_id: int) -> float:
    global _FAISS_INDEX
    if _FAISS_INDEX is None:
        _init_worker()

    rng = np.random.default_rng(BASE_SEED + task_id)
    queries = rng.standard_normal((QUERY_BATCH_SIZE, DIMENSION), dtype=np.float32)

    start = time.perf_counter()
    _FAISS_INDEX.search(queries, K_NEIGHBORS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms


def run_scaling_tier(camera_count: int) -> dict:
    total_requests = camera_count * REQUESTS_PER_CAMERA
    task_ids = [camera_count * 10_000 + offset for offset in range(total_requests)]

    wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=camera_count, initializer=_init_worker) as executor:
        latencies_ms = list(executor.map(camera_stream_worker, task_ids))
    wall_seconds = time.perf_counter() - wall_start

    throughput_ips = total_requests / wall_seconds if wall_seconds > 0 else 0.0

    return {
        "camera_count": camera_count,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_ips": float(throughput_ips),
    }


def plot_results(results_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    axes[0].plot(
        results_df["camera_count"],
        results_df["throughput_ips"],
        color="#1f4e79",
        marker="o",
        linewidth=2.5,
        markersize=7,
    )
    axes[0].set_title("Throughput (FPS) vs. Camera Count")
    axes[0].set_xlabel("Concurrent Camera Streams")
    axes[0].set_ylabel("Throughput (Images / Second)")
    axes[0].set_xticks(results_df["camera_count"])
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    axes[1].plot(
        results_df["camera_count"],
        results_df["p95_latency_ms"],
        color="#b22222",
        marker="o",
        linewidth=2.5,
        markersize=7,
    )
    axes[1].set_title("Latency (ms) vs. Camera Count")
    axes[1].set_xlabel("Concurrent Camera Streams")
    axes[1].set_ylabel("P95 Latency (ms)")
    axes[1].set_xticks(results_df["camera_count"])
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    fig.suptitle("Industrial Multi-Tenancy Validation for the Edge AI Trilemma", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = []

    for camera_count in CAMERA_COUNTS:
        print(f"Running concurrency tier with {camera_count} camera stream(s)...")
        result = run_scaling_tier(camera_count)
        records.append(result)
        print(
            f"Completed {camera_count:>2} cameras | "
            f"P95 latency: {result['p95_latency_ms']:.3f} ms | "
            f"Throughput: {result['throughput_ips']:.3f} images/s"
        )

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False)
    plot_results(results_df)

    print(f"\nSaved metrics to {OUTPUT_CSV}")
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
