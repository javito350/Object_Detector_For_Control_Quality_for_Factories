"""Profile backbone versus retrieval CPU cost under concurrent camera streams.

The script simulates CPU-only inspection workloads with a heavy convolutional
backbone stage and a FAISS IVF-PQ retrieval stage. Each concurrency level is
profiled with cProfile/pstats and summarized with explicit stage percentages.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import faiss
except ImportError:  # pragma: no cover - optional fallback for non-FAISS environments
    faiss = None  # type: ignore[assignment]


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_CONCURRENCY_LEVELS = (1, 4, 8, 16)
DEFAULT_ITERATIONS_PER_STREAM = 4
DEFAULT_IMAGE_SIZE = 224
DEFAULT_DIMENSION = 1536
DEFAULT_QUERY_BATCH = 64
DEFAULT_DATABASE_SIZE = 25_000
DEFAULT_TRAIN_SIZE = 8_192
DEFAULT_TOP_K = 1


@dataclass(frozen=True)
class BackboneKernels:
    weight_1: Tensor
    weight_2: Tensor
    weight_3: Tensor


@dataclass(frozen=True)
class SimulationContext:
    image_size: int
    dimension: int
    query_batch: int
    train_size: int
    database_size: int
    retrieval_mode: str
    sleep_seconds: float
    top_k: int
    seed: int
    iterations_per_stream: int


@dataclass(frozen=True)
class WorkerSummary:
    profile_path: str
    backbone_seconds: float
    retrieval_seconds: float
    wall_seconds: float


def _configure_threading() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    if faiss is not None:
        faiss.omp_set_num_threads(1)


def build_backbone_kernels(seed: int) -> BackboneKernels:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight_1 = torch.randn((64, 3, 7, 7), generator=generator, dtype=torch.float32)
    weight_2 = torch.randn((128, 64, 3, 3), generator=generator, dtype=torch.float32)
    weight_3 = torch.randn((256, 128, 3, 3), generator=generator, dtype=torch.float32)
    return BackboneKernels(weight_1=weight_1, weight_2=weight_2, weight_3=weight_3)


def build_faiss_index(dimension: int, train_size: int, database_size: int, seed: int) -> object:
    if faiss is None:
        return None

    rng = np.random.default_rng(seed)
    training_vectors = rng.standard_normal((train_size, dimension), dtype=np.float32)
    database_vectors = rng.standard_normal((database_size, dimension), dtype=np.float32)

    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, 32, 48, 8)
    index.train(training_vectors)
    index.add(database_vectors)
    return index


def build_simulation_context(
    image_size: int,
    dimension: int,
    query_batch: int,
    train_size: int,
    database_size: int,
    retrieval_mode: str,
    sleep_seconds: float,
    seed: int,
    top_k: int,
    iterations_per_stream: int,
) -> SimulationContext:
    return SimulationContext(
        image_size=image_size,
        dimension=dimension,
        query_batch=query_batch,
        train_size=train_size,
        database_size=database_size,
        retrieval_mode=retrieval_mode,
        sleep_seconds=sleep_seconds,
        top_k=top_k,
        seed=seed,
        iterations_per_stream=iterations_per_stream,
    )


def run_backbone_pass(image_tensor: torch.Tensor, kernels: BackboneKernels) -> float:
    with torch.inference_mode():
        output = F.conv2d(image_tensor, kernels.weight_1, stride=2, padding=3)
        output = F.relu(output)
        output = F.conv2d(output, kernels.weight_2, stride=1, padding=1)
        output = F.relu(output)
        output = F.conv2d(output, kernels.weight_3, stride=1, padding=1)
        output = F.adaptive_avg_pool2d(output, (7, 7))
        return float(output.sum().item())


def run_retrieval_pass(context: SimulationContext, stream_index: int, faiss_index: object | None) -> float:
    if context.retrieval_mode == "sleep":
        time.sleep(context.sleep_seconds)
        return 0.0

    if faiss is None or faiss_index is None:
        raise RuntimeError("FAISS retrieval requested but the index is unavailable")

    rng = np.random.default_rng(context.seed + 10_000 + stream_index)
    query = rng.standard_normal((context.query_batch, context.dimension), dtype=np.float32)
    distances, indices = faiss_index.search(query, context.top_k)
    return float(distances.sum() + indices.sum())


def run_camera_stream(stream_index: int, context: SimulationContext, profile_dir: str) -> WorkerSummary:
    kernels = build_backbone_kernels(seed=context.seed + 1 + stream_index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(context.seed + 2 + stream_index)
    image_tensor = torch.randn((1, 3, context.image_size, context.image_size), generator=generator, dtype=torch.float32)
    faiss_index = None
    if context.retrieval_mode == "faiss":
        faiss_index = build_faiss_index(
            dimension=context.dimension,
            train_size=context.train_size,
            database_size=context.database_size,
            seed=context.seed + 3 + stream_index,
        )

    profiler = cProfile.Profile()
    backbone_seconds = 0.0
    retrieval_seconds = 0.0
    profile_path = Path(profile_dir) / f"stream_{stream_index:02d}_pid_{os.getpid()}.prof"

    def workload() -> None:
        nonlocal backbone_seconds, retrieval_seconds
        for _ in range(context.iterations_per_stream):
            backbone_start = time.perf_counter()
            _ = run_backbone_pass(image_tensor, kernels)
            backbone_seconds += time.perf_counter() - backbone_start

            retrieval_start = time.perf_counter()
            _ = run_retrieval_pass(context, stream_index, faiss_index)
            retrieval_seconds += time.perf_counter() - retrieval_start

    wall_start = time.perf_counter()
    profiler.runcall(workload)
    wall_seconds = time.perf_counter() - wall_start
    profiler.dump_stats(str(profile_path))

    return WorkerSummary(
        profile_path=str(profile_path),
        backbone_seconds=backbone_seconds,
        retrieval_seconds=retrieval_seconds,
        wall_seconds=wall_seconds,
    )


def aggregate_profiles(worker_summaries: Iterable[WorkerSummary]) -> tuple[pstats.Stats, dict[str, float]]:
    summaries = list(worker_summaries)
    if not summaries:
        raise ValueError("At least one worker profile is required")

    stats = pstats.Stats(summaries[0].profile_path)
    for summary in summaries[1:]:
        stats.add(summary.profile_path)

    totals = {
        "backbone_seconds": sum(summary.backbone_seconds for summary in summaries),
        "retrieval_seconds": sum(summary.retrieval_seconds for summary in summaries),
        "wall_seconds": sum(summary.wall_seconds for summary in summaries),
    }
    return stats, totals


def print_concurrency_report(concurrency: int, iterations: int, stats: pstats.Stats, totals: dict[str, float]) -> None:
    active_seconds = totals["backbone_seconds"] + totals["retrieval_seconds"]
    backbone_pct = 100.0 * totals["backbone_seconds"] / active_seconds if active_seconds > 0 else 0.0
    retrieval_pct = 100.0 * totals["retrieval_seconds"] / active_seconds if active_seconds > 0 else 0.0
    wall_ms = totals["wall_seconds"] * 1000.0
    backbone_ms = totals["backbone_seconds"] * 1000.0
    retrieval_ms = totals["retrieval_seconds"] * 1000.0

    print("\n" + "=" * 96)
    print(f"Concurrency level: {concurrency} stream(s) | Iterations per stream: {iterations}")
    print("-" * 96)
    print(f"Wall time: {wall_ms:.2f} ms")
    print(f"Backbone CPU time: {backbone_ms:.2f} ms ({backbone_pct:.2f}%)")
    print(f"Retrieval CPU time: {retrieval_ms:.2f} ms ({retrieval_pct:.2f}%)")
    print("-" * 96)
    print("Top cumulative-time functions")
    print("-" * 96)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CPU concurrency for backbone versus retrieval.")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=list(DEFAULT_CONCURRENCY_LEVELS))
    parser.add_argument("--iterations-per-stream", type=int, default=DEFAULT_ITERATIONS_PER_STREAM)
    parser.add_argument("--retrieval-mode", choices=["faiss", "sleep"], default="faiss")
    parser.add_argument("--sleep-ms", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--query-batch", type=int, default=DEFAULT_QUERY_BATCH)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--database-size", type=int, default=DEFAULT_DATABASE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> None:
    _configure_threading()
    args = parse_args()

    context = build_simulation_context(
        image_size=args.image_size,
        dimension=args.dimension,
        query_batch=args.query_batch,
        train_size=args.train_size,
        database_size=args.database_size,
        retrieval_mode=args.retrieval_mode,
        sleep_seconds=args.sleep_ms / 1000.0,
        seed=args.seed,
        top_k=args.top_k,
        iterations_per_stream=args.iterations_per_stream,
    )

    print("CPU CONCURRENCY PROFILE")
    print(f"Retrieval mode: {args.retrieval_mode}")
    print(f"Backbone input size: 1 x 3 x {args.image_size} x {args.image_size}")
    print(f"FAISS dimension: {args.dimension} | Query batch: {args.query_batch}")

    with tempfile.TemporaryDirectory(prefix="concurrency_profiles_") as profile_dir:
        for concurrency in args.concurrency_levels:
            if concurrency < 1:
                raise ValueError(f"Concurrency level must be positive, got {concurrency}")

            print(f"\nRunning {concurrency} concurrent stream(s)...")
            wall_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=concurrency) as executor:
                worker_summaries = list(
                    executor.map(
                        run_camera_stream,
                        range(concurrency),
                        [context] * concurrency,
                        [profile_dir] * concurrency,
                    )
                )
            wall_seconds = time.perf_counter() - wall_start

            stats, totals = aggregate_profiles(worker_summaries)
            totals["wall_seconds"] = wall_seconds
            print_concurrency_report(concurrency, args.iterations_per_stream, stats, totals)


if __name__ == "__main__":
    main()