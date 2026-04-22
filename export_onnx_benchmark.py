from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort
import torch
from torchvision import models


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = ROOT_DIR / "models" / "resnet50_opt.onnx"
DEFAULT_OUTPUT_TXT = ROOT_DIR / "results" / "onnx_latency.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the frozen Wide ResNet-50 model to ONNX and benchmark CPU latency."
    )
    parser.add_argument("--onnx-output", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--opset-version", type=int, default=18)
    return parser.parse_args()


def set_low_resource_mode() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def build_frozen_model() -> torch.nn.Module:
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, image_size: int, opset_version: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


def benchmark_onnx(onnx_path: Path, image_size: int, num_images: int) -> float:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    warmup_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    session.run(None, {input_name: warmup_input})

    latencies_ms: list[float] = []
    for _ in range(num_images):
        input_tensor = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {input_name: input_tensor})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(latencies_ms))


def save_metric(output_path: Path, onnx_path: Path, avg_latency_ms: float, num_images: int, image_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ONNX CPU Benchmark\n")
        handle.write(f"Model: {onnx_path}\n")
        handle.write("Provider: CPUExecutionProvider\n")
        handle.write(f"Image size: {image_size}x{image_size}\n")
        handle.write(f"Images benchmarked: {num_images}\n")
        handle.write(f"Average latency per image (ms): {avg_latency_ms:.6f}\n")


def main() -> None:
    args = parse_args()
    set_low_resource_mode()

    model = build_frozen_model()
    export_to_onnx(
        model=model,
        onnx_path=args.onnx_output,
        image_size=args.image_size,
        opset_version=args.opset_version,
    )
    avg_latency_ms = benchmark_onnx(
        onnx_path=args.onnx_output,
        image_size=args.image_size,
        num_images=args.num_images,
    )
    print(f"Average ONNX CPU latency per image: {avg_latency_ms:.6f} ms")
    save_metric(
        output_path=args.metrics_output,
        onnx_path=args.onnx_output,
        avg_latency_ms=avg_latency_ms,
        num_images=args.num_images,
        image_size=args.image_size,
    )
    print(f"Saved ONNX model to {args.onnx_output}")
    print(f"Saved latency metric to {args.metrics_output}")


if __name__ == "__main__":
    main()
