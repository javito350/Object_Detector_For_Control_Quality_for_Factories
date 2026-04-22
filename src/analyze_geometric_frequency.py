"""Quantify geometric frequency content in MVTec AD categories.

This script measures a simple high-frequency energy score for all images in a
directory by computing the 2D FFT magnitude spectrum, suppressing a configurable
low-frequency center region, and averaging the remaining energy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_IMAGE_SIZE = 256
DEFAULT_CUTOFF_RADIUS = 24


def list_image_paths(directory: str | Path) -> list[Path]:
    """Return all supported image files in a directory, sorted deterministically."""

    image_dir = Path(directory)
    if not image_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {image_dir}")

    image_paths: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(image_dir.glob(f"*{extension}"))
        image_paths.extend(image_dir.glob(f"*{extension.upper()}"))

    unique_paths = {str(path).lower(): path for path in image_paths}
    return sorted(unique_paths.values())


def load_grayscale_image(image_path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load an image, convert it to grayscale, and resize it to a fixed square."""

    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def compute_high_frequency_energy_score(image: np.ndarray, cutoff_radius: int = DEFAULT_CUTOFF_RADIUS) -> float:
    """Compute a scalar high-frequency energy score from a grayscale image.

    The score is the sum of FFT magnitudes outside a low-frequency circular mask.
    The image is expected to be a 2D grayscale array.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    if cutoff_radius < 0:
        raise ValueError(f"cutoff_radius must be non-negative, got {cutoff_radius}")

    normalized = image.astype(np.float32) / 255.0
    fft = np.fft.fft2(normalized)
    shifted = np.fft.fftshift(fft)
    magnitude = np.abs(shifted)

    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    high_pass_mask = distance > float(cutoff_radius)

    high_frequency_energy = float(magnitude[high_pass_mask].sum())
    return high_frequency_energy


def analyze_directory(directory: str | Path, image_size: int = DEFAULT_IMAGE_SIZE, cutoff_radius: int = DEFAULT_CUTOFF_RADIUS) -> dict[str, float | int]:
    """Compute per-image scores and return directory-level summary statistics."""

    image_paths = list_image_paths(directory)
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {directory}")

    scores: list[float] = []
    for image_path in image_paths:
        grayscale = load_grayscale_image(image_path, image_size=image_size)
        score = compute_high_frequency_energy_score(grayscale, cutoff_radius=cutoff_radius)
        scores.append(score)

    scores_array = np.asarray(scores, dtype=np.float64)
    return {
        "image_count": int(len(scores)),
        "average_geometric_frequency_score": float(scores_array.mean()),
        "std_geometric_frequency_score": float(scores_array.std(ddof=0)),
        "min_geometric_frequency_score": float(scores_array.min()),
        "max_geometric_frequency_score": float(scores_array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze geometric high-frequency energy for an image directory.")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing images, for example data/mvtec/screw/train/good",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Resize each image to a square of this size before FFT.",
    )
    parser.add_argument(
        "--cutoff-radius",
        type=int,
        default=DEFAULT_CUTOFF_RADIUS,
        help="Radius of the low-frequency center region to exclude from the score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_directory(args.directory, image_size=args.image_size, cutoff_radius=args.cutoff_radius)

    print(f"Directory: {Path(args.directory)}")
    print(f"Images processed: {summary['image_count']}")
    print(f"Image size: {args.image_size} x {args.image_size}")
    print(f"High-pass cutoff radius: {args.cutoff_radius}")
    print(f"Geometric Frequency Score: {summary['average_geometric_frequency_score']:.6f}")
    print(f"Score std: {summary['std_geometric_frequency_score']:.6f}")
    print(f"Score min: {summary['min_geometric_frequency_score']:.6f}")
    print(f"Score max: {summary['max_geometric_frequency_score']:.6f}")


if __name__ == "__main__":
    main()