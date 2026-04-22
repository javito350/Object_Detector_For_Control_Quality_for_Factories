"""Build and load an HDF5 cache of frozen Wide ResNet-50 patch embeddings.

The cache stores one dense patch descriptor tensor per MVTec AD training image.
Each descriptor tensor is shaped [P, D], where P is the number of pooled patch
locations and D is the concatenated layer2/layer3 feature dimension.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.symmetry_feature_extractor import SquarePad


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_IMAGE_SIZE = 256
DEFAULT_POOL_SIZE = 28
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class CachedEmbeddings:
    """Container for an in-memory embedding cache."""

    embeddings: np.ndarray
    image_paths: tuple[str, ...]
    category: str
    image_size: int
    patch_grid_size: int
    feature_dim: int
    source_split: str

    @property
    def flat_embeddings(self) -> np.ndarray:
        """Return the cache flattened to [N * P, D] for FAISS ingestion."""
        return self.embeddings.reshape(-1, self.feature_dim)


class MVTecTrainImageDataset(Dataset):
    """Deterministic loader for MVTec AD train/good images."""

    def __init__(self, dataset_path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        self.category_root = Path(dataset_path)
        if not self.category_root.exists():
            raise FileNotFoundError(f"Missing dataset directory: {self.category_root}")
        if not self.category_root.is_dir():
            raise NotADirectoryError(f"dataset_path must point to a directory: {self.category_root}")

        self.image_paths = self._collect_image_paths(self.category_root)
        if not self.image_paths:
            raise FileNotFoundError(f"No training images found in {self.category_root}")

        self.transform = T.Compose(
            [
                SquarePad(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _collect_image_paths(directory: Path) -> list[Path]:
        paths: list[Path] = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(directory.glob(f"*{extension}"))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(image_path)


class WideResNetPatchExtractor(torch.nn.Module):
    """Frozen Wide ResNet-50-2 backbone with layer2/layer3 patch hooks."""

    def __init__(self, device: str = "cpu", pool_size: int = DEFAULT_POOL_SIZE) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.pool_size = pool_size

        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.backbone = models.wide_resnet50_2(weights=weights)
        self.backbone.eval()
        self.backbone.to(self.device)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self._activations: dict[str, Tensor] = {}

        def capture(name: str):
            def hook(_: torch.nn.Module, __: tuple[Tensor, ...], output: Tensor) -> None:
                self._activations[name] = output

            return hook

        self.backbone.layer2.register_forward_hook(capture("layer2"))
        self.backbone.layer3.register_forward_hook(capture("layer3"))

    @property
    def feature_dim(self) -> int:
        return 1536

    @property
    def patch_count(self) -> int:
        return self.pool_size * self.pool_size

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, C, H, W], got shape {tuple(images.shape)}")

        self._activations.clear()
        with torch.inference_mode():
            _ = self.backbone(images.to(self.device))
            try:
                layer2 = self._activations["layer2"]
                layer3 = self._activations["layer3"]
            except KeyError as exc:
                raise RuntimeError("Backbone hooks did not capture all target layers") from exc

            pooled_layer2 = F.adaptive_avg_pool2d(layer2, (self.pool_size, self.pool_size))
            pooled_layer3 = F.adaptive_avg_pool2d(layer3, (self.pool_size, self.pool_size))
            combined = torch.cat([pooled_layer2, pooled_layer3], dim=1)
            batch_size, channels, height, width = combined.shape
            return combined.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).cpu()


def _decode_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def build_embedding_cache(
    dataset_path: str | Path,
    output_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
    device: str = "cpu",
    compression: str | None = None,
) -> Path:
    """Extract and persist patch embeddings for the MVTec train/good split."""

    dataset = MVTecTrainImageDataset(dataset_path=dataset_path, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    extractor = WideResNetPatchExtractor(device=device)

    total_images = len(dataset)
    patch_count = extractor.patch_count
    feature_dim = extractor.feature_dim

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression_kwargs: dict[str, object] = {}
    if compression is not None:
        if compression not in {"gzip", "lzf"}:
            raise ValueError("compression must be one of: None, gzip, lzf")
        compression_kwargs["compression"] = compression
        if compression == "gzip":
            compression_kwargs["compression_opts"] = 4

    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
        embeddings_ds = handle.create_dataset(
            "embeddings",
            shape=(total_images, patch_count, feature_dim),
            dtype=np.float32,
            chunks=(min(batch_size, total_images), patch_count, feature_dim),
            **compression_kwargs,
        )
        paths_ds = handle.create_dataset("image_paths", shape=(total_images,), dtype=string_dtype)

        handle.attrs["dataset_path"] = str(Path(dataset_path))
        handle.attrs["source_split"] = "train/good"
        handle.attrs["image_size"] = int(image_size)
        handle.attrs["patch_grid_size"] = int(extractor.pool_size)
        handle.attrs["feature_dim"] = int(feature_dim)
        handle.attrs["patch_count"] = int(patch_count)
        handle.attrs["backbone"] = "wide_resnet50_2"

        write_index = 0
        for batch_images, batch_paths in dataloader:
            batch_embeddings = extractor(batch_images).numpy().astype(np.float32, copy=False)
            batch_size_actual = batch_embeddings.shape[0]
            end_index = write_index + batch_size_actual

            embeddings_ds[write_index:end_index] = batch_embeddings
            paths_ds[write_index:end_index] = batch_paths
            write_index = end_index

        if write_index != total_images:
            raise RuntimeError(f"Expected to write {total_images} images, wrote {write_index}")

    return output_path


def load_cached_embeddings(filepath: str | Path) -> CachedEmbeddings:
    """Load a cache file fully into memory for downstream FAISS experiments."""

    cache_path = Path(filepath)
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    with h5py.File(cache_path, "r") as handle:
        embeddings = np.asarray(handle["embeddings"], dtype=np.float32)
        image_paths = tuple(_decode_str(value) for value in handle["image_paths"][:])
        dataset_path_attr = _decode_str(handle.attrs.get("dataset_path", ""))
        category = _decode_str(handle.attrs.get("category", Path(dataset_path_attr).parent.parent.name if dataset_path_attr else "unknown"))
        source_split = _decode_str(handle.attrs.get("source_split", "unknown"))
        image_size = int(handle.attrs.get("image_size", DEFAULT_IMAGE_SIZE))
        patch_grid_size = int(handle.attrs.get("patch_grid_size", DEFAULT_POOL_SIZE))
        feature_dim = int(handle.attrs.get("feature_dim", embeddings.shape[-1]))

    return CachedEmbeddings(
        embeddings=embeddings,
        image_paths=image_paths,
        category=category,
        image_size=image_size,
        patch_grid_size=patch_grid_size,
        feature_dim=feature_dim,
        source_split=source_split,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HDF5 cache from an image directory of normal samples.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ROOT_DIR / "data" / "mvtec" / "cable" / "train" / "good",
        help="Path to normal training images, e.g. data/mvtec/screw/train/good",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ROOT_DIR / "cache_cable.h5",
        help="Output HDF5 file path, e.g. cache_screw.h5",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--compression", type=str, default="none", choices=["none", "gzip", "lzf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    cache_path = build_embedding_cache(
        dataset_path=args.dataset_path,
        output_path=args.output_file,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        device=args.device,
        compression=compression,
    )
    print(f"Saved embedding cache to {cache_path}")


if __name__ == "__main__":
    main()