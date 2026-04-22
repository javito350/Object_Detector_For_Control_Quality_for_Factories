"""
dino_feature_extractor.py
─────────────────────────────────────────────────────────────────────────────
Drop-in replacement for SymmetryAwareFeatureExtractor that uses a frozen
DINOv2 ViT-B/14 backbone instead of Wide ResNet-50.

PUBLIC CONTRACT (identical to SymmetryAwareFeatureExtractor):
  extractor = DINOv2FeatureExtractor(device="cpu")
  features  = extractor.extract_patch_features(x, apply_p4m=True)
  # → np.ndarray  shape [N_patches, FEATURE_DIM]

TENSOR SHAPE WALK-THROUGH
─────────────────────────
ViT-B/14 with input resolution R:
  patch_size  = 14
  grid_side   = R // 14          e.g. 224→16, 392→28
  n_patches   = grid_side²       e.g. 256,    784
  n_tokens    = n_patches + 1    (prepend CLS token)
  hidden_dim  = 768              (ViT-B width)

After one transformer block:
  output shape = [B, n_tokens, 768]

We tap TWO intermediate blocks (early-mid + late-mid, analogous to
ResNet layer2 + layer3) and strip the CLS token from each:
  block_8  → [B, n_patches, 768]
  block_11 → [B, n_patches, 768]

Concatenate along the feature axis:
  [B, n_patches, 1536]

Reshape to spatial grid then flatten across batch:
  [B, grid_side, grid_side, 1536]
  → [B * grid_side², 1536]          ← identical shape contract as ResNet

GRID-SIZE COMPATIBILITY WITH anomaly_inspector.py
──────────────────────────────────────────────────
anomaly_inspector.py derives the grid with:
    grid_size = int(np.sqrt(patches_per_image))
and later calls cv2.resize(patch_grid[i], (W, H)).

This is resolution-agnostic as long as grid_side² == patches_per_image
and the grid is square — both are guaranteed here.

Recommended img_size values:
  224  → 16×16 grid  (256 patches)   — faster, lighter
  392  → 28×28 grid  (784 patches)   — matches ResNet exactly, better heatmaps
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
DINO_PATCH_SIZE: int = 14          # ViT-B/14 patch stride in pixels
DINO_HIDDEN_DIM: int = 768         # ViT-B channel width
DINO_N_BLOCKS: int = 12            # total transformer blocks (0-indexed 0..11)

# We mirror the PatchCore two-layer strategy:
#   block 8  ≈ ResNet layer2  (captures mid-level geometry)
#   block 11 ≈ ResNet layer3  (captures high-level semantics)
# Concatenating both gives 768 + 768 = 1536 — identical to WRN-50.
INTERMEDIATE_BLOCKS: tuple[int, int] = (8, 11)

FEATURE_DIM: int = DINO_HIDDEN_DIM * len(INTERMEDIATE_BLOCKS)  # 1536


# ─── SquarePad (identical copy kept here for module self-containment) ─────────
class SquarePad:
    """
    Pads a PIL Image or Tensor to a square without distorting aspect ratio.
    Required before p4m augmentation (rotation assumes square input).
    """
    def __call__(self, image: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            raise TypeError(f"Expected PIL Image or torch.Tensor, got {type(image)}")

        max_dim = max(w, h)
        pad_left   = (max_dim - w) // 2
        pad_right  = max_dim - w - pad_left
        pad_top    = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        return TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom),
                      fill=0, padding_mode="constant")


# ─── Main extractor ───────────────────────────────────────────────────────────
class DINOv2FeatureExtractor(nn.Module):
    """
    Drop-in replacement for SymmetryAwareFeatureExtractor.

    Uses a frozen DINOv2 ViT-B/14 backbone.  Intermediate patch-token
    features from two transformer blocks are concatenated and returned as a
    flat numpy array of shape  [N_patches, 1536],  matching the contract
    expected by MemoryBank / FAISS IVF-PQ downstream.

    Parameters
    ----------
    img_size : int
        Must be divisible by 14.
        224  → 16×16 patch grid  (fast, smaller memory bank)
        392  → 28×28 patch grid  (matches WRN-50 spatial resolution exactly)
    device : str | None
        "cpu" | "cuda" | None (auto-detect).
    intermediate_blocks : tuple[int, int]
        Indices of the two ViT blocks whose outputs are concatenated.
        Defaults to (8, 11) — analogous to ResNet layer2 + layer3.
    """

    def __init__(
        self,
        img_size: int = 224,
        device: Optional[str] = None,
        intermediate_blocks: tuple[int, int] = INTERMEDIATE_BLOCKS,
    ) -> None:
        super().__init__()

        # ── Validate img_size ────────────────────────────────────────────────
        if img_size % DINO_PATCH_SIZE != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={DINO_PATCH_SIZE}. "
                f"Recommended values: 224 or 392."
            )

        self.img_size = img_size
        self.grid_side: int = img_size // DINO_PATCH_SIZE   # e.g. 16 or 28
        self.n_patches: int = self.grid_side ** 2            # e.g. 256 or 784
        self.feature_dim: int = FEATURE_DIM                  # 1536

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Validate block indices ───────────────────────────────────────────
        for blk_idx in intermediate_blocks:
            if not (0 <= blk_idx < DINO_N_BLOCKS):
                raise ValueError(
                    f"Block index {blk_idx} is out of range for ViT-B "
                    f"(valid range 0..{DINO_N_BLOCKS - 1})."
                )
        self.intermediate_blocks = intermediate_blocks

        # ── Load frozen DINOv2 ViT-B/14 ─────────────────────────────────────
        LOGGER.info("Loading DINOv2 ViT-B/14 from torch.hub …")
        # facebookresearch/dinov2 — loads the ViT-B variant with 14-pixel patches.
        # This call downloads ~330 MB on first run; subsequent calls use cache.
        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
            verbose=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # Completely freeze — zero gradient updates, ever.
        for param in self.model.parameters():
            param.requires_grad = False

        # ── Register hooks on the two chosen transformer blocks ──────────────
        # Each ViT block outputs [B, 1 + n_patches, hidden_dim].
        # We cache the raw block output; CLS stripping happens in the forward.
        self._block_outputs: dict[int, torch.Tensor] = {}

        for blk_idx in self.intermediate_blocks:
            self.model.blocks[blk_idx].register_forward_hook(
                self._make_hook(blk_idx)
            )

        LOGGER.info(
            "DINOv2FeatureExtractor ready | "
            f"img_size={img_size} | grid={self.grid_side}×{self.grid_side} | "
            f"n_patches={self.n_patches} | feature_dim={self.feature_dim} | "
            f"blocks={intermediate_blocks} | device={self.device}"
        )

    # ── Hook factory ──────────────────────────────────────────────────────────
    def _make_hook(self, blk_idx: int):
        """Returns a forward hook that caches the block output tensor."""
        def _hook(module: nn.Module, inp, output: torch.Tensor) -> None:
            # output: [B, 1 + n_patches, hidden_dim]
            self._block_outputs[blk_idx] = output
        return _hook

    # ── p4m orbit (identical to SymmetryAwareFeatureExtractor) ───────────────
    def _generate_p4m_orbit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expands a batch of square image tensors under the p4m group.

        Input  : [B, C, H, W]   — square, H == W
        Output : [B*8, C, H, W] — 4 rotations × 2 reflections per image
        """
        assert x.shape[-2] == x.shape[-1], (
            "Input tensor must be square for p4m augmentation. "
            "Apply SquarePad before calling this method."
        )
        orbit: list[torch.Tensor] = []
        for k in range(4):                          # 0°, 90°, 180°, 270°
            rotated = torch.rot90(x, k, dims=[-2, -1])
            orbit.append(rotated)                   # original rotation
            orbit.append(TF.hflip(rotated))         # + horizontal flip
        return torch.cat(orbit, dim=0)              # [B*8, C, H, W]

    # ── Core extraction ───────────────────────────────────────────────────────
    def extract_patch_features(
        self,
        x: torch.Tensor,
        apply_p4m: bool = True,
    ) -> np.ndarray:
        """
        Extract dense spatial patch features for FAISS ingestion.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 3, H, W].  H and W must equal self.img_size after
            upstream preprocessing.  For support images with p4m, typically
            B=1 (single reference); for inference, B≥1 is fine.
        apply_p4m : bool
            True  → expands each image in the batch ×8 (offline commissioning).
            False → no augmentation (online inference — single forward pass).

        Returns
        -------
        np.ndarray
            Shape [B_eff × grid_side², 1536] where B_eff = B*8 if apply_p4m
            else B.  Values are float32, on CPU, ready for FAISS .add() or
            .search().

        Tensor shape walkthrough (ViT-B/14, img_size=224):
        ┌────────────────────────────────────────────────────────────────────┐
        │ Input x              [B, 3, 224, 224]                             │
        │   └─ after p4m       [B*8, 3, 224, 224]   (if apply_p4m)         │
        │ After model forward:                                              │
        │   block_8 output     [B*8, 257, 768]   (257 = 1_CLS + 256_patch) │
        │   block_11 output    [B*8, 257, 768]                             │
        │ Strip CLS ([:,1:,:]):                                             │
        │   f_early            [B*8, 256, 768]                             │
        │   f_late             [B*8, 256, 768]                             │
        │ Concatenate on dim=2:                                             │
        │   combined           [B*8, 256, 1536]                            │
        │ Reshape to spatial:                                               │
        │   spatial            [B*8, 16, 16, 1536]                         │
        │ Flatten batch×spatial:                                            │
        │   output             [B*8 × 256, 1536]  ← FAISS input            │
        └────────────────────────────────────────────────────────────────────┘
        """
        # ── Input validation ─────────────────────────────────────────────────
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"x must be [B, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"x must have 3 channels (RGB), got {x.shape[1]}")
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"Spatial dimensions must be {self.img_size}×{self.img_size}, "
                f"got {x.shape[-2]}×{x.shape[-1]}. "
                f"Check that your DataLoader resizes to img_size={self.img_size}."
            )

        # ── Clear stale hook buffers ──────────────────────────────────────────
        self._block_outputs.clear()

        with torch.no_grad():
            # ── Apply geometry prior ─────────────────────────────────────────
            if apply_p4m:
                # Requires square input — guaranteed by SquarePad in the transform.
                batch_x = self._generate_p4m_orbit(x)   # [B*8, 3, H, W]
            else:
                batch_x = x                               # [B, 3, H, W]

            batch_x = batch_x.to(self.device)
            B_eff = batch_x.shape[0]   # effective batch size after augmentation

            # ── Forward pass — populates hook buffers ────────────────────────
            # DINOv2's forward_features returns the final-layer CLS token as a
            # 1-D vector, but we only need the side-effect on the hook buffers.
            _ = self.model.forward_features(batch_x)
            # Each self._block_outputs[blk] is now [B_eff, 1+n_patches, 768].

            # ── Verify hooks fired ───────────────────────────────────────────
            for blk_idx in self.intermediate_blocks:
                if blk_idx not in self._block_outputs:
                    raise RuntimeError(
                        f"Hook for block {blk_idx} did not fire. "
                        f"Check that the DINOv2 model has at least {blk_idx+1} blocks."
                    )

            # ── Strip CLS token and retrieve patch tokens ────────────────────
            # Token layout: position 0 = CLS, positions 1.. = spatial patches
            # Patch order: row-major raster scan (top-left → bottom-right).
            blk_early, blk_late = self.intermediate_blocks

            # [B_eff, n_patches, 768]
            f_early: torch.Tensor = self._block_outputs[blk_early][:, 1:, :]
            f_late:  torch.Tensor = self._block_outputs[blk_late][:, 1:, :]

            # ── Validate patch count ─────────────────────────────────────────
            if f_early.shape[1] != self.n_patches:
                raise RuntimeError(
                    f"Expected {self.n_patches} patch tokens from block {blk_early}, "
                    f"got {f_early.shape[1]}. "
                    f"The model was likely called with a different image size than "
                    f"img_size={self.img_size}."
                )

            # ── Concatenate mid-level and late-level features ────────────────
            # Mirrors the ResNet strategy: layer2 (geometry) ∥ layer3 (semantics)
            # [B_eff, n_patches, 1536]
            combined: torch.Tensor = torch.cat([f_early, f_late], dim=2)

            # ── Reshape to explicit spatial grid ─────────────────────────────
            # [B_eff, grid_side, grid_side, 1536]
            spatial: torch.Tensor = combined.reshape(
                B_eff, self.grid_side, self.grid_side, self.feature_dim
            )

            # ── Flatten batch and spatial dimensions ─────────────────────────
            # [B_eff * n_patches, 1536]  ← exact shape contract for FAISS
            patch_descriptors: torch.Tensor = spatial.reshape(-1, self.feature_dim)

            return patch_descriptors.cpu().numpy().astype(np.float32)
