import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
import numpy as np
from typing import Dict, List
from PIL import Image

class SquarePad:
    """
    Pads a rectangular PIL Image or Tensor into a perfect square using black pixels.
    This prevents structural distortion when rotating elongated objects (like the MVTec toothbrush).
    """
    def __call__(self, image):
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            raise TypeError("Input must be a PIL Image or PyTorch Tensor")

        max_dim = max(w, h)
        
        # Calculate padding for left/right and top/bottom
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        
        return TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')


class SymmetryAwareFeatureExtractor(nn.Module):
    """
    Extracts dense local patch features using a frozen Wide ResNet50 backbone.
    Implements p4m discrete symmetry augmentation in the input space to ensure
    rotational and reflectional robustness without gradient updates.
    """
    def __init__(self, backbone: str = "wide_resnet50_2", device: str = None):
        super().__init__()
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Pretrained Backbone
        if backbone == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Backbone {backbone} not supported")
            
        self.model.to(self.device)
        self.model.eval() # Strictly evaluation mode

        # Completely freeze the network - Zero gradient updates allowed
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Setup Hooks for Mid-Level Feature Extraction (PatchCore Standard)
        # We extract from layer2 and layer3 as they balance fine detail and global context
        self.features = {}
        
        def hook_fn(layer_name):
            def hook(module, input, output):
                self.features[layer_name] = output
            return hook

        self.model.layer2.register_forward_hook(hook_fn('layer2'))
        self.model.layer3.register_forward_hook(hook_fn('layer3'))

    def _generate_p4m_orbit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the p4m discrete symmetry group to a square image tensor.
        Input: [B, C, H, W] (Usually B=1 for a single reference image)
        Output: [B*8, C, H, W] containing all orthogonal rotations and reflections.
        """
        assert x.shape[-2] == x.shape[-1], "Tensor must be square before p4m augmentation."
        
        orbit = []
        for k in range(4): # 0, 90, 180, 270 degrees
            rotated = torch.rot90(x, k, dims=[-2, -1])
            orbit.append(rotated)
            orbit.append(TF.hflip(rotated)) # Reflection
            
        return torch.cat(orbit, dim=0)

    def extract_patch_features(self, x: torch.Tensor, apply_p4m: bool = True) -> np.ndarray:
        """
        Extracts patch descriptors for the FAISS memory bank.
        
        Args:
            x: Input image tensor of shape [1, C, H, W]
            apply_p4m: If True, applies 8x symmetry expansion (for MVTec). 
                       If False, extracts features for the original image only (for Side-view case studies).
                       
        Returns:
            np.ndarray of shape [N_patches, Feature_Dimension]
        """
        self.features.clear() # Reset hooks
        
        with torch.no_grad(): # Ensure no gradients are tracked
            # Apply geometry prior if requested
            if apply_p4m:
                batch_x = self._generate_p4m_orbit(x)
            else:
                batch_x = x
                
            batch_x = batch_x.to(self.device)
            
            # Forward pass to trigger hooks
            _ = self.model(batch_x)
            
            f2 = self.features['layer2'] # Shape: [Batch, Channels, H, W]
            f3 = self.features['layer3']

            # Adaptive pooling to standardize spatial dimensions (e.g., 28x28 patches)
            f2_pooled = F.adaptive_avg_pool2d(f2, (28, 28))
            f3_pooled = F.adaptive_avg_pool2d(f3, (28, 28))

            # Concatenate features along the channel dimension
            # For WRN50: layer2 is 512, layer3 is 1024. Total patch dim = 1536
            combined_features = torch.cat([f2_pooled, f3_pooled], dim=1)
            
            # Reshape from [Batch, Channels, H, W] to [Batch * H * W, Channels]
            B, C, H, W = combined_features.shape
            patch_descriptors = combined_features.permute(0, 2, 3, 1).reshape(-1, C)

            # Move to CPU and convert to numpy for FAISS ingestion
            return patch_descriptors.cpu().numpy()