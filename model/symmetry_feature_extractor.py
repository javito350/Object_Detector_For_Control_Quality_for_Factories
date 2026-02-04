import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple
import cv2
from scipy import ndimage

class SymmetryAwareFeatureExtractor(nn.Module):
    """
    PhD-level feature extractor with explicit symmetry exploitation.
    Extracts multi-scale features with symmetry constraints.
    """
    
    def __init__(self, backbone: str = "wide_resnet50_2", 
                 symmetry_type: str = "both",  # 'horizontal', 'vertical', 'both', 'rotational'
                 device: str = "cuda"):
        super().__init__()
        
        self.symmetry_type = symmetry_type
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load pretrained backbone
        if backbone == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=True, progress=True)
        elif backbone == "resnet18":
            model = models.resnet18(pretrained=True, progress=True)
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # Extract intermediate layers for multi-scale features
        self.layer1 = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        
        # Store output dimensions for each layer
        self.feature_dims = {
            'layer1': 256,
            'layer2': 512, 
            'layer3': 1024
        }
        
        # Learnable symmetry weights
        self.symmetry_weights = nn.ParameterDict({
            'horizontal': nn.Parameter(torch.ones(1)),
            'vertical': nn.Parameter(torch.ones(1)),
            'rotational': nn.Parameter(torch.ones(1))
        })
        
        self.to(self.device)
        
        # Freeze backbone (only symmetry weights are trainable)
        for name, param in self.named_parameters():
            if 'symmetry_weights' not in name:
                param.requires_grad = False
                
        self.eval()
    
    def extract_symmetry_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features with explicit symmetry constraints.
        Returns dictionary with original and symmetric features.
        """
        features = {}
        
        # Original image features (multi-scale)
        with torch.no_grad():
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            
            # Adaptive pooling to fixed spatial dimensions
            features['original_layer1'] = F.adaptive_avg_pool2d(f1, (28, 28))
            features['original_layer2'] = F.adaptive_avg_pool2d(f2, (28, 28))
            features['original_layer3'] = F.adaptive_avg_pool2d(f3, (28, 28))
        
        # Extract symmetric features if enabled
        if self.symmetry_type in ['horizontal', 'both']:
            x_h = torch.flip(x, dims=[3])  # Horizontal flip
            with torch.no_grad():
                f1_h = self.layer1(x_h)
                f2_h = self.layer2(f1_h)
                f3_h = self.layer3(f2_h)
                
                features['horizontal_layer1'] = F.adaptive_avg_pool2d(f1_h, (28, 28))
                features['horizontal_layer2'] = F.adaptive_avg_pool2d(f2_h, (28, 28))
                features['horizontal_layer3'] = F.adaptive_avg_pool2d(f3_h, (28, 28))
        
        if self.symmetry_type in ['vertical', 'both']:
            x_v = torch.flip(x, dims=[2])  # Vertical flip
            with torch.no_grad():
                f1_v = self.layer1(x_v)
                f2_v = self.layer2(f1_v)
                f3_v = self.layer3(f2_v)
                
                features['vertical_layer1'] = F.adaptive_avg_pool2d(f1_v, (28, 28))
                features['vertical_layer2'] = F.adaptive_avg_pool2d(f2_v, (28, 28))
                features['vertical_layer3'] = F.adaptive_avg_pool2d(f3_v, (28, 28))
        
        if self.symmetry_type == 'rotational':
            x_r = torch.rot90(x, k=2, dims=[2, 3])  # 180-degree rotation
            with torch.no_grad():
                f1_r = self.layer1(x_r)
                f2_r = self.layer2(f1_r)
                f3_r = self.layer3(f2_r)
                
                features['rotational_layer1'] = F.adaptive_avg_pool2d(f1_r, (28, 28))
                features['rotational_layer2'] = F.adaptive_avg_pool2d(f2_r, (28, 28))
                features['rotational_layer3'] = F.adaptive_avg_pool2d(f3_r, (28, 28))
        
        return features
    
    def compute_symmetry_consistency(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute symmetry consistency scores between original and symmetric features.
        Lower score = more symmetric (good for normal samples).
        """
        consistency_scores = []
        
        # Compare each symmetric view with original
        for layer in ['layer1', 'layer2', 'layer3']:
            original_key = f'original_{layer}'
            original_feat = features[original_key]
            
            # Horizontal symmetry consistency
            if f'horizontal_{layer}' in features:
                horizontal_feat = features[f'horizontal_{layer}']
                # Flip back for comparison
                horizontal_feat = torch.flip(horizontal_feat, dims=[3])
                diff_h = F.mse_loss(original_feat, horizontal_feat, reduction='none')
                diff_h = diff_h.mean(dim=[1, 2, 3])  # Average over spatial dimensions
                consistency_scores.append(diff_h * self.symmetry_weights['horizontal'])
            
            # Vertical symmetry consistency
            if f'vertical_{layer}' in features:
                vertical_feat = features[f'vertical_{layer}']
                vertical_feat = torch.flip(vertical_feat, dims=[2])
                diff_v = F.mse_loss(original_feat, vertical_feat, reduction='none')
                diff_v = diff_v.mean(dim=[1, 2, 3])
                consistency_scores.append(diff_v * self.symmetry_weights['vertical'])
            
            # Rotational symmetry consistency
            if f'rotational_{layer}' in features:
                rotational_feat = features[f'rotational_{layer}']
                rotational_feat = torch.rot90(rotational_feat, k=2, dims=[2, 3])
                diff_r = F.mse_loss(original_feat, rotational_feat, reduction='none')
                diff_r = diff_r.mean(dim=[1, 2, 3])
                consistency_scores.append(diff_r * self.symmetry_weights['rotational'])
        
        if consistency_scores:
            result = torch.stack(consistency_scores).mean(dim=0)
            return result.detach()  # Added .detach() to prevent gradient issues
        else:
            return torch.zeros(1, device=self.device)
    
    def extract_patch_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract patch-level features for memory bank.
        Enhanced with symmetry-aware features.
        """
        # Get all features
        features = self.extract_symmetry_features(x)
        
        # Combine features from all layers and views
        patch_features = []
        
        for key, feat in features.items():
            B, C, H, W = feat.shape
            
            # Reshape to patch-level: (B, C, H, W) -> (B*H*W, C)
            patches = feat.permute(0, 2, 3, 1).reshape(-1, C)
            patch_features.append(patches.detach().cpu().numpy())  # Added .detach()
        
        # Concatenate all features
        combined_features = np.concatenate(patch_features, axis=1)
        
        # Also compute symmetry consistency and append as additional feature
        symmetry_scores = self.compute_symmetry_consistency(features)
        symmetry_scores_np = symmetry_scores.detach().cpu().numpy()  # Added .detach()
        
        # Repeat symmetry score for each patch
        _, _, H, W = features['original_layer1'].shape
        symmetry_expanded = np.repeat(symmetry_scores_np[:, np.newaxis], H * W, axis=1).reshape(-1, 1)
        
        # Combine with patch features
        final_features = np.concatenate([combined_features, symmetry_expanded], axis=1)
        
        return final_features
    
    def extract_symmetry_heatmap(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract per-pixel symmetry inconsistency heatmap.
        Useful for defect localization.
        """
        features = self.extract_symmetry_features(x)
        
        B, C, H, W = features['original_layer1'].shape
        heatmap = np.zeros((B, H, W))
        
        # Compute pixel-level symmetry differences
        for layer in ['layer1', 'layer2', 'layer3']:
            original = features[f'original_{layer}']
            
            if f'horizontal_{layer}' in features:
                horizontal = features[f'horizontal_{layer}']
                horizontal = torch.flip(horizontal, dims=[3])
                diff_h = torch.abs(original - horizontal).mean(dim=1)  # Average over channels
                # Upsample to original layer1 resolution
                if layer != 'layer1':
                    diff_h = F.interpolate(diff_h.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
                heatmap += diff_h.detach().cpu().numpy() * self.symmetry_weights['horizontal'].item()  # Added .detach()
            
            if f'vertical_{layer}' in features:
                vertical = features[f'vertical_{layer}']
                vertical = torch.flip(vertical, dims=[2])
                diff_v = torch.abs(original - vertical).mean(dim=1)
                if layer != 'layer1':
                    diff_v = F.interpolate(diff_v.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
                heatmap += diff_v.detach().cpu().numpy() * self.symmetry_weights['vertical'].item()  # Added .detach()
        
        # Average over layers
        heatmap = heatmap / 3.0  # 3 layers
        
        # Normalize to [0, 1]
        for i in range(B):
            if heatmap[i].max() > 0:
                heatmap[i] = (heatmap[i] - heatmap[i].min()) / (heatmap[i].max() - heatmap[i].min())
        
        return heatmap