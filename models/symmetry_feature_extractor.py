import torch  # Import torch for tensors and model layers.
import torch.nn as nn  # Import nn for defining neural network modules.
import torch.nn.functional as F  # Import functional API for operations.
import torchvision.models as models  # Import pretrained vision models.
import numpy as np  # Import NumPy for array operations.
from typing import Dict, List, Tuple  # Import typing helpers.
import cv2  # Import OpenCV for resizing and image operations.
from scipy import ndimage  # Import ndimage for potential filtering.

# Feature extractor that uses symmetry to improve defect detection.
class SymmetryAwareFeatureExtractor(nn.Module):  # Define the extractor as a PyTorch module.
    # Initialize the extractor and choose the backbone.
    def __init__(self, backbone: str = "wide_resnet50_2",  # Select backbone model.
                 symmetry_type: str = "both",  # Choose symmetry type.
                 device: str = "cuda"):  # Choose device preference.
        super().__init__()  # Initialize nn.Module parent.
        
        self.symmetry_type = symmetry_type  # Store chosen symmetry type.
        self.device = device if torch.cuda.is_available() else "cpu"  # Use GPU if available.
        
        # Load pretrained backbone.
        if backbone == "wide_resnet50_2":  # Use wide ResNet50 if selected.
            model = models.wide_resnet50_2(pretrained=True, progress=True)  # Load pretrained model.
        elif backbone == "resnet18":  # Use ResNet18 if selected.
            model = models.resnet18(pretrained=True, progress=True)  # Load pretrained model.
        else:  # If the backbone name is unsupported.
            raise ValueError(f"Backbone {backbone} not supported")  # Raise an error.
        
        # Extract intermediate layers for multi-scale features.
        self.layer1 = nn.Sequential(  # Create first feature block.
            model.conv1, model.bn1, model.relu, model.maxpool,  # Use early layers.
            model.layer1  # Add layer1 from backbone.
        )  # End layer1 block.
        self.layer2 = model.layer2  # Keep layer2 from backbone.
        self.layer3 = model.layer3  # Keep layer3 from backbone.
        
        # Store output dimensions for each layer.
        self.feature_dims = {  # Map layer name to channel count.
            'layer1': 256,  # Layer1 channels.
            'layer2': 512,  # Layer2 channels.
            'layer3': 1024  # Layer3 channels.
        }  # End feature dims dict.
        
        # Learnable symmetry weights.
        self.symmetry_weights = nn.ParameterDict({  # Store weights for each symmetry type.
            'horizontal': nn.Parameter(torch.ones(1)),  # Horizontal symmetry weight.
            'vertical': nn.Parameter(torch.ones(1)),  # Vertical symmetry weight.
            'rotational': nn.Parameter(torch.ones(1))  # Rotational symmetry weight.
        })  # End symmetry weights.
        
        self.to(self.device)  # Move model to the chosen device.
        
        # Freeze backbone (only symmetry weights are trainable).
        for name, param in self.named_parameters():  # Loop over all parameters.
            if 'symmetry_weights' not in name:  # If not a symmetry weight.
                param.requires_grad = False  # Freeze parameter.
                
        self.eval()  # Put model in eval mode.
    
    # Extract features for original and symmetric views.
    def extract_symmetry_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # Return feature dict.
        # Extract features with explicit symmetry constraints.
        features = {}  # Initialize feature dictionary.
        
        # Original image features (multi-scale).
        with torch.no_grad():  # Disable gradients.
            f1 = self.layer1(x)  # Extract layer1 features.
            f2 = self.layer2(f1)  # Extract layer2 features.
            f3 = self.layer3(f2)  # Extract layer3 features.
            
            # Adaptive pooling to fixed spatial dimensions.
            features['original_layer1'] = F.adaptive_avg_pool2d(f1, (28, 28))  # Pool layer1.
            features['original_layer2'] = F.adaptive_avg_pool2d(f2, (28, 28))  # Pool layer2.
            features['original_layer3'] = F.adaptive_avg_pool2d(f3, (28, 28))  # Pool layer3.
        
        # Extract symmetric features if enabled.
        if self.symmetry_type in ['horizontal', 'both']:  # If horizontal symmetry is used.
            x_h = torch.flip(x, dims=[3])  # Flip image horizontally.
            with torch.no_grad():  # Disable gradients.
                f1_h = self.layer1(x_h)  # Extract layer1 for flipped image.
                f2_h = self.layer2(f1_h)  # Extract layer2 for flipped image.
                f3_h = self.layer3(f2_h)  # Extract layer3 for flipped image.
                
                features['horizontal_layer1'] = F.adaptive_avg_pool2d(f1_h, (28, 28))  # Pool layer1.
                features['horizontal_layer2'] = F.adaptive_avg_pool2d(f2_h, (28, 28))  # Pool layer2.
                features['horizontal_layer3'] = F.adaptive_avg_pool2d(f3_h, (28, 28))  # Pool layer3.
        
        if self.symmetry_type in ['vertical', 'both']:  # If vertical symmetry is used.
            x_v = torch.flip(x, dims=[2])  # Flip image vertically.
            with torch.no_grad():  # Disable gradients.
                f1_v = self.layer1(x_v)  # Extract layer1 for flipped image.
                f2_v = self.layer2(f1_v)  # Extract layer2 for flipped image.
                f3_v = self.layer3(f2_v)  # Extract layer3 for flipped image.
                
                features['vertical_layer1'] = F.adaptive_avg_pool2d(f1_v, (28, 28))  # Pool layer1.
                features['vertical_layer2'] = F.adaptive_avg_pool2d(f2_v, (28, 28))  # Pool layer2.
                features['vertical_layer3'] = F.adaptive_avg_pool2d(f3_v, (28, 28))  # Pool layer3.
        
        if self.symmetry_type == 'rotational':  # If rotational symmetry is used.
            x_r = torch.rot90(x, k=2, dims=[2, 3])  # Rotate image 180 degrees.
            with torch.no_grad():  # Disable gradients.
                f1_r = self.layer1(x_r)  # Extract layer1 for rotated image.
                f2_r = self.layer2(f1_r)  # Extract layer2 for rotated image.
                f3_r = self.layer3(f2_r)  # Extract layer3 for rotated image.
                
                features['rotational_layer1'] = F.adaptive_avg_pool2d(f1_r, (28, 28))  # Pool layer1.
                features['rotational_layer2'] = F.adaptive_avg_pool2d(f2_r, (28, 28))  # Pool layer2.
                features['rotational_layer3'] = F.adaptive_avg_pool2d(f3_r, (28, 28))  # Pool layer3.
        
        return features  # Return the feature dictionary.
    
    # Compute how consistent the symmetric features are.
    def compute_symmetry_consistency(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:  # Return scores.
        # Compute symmetry consistency scores between original and symmetric features.
        consistency_scores = []  # Collect scores for each symmetry type.
        
        # Compare each symmetric view with original.
        for layer in ['layer1', 'layer2', 'layer3']:  # Loop over each layer.
            original_key = f'original_{layer}'  # Build original key name.
            original_feat = features[original_key]  # Get original features.
            
            # Horizontal symmetry consistency.
            if f'horizontal_{layer}' in features:  # If horizontal features exist.
                horizontal_feat = features[f'horizontal_{layer}']  # Get horizontal features.
                horizontal_feat = torch.flip(horizontal_feat, dims=[3])  # Flip back to compare.
                diff_h = F.mse_loss(original_feat, horizontal_feat, reduction='none')  # Per-pixel MSE.
                diff_h = diff_h.mean(dim=[1, 2, 3])  # Average spatially.
                consistency_scores.append(diff_h * self.symmetry_weights['horizontal'])  # Weight and store.
            
            # Vertical symmetry consistency.
            if f'vertical_{layer}' in features:  # If vertical features exist.
                vertical_feat = features[f'vertical_{layer}']  # Get vertical features.
                vertical_feat = torch.flip(vertical_feat, dims=[2])  # Flip back to compare.
                diff_v = F.mse_loss(original_feat, vertical_feat, reduction='none')  # Per-pixel MSE.
                diff_v = diff_v.mean(dim=[1, 2, 3])  # Average spatially.
                consistency_scores.append(diff_v * self.symmetry_weights['vertical'])  # Weight and store.
            
            # Rotational symmetry consistency.
            if f'rotational_{layer}' in features:  # If rotational features exist.
                rotational_feat = features[f'rotational_{layer}']  # Get rotational features.
                rotational_feat = torch.rot90(rotational_feat, k=2, dims=[2, 3])  # Rotate back to compare.
                diff_r = F.mse_loss(original_feat, rotational_feat, reduction='none')  # Per-pixel MSE.
                diff_r = diff_r.mean(dim=[1, 2, 3])  # Average spatially.
                consistency_scores.append(diff_r * self.symmetry_weights['rotational'])  # Weight and store.
        
        if consistency_scores:  # If any scores were collected.
            result = torch.stack(consistency_scores).mean(dim=0)  # Average across types.
            return result.detach()  # Return detached scores to avoid gradients.
        else:  # If no symmetry features were used.
            return torch.zeros(1, device=self.device)  # Return zeros.
    
    # Extract patch-level features for the memory bank.
    def extract_patch_features(self, x: torch.Tensor) -> np.ndarray:  # Return features as numpy.
        # Extract patch-level features for memory bank.
        features = self.extract_symmetry_features(x)  # Compute symmetry features.
        
        # Combine features from all layers and views.
        patch_features = []  # Store flattened patch features.
        
        for key, feat in features.items():  # Loop through each feature map.
            B, C, H, W = feat.shape  # Read shape for reshaping.
            
            # Reshape to patch-level: (B, C, H, W) -> (B*H*W, C).
            patches = feat.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten spatial dims.
            patch_features.append(patches.detach().cpu().numpy())  # Detach and move to CPU.
        
        # Concatenate all features.
        combined_features = np.concatenate(patch_features, axis=1)  # Combine across layers.
        
        # Also compute symmetry consistency and append as additional feature.
        symmetry_scores = self.compute_symmetry_consistency(features)  # Compute symmetry score.
        symmetry_scores_np = symmetry_scores.detach().cpu().numpy()  # Convert to numpy.
        
        # Repeat symmetry score for each patch.
        _, _, H, W = features['original_layer1'].shape  # Get patch grid size.
        symmetry_expanded = np.repeat(symmetry_scores_np[:, np.newaxis], H * W, axis=1).reshape(-1, 1)  # Expand.
        
        # Combine with patch features.
        final_features = np.concatenate([combined_features, symmetry_expanded], axis=1)  # Append symmetry score.
        
        return final_features  # Return combined features.
    
    # Build a heatmap showing symmetry inconsistencies.
    def extract_symmetry_heatmap(self, x: torch.Tensor) -> np.ndarray:  # Return heatmap per image.
        # Extract per-pixel symmetry inconsistency heatmap.
        features = self.extract_symmetry_features(x)  # Compute symmetry features.
        
        B, C, H, W = features['original_layer1'].shape  # Get base feature size.
        heatmap = np.zeros((B, H, W))  # Initialize heatmap.
        
        # Compute pixel-level symmetry differences.
        for layer in ['layer1', 'layer2', 'layer3']:  # Loop over layers.
            original = features[f'original_{layer}']  # Get original features.
            
            if f'horizontal_{layer}' in features:  # If horizontal features exist.
                horizontal = features[f'horizontal_{layer}']  # Get horizontal features.
                horizontal = torch.flip(horizontal, dims=[3])  # Flip back to compare.
                diff_h = torch.abs(original - horizontal).mean(dim=1)  # Average over channels.
                if layer != 'layer1':  # If not base layer.
                    diff_h = F.interpolate(diff_h.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)  # Upsample.
                heatmap += diff_h.detach().cpu().numpy() * self.symmetry_weights['horizontal'].item()  # Add weighted map.
            
            if f'vertical_{layer}' in features:  # If vertical features exist.
                vertical = features[f'vertical_{layer}']  # Get vertical features.
                vertical = torch.flip(vertical, dims=[2])  # Flip back to compare.
                diff_v = torch.abs(original - vertical).mean(dim=1)  # Average over channels.
                if layer != 'layer1':  # If not base layer.
                    diff_v = F.interpolate(diff_v.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)  # Upsample.
                heatmap += diff_v.detach().cpu().numpy() * self.symmetry_weights['vertical'].item()  # Add weighted map.
        
        # Average over layers.
        heatmap = heatmap / 3.0  # Divide by number of layers.
        
        # Normalize to [0, 1].
        for i in range(B):  # Loop over each image.
            if heatmap[i].max() > 0:  # If heatmap has non-zero values.
                heatmap[i] = (heatmap[i] - heatmap[i].min()) / (heatmap[i].max() - heatmap[i].min())  # Normalize.
        
        return heatmap  # Return the heatmap.