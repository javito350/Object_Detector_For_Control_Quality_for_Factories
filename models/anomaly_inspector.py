import torch  # Import PyTorch for tensors and model execution.
import numpy as np  # Import NumPy for array math.
from scipy.ndimage import gaussian_filter, median_filter  # Import filters for smoothing heatmaps.
from skimage import morphology, measure  # Import tools for morphological cleanup and labeling.
from typing import Dict, List, Tuple, Optional  # Import typing helpers for readability.
import cv2  # Import OpenCV for resizing and contour operations.
from dataclasses import dataclass  # Import dataclass to define simple data containers.
from enum import Enum  # Import Enum for defect labels.
import warnings  # Import warnings to silence noisy libraries.
warnings.filterwarnings('ignore')  # Silence warning messages for a cleaner console.

# Define the list of defect types the model can report.
class DefectType(Enum):  # Create an enum to hold defect labels.
    CRACK = "crack"  # A crack-like defect.
    SCRATCH = "scratch"  # A scratch-like defect.
    DIRT = "dirt"  # Dirt or contamination defect.
    DEFORMATION = "deformation"  # Shape deformation defect.
    DISCOLORATION = "discoloration"  # Color shift defect.
    SYMMETRY_BREAK = "symmetry_break"  # Symmetry break defect.
    UNKNOWN = "unknown"  # Unknown or unclassified defect.

# Use a dataclass to store all results from one inspection.
@dataclass  # Mark this class as a dataclass for easier init and printing.
class InspectionResult:  # Store all outputs from a single image inspection.
    image_score: float  # Overall anomaly score for the image.
    anomaly_map: np.ndarray  # Heatmap showing where anomalies are.
    symmetry_map: np.ndarray  # Heatmap showing symmetry breaks.
    binary_mask: np.ndarray  # Binary mask of detected anomaly regions.
    defect_type: DefectType  # Predicted defect type label.
    confidence: float  # Confidence score for the defect decision.
    bbox: Optional[Tuple[int, int, int, int]] = None  # Optional bounding box for the defect.
    severity: float = 0.0  # Severity score of the defect.
    metadata: Dict = None  # Extra debug or measurement info.
    
    def is_defective(self, threshold: float = 0.5) -> bool:  # Simple helper to check defect flag.
        return self.image_score > threshold  # Defect if score is above the threshold.

# Main inspector that combines symmetry and memory bank logic.
class EnhancedAnomalyInspector:  # Define the core anomaly detection class.
    # Create the inspector and set up its feature extractor and memory bank.
    def __init__(self,  # Start the constructor definition.
                 backbone: str = "wide_resnet50_2",  # Name of the CNN backbone.
                 symmetry_type: str = "both",  # Which symmetry types to use.
                 device: str = "cuda",  # Desired device (GPU if available).
                 coreset_percentage: float = 0.1):  # Fraction of features to keep.
        
        from .symmetry_feature_extractor import SymmetryAwareFeatureExtractor  # Import feature extractor.
        from .memory_bank import MemoryBank  # Import memory bank implementation.
        
        self.device = device if torch.cuda.is_available() else "cpu"  # Choose CPU if GPU is unavailable.
        self.feature_extractor = SymmetryAwareFeatureExtractor(  # Create the symmetry-aware extractor.
            backbone=backbone,  # Set which CNN backbone to use.
            symmetry_type=symmetry_type,  # Set which symmetry to test.
            device=self.device  # Set device for the extractor.
        )  # Finish creating the feature extractor.
        
        # Calculate feature dimension (including symmetry features).
        feat_dims = self.feature_extractor.feature_dims  # Read feature sizes from extractor.
        
        # Number of views depends on symmetry_type.
        if symmetry_type == "both":  # If horizontal and vertical are both used.
            num_views = 3  # Original + horizontal + vertical.
        elif symmetry_type in ["horizontal", "vertical"]:  # If only one flip is used.
            num_views = 2  # Original + one symmetry view.
        elif symmetry_type == "rotational":  # If rotation symmetry is used.
            num_views = 2  # Original + rotated view.
        else:  # If no symmetry is used.
            num_views = 1  # Only the original view.
        
        total_dim = sum(feat_dims.values()) * num_views  # Compute total feature size across views.
        
        # Add 1 for symmetry consistency score.
        total_dim += 1  # Add one extra dimension for symmetry score.
        
        print(f"Initializing memory bank with dimension: {total_dim}")  # Print memory bank size.
        
        self.memory_bank = MemoryBank(total_dim, use_gpu=(self.device == "cuda"))  # Create memory bank.
        self.coreset_percentage = coreset_percentage  # Store coreset percentage.
        self.symmetry_type = symmetry_type  # Store symmetry mode.
        
        # Adaptive thresholds (learned from data).
        self.pixel_threshold = 0.5  # Per-pixel threshold for masks.
        self.image_threshold = 0.5  # Per-image threshold for defect decision.
        self.symmetry_threshold = 0.3  # Threshold for symmetry consistency.
        
        # Post-processing parameters.
        self.gaussian_sigma = 4  # Blur strength for smoothing heatmaps.
        self.min_anomaly_size = 50  # Minimum defect size in pixels.
        
        # Statistical parameters.
        self.normal_stats = None  # Placeholder for mean and std of normal distances.
        
    def fit(self, dataloader) -> None:  # Train on normal samples only.
        # Explain the start of training.
        print("Building memory bank from normal samples...")  # Print status for user.
        
        all_features = []  # Store all extracted patch features.
        all_symmetry_scores = []  # Store all symmetry consistency scores.
        
        for batch_idx, (images, labels, _) in enumerate(dataloader):  # Loop through dataloader batches.
            if batch_idx % 10 == 0:  # Print progress every 10 batches.
                print(f"Processing batch {batch_idx}/{len(dataloader)}")  # Show progress.
            
            images = images.to(self.device)  # Move images to the selected device.
            
            # Extract features with symmetry information.
            features = self.feature_extractor.extract_patch_features(images)  # Get patch features.
            all_features.append(features)  # Save features for memory bank.
            
            # Also extract symmetry scores for statistical modeling.
            symmetry_features = self.feature_extractor.extract_symmetry_features(images)  # Get symmetry feats.
            symmetry_scores = self.feature_extractor.compute_symmetry_consistency(symmetry_features)  # Score symmetry.
            all_symmetry_scores.append(symmetry_scores.cpu().numpy())  # Store symmetry scores.
        
        # Build memory bank with coreset.
        self.memory_bank.build(all_features, self.coreset_percentage)  # Build memory bank.
        print("Memory bank built successfully!")  # Confirm build success.
        
        # Learn statistical thresholds.
        self._learn_statistical_thresholds(dataloader)  # Compute stats for normalization.
        
        # Learn symmetry statistics from normal samples.
        all_symmetry_scores = np.concatenate(all_symmetry_scores)  # Flatten symmetry scores.
        self.symmetry_threshold = np.percentile(all_symmetry_scores, 95)  # Set threshold at 95th percentile.
        print(f"Symmetry threshold (95th percentile): {self.symmetry_threshold:.4f}")  # Print threshold.
        
        # Store symmetry statistics.
        self.symmetry_mean = np.mean(all_symmetry_scores)  # Store average symmetry score.
        self.symmetry_std = np.std(all_symmetry_scores)  # Store symmetry score spread.
    
    def predict(self, images: torch.Tensor) -> List[InspectionResult]:  # Predict defects on a batch.
        # Explain what this function returns.
        results = []  # Collect results for each image.
        
        with torch.no_grad():  # Disable gradients for faster inference.
            images = images.to(self.device)  # Move images to device.
            B, C, H, W = images.shape  # Read batch size and image shape.
            
            # Extract patch features for memory bank comparison.
            batch_features = self.feature_extractor.extract_patch_features(images)  # Compute patch features.
            
            # Calculate patch-level anomaly scores.
            patch_scores = self._calculate_patch_scores(batch_features, B, H, W)  # Score patches.
            
            # Extract symmetry heatmap.
            symmetry_maps = self.feature_extractor.extract_symmetry_heatmap(images)  # Compute symmetry maps.
            
            # Process each image in batch.
            for i in range(B):  # Loop over each image in the batch.
                # Create appearance-based anomaly heatmap.
                appearance_map = self._create_anomaly_map(patch_scores[i], H, W)  # Upsample patch scores.
                
                # Get symmetry map for this image.
                symmetry_map = symmetry_maps[i]  # Select symmetry map for this image.
                
                # Ensure symmetry map is 2D (remove batch dimension if present).
                if symmetry_map.ndim == 3 and symmetry_map.shape[0] == 1:  # Check for extra dim.
                    symmetry_map = symmetry_map[0]  # Remove the extra dimension.
                
                # Store the original symmetry map for the result.
                original_symmetry_map = symmetry_map.copy()  # Keep a copy (unused but kept).
                
                # Fuse appearance and symmetry information.
                fused_map = self._fuse_maps(appearance_map, symmetry_map)  # Combine heatmaps.
                
                # Post-process fused heatmap.
                fused_map = self._post_process(fused_map)  # Clean and smooth the map.
                
                # Calculate image-level score (consider both appearance and symmetry).
                appearance_score = np.max(appearance_map)  # Highest appearance anomaly.
                
                # Use the upscaled symmetry map for symmetry_score calculation.
                symmetry_map_upscaled = symmetry_map  # Default to the same map.
                if symmetry_map_upscaled.shape != appearance_map.shape:  # Resize if shape differs.
                    h, w = appearance_map.shape  # Get target size.
                    symmetry_map_upscaled = cv2.resize(symmetry_map_upscaled, (w, h), interpolation=cv2.INTER_CUBIC)  # Resize.
                
                symmetry_score = np.max(symmetry_map_upscaled)  # Highest symmetry anomaly.
                image_score = 0.5 * appearance_score + 0.5 * symmetry_score  # Combine into one score.
                
                # Create binary mask.
                binary_mask = (fused_map > self.pixel_threshold).astype(np.uint8)  # Threshold to binary.
                
                # Identify defect type (consider symmetry information).
                defect_type = self._classify_defect(binary_mask, fused_map, symmetry_map_upscaled)  # Classify defect.
                
                # Calculate severity (weighted by both appearance and symmetry).
                severity = self._calculate_severity(fused_map, binary_mask, symmetry_map_upscaled)  # Calculate severity.
                
                # Find bounding box if defect exists.
                bbox = self._find_defect_bbox(binary_mask) if np.any(binary_mask) else None  # Find bounding box.
                
                # Compute confidence (combination of scores).
                anomaly_pixels = fused_map[binary_mask == 1] if np.any(binary_mask) else np.array([0])  # Grab anomaly values.
                confidence = float(np.mean(anomaly_pixels)) if len(anomaly_pixels) > 0 else 0.0  # Average anomaly.
                
                # Create result object with symmetry information.
                result = InspectionResult(  # Build the result object.
                    image_score=float(image_score),  # Store final image score.
                    anomaly_map=fused_map,  # Store fused anomaly map.
                    symmetry_map=symmetry_map_upscaled,  # Store resized symmetry map.
                    binary_mask=binary_mask,  # Store binary mask.
                    defect_type=defect_type,  # Store defect label.
                    confidence=confidence,  # Store confidence value.
                    bbox=bbox,  # Store bounding box if any.
                    severity=severity,  # Store severity score.
                    metadata={  # Store extra metadata for debugging.
                        'image_shape': (H, W),  # Store image shape.
                        'num_anomaly_pixels': np.sum(binary_mask),  # Count anomaly pixels.
                        'appearance_score': float(appearance_score),  # Save appearance score.
                        'symmetry_score': float(symmetry_score),  # Save symmetry score.
                        'symmetry_mean': float(np.mean(symmetry_map_upscaled)),  # Save symmetry mean.
                        'symmetry_std': float(np.std(symmetry_map_upscaled))  # Save symmetry std.
                    }  # End metadata.
                )  # Finish creating result.
                
                results.append(result)  # Append result to list.
        
        return results  # Return all results.
    
    def _calculate_patch_scores(self, features: np.ndarray, B: int, H: int, W: int) -> np.ndarray:  # Score patches.
        # Query memory bank for nearest neighbors.
        distances, _ = self.memory_bank.query(features, k=1)  # Find nearest distance for each patch.
        
        # Statistical normalization if we have normal statistics.
        if self.normal_stats is not None:  # Check if stats are available.
            mean_dist, std_dist = self.normal_stats  # Unpack mean and std.
            distances = (distances - mean_dist) / (std_dist + 1e-8)  # Normalize distances.
            distances = np.maximum(distances, 0)  # Keep only positive deviations.
        
        # Reshape to batch size and spatial dimensions.
        patch_h, patch_w = H // 8, W // 8  # Assume 8x downsampling.
        scores = distances.reshape(B, patch_h, patch_w)  # Reshape to patch grid.
        
        return scores  # Return the patch-level scores.
    
    def _create_anomaly_map(self, patch_scores: np.ndarray, H: int, W: int) -> np.ndarray:  # Upsample patch scores.
        # Reshape to 2D.
        patch_h, patch_w = patch_scores.shape  # Read patch grid shape.
        
        # Edge-aware upsampling (bicubic for smoother results).
        heatmap = cv2.resize(patch_scores, (W, H), interpolation=cv2.INTER_CUBIC)  # Upsample.
        
        # Apply adaptive Gaussian smoothing (stronger smoothing for high-variance areas).
        sigma = self.gaussian_sigma  # Read smoothing strength.
        heatmap = gaussian_filter(heatmap, sigma=sigma)  # Apply Gaussian blur.
        
        # Apply median filter to remove salt-and-pepper noise.
        heatmap = median_filter(heatmap, size=3)  # Apply median filter.
        
        # Normalize to [0, 1] with robust statistics.
        q1, q99 = np.percentile(heatmap, [1, 99])  # Compute robust percentiles.
        if q99 > q1:  # Normal case where spread exists.
            heatmap = np.clip((heatmap - q1) / (q99 - q1), 0, 1)  # Normalize by percentiles.
        elif heatmap.max() > 0:  # Fallback if percentiles are equal.
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize by min/max.
        
        return heatmap  # Return normalized heatmap.
    
    def _fuse_maps(self, appearance_map: np.ndarray, symmetry_map: np.ndarray) -> np.ndarray:  # Combine maps.
        # Upscale symmetry map to match appearance map size.
        if symmetry_map.shape != appearance_map.shape:  # Resize if needed.
            H, W = appearance_map.shape  # Get target shape.
            symmetry_map = cv2.resize(symmetry_map, (W, H), interpolation=cv2.INTER_CUBIC)  # Resize.
        
        # Normalize symmetry map.
        if symmetry_map.max() > 0:  # If symmetry has non-zero values.
            symmetry_norm = symmetry_map / symmetry_map.max()  # Normalize to 0..1.
        else:  # If symmetry map is all zeros.
            symmetry_norm = symmetry_map  # Keep as is.
        
        # Weighted fusion - symmetry gets equal or higher weight (one-shot learning emphasis).
        symmetry_weight = 0.5 + 0.3 * (symmetry_norm.max() > self.symmetry_threshold)  # Weight for symmetry.
        appearance_weight = 1.0 - symmetry_weight  # Weight for appearance.
        
        fused = appearance_weight * appearance_map + symmetry_weight * symmetry_norm  # Combine maps.
        
        # Ensure range [0, 1].
        fused = np.clip(fused, 0, 1)  # Clip to valid range.
        
        return fused  # Return fused heatmap.
    
    def _post_process(self, heatmap: np.ndarray) -> np.ndarray:  # Clean the heatmap.
        # Adaptive thresholding using Otsu's method.
        _, binary_otsu = cv2.threshold((heatmap * 255).astype(np.uint8), 0, 255,  # Compute Otsu threshold.
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Use binary + Otsu.
        binary_otsu = binary_otsu / 255.0  # Convert to 0..1.
        
        # Remove small anomalies with connectivity analysis.
        labeled = measure.label(binary_otsu)  # Label connected regions.
        regions = measure.regionprops(labeled)  # Get region properties.
        
        # Create mask keeping only large enough regions.
        large_regions_mask = np.zeros_like(binary_otsu, dtype=bool)  # Initialize mask.
        for region in regions:  # Loop over each region.
            if region.area >= self.min_anomaly_size:  # Keep only large regions.
                large_regions_mask[region.coords[:, 0], region.coords[:, 1]] = True  # Mark pixels.
        
        # Apply morphological operations.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Create round kernel.
        cleaned = cv2.morphologyEx(large_regions_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # Close gaps.
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)  # Remove small noise.
        
        # Apply mask to heatmap.
        heatmap = heatmap * cleaned  # Zero out removed regions.
        
        # Final Gaussian smoothing.
        heatmap = gaussian_filter(heatmap, sigma=2)  # Smooth again.
        
        return heatmap  # Return processed heatmap.
    
    def _classify_defect(self, binary_mask: np.ndarray,  # Determine defect type.
                        heatmap: np.ndarray,  # Provide intensity map.
                        symmetry_map: np.ndarray) -> DefectType:  # Provide symmetry map.
        # If no defect pixels, return unknown.
        if not np.any(binary_mask):  # Check for any positive pixels.
            return DefectType.UNKNOWN  # Return unknown type.
        
        # Calculate shape features.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours.
        
        if not contours:  # If no contours found.
            return DefectType.UNKNOWN  # Return unknown type.
        
        # Use largest contour.
        contour = max(contours, key=cv2.contourArea)  # Choose the biggest contour.
        
        # Calculate shape features.
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box.
        aspect_ratio = w / h if h > 0 else 1  # Compute aspect ratio.
        
        area = cv2.contourArea(contour)  # Compute area.
        perimeter = cv2.arcLength(contour, True)  # Compute perimeter.
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0  # Compute circularity.
        
        # Calculate intensity features.
        anomaly_intensities = heatmap[binary_mask == 1]  # Extract anomaly values.
        intensity_mean = np.mean(anomaly_intensities)  # Compute mean intensity.
        intensity_std = np.std(anomaly_intensities)  # Compute intensity spread.
        
        # Calculate symmetry features in defect region.
        symmetry_in_defect = symmetry_map[binary_mask == 1]  # Extract symmetry values.
        symmetry_mean = np.mean(symmetry_in_defect) if len(symmetry_in_defect) > 0 else 0  # Symmetry mean.
        symmetry_max = np.max(symmetry_in_defect) if len(symmetry_in_defect) > 0 else 0  # Symmetry max.
        
        # Enhanced classification rules.
        if symmetry_max > self.symmetry_threshold:  # If symmetry break is strong.
            return DefectType.SYMMETRY_BREAK  # Classify as symmetry break.
        elif aspect_ratio > 5 or aspect_ratio < 0.2:  # Long thin shapes.
            return DefectType.CRACK  # Classify as crack.
        elif circularity < 0.3:  # Very non-circular shapes.
            return DefectType.SCRATCH  # Classify as scratch.
        elif intensity_std > 0.25:  # High intensity variation.
            return DefectType.DIRT  # Classify as dirt.
        elif intensity_mean > 0.75:  # Very bright anomalies.
            return DefectType.DISCOLORATION  # Classify as discoloration.
        elif area > 0.1 * binary_mask.size:  # Large area defects.
            return DefectType.DEFORMATION  # Classify as deformation.
        else:  # If none of the rules match.
            return DefectType.UNKNOWN  # Return unknown.
    
    def _calculate_severity(self, heatmap: np.ndarray,  # Compute severity score.
                           binary_mask: np.ndarray,  # Provide binary mask.
                           symmetry_map: np.ndarray) -> float:  # Provide symmetry map.
        # If no defect pixels, severity is zero.
        if not np.any(binary_mask):  # Check for any anomalies.
            return 0.0  # Return zero severity.
        
        # Area-based severity.
        anomaly_area = np.sum(binary_mask)  # Count anomaly pixels.
        total_area = binary_mask.size  # Count total pixels.
        area_ratio = anomaly_area / total_area  # Compute area ratio.
        
        # Intensity-based severity.
        anomaly_intensities = heatmap[binary_mask == 1]  # Extract anomaly values.
        max_intensity = np.max(anomaly_intensities) if len(anomaly_intensities) > 0 else 0  # Max intensity.
        mean_intensity = np.mean(anomaly_intensities) if len(anomaly_intensities) > 0 else 0  # Mean intensity.
        
        # Symmetry-based severity.
        symmetry_in_defect = symmetry_map[binary_mask == 1]  # Extract symmetry values.
        symmetry_severity = np.mean(symmetry_in_defect) if len(symmetry_in_defect) > 0 else 0  # Symmetry mean.
        
        # Combined severity score with symmetry consideration.
        severity = (0.3 * area_ratio +  # Weight for area size.
                   0.3 * max_intensity +  # Weight for strongest anomaly value.
                   0.2 * mean_intensity +  # Weight for average anomaly strength.
                   0.2 * symmetry_severity)  # Weight for symmetry break strength.
        
        return float(np.clip(severity, 0, 1))  # Clamp severity to 0..1 and return.
    
    def _find_defect_bbox(self, binary_mask: np.ndarray) -> Tuple[int, int, int, int]:  # Find defect bounds.
        # Find bounding box of defect region with adaptive margin.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours.
        
        if not contours:  # If no contours exist.
            return None  # Return no bounding box.
        
        # Find union of all defect regions.
        all_points = np.vstack(contours)  # Stack all contour points.
        x, y, w, h = cv2.boundingRect(all_points)  # Compute bounding rectangle.
        
        # Adaptive margin based on defect size.
        margin_ratio = 0.1  # Margin as a fraction of defect size.
        margin_x = int(max(5, w * margin_ratio))  # Horizontal margin.
        margin_y = int(max(5, h * margin_ratio))  # Vertical margin.
        
        x = max(0, x - margin_x)  # Expand left edge with margin.
        y = max(0, y - margin_y)  # Expand top edge with margin.
        w = min(binary_mask.shape[1] - x, w + 2 * margin_x)  # Expand width with margin.
        h = min(binary_mask.shape[0] - y, h + 2 * margin_y)  # Expand height with margin.
        
        return (x, y, x + w, y + h)  # Return box in (x1, y1, x2, y2) format.
    
    def _learn_statistical_thresholds(self, dataloader) -> None:  # Learn thresholds from training data.
        # Learn statistical thresholds from normal training data.
        print("Learning statistical thresholds...")  # Print status.
        
        all_distances = []  # Collect all distances to compute stats.
        
        for images, labels, _ in dataloader:  # Loop over batches.
            images = images.to(self.device)  # Move images to device.
            features = self.feature_extractor.extract_patch_features(images)  # Extract features.
            distances, _ = self.memory_bank.query(features, k=1)  # Query memory bank.
            all_distances.extend(distances.flatten())  # Store distances.
        
        # Compute statistics.
        all_distances = np.array(all_distances)  # Convert list to array.
        self.normal_stats = (np.mean(all_distances), np.std(all_distances))  # Store mean and std.
        
        # Set thresholds based on statistical percentiles.
        self.image_threshold = np.percentile(all_distances, 97)  # 97th percentile for image-level.
        self.pixel_threshold = np.percentile(all_distances, 95)  # 95th percentile for pixel-level.
        
        # Normalize thresholds if we have statistics.
        if self.normal_stats[1] > 0:  # Avoid divide by zero.
            self.image_threshold = (self.image_threshold - self.normal_stats[0]) / self.normal_stats[1]  # Normalize image.
            self.pixel_threshold = (self.pixel_threshold - self.normal_stats[0]) / self.normal_stats[1]  # Normalize pixel.
        
        print("Learned statistical thresholds:")  # Print header.
        print(f"  Normal mean: {self.normal_stats[0]:.4f}, std: {self.normal_stats[1]:.4f}")  # Print stats.
        print(f"  Image threshold (normalized): {self.image_threshold:.4f}")  # Print image threshold.
        print(f"  Pixel threshold (normalized): {self.pixel_threshold:.4f}")  # Print pixel threshold.