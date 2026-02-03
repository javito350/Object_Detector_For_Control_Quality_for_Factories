import torch
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage import morphology, measure
from typing import Dict, List, Tuple, Optional
import cv2
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class DefectType(Enum):
    CRACK = "crack"
    SCRATCH = "scratch" 
    DIRT = "dirt"
    DEFORMATION = "deformation"
    DISCOLORATION = "discoloration"
    SYMMETRY_BREAK = "symmetry_break"
    UNKNOWN = "unknown"

@dataclass
class InspectionResult:
    """PhD-level inspection result container"""
    image_score: float
    anomaly_map: np.ndarray
    symmetry_map: np.ndarray
    binary_mask: np.ndarray
    defect_type: DefectType
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    severity: float = 0.0
    metadata: Dict = None
    
    def is_defective(self, threshold: float = 0.5) -> bool:
        return self.image_score > threshold

class EnhancedAnomalyInspector:
    """
    PhD-level industrial defect inspector with symmetry exploitation.
    Combines PatchCore memory bank with explicit symmetry analysis.
    """
    
    def __init__(self, 
                 backbone: str = "wide_resnet50_2",
                 symmetry_type: str = "both",
                 device: str = "cuda",
                 coreset_percentage: float = 0.1):
        
        from .symmetry_feature_extractor import SymmetryAwareFeatureExtractor
        from .memory_bank import MemoryBank
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SymmetryAwareFeatureExtractor(
            backbone=backbone, 
            symmetry_type=symmetry_type,
            device=self.device
        )
        
        # Calculate feature dimension (including symmetry features)
        feat_dims = self.feature_extractor.feature_dims
        
        # Number of views depends on symmetry_type
        if symmetry_type == "both":
            num_views = 3  # original + horizontal + vertical
        elif symmetry_type in ["horizontal", "vertical"]:
            num_views = 2  # original + one symmetry
        elif symmetry_type == "rotational":
            num_views = 2  # original + rotational
        else:
            num_views = 1  # just original
        
        total_dim = sum(feat_dims.values()) * num_views
        
        # Add 1 for symmetry consistency score
        total_dim += 1
        
        print(f"Initializing memory bank with dimension: {total_dim}")
        
        self.memory_bank = MemoryBank(total_dim, use_gpu=(self.device == "cuda"))
        self.coreset_percentage = coreset_percentage
        self.symmetry_type = symmetry_type
        
        # Adaptive thresholds (learned from data)
        self.pixel_threshold = 0.5
        self.image_threshold = 0.5
        self.symmetry_threshold = 0.3
        
        # Post-processing parameters
        self.gaussian_sigma = 4
        self.min_anomaly_size = 50  # pixels
        
        # Statistical parameters
        self.normal_stats = None  # Will store mean and std of normal distances
        
    def fit(self, dataloader) -> None:
        """Train on normal samples only - learns both appearance and symmetry"""
        print("Building memory bank from normal samples...")
        
        all_features = []
        all_symmetry_scores = []
        
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")
            
            images = images.to(self.device)
            
            # Extract features with symmetry information
            features = self.feature_extractor.extract_patch_features(images)
            all_features.append(features)
            
            # Also extract symmetry scores for statistical modeling
            symmetry_features = self.feature_extractor.extract_symmetry_features(images)
            symmetry_scores = self.feature_extractor.compute_symmetry_consistency(symmetry_features)
            all_symmetry_scores.append(symmetry_scores.cpu().numpy())
        
        # Build memory bank with coreset
        self.memory_bank.build(all_features, self.coreset_percentage)
        print("Memory bank built successfully!")
        
        # Learn statistical thresholds
        self._learn_statistical_thresholds(dataloader)
        
        # Learn symmetry statistics from normal samples
        all_symmetry_scores = np.concatenate(all_symmetry_scores)
        self.symmetry_threshold = np.percentile(all_symmetry_scores, 95)
        print(f"Symmetry threshold (95th percentile): {self.symmetry_threshold:.4f}")
        
        # Store symmetry statistics
        self.symmetry_mean = np.mean(all_symmetry_scores)
        self.symmetry_std = np.std(all_symmetry_scores)
    
    def predict(self, images: torch.Tensor) -> List[InspectionResult]:
        """
        Predict defects in batch of images with symmetry analysis.
        
        Returns:
            List of InspectionResult objects with symmetry information
        """
        results = []
        
        with torch.no_grad():
            images = images.to(self.device)
            B, C, H, W = images.shape
            
            # Extract patch features for memory bank comparison
            batch_features = self.feature_extractor.extract_patch_features(images)
            
            # Calculate patch-level anomaly scores
            patch_scores = self._calculate_patch_scores(batch_features, B, H, W)
            
            # Extract symmetry heatmap
            symmetry_maps = self.feature_extractor.extract_symmetry_heatmap(images)
            
            # Process each image in batch
            for i in range(B):
                # Create appearance-based anomaly heatmap
                appearance_map = self._create_anomaly_map(patch_scores[i], H, W)
                
                # Get symmetry map for this image
                symmetry_map = symmetry_maps[i]
                
                # Ensure symmetry map is 2D (remove batch dimension if present)
                if symmetry_map.ndim == 3 and symmetry_map.shape[0] == 1:
                    symmetry_map = symmetry_map[0]
                
                # Store the original symmetry map for the result
                original_symmetry_map = symmetry_map.copy()
                
                # Fuse appearance and symmetry information
                # Symmetry breaks are strong indicators of defects
                fused_map = self._fuse_maps(appearance_map, symmetry_map)
                
                # Post-process fused heatmap
                fused_map = self._post_process(fused_map)
                
                # Calculate image-level score (consider both appearance and symmetry)
                appearance_score = np.max(appearance_map)
                
                # Use the upscaled symmetry map for symmetry_score calculation
                symmetry_map_upscaled = symmetry_map
                if symmetry_map_upscaled.shape != appearance_map.shape:
                    h, w = appearance_map.shape
                    symmetry_map_upscaled = cv2.resize(symmetry_map_upscaled, (w, h), interpolation=cv2.INTER_CUBIC)
                
                symmetry_score = np.max(symmetry_map_upscaled)
                image_score = 0.7 * appearance_score + 0.3 * symmetry_score
                
                # Create binary mask
                binary_mask = (fused_map > self.pixel_threshold).astype(np.uint8)
                
                # Identify defect type (consider symmetry information)
                defect_type = self._classify_defect(binary_mask, fused_map, symmetry_map_upscaled)
                
                # Calculate severity (weighted by both appearance and symmetry)
                severity = self._calculate_severity(fused_map, binary_mask, symmetry_map_upscaled)
                
                # Find bounding box if defect exists
                bbox = self._find_defect_bbox(binary_mask) if np.any(binary_mask) else None
                
                # Compute confidence (combination of scores)
                anomaly_pixels = fused_map[binary_mask == 1] if np.any(binary_mask) else np.array([0])
                confidence = float(np.mean(anomaly_pixels)) if len(anomaly_pixels) > 0 else 0.0
                
                # Create result object with symmetry information
                # Store the upscaled symmetry map in the result
                result = InspectionResult(
                    image_score=float(image_score),
                    anomaly_map=fused_map,
                    symmetry_map=symmetry_map_upscaled,  # Use upscaled map
                    binary_mask=binary_mask,
                    defect_type=defect_type,
                    confidence=confidence,
                    bbox=bbox,
                    severity=severity,
                    metadata={
                        'image_shape': (H, W),
                        'num_anomaly_pixels': np.sum(binary_mask),
                        'appearance_score': float(appearance_score),
                        'symmetry_score': float(symmetry_score),
                        'symmetry_mean': float(np.mean(symmetry_map_upscaled)),
                        'symmetry_std': float(np.std(symmetry_map_upscaled))
                    }
                )
                
                results.append(result)
        
        return results
    
    def _calculate_patch_scores(self, features: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
        """Calculate anomaly scores for each patch with statistical normalization"""
        # Query memory bank for nearest neighbors
        distances, _ = self.memory_bank.query(features, k=1)
        
        # Statistical normalization if we have normal statistics
        if self.normal_stats is not None:
            mean_dist, std_dist = self.normal_stats
            distances = (distances - mean_dist) / (std_dist + 1e-8)
            distances = np.maximum(distances, 0)  # Only positive deviations matter
        
        # Reshape to batch size and spatial dimensions
        patch_h, patch_w = H // 8, W // 8  # Assuming 8x downsampling
        scores = distances.reshape(B, patch_h, patch_w)
        
        return scores
    
    def _create_anomaly_map(self, patch_scores: np.ndarray, H: int, W: int) -> np.ndarray:
        """Upsample patch scores to full resolution heatmap with edge-aware interpolation"""
        # Reshape to 2D
        patch_h, patch_w = patch_scores.shape
        
        # Edge-aware upsampling (bicubic for smoother results)
        heatmap = cv2.resize(patch_scores, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive Gaussian smoothing (stronger smoothing for high-variance areas)
        sigma = self.gaussian_sigma
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Apply median filter to remove salt-and-pepper noise
        heatmap = median_filter(heatmap, size=3)
        
        # Normalize to [0, 1] with robust statistics
        q1, q99 = np.percentile(heatmap, [1, 99])
        if q99 > q1:
            heatmap = np.clip((heatmap - q1) / (q99 - q1), 0, 1)
        elif heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def _fuse_maps(self, appearance_map: np.ndarray, symmetry_map: np.ndarray) -> np.ndarray:
        """
        Fuse appearance and symmetry maps.
        Symmetry breaks are particularly strong indicators for industrial objects.
        """
        # Upscale symmetry map to match appearance map size
        if symmetry_map.shape != appearance_map.shape:
            H, W = appearance_map.shape
            symmetry_map = cv2.resize(symmetry_map, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Normalize symmetry map
        if symmetry_map.max() > 0:
            symmetry_norm = symmetry_map / symmetry_map.max()
        else:
            symmetry_norm = symmetry_map
        
        # Weighted fusion - symmetry gets higher weight when it's significant
        symmetry_weight = 0.3 + 0.4 * (symmetry_norm.max() > self.symmetry_threshold)
        appearance_weight = 1.0 - symmetry_weight
        
        fused = appearance_weight * appearance_map + symmetry_weight * symmetry_norm
        
        # Ensure range [0, 1]
        fused = np.clip(fused, 0, 1)
        
        return fused
    
    def _post_process(self, heatmap: np.ndarray) -> np.ndarray:
        """Advanced post-processing of anomaly heatmap"""
        # Adaptive thresholding using Otsu's method
        _, binary_otsu = cv2.threshold((heatmap * 255).astype(np.uint8), 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_otsu = binary_otsu / 255.0
        
        # Remove small anomalies with connectivity analysis
        labeled = measure.label(binary_otsu)
        regions = measure.regionprops(labeled)
        
        # Create mask keeping only large enough regions
        large_regions_mask = np.zeros_like(binary_otsu, dtype=bool)
        for region in regions:
            if region.area >= self.min_anomaly_size:
                large_regions_mask[region.coords[:, 0], region.coords[:, 1]] = True
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(large_regions_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to heatmap
        heatmap = heatmap * cleaned
        
        # Final Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        return heatmap
    
    def _classify_defect(self, binary_mask: np.ndarray, 
                        heatmap: np.ndarray, 
                        symmetry_map: np.ndarray) -> DefectType:
        """Enhanced defect classification using both appearance and symmetry"""
        if not np.any(binary_mask):
            return DefectType.UNKNOWN
        
        # Calculate shape features
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return DefectType.UNKNOWN
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        
        # Calculate intensity features
        anomaly_intensities = heatmap[binary_mask == 1]
        intensity_mean = np.mean(anomaly_intensities)
        intensity_std = np.std(anomaly_intensities)
        
        # Calculate symmetry features in defect region
        symmetry_in_defect = symmetry_map[binary_mask == 1]
        symmetry_mean = np.mean(symmetry_in_defect) if len(symmetry_in_defect) > 0 else 0
        symmetry_max = np.max(symmetry_in_defect) if len(symmetry_in_defect) > 0 else 0
        
        # Enhanced classification rules
        if symmetry_max > self.symmetry_threshold:
            return DefectType.SYMMETRY_BREAK
        elif aspect_ratio > 5 or aspect_ratio < 0.2:
            return DefectType.CRACK
        elif circularity < 0.3:
            return DefectType.SCRATCH
        elif intensity_std > 0.25:
            return DefectType.DIRT
        elif intensity_mean > 0.75:
            return DefectType.DISCOLORATION
        elif area > 0.1 * binary_mask.size:  # Large area defect
            return DefectType.DEFORMATION
        else:
            return DefectType.UNKNOWN
    
    def _calculate_severity(self, heatmap: np.ndarray, 
                           binary_mask: np.ndarray,
                           symmetry_map: np.ndarray) -> float:
        """Enhanced severity calculation considering symmetry"""
        if not np.any(binary_mask):
            return 0.0
        
        # Area-based severity
        anomaly_area = np.sum(binary_mask)
        total_area = binary_mask.size
        area_ratio = anomaly_area / total_area
        
        # Intensity-based severity
        anomaly_intensities = heatmap[binary_mask == 1]
        max_intensity = np.max(anomaly_intensities) if len(anomaly_intensities) > 0 else 0
        mean_intensity = np.mean(anomaly_intensities) if len(anomaly_intensities) > 0 else 0
        
        # Symmetry-based severity
        symmetry_in_defect = symmetry_map[binary_mask == 1]
        symmetry_severity = np.mean(symmetry_in_defect) if len(symmetry_in_defect) > 0 else 0
        
        # Combined severity score with symmetry consideration
        severity = (0.3 * area_ratio + 
                   0.3 * max_intensity + 
                   0.2 * mean_intensity + 
                   0.2 * symmetry_severity)
        
        return float(np.clip(severity, 0, 1))
    
    def _find_defect_bbox(self, binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Find bounding box of defect region with adaptive margin"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find union of all defect regions
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Adaptive margin based on defect size
        margin_ratio = 0.1
        margin_x = int(max(5, w * margin_ratio))
        margin_y = int(max(5, h * margin_ratio))
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(binary_mask.shape[1] - x, w + 2 * margin_x)
        h = min(binary_mask.shape[0] - y, h + 2 * margin_y)
        
        return (x, y, x + w, y + h)
    
    def _learn_statistical_thresholds(self, dataloader) -> None:
        """Learn statistical thresholds from normal training data"""
        print("Learning statistical thresholds...")
        
        all_distances = []
        
        for images, labels, _ in dataloader:
            images = images.to(self.device)
            features = self.feature_extractor.extract_patch_features(images)
            distances, _ = self.memory_bank.query(features, k=1)
            all_distances.extend(distances.flatten())
        
        # Compute statistics
        all_distances = np.array(all_distances)
        self.normal_stats = (np.mean(all_distances), np.std(all_distances))
        
        # Set thresholds based on statistical percentiles
        self.image_threshold = np.percentile(all_distances, 97)  # 97th percentile for image-level
        self.pixel_threshold = np.percentile(all_distances, 95)  # 95th percentile for pixel-level
        
        # Normalize thresholds if we have statistics
        if self.normal_stats[1] > 0:
            self.image_threshold = (self.image_threshold - self.normal_stats[0]) / self.normal_stats[1]
            self.pixel_threshold = (self.pixel_threshold - self.normal_stats[0]) / self.normal_stats[1]
        
        print(f"Learned statistical thresholds:")
        print(f"  Normal mean: {self.normal_stats[0]:.4f}, std: {self.normal_stats[1]:.4f}")
        print(f"  Image threshold (normalized): {self.image_threshold:.4f}")
        print(f"  Pixel threshold (normalized): {self.pixel_threshold:.4f}")