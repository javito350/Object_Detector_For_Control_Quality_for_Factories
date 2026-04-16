import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import time
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from .symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from .memory_bank import MemoryBank
from .thresholding import EVTCalibrator


LOGGER = logging.getLogger(__name__)

class DefectType(Enum):
    NOMINAL = "nominal"
    ANOMALY = "anomaly"

@dataclass
class InspectionResult:
    image_score: float         # Continuous anomaly score (max patch distance)
    anomaly_map: np.ndarray    # Spatial heatmap
    binary_mask: np.ndarray    # Binarized mask based on pixel threshold
    defect_type: DefectType    # Classification based on EVT threshold
    inference_time_ms: float   # Explicit latency tracking for your paper
    is_defective: bool

class EnhancedAnomalyInspector:
    """
    Core anomaly detection pipeline.
    Integrates the frozen WRN50 backbone, p4m symmetry augmentation, 
    FAISS IVF-PQ memory bank, and EVT statistical thresholding.
    """
    def __init__(self, 
                 backbone: str = "wide_resnet50_2",
                 device: str = None,
                 coreset_percentage: float = 0.1,
                 use_pq: bool = True):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = SymmetryAwareFeatureExtractor(
            backbone=backbone, 
            device=self.device
        )
        
        # WRN50 layer2 + layer3 concatenated = 1536 dimensions
        feature_dim = 1536 
        
        self.memory_bank = MemoryBank(dimension=feature_dim, use_gpu=(self.device == "cuda"), use_pq=use_pq)
        self.coreset_percentage = coreset_percentage
        
        # Will be calibrated via EVT
        self.image_threshold = 0.0 
        self.pixel_threshold = 0.0

    def fit(self, dataloader, apply_p4m: bool = True) -> None:
        """
        Builds the memory bank from the few-shot normal support set (N <= 10).
        """
        print("Building memory bank from normal samples...")
        all_features = []
        sample_count = 0
        
        for batch_idx, (images, _, _) in enumerate(dataloader):
            if images is None or images.ndim != 4:
                raise ValueError(f"Batch {batch_idx} has invalid image tensor shape: {getattr(images, 'shape', None)}")
            images = images.to(self.device)
            sample_count += int(images.shape[0])
            # Extract features (applies 8x augmentation if apply_p4m=True)
            features = self.feature_extractor.extract_patch_features(images, apply_p4m=apply_p4m)
            if features is None or len(features) == 0:
                raise ValueError(f"Empty feature batch extracted at batch {batch_idx}")
            all_features.append(features)

        if sample_count == 0:
            raise ValueError("No training samples found in dataloader; cannot build memory bank")
            
        # Build FAISS Index
        self.memory_bank.build(all_features, coreset_percentage=self.coreset_percentage)
        print("Memory bank built successfully!")
        
        # Calibrate Thresholds using Extreme Value Theory
        self._calibrate_evt_thresholds(all_features)

    def _calibrate_evt_thresholds(self, features_list: list) -> None:
        print("Calibrating thresholds using Extreme Value Theory (GPD)...")
        # Query the training features against themselves to get baseline nominal distances
        training_features = np.vstack(features_list)
        distances, _ = self.memory_bank.query(training_features, k=1)
        
        # Fit EVT for a strict 1% False Positive Rate
        calibrator = EVTCalibrator(tail_fraction=0.10, target_fpr=0.01)
        self.image_threshold = calibrator.fit(distances.flatten())
        self.pixel_threshold = self.image_threshold * 0.9 # Slightly lower threshold for spatial mapping
        
        print(f"EVT Calibration Complete. Decision Boundary (tau): {self.image_threshold:.4f}")

    def predict(self, images: torch.Tensor, apply_p4m: bool = False) -> list:
        """
        Runs inference on test images.
        NOTE: apply_p4m is False during inference; we only augment the support set!
        """
        results = []

        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images)}")
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"images must have 3 channels (RGB), got {images.shape[1]}")
        if not self.memory_bank.is_trained:
            raise RuntimeError("Memory bank is not trained. Call fit(...) before predict(...)")
        
        with torch.no_grad():
            images = images.to(self.device)
            B, C, H, W = images.shape
            
            # --- EDGE INFERENCE TIMING BLOCK ---
            start_time = time.perf_counter()
            
            # 1. Extract patch features (single forward pass, no augmentation)
            batch_features = self.feature_extractor.extract_patch_features(images, apply_p4m=apply_p4m)
            if not np.isfinite(batch_features).all():
                raise ValueError("Extracted features contain NaN/Inf values")
            
            # 2. FAISS Retrieval
            try:
                distances, _ = self.memory_bank.query(batch_features, k=1)
            except Exception as exc:
                raise RuntimeError(f"Memory bank query failed during prediction: {exc}") from exc
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000.0
            # ------------------------------------
            
            # Reshape distances back into spatial heatmaps (assuming 28x28 feature grid)
            patches_per_image = distances.shape[0] // B
            grid_size = int(np.sqrt(patches_per_image)) if patches_per_image > 0 else 0
            if grid_size <= 0 or grid_size * grid_size != patches_per_image:
                raise ValueError(
                    f"Cannot reshape patch distances into square grid: total={distances.shape[0]}, batch={B}"
                )
            patch_h, patch_w = grid_size, grid_size
            distances_reshaped = distances.reshape(B, patch_h, patch_w)
            
            for i in range(B):
                # Image-level anomaly score is the maximum patch distance
                image_score = float(np.max(distances_reshaped[i]))
                is_defective = image_score > self.image_threshold
                defect_type = DefectType.ANOMALY if is_defective else DefectType.NOMINAL
                
                # Upsample heatmap to original image size
                anomaly_map = cv2.resize(distances_reshaped[i], (W, H), interpolation=cv2.INTER_CUBIC)
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                
                # Binarize mask
                binary_mask = (anomaly_map > self.pixel_threshold).astype(np.uint8)

                if not np.isfinite(image_score):
                    LOGGER.warning("Non-finite image score detected; replacing with 0.0")
                    image_score = 0.0
                
                results.append(InspectionResult(
                    image_score=image_score,
                    anomaly_map=anomaly_map,
                    binary_mask=binary_mask,
                    defect_type=defect_type,
                    inference_time_ms=inference_time_ms,
                    is_defective=is_defective
                ))
                
        return results