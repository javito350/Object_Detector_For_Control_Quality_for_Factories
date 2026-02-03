import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch
from typing import Dict, List, Tuple, Optional
import json

class AdvancedMetrics:
    """
    PhD-level evaluation metrics for anomaly detection.
    Includes pixel-level and image-level metrics.
    """
    
    @staticmethod
    def compute_auroc(pred_scores: np.ndarray, gt_labels: np.ndarray) -> float:
        """Compute Area Under ROC Curve"""
        return roc_auc_score(gt_labels, pred_scores)
    
    @staticmethod
    def compute_ap(pred_scores: np.ndarray, gt_labels: np.ndarray) -> float:
        """Compute Average Precision"""
        return average_precision_score(gt_labels, pred_scores)
    
    @staticmethod
    def compute_pro_curve(pred_masks: np.ndarray, gt_masks: np.ndarray, 
                         max_fpr: float = 0.3) -> float:
        """
        Compute Per-Region Overlap (PRO) Curve - SOTA metric for anomaly localization.
        
        Args:
            pred_masks: (N, H, W) predicted anomaly masks
            gt_masks: (N, H, W) ground truth masks
            max_fpr: Maximum false positive rate
            
        Returns:
            PRO score
        """
        N = len(pred_masks)
        pro_scores = []
        
        for i in range(N):
            pred_mask = pred_masks[i].flatten()
            gt_mask = gt_masks[i].flatten()
            
            # Only consider regions with anomalies
            anomaly_regions = np.where(gt_mask > 0)[0]
            if len(anomaly_regions) == 0:
                continue
            
            # Sort predictions
            sorted_indices = np.argsort(pred_mask)[::-1]
            
            # Calculate overlap at different thresholds
            tp, fp = 0, 0
            total_anomaly_pixels = len(anomaly_regions)
            
            for idx in sorted_indices:
                if idx in anomaly_regions:
                    tp += 1
                else:
                    fp += 1
                
                fpr = fp / (len(pred_mask) - total_anomaly_pixels)
                if fpr > max_fpr:
                    break
            
            pro_score = tp / total_anomaly_pixels
            pro_scores.append(pro_score)
        
        return np.mean(pro_scores) if pro_scores else 0.0
    
    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        """Compute Intersection over Union"""
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
        """Compute Dice Score (F1 score for segmentation)"""
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        
        if intersection == 0:
            return 0.0
        
        return 2 * intersection / (pred_binary.sum() + gt_binary.sum())
    
    @staticmethod
    def compute_all_metrics(pred_scores: np.ndarray, gt_labels: np.ndarray,
                           pred_masks: np.ndarray, gt_masks: np.ndarray) -> Dict:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Image-level metrics
        metrics['auroc'] = AdvancedMetrics.compute_auroc(pred_scores, gt_labels)
        metrics['ap'] = AdvancedMetrics.compute_ap(pred_scores, gt_labels)
        
        # Pixel-level metrics
        if pred_masks is not None and gt_masks is not None:
            metrics['pro'] = AdvancedMetrics.compute_pro_curve(pred_masks, gt_masks)
            
            # Compute IoU and Dice for each image
            ious = []
            dices = []
            
            for i in range(len(pred_masks)):
                iou = AdvancedMetrics.compute_iou(pred_masks[i], gt_masks[i])
                dice = AdvancedMetrics.compute_dice_score(pred_masks[i], gt_masks[i])
                ious.append(iou)
                dices.append(dice)
            
            metrics['mean_iou'] = np.mean(ious)
            metrics['mean_dice'] = np.mean(dices)
            
            # Additional statistics
            metrics['detection_rate'] = np.mean(pred_scores > 0.5)
            metrics['false_positive_rate'] = np.mean((pred_scores > 0.5) & (gt_labels == 0))
        
        return metrics
    
    @staticmethod
    def save_metrics(metrics: Dict, path: str) -> None:
        """Save metrics to JSON file"""
        with open(path, 'w') as f:
            # Convert numpy types to Python types
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_serializable[key] = float(value)
                else:
                    metrics_serializable[key] = value
            
            json.dump(metrics_serializable, f, indent=4)