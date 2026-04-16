import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Optional, Tuple
import torch
from models.anomaly_inspector import InspectionResult

class AnomalyVisualizer:
    """
    Visualization for anomaly detection results.
    """
    
    @staticmethod
    def visualize_result(image: np.ndarray, result: InspectionResult,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of inspection results.
        
        Args:
            image: Original RGB image (H, W, 3)
            result: InspectionResult object
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Anomaly heatmap
        heatmap = axes[0, 1].imshow(result.anomaly_map, cmap='jet')
        axes[0, 1].set_title(f'Anomaly Heatmap (Score: {result.image_score:.3f})')
        axes[0, 1].axis('off')
        plt.colorbar(heatmap, ax=axes[0, 1])
        
        # Binary mask
        axes[0, 2].imshow(result.binary_mask, cmap='gray')
        axes[0, 2].set_title(f'Binary Mask (Defect: {result.defect_type.value})')
        axes[0, 2].axis('off')
        
        # Overlay on original
        overlay = image.copy()
        mask_rgb = np.stack([result.binary_mask * 255, 
                           np.zeros_like(result.binary_mask), 
                           np.zeros_like(result.binary_mask)], axis=-1)
        overlay = cv2.addWeighted(overlay, 0.7, mask_rgb.astype(np.uint8), 0.3, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Defect Overlay')
        
        # Draw bounding box if exists
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='r', facecolor='none')
            axes[1, 0].add_patch(rect)
        
        axes[1, 0].axis('off')
        
        # Histogram of anomaly scores
        axes[1, 1].hist(result.anomaly_map.flatten(), bins=50, alpha=0.7)
        axes[1, 1].axvline(x=result.image_score, color='r', linestyle='--', 
                          label=f'Image Score: {result.image_score:.3f}')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Score Distribution')
        axes[1, 1].legend()
        
        # Metrics summary
        axes[1, 2].axis('off')
        summary_text = (
            f'Defect Type: {result.defect_type.value}\n'
            f'Confidence: {result.confidence:.3f}\n'
            f'Severity: {result.severity:.3f}\n'
            f'Anomaly Pixels: {result.metadata["num_anomaly_pixels"]}\n'
            f'Is Defective: {result.is_defective()}'
        )
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12,
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Industrial Inspection Result - {result.defect_type.value}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_comparison_grid(images: List[np.ndarray], 
                              results: List[InspectionResult],
                              n_cols: int = 4) -> plt.Figure:
        """
        Create grid comparison of multiple inspection results.
        """
        n_images = len(images)
        n_rows = int(np.ceil(n_images / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, 5 * n_rows))
        
        for i in range(n_images):
            row = i // n_cols
            col = (i % n_cols) * 2
            
            # Original image
            axes[row, col].imshow(images[i])
            axes[row, col].set_title(f'Image {i+1}')
            axes[row, col].axis('off')
            
            # Anomaly overlay
            overlay = images[i].copy()
            mask_rgb = np.stack([results[i].binary_mask * 255, 0, 0], axis=-1)
            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb.astype(np.uint8), 0.3, 0)
            axes[row, col+1].imshow(overlay)
            
            title = (f'Defect: {results[i].defect_type.value}\n'
                    f'Score: {results[i].image_score:.3f}')
            axes[row, col+1].set_title(title)
            axes[row, col+1].axis('off')
        
        # Hide empty subplots
        for i in range(n_images, n_rows * n_cols):
            row = i // n_cols
            col = (i % n_cols) * 2
            axes[row, col].axis('off')
            axes[row, col+1].axis('off')
        
        plt.suptitle('Batch Inspection Results', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auroc: float) -> plt.Figure:
        """Plot ROC curve with AUC score"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auroc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return fig