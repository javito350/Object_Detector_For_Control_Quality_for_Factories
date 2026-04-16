"""
Grid Sweep Evaluation Script for Few-Shot Anomaly Detection
Evaluates EnhancedAnomalyInspector on MVTec AD dataset
Outputs academic metrics for research papers: AUROC (image/pixel), latency, FN
"""

import os
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
import cv2
import warnings

warnings.filterwarnings('ignore')

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data" / "mvtec_toothbrush"
TEST_DIR = DATA_DIR / "test"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

sys.path.append(str(ROOT_DIR))

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SquarePad
from models.memory_bank import MemoryBank


def load_checkpoint_into_inspector(checkpoint_path):
    """
    Loads a checkpoint dict and reconstructs an EnhancedAnomalyInspector instance.
    Handles the PyTorch serialization workaround for nested hook functions.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load with weights_only=False to handle FAISS IndexIVFPQ objects
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Initialize fresh inspector
    inspector = EnhancedAnomalyInspector(
        backbone="wide_resnet50_2",
        device=checkpoint['device'],
        coreset_percentage=checkpoint['coreset_percentage'],
        use_pq=True
    )
    
    # Restore feature extractor weights
    inspector.feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])
    
    # Restore memory bank
    inspector.memory_bank.index = checkpoint['memory_bank_index']
    inspector.memory_bank.dimension = checkpoint['memory_bank_dimension']
    inspector.memory_bank.is_trained = True  # Mark as trained
    
    # Restore thresholds
    inspector.image_threshold = checkpoint['image_threshold']
    inspector.pixel_threshold = checkpoint['pixel_threshold']
    
    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Device: {inspector.device}")
    print(f"  Image Threshold (tau): {inspector.image_threshold:.4f}")
    print(f"  Pixel Threshold (tau_px): {inspector.pixel_threshold:.4f}")
    
    return inspector


def get_transform():
    """
    Returns the inference transform pipeline with SquarePad to preserve aspect ratio.
    """
    return transforms.Compose([
        SquarePad(),  # Prevent distortion on rectangular images
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_ground_truth_mask(mask_path):
    """Load ground truth mask and normalize to [0, 1]."""
    if not Path(mask_path).exists():
        return None
    mask = Image.open(mask_path).convert('L')  # Grayscale
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    return mask_array


def evaluate_grid_sweep(inspector, test_dir, ground_truth_dir):
    """
    Evaluates the inspector on all test images.
    Returns metrics dict and detailed results.
    """
    transform = get_transform()
    
    # Lists to accumulate predictions and ground truth
    image_scores = []
    image_labels = []
    pixel_scores_list = []  # List of flattened anomaly maps
    pixel_labels_list = []   # List of flattened ground truth masks
    inference_times = []
    false_negatives = 0
    
    print("\n" + "=" * 70)
    print("EVALUATING TEST SAMPLES")
    print("=" * 70)
    
    # 1. Process GOOD samples (class 0)
    good_dir = test_dir / "good"
    if good_dir.exists():
        good_images = sorted(good_dir.glob("*.png"))
        print(f"\nProcessing {len(good_images)} GOOD samples...")
        
        for img_path in good_images:
            try:
                # Load and preprocess
                img_pil = Image.open(img_path).convert('RGB')
                img_tensor = transform(img_pil).unsqueeze(0)
                h, w = img_pil.size[::-1]  # Original dimensions
                
                # Move to inspector device
                img_tensor = img_tensor.to(inspector.device)
                
                # Predict
                results = inspector.predict(img_tensor, apply_p4m=False)
                result = results[0]
                
                # Accumulate image-level metrics
                image_scores.append(result.image_score)
                image_labels.append(0)  # Good = 0
                inference_times.append(result.inference_time_ms)
                
                # Accumulate pixel-level metrics (good images have all-zero ground truth)
                dummy_ground_truth = np.zeros_like(result.anomaly_map, dtype=np.float32)
                pixel_scores_list.append(result.anomaly_map.flatten())
                pixel_labels_list.append(dummy_ground_truth.flatten())
                
                print(f"  ✓ {img_path.name}: score={result.image_score:.4f}, time={result.inference_time_ms:.2f}ms")
                
            except Exception as e:
                print(f"  ✗ {img_path.name}: {str(e)}")
    
    # 2. Process DEFECTIVE samples (class 1)
    defective_dir = test_dir / "defective"
    if defective_dir.exists():
        defective_images = sorted(defective_dir.glob("*.png"))
        print(f"\nProcessing {len(defective_images)} DEFECTIVE samples...")
        
        for img_path in defective_images:
            try:
                # Extract image index for ground truth matching
                img_idx = img_path.stem  # e.g., "000"
                mask_name = f"{img_idx}_mask.png"
                mask_path = ground_truth_dir / "defective" / mask_name
                
                # Load and preprocess image
                img_pil = Image.open(img_path).convert('RGB')
                img_tensor = transform(img_pil).unsqueeze(0)
                h_orig, w_orig = img_pil.size[::-1]  # Original dimensions
                
                # Move to inspector device
                img_tensor = img_tensor.to(inspector.device)
                
                # Predict
                results = inspector.predict(img_tensor, apply_p4m=False)
                result = results[0]
                
                # Accumulate image-level metrics
                image_scores.append(result.image_score)
                image_labels.append(1)  # Defective = 1
                inference_times.append(result.inference_time_ms)
                
                # Count false negatives (missed detections)
                if not result.is_defective:  # Model predicted as nominal but it's defective
                    false_negatives += 1
                
                # Accumulate pixel-level metrics with ground truth mask
                gt_mask = load_ground_truth_mask(mask_path)
                if gt_mask is not None:
                    # Resize anomaly map to original image dimensions if needed
                    anomaly_map_resized = cv2.resize(result.anomaly_map, (w_orig, h_orig), 
                                                      interpolation=cv2.INTER_CUBIC)
                    pixel_scores_list.append(anomaly_map_resized.flatten())
                    pixel_labels_list.append(gt_mask.flatten())
                else:
                    # Fallback if mask not found
                    pixel_scores_list.append(result.anomaly_map.flatten())
                    pixel_labels_list.append(np.ones_like(result.anomaly_map).flatten())
                
                print(f"  ✓ {img_path.name}: score={result.image_score:.4f}, time={result.inference_time_ms:.2f}ms, "
                      f"detected={result.is_defective}")
                
            except Exception as e:
                print(f"  ✗ {img_path.name}: {str(e)}")
    
    # Convert to numpy arrays
    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)
    pixel_scores_all = np.concatenate(pixel_scores_list)
    pixel_labels_all = np.concatenate(pixel_labels_list)
    
    # Compute metrics
    try:
        image_auroc = roc_auc_score(image_labels, image_scores)
    except:
        image_auroc = np.nan
        print("⚠ Warning: Could not compute image-level AUROC")
    
    try:
        pixel_auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
    except:
        pixel_auroc = np.nan
        print("⚠ Warning: Could not compute pixel-level AUROC")
    
    avg_latency_ms = np.mean(inference_times)
    
    metrics = {
        'image_auroc': image_auroc,
        'pixel_auroc': pixel_auroc,
        'avg_latency_ms': avg_latency_ms,
        'false_negatives': false_negatives,
        'total_defective': len([1 for l in image_labels if l == 1]),
        'total_good': len([0 for l in image_labels if l == 0]),
        'total_images': len(image_labels),
    }
    
    return metrics


def print_results_table(metrics):
    """Print formatted ASCII table of evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Build table
    lines = []
    lines.append("┌" + "─" * 68 + "┐")
    lines.append("│ METRIC                                        VALUE              │")
    lines.append("├" + "─" * 68 + "┤")
    lines.append(f"│ Image-Level AUROC (Higher is Better)          {metrics['image_auroc']:>15.4f} │")
    lines.append(f"│ Pixel-Level AUROC (Higher is Better)          {metrics['pixel_auroc']:>15.4f} │")
    lines.append(f"│ Average Inference Latency (ms)                {metrics['avg_latency_ms']:>15.2f} │")
    lines.append(f"│ Total False Negatives (Critical)              {metrics['false_negatives']:>15d} │")
    lines.append("├" + "─" * 68 + "┤")
    lines.append(f"│ Test Set Composition                                            │")
    lines.append(f"│   - Good Samples                              {metrics['total_good']:>15d} │")
    lines.append(f"│   - Defective Samples                         {metrics['total_defective']:>15d} │")
    lines.append(f"│   - Total Images                              {metrics['total_images']:>15d} │")
    lines.append("└" + "─" * 68 + "┘")
    
    for line in lines:
        print(line)
    
    print("\n" + "=" * 70)
    print("NOTES FOR PAPER:")
    print("=" * 70)
    print(f"• Report Image-Level AUROC: {metrics['image_auroc']:.4f}")
    print(f"• Report Pixel-Level AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"• Report Inference Time: {metrics['avg_latency_ms']:.2f} ms per image")
    print(f"• Critical Production Metric - False Negatives: {metrics['false_negatives']}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("FEW-SHOT ANOMALY DETECTION - GRID SWEEP EVALUATION")
    print("=" * 70)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Test Data: {TEST_DIR}")
    print(f"Checkpoint: {WEIGHTS_DIR / 'calibrated_inspector.pth'}")
    
    # Load checkpoint and reconstruct model
    checkpoint_path = WEIGHTS_DIR / "calibrated_inspector.pth"
    inspector = load_checkpoint_into_inspector(checkpoint_path)
    
    # Run grid sweep evaluation
    metrics = evaluate_grid_sweep(inspector, TEST_DIR, GROUND_TRUTH_DIR)
    
    # Print results
    print_results_table(metrics)


if __name__ == "__main__":
    main()