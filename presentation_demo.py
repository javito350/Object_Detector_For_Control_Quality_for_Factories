"""
G-CNN Defect Detection System - Presentation Demo
High-quality anomaly detection for water bottle inspection
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure proper encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import model
from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

class PresentationDemo:
    def __init__(self, model_path='sensitive_inspector.pth'):
        print("=" * 70)
        print("G-CNN DEFECT DETECTION SYSTEM")
        print("Automated Quality Control for Manufacturing")
        print("=" * 70)
        
        # Load model
        print("\n[1/3] Loading trained model...")
        self.inspector = torch.load(model_path, map_location='cpu', weights_only=False)
        print("      Model loaded successfully!")
        
        # Setup preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Get threshold
        self.threshold = self.inspector.image_threshold
        if hasattr(self.threshold, 'item'):
            self.threshold = self.threshold.item()
        
        print(f"[2/3] System configured")
        print(f"      Threshold: {self.threshold:.4f}")
        print(f"      Backbone: {getattr(self.inspector, 'feature_extractor', 'Unknown')}")
        print("[3/3] Ready for inspection!")
        print("=" * 70)
    
    def inspect_image(self, image_path, save_visualization=True):
        """Inspect a single image and return results"""
        if not os.path.exists(image_path):
            print(f"\nERROR: Image not found: {image_path}")
            return None
        
        # Load image
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0)
        
        # Run inspection
        results = self.inspector.predict(img_tensor)
        result = results[0]
        
        # Determine status
        score = result.image_score
        is_defective = score > self.threshold
        
        # Print results
        print(f"\n{'='*70}")
        print(f"IMAGE: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        if is_defective:
            print(f"STATUS: DEFECTIVE")
            print(f"  >> Defect Type: {result.defect_type.value.upper()}")
            print(f"  >> Severity: {result.severity*100:.1f}%")
            print(f"  >> Confidence: {result.confidence*100:.1f}%")
        else:
            print(f"STATUS: GOOD (Ready for Sale)")
            print(f"  >> Quality Score: {(1-score/self.threshold)*100:.1f}%")
        
        print(f"\nTECHNICAL DETAILS:")
        print(f"  Anomaly Score: {score:.4f}")
        print(f"  Threshold: {self.threshold:.4f}")
        print(f"  Difference: {(score-self.threshold):.4f}")
        
        if result.metadata:
            meta = result.metadata
            print(f"  Anomaly Pixels: {meta.get('num_anomaly_pixels', 0)}")
            print(f"  Appearance Score: {meta.get('appearance_score', 0):.4f}")
            print(f"  Symmetry Score: {meta.get('symmetry_score', 0):.4f}")
        
        if result.bbox and is_defective:
            x1, y1, x2, y2 = result.bbox
            print(f"  Defect Location: [{x1}, {y1}] to [{x2}, {y2}]")
        
        # Save visualization
        if save_visualization:
            self.save_visualization(image_path, img_pil, result, is_defective)
        
        return result, is_defective
    
    def save_visualization(self, image_path, img_pil, result, is_defective):
        """Create and save visualization with heatmap"""
        output_dir = Path("presentation_results")
        output_dir.mkdir(exist_ok=True)
        
        img_name = Path(image_path).stem
        
        # Convert PIL to numpy
        img_np = np.array(img_pil)
        
        # Resize heatmaps to match image size
        h, w = img_np.shape[:2]
        anomaly_map = cv2.resize(result.anomaly_map, (w, h))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Anomaly heatmap
        im1 = axes[1].imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Anomaly Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = img_np.copy()
        heatmap_colored = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
        
        # Draw bounding box if defective
        if is_defective and result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        axes[2].imshow(overlay)
        status_text = "DEFECTIVE" if is_defective else "GOOD"
        color = 'red' if is_defective else 'green'
        axes[2].set_title(f'Result: {status_text}', fontsize=14, fontweight='bold', color=color)
        axes[2].axis('off')
        
        # Add score text
        score_text = f"Score: {result.image_score:.4f}\nThreshold: {self.threshold:.4f}"
        fig.text(0.5, 0.02, score_text, ha='center', fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"{img_name}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {output_path}")
    
    def batch_inspect(self, image_dir):
        """Inspect all images in a directory"""
        print(f"\n\nBATCH INSPECTION MODE")
        print(f"Directory: {image_dir}")
        print("=" * 70)
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(image_dir).glob(ext))
        
        if not image_files:
            print("No images found!")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Inspect each image
        results_summary = []
        for img_path in sorted(image_files):
            result, is_defective = self.inspect_image(str(img_path), save_visualization=True)
            results_summary.append({
                'image': img_path.name,
                'status': 'DEFECTIVE' if is_defective else 'GOOD',
                'score': result.image_score,
                'defect_type': result.defect_type.value if is_defective else 'N/A'
            })
        
        # Summary report
        print(f"\n\n{'='*70}")
        print("INSPECTION SUMMARY")
        print(f"{'='*70}")
        print(f"{'Image':<30} {'Status':<12} {'Score':<10} {'Defect Type'}")
        print("-" * 70)
        
        good_count = 0
        defect_count = 0
        
        for r in results_summary:
            status_symbol = "[X]" if r['status'] == 'DEFECTIVE' else "[OK]"
            print(f"{r['image']:<30} {status_symbol} {r['status']:<8} {r['score']:<10.4f} {r['defect_type']}")
            if r['status'] == 'GOOD':
                good_count += 1
            else:
                defect_count += 1
        
        print("-" * 70)
        print(f"Total: {len(results_summary)} images")
        print(f"Good: {good_count} ({good_count/len(results_summary)*100:.1f}%)")
        print(f"Defective: {defect_count} ({defect_count/len(results_summary)*100:.1f}%)")
        print(f"{'='*70}")
        
        # Save summary to file
        summary_path = Path("presentation_results") / "inspection_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("G-CNN DEFECT DETECTION - INSPECTION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Image':<30} {'Status':<12} {'Score':<10} {'Defect Type'}\n")
            f.write("-" * 70 + "\n")
            for r in results_summary:
                f.write(f"{r['image']:<30} {r['status']:<12} {r['score']:<10.4f} {r['defect_type']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total: {len(results_summary)}\n")
            f.write(f"Good: {good_count} ({good_count/len(results_summary)*100:.1f}%)\n")
            f.write(f"Defective: {defect_count} ({defect_count/len(results_summary)*100:.1f}%)\n")
        
        print(f"\nSummary saved to: {summary_path}")

def main():
    # Initialize demo
    demo = PresentationDemo()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            demo.batch_inspect(path)
        else:
            demo.inspect_image(path, save_visualization=True)
    else:
        # Default: inspect all test images
        test_dir = "data/water_bottles/test"
        if os.path.exists(test_dir):
            demo.batch_inspect(test_dir)
        else:
            print("\nUsage:")
            print("  python presentation_demo.py <image_path>         # Single image")
            print("  python presentation_demo.py <directory_path>     # Batch mode")
            print("  python presentation_demo.py                      # Default test set")

if __name__ == "__main__":
    main()
