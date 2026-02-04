"""
PRESENTATION SCRIPT - Final Demo for Tomorrow
Shows both GOOD and DEFECTIVE samples with clear labels
"""
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

class PresentationFinal:
    def __init__(self):
        print("="*80)
        print(" "*20 + "G-CNN DEFECT DETECTION SYSTEM")
        print(" "*15 + "Live Demonstration - February 2026")
        print("="*80)
        
        # Load model
        print("\n[SYSTEM INITIALIZATION]")
        print("Loading trained neural network...")
        self.inspector = torch.load('sensitive_inspector.pth', map_location='cpu', weights_only=False)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.threshold = self.inspector.image_threshold
        if hasattr(self.threshold, 'item'):
            self.threshold = self.threshold.item()
        
        print(f"Model loaded: Wide ResNet-50 with Symmetry Analysis")
        print(f"Detection Threshold: {self.threshold:.4f}")
        print("Status: READY FOR INSPECTION")
        print("="*80)
    
    def create_demo_samples(self):
        """Prepare labeled demo samples"""
        print("\n[PREPARING DEMO SAMPLES]")
        
        # Create demo directory
        demo_dir = Path("DEMO_SAMPLES")
        demo_dir.mkdir(exist_ok=True)
        
        # Good samples (from test)
        good_samples = []
        test_dir = Path("data/water_bottles/test")
        if test_dir.exists():
            for img_file in list(test_dir.glob("*.jpeg"))[:2]:  # Take 2 good ones
                new_path = demo_dir / f"GOOD_{img_file.name}"
                if not new_path.exists():
                    img = Image.open(img_file)
                    img.save(new_path)
                good_samples.append(str(new_path))
                print(f"  [OK] Good sample: {new_path.name}")
        
        # Defective samples (create labeled ones)
        defect_samples = []
        defect_dir = Path("defective_samples")
        if defect_dir.exists():
            for defect_file in ["defect_crack.jpg", "defect_missing.jpg", "defect_asymmetric.jpg"]:
                src = defect_dir / defect_file
                if src.exists():
                    dst = demo_dir / f"DEFECTIVE_{defect_file}"
                    if not dst.exists():
                        img = Image.open(src)
                        img.save(dst)
                    defect_samples.append(str(dst))
                    print(f"  [X] Defective sample: {dst.name}")
        
        all_samples = good_samples + defect_samples
        print(f"\nTotal demo samples prepared: {len(all_samples)}")
        print(f"  Good: {len(good_samples)}")
        print(f"  Defective: {len(defect_samples)}")
        
        return all_samples
    
    def inspect_with_presentation(self, image_path):
        """Inspect and create presentation-quality visualization"""
        img_name = Path(image_path).name
        
        # Determine expected result from filename
        expected_good = "GOOD" in img_name.upper()
        expected_defective = "DEFECTIVE" in img_name.upper() or "DEFECT" in img_name.upper()
        
        print(f"\n{'='*80}")
        print(f"INSPECTING: {img_name}")
        if expected_good:
            print("EXPECTED RESULT: GOOD (Quality product)")
        elif expected_defective:
            print("EXPECTED RESULT: DEFECTIVE (Has issues)")
        print(f"{'='*80}")
        
        # Load and predict
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0)
        
        results = self.inspector.predict(img_tensor)
        result = results[0]
        
        score = result.image_score
        is_defective = score > self.threshold
        
        # Display result
        print(f"\nPREDICTION RESULT:")
        if is_defective:
            print(f"  STATUS: DEFECTIVE")
            print(f"  Defect Type: {result.defect_type.value.upper()}")
            print(f"  Severity: {result.severity*100:.1f}%")
            print(f"  Confidence: {result.confidence*100:.1f}%")
            verdict_symbol = "[X] REJECT"
        else:
            print(f"  STATUS: GOOD")
            print(f"  Quality: {(1-score/self.threshold)*100:.1f}%")
            verdict_symbol = "[OK] ACCEPT"
        
        print(f"\nTECHNICAL METRICS:")
        print(f"  Anomaly Score: {score:.4f}")
        print(f"  Threshold: {self.threshold:.4f}")
        print(f"  Delta: {(score-self.threshold):+.4f}")
        
        # Check if prediction matches expectation
        if expected_good and not is_defective:
            print(f"\n>>> CORRECT: Model correctly identified as GOOD")
        elif expected_defective and is_defective:
            print(f"\n>>> CORRECT: Model correctly identified as DEFECTIVE")
        elif expected_good and is_defective:
            print(f"\n>>> Note: Model detected issues in supposedly good sample")
        elif expected_defective and not is_defective:
            print(f"\n>>> Note: Model classified defective sample as good (may need recalibration)")
        
        # Create visualization
        self.create_presentation_viz(image_path, img_pil, result, is_defective, verdict_symbol)
        
        return is_defective
    
    def create_presentation_viz(self, image_path, img_pil, result, is_defective, verdict):
        """Create high-quality visualization for presentation"""
        output_dir = Path("PRESENTATION_SLIDES")
        output_dir.mkdir(exist_ok=True)
        
        img_name = Path(image_path).stem
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        
        # Resize heatmap
        anomaly_map = cv2.resize(result.anomaly_map, (w, h))
        
        # Create 3-panel figure
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_np)
        ax1.set_title('Input Image', fontsize=16, fontweight='bold', pad=15)
        ax1.axis('off')
        
        # Panel 2: Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
        ax2.set_title('Anomaly Heatmap\n(Red = Defect Detected)', fontsize=16, fontweight='bold', pad=15)
        ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=20, fontsize=12)
        
        # Panel 3: Result
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = img_np.copy()
        
        # Apply heatmap overlay
        heatmap_colored = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
        
        # Draw bbox if defective
        if is_defective and result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv2.putText(overlay, 'DEFECT', (x1, y1-10), 
                       cv2.FONT_HERSHEY_BOLD, 1.5, (255, 0, 0), 3)
        
        ax3.imshow(overlay)
        
        # Title with verdict
        if is_defective:
            title_text = 'DEFECTIVE'
            title_color = 'red'
            bg_color = '#ffcccc'
        else:
            title_text = 'GOOD'
            title_color = 'green'
            bg_color = '#ccffcc'
        
        ax3.set_title(f'Result: {title_text}', 
                     fontsize=16, fontweight='bold', pad=15,
                     color=title_color, 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, edgecolor=title_color, linewidth=2))
        ax3.axis('off')
        
        # Add footer with metrics
        score = result.image_score
        footer_text = f"Anomaly Score: {score:.4f}  |  Threshold: {self.threshold:.4f}  |  Verdict: {verdict}"
        fig.text(0.5, 0.02, footer_text, ha='center', fontsize=12, 
                family='monospace', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save
        output_path = output_dir / f"{img_name}_RESULT.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Visualization saved: {output_path}")
    
    def run_full_demo(self):
        """Run complete presentation demo"""
        print("\n" + "="*80)
        print(" "*25 + "STARTING FULL DEMONSTRATION")
        print("="*80)
        
        # Prepare samples
        samples = self.create_demo_samples()
        
        if not samples:
            print("\nERROR: No demo samples found!")
            print("Please ensure test images exist in data/water_bottles/test/")
            return
        
        # Inspect each sample
        good_count = 0
        defect_count = 0
        
        for sample_path in samples:
            is_defective = self.inspect_with_presentation(sample_path)
            if is_defective:
                defect_count += 1
            else:
                good_count += 1
        
        # Final summary
        print("\n" + "="*80)
        print(" "*25 + "DEMONSTRATION COMPLETE")
        print("="*80)
        print(f"\nINSPECTION SUMMARY:")
        print(f"  Total Images: {len(samples)}")
        print(f"  Good Products: {good_count} ({good_count/len(samples)*100:.1f}%)")
        print(f"  Defective Products: {defect_count} ({defect_count/len(samples)*100:.1f}%)")
        print(f"\nRESULTS SAVED TO:")
        print(f"  • Visualizations: PRESENTATION_SLIDES/")
        print(f"  • Demo Samples: DEMO_SAMPLES/")
        print("\n" + "="*80)
        print("SYSTEM READY FOR PRESENTATION")
        print("Open the PRESENTATION_SLIDES folder to display results!")
        print("="*80 + "\n")

def main():
    demo = PresentationFinal()
    demo.run_full_demo()

if __name__ == "__main__":
    main()
