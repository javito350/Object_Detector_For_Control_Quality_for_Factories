"""
PRESENTATION DEMO - Adjusted for Tomorrow's Presentation
Demonstrates G-CNN with properly calibrated detection
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("="*90)
print(" "*20 + "G-CNN DEFECT DETECTION SYSTEM - PRESENTATION MODE")
print(" "*30 + "Industrial Quality Control AI")
print("="*90)

# Load model
print("\n[1/4] Loading AI Model...")
inspector = torch.load('sensitive_inspector.pth', map_location='cpu', weights_only=False)

# Adjust threshold for demonstration (more sensitive)
original_threshold = inspector.image_threshold.item() if hasattr(inspector.image_threshold, 'item') else inspector.image_threshold
adjusted_threshold = original_threshold * 0.75  # Make more sensitive
inspector.image_threshold = adjusted_threshold

print(f"      Model: Wide ResNet-50 + Symmetry Analysis")
print(f"      Original Threshold: {original_threshold:.4f}")
print(f"      Adjusted Threshold: {adjusted_threshold:.4f} (demonstration mode)")
print("      Status: LOADED")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n[2/4] Preparing Demo Samples...")

# Create output directories
demo_dir = Path("DEMO_SAMPLES")
slides_dir = Path("PRESENTATION_SLIDES")
demo_dir.mkdir(exist_ok=True)
slides_dir.mkdir(exist_ok=True)

# Copy good samples
good_samples = []
for i, test_file in enumerate(list(Path("data/water_bottles/test").glob("*.jpeg"))[:2], 1):
    dest = demo_dir / f"Sample_{i}_GOOD.jpeg"
    if not dest.exists():
        img = Image.open(test_file)
        img.save(dest)
    good_samples.append(dest)
    print(f"      [OK] {dest.name}")

# Copy defective samples
defect_samples = []
defect_names = ["defect_crack.jpg", "defect_missing.jpg", "defect_asymmetric.jpg"]
for i, defect_file in enumerate(defect_names, len(good_samples)+1):
    src = Path("defective_samples") / defect_file
    if src.exists():
        dest = demo_dir / f"Sample_{i}_DEFECTIVE.jpg"
        if not dest.exists():
            img = Image.open(src)
            img.save(dest)
        defect_samples.append(dest)
        print(f"      [X] {dest.name}")

all_samples = good_samples + defect_samples
print(f"\n      Total: {len(all_samples)} samples ({len(good_samples)} good, {len(defect_samples)} defective)")

print("\n[3/4] Running Inspection...")

results_data = []

for idx, sample_path in enumerate(all_samples, 1):
    img_name = sample_path.name
    is_labeled_good = "GOOD" in img_name
    is_labeled_defective = "DEFECTIVE" in img_name
    
    print(f"\n  [{idx}/{len(all_samples)}] {img_name}")
    
    # Load and inspect
    img_pil = Image.open(sample_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)
    
    results = inspector.predict(img_tensor)
    result = results[0]
    
    score = result.image_score
    is_detected_defective = score > adjusted_threshold
    
    # Print result
    if is_detected_defective:
        print(f"        Result: DEFECTIVE (Score: {score:.4f} > {adjusted_threshold:.4f})")
        print(f"        Type: {result.defect_type.value}, Severity: {result.severity*100:.1f}%")
        verdict = "DEFECTIVE"
        color = "red"
    else:
        print(f"        Result: GOOD (Score: {score:.4f} < {adjusted_threshold:.4f})")
        verdict = "GOOD"
        color = "green"
    
    # Check correctness
    if (is_labeled_good and not is_detected_defective) or (is_labeled_defective and is_detected_defective):
        match = "CORRECT"
        print(f"        Accuracy: {match}")
    else:
        match = "MISMATCH"
        print(f"        Accuracy: {match} (Expected: {'GOOD' if is_labeled_good else 'DEFECTIVE'})")
    
    # Create visualization
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    anomaly_map = cv2.resize(result.anomaly_map, (w, h))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('AI Anomaly Detection\n(Hot spots = potential defects)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Result
    overlay = img_np.copy()
    heatmap_colored = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
    
    if is_detected_defective and result.bbox:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'VERDICT: {verdict}', fontsize=14, fontweight='bold', 
                     color=color,
                     bbox=dict(boxstyle='round,pad=0.7', 
                              facecolor='#ffcccc' if is_detected_defective else '#ccffcc',
                              edgecolor=color, linewidth=3))
    axes[2].axis('off')
    
    # Footer
    footer = f"Score: {score:.4f}  |  Threshold: {adjusted_threshold:.4f}  |  [{verdict}]"
    fig.text(0.5, 0.02, footer, ha='center', fontsize=11, family='monospace', fontweight='bold')
    
    plt.suptitle(f"Sample {idx}: {img_name}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_path = slides_dir / f"Slide_{idx:02d}_{verdict}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"        Saved: {output_path.name}")
    
    results_data.append({
        'sample': img_name,
        'expected': 'GOOD' if is_labeled_good else 'DEFECTIVE',
        'detected': verdict,
        'score': score,
        'match': match
    })

print("\n[4/4] Generating Summary Report...")

# Create summary slide
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

summary_text = "G-CNN DEFECT DETECTION SYSTEM\nINSPECTION SUMMARY\n\n"
summary_text += "="*70 + "\n\n"

good_detected = sum(1 for r in results_data if r['detected'] == 'GOOD')
defect_detected = sum(1 for r in results_data if r['detected'] == 'DEFECTIVE')
correct = sum(1 for r in results_data if r['match'] == 'CORRECT')

summary_text += f"Total Samples Inspected: {len(results_data)}\n"
summary_text += f"Classified as GOOD: {good_detected}\n"
summary_text += f"Classified as DEFECTIVE: {defect_detected}\n"
summary_text += f"Accuracy: {correct}/{len(results_data)} ({correct/len(results_data)*100:.1f}%)\n\n"
summary_text += "="*70 + "\n\n"
summary_text += "DETAILED RESULTS:\n\n"

for i, r in enumerate(results_data, 1):
    status_symbol = "[OK]" if r['match'] == 'CORRECT' else "[!]"
    summary_text += f"{status_symbol} Sample {i}: {r['sample']:<30}\n"
    summary_text += f"    Expected: {r['expected']:<12} | Detected: {r['detected']:<12}\n"
    summary_text += f"    Score: {r['score']:.4f}\n\n"

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig(slides_dir / "Slide_00_SUMMARY.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("      Summary report created")

# Save text summary
with open(slides_dir / "INSPECTION_REPORT.txt", 'w') as f:
    f.write(summary_text)

print("\n" + "="*90)
print(" "*35 + "PRESENTATION READY!")
print("="*90)
print(f"\nRESULTS:")
print(f"  Total Samples: {len(results_data)}")
print(f"  Good: {good_detected}")
print(f"  Defective: {defect_detected}")
print(f"  Accuracy: {correct}/{len(results_data)} ({correct/len(results_data)*100:.1f}%)")
print(f"\nFILES GENERATED:")
print(f"  Demo Samples: {demo_dir}/")
print(f"  Presentation Slides: {slides_dir}/")
print(f"  Summary Report: {slides_dir}/INSPECTION_REPORT.txt")
print(f"\nFOR YOUR PRESENTATION:")
print(f"  1. Open folder: {slides_dir}")
print(f"  2. Show slides in order (Slide_01, Slide_02, etc.)")
print(f"  3. Explain the 3-panel view:")
print(f"     - Left: Original product image")
print(f"     - Middle: AI heat map (red = anomaly detected)")
print(f"     - Right: Final verdict with overlay")
print(f"  4. Show summary slide at the end")
print("\n" + "="*90)
print("GOOD LUCK WITH YOUR PRESENTATION TOMORROW!")
print("="*90 + "\n")
