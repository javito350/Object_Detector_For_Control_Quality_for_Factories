"""
Enhanced Visual Demo - Shows Symmetry Analysis
Demonstrates one-shot learning with symmetry exploitation
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from pathlib import Path
import sys

from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("="*70)
print("G-CNN DEFECT DETECTION - ONE-SHOT SYMMETRY ANALYSIS")
print("="*70)

# Load model
print("\nLoading model...")
model_path = 'calibrated_inspector.pth' if os.path.exists('calibrated_inspector.pth') else 'sensitive_inspector.pth'
inspector = torch.load(model_path, map_location='cpu', weights_only=False)
threshold = inspector.image_threshold
if hasattr(threshold, 'item'):
    threshold = threshold.item()

print(f"Model: {model_path}")
print(f"Training: ONE normal image + symmetry analysis")
print(f"Threshold: {threshold:.4f}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def show_symmetry_analysis(image_path, result, is_defective, score):
    """Display 4-panel analysis showing symmetry detection"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Resize for display
    max_dim = 600
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w
    
    # Create 2x2 panel display
    panel_h, panel_w = h, w
    display = np.zeros((panel_h * 2 + 20, panel_w * 2 + 20, 3), dtype=np.uint8)
    display.fill(50)
    
    # Panel 1: Original Image
    display[10:10+panel_h, 10:10+panel_w] = img
    cv2.putText(display, "ORIGINAL", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 2: Horizontally Flipped (for symmetry check)
    img_flipped = cv2.flip(img, 1)
    display[10:10+panel_h, 10+panel_w+10:10+panel_w*2+10] = img_flipped
    cv2.putText(display, "FLIPPED (Symmetry)", (10+panel_w+15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 3: Symmetry Map
    symmetry_map = result.symmetry_map
    if symmetry_map.shape != (panel_h, panel_w):
        symmetry_map = cv2.resize(symmetry_map, (panel_w, panel_h))
    
    symmetry_colored = cv2.applyColorMap((symmetry_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    symmetry_colored = cv2.cvtColor(symmetry_colored, cv2.COLOR_BGR2RGB)
    display[10+panel_h+10:10+panel_h*2+10, 10:10+panel_w] = symmetry_colored
    cv2.putText(display, "SYMMETRY BREAKS", (15, 10+panel_h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 4: Final Result with Verdict
    anomaly_map = result.anomaly_map
    if anomaly_map.shape != (panel_h, panel_w):
        anomaly_map = cv2.resize(anomaly_map, (panel_w, panel_h))
    
    heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    
    # Add verdict banner
    verdict = "DEFECTIVE" if is_defective else "GOOD"
    color = (255, 0, 0) if is_defective else (0, 255, 0)
    cv2.rectangle(blended, (0, 0), (panel_w, 50), (0, 0, 0), -1)
    cv2.putText(blended, verdict, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    display[10+panel_h+10:10+panel_h*2+10, 10+panel_w+10:10+panel_w*2+10] = blended
    cv2.putText(display, "DETECTION RESULT", (10+panel_w+15, 10+panel_h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add bottom info bar
    info_bar = np.zeros((60, panel_w * 2 + 20, 3), dtype=np.uint8)
    info_bar.fill(30)
    
    # Score and metadata
    info_text = f"Score: {score:.4f} | Threshold: {threshold:.4f} | Symmetry Score: {result.metadata.get('symmetry_score', 0):.4f}"
    cv2.putText(info_bar, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Method explanation
    method_text = "Method: ONE-SHOT LEARNING + SYMMETRY ANALYSIS"
    cv2.putText(info_bar, method_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Combine display and info bar
    final_display = np.vstack([display, info_bar])
    
    # Show
    window_name = f"Symmetry Analysis: {os.path.basename(image_path)}"
    cv2.imshow(window_name, final_display)
    
    print(f"\n{'='*70}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Verdict: {verdict}")
    print(f"Anomaly Score: {score:.4f}")
    print(f"Symmetry Score: {result.metadata.get('symmetry_score', 0):.4f}")
    print(f"Appearance Score: {result.metadata.get('appearance_score', 0):.4f}")
    print(f"\nKey Insight: Symmetry breaks indicate defects")
    print(f"Training Data: Just ONE normal image!")
    print("="*70)
    print("Press any key to continue...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inspect_image(image_path):
    """Inspect with symmetry analysis"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)
    
    results = inspector.predict(img_tensor)
    result = results[0]
    
    score = result.image_score
    is_defective = score > threshold
    
    show_symmetry_analysis(image_path, result, is_defective, score)

def batch_inspect(directory):
    """Batch inspection with symmetry analysis"""
    print(f"\nInspecting: {directory}")
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(directory).glob(ext))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    for img_path in sorted(image_files):
        inspect_image(str(img_path))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            batch_inspect(path)
        else:
            inspect_image(path)
    else:
        test_dir = "data/water_bottles/test"
        if os.path.exists(test_dir):
            batch_inspect(test_dir)
        else:
            print("\nUsage:")
            print("  python symmetry_demo.py <image_path>")
            print("  python symmetry_demo.py <directory_path>")
