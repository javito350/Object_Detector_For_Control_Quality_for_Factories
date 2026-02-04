"""
G-CNN Defect Detection System - Visual Demo
Shows images with detection results overlaid
"""
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from pathlib import Path
import sys

from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("="*70)
print("G-CNN DEFECT DETECTION SYSTEM")
print("="*70)

# Load model - try calibrated version first
print("\nLoading model...")
model_path = 'calibrated_inspector.pth' if os.path.exists('calibrated_inspector.pth') else 'sensitive_inspector.pth'
inspector = torch.load(model_path, map_location='cpu', weights_only=False)
threshold = inspector.image_threshold
if hasattr(threshold, 'item'):
    threshold = threshold.item()

print(f"Model loaded from: {model_path}")
print(f"Detection threshold: {threshold:.4f}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def show_result(image_path, result, is_defective, score):
    """Display image with result overlay"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Resize for display
    max_dim = 800
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w
    
    # Create result overlay
    overlay = img.copy()
    
    # Resize heatmap to match image
    heatmap = cv2.resize(result.anomaly_map, (w, h))
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    blended = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    # Add text overlay
    verdict = "DEFECTIVE" if is_defective else "GOOD"
    color = (255, 0, 0) if is_defective else (0, 255, 0)
    
    # Draw banner at top
    cv2.rectangle(blended, (0, 0), (w, 80), (0, 0, 0), -1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blended, verdict, (20, 55), font, 2, color, 4)
    
    # Add score
    score_text = f"Score: {score:.4f}"
    cv2.putText(blended, score_text, (w - 250, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw bounding box if defective
    if is_defective and result.bbox:
        x1, y1, x2, y2 = result.bbox
        # Scale bbox to display size
        scale_x = w / result.anomaly_map.shape[1]
        scale_y = h / result.anomaly_map.shape[0]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    # Show image
    cv2.imshow(f"Detection Result: {os.path.basename(image_path)}", blended)
    print(f"\nDisplaying: {os.path.basename(image_path)}")
    print(f"Verdict: {verdict}")
    print(f"Score: {score:.4f} (Threshold: {threshold:.4f})")
    print("Press any key to continue...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inspect_image(image_path):
    """Inspect single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Load and predict
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)
    
    results = inspector.predict(img_tensor)
    result = results[0]
    
    score = result.image_score
    is_defective = score > threshold
    
    # Show visual result
    show_result(image_path, result, is_defective, score)

def batch_inspect(directory):
    """Inspect all images in directory"""
    print(f"\nInspecting images in: {directory}")
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(directory).glob(ext))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    for img_path in sorted(image_files):
        inspect_image(str(img_path))

# Main
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            batch_inspect(path)
        else:
            inspect_image(path)
    else:
        # Default: test directory
        test_dir = "data/water_bottles/test"
        if os.path.exists(test_dir):
            batch_inspect(test_dir)
        else:
            print("\nUsage:")
            print("  python demo.py <image_path>")
            print("  python demo.py <directory_path>")
