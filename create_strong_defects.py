"""
Create strong, realistic defective samples that will be detected
"""
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os
import numpy as np
import cv2

print("="*70)
print("CREATING REALISTIC DEFECTIVE SAMPLES")
print("="*70)

# Load good reference
good_path = "data/water_bottles/train/good/example_good_water_bottle.jpeg"
img = Image.open(good_path).convert('RGB')
img_np = np.array(img)

output_dir = "defective_samples"
os.makedirs(output_dir, exist_ok=True)

def create_strong_defect(name, func):
    """Create a defect using custom function"""
    print(f"\nCreating: {name}")
    result = func(img_np.copy())
    result_pil = Image.fromarray(result.astype(np.uint8))
    
    output_path = os.path.join(output_dir, f"{name}.jpg")
    result_pil.save(output_path, quality=95)
    print(f"  Saved: {output_path}")
    return output_path

# 1. SEVERE CRACK - breaks symmetry
def add_crack(img):
    result = img.copy()
    h, w = result.shape[:2]
    # Diagonal crack
    for i in range(100):
        y = h//3 + i*4
        x = w//2 - 30 + i
        if y < h and x < w:
            cv2.circle(result, (x, y), 8, (30, 30, 30), -1)
    return result

# 2. LARGE DENT - major deformation
def add_dent(img):
    result = img.copy()
    h, w = result.shape[:2]
    center = (w//2 + 50, h//2)
    # Create dark circular dent
    cv2.circle(result, center, 80, (40, 40, 40), -1)
    cv2.circle(result, center, 60, (60, 60, 60), -1)
    # Add some highlights
    cv2.circle(result, (center[0]-20, center[1]-20), 15, (100, 100, 100), -1)
    return result

# 3. MISSING PART - chunk removed
def add_missing_part(img):
    result = img.copy()
    h, w = result.shape[:2]
    # Black out a section (simulates missing piece)
    result[h//3:h//3+200, w//4:w//4+150] = [0, 0, 0]
    return result

# 4. CONTAMINATION - large stain
def add_contamination(img):
    result = img.copy()
    h, w = result.shape[:2]
    # Brown/yellow stain
    overlay = result.copy()
    cv2.circle(overlay, (w//2-100, h//2), 150, (80, 120, 180), -1)
    result = cv2.addWeighted(result, 0.5, overlay, 0.5, 0)
    return result

# 5. WRONG COLOR - severe color shift
def add_color_defect(img):
    result = img.copy().astype(np.float32)
    # Shift to red/yellow (wrong tint)
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.5, 0, 255)  # More red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.3, 0, 255)  # More green
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.6, 0, 255)  # Less blue
    return result

# 6. ASYMMETRIC DAMAGE - breaks symmetry completely
def add_asymmetric_damage(img):
    result = img.copy()
    h, w = result.shape[:2]
    # Damage only left side
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(damage_mask, (0, h//4), (w//3, 3*h//4), 255, -1)
    result[damage_mask > 0] = result[damage_mask > 0] * 0.4
    return result

# Create all defects
print("\nGenerating defective samples...")
files = []

files.append(create_strong_defect("defect_crack", add_crack))
files.append(create_strong_defect("defect_dent", add_dent))
files.append(create_strong_defect("defect_missing", add_missing_part))
files.append(create_strong_defect("defect_contamination", add_contamination))
files.append(create_strong_defect("defect_color", add_color_defect))
files.append(create_strong_defect("defect_asymmetric", add_asymmetric_damage))

print("\n" + "="*70)
print(f"Created {len(files)} defective samples in {output_dir}/")
print("="*70)
print("\nFiles created:")
for f in files:
    print(f"  - {os.path.basename(f)}")

print("\nTest them with:")
print(f"  python presentation_demo.py {output_dir}/")
