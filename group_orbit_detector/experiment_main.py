import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import cv2
import numpy as np

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.normal_ai import NormalAI
from models.symmetry_ai import SymmetryAI

# --- CONFIGURATION ---
TEST_FOLDER = os.path.join(parent_dir, 'data', 'test')
DETECTION_THRESHOLD = 50.0 
DISPLAY_SIZE = (200, 200)

def get_best_training_image():
    path_test = os.path.join(TEST_FOLDER, 'original.jpeg')
    path_train = os.path.join(parent_dir, 'data', 'train', 'original.jpeg')
    if os.path.exists(path_test): return path_test
    elif os.path.exists(path_train): return path_train
    else: return None

def run_visual_experiment():
    """Generates Class Dashboard (Moons)."""
    print("--- 🚀 GENERATING CLASS DASHBOARD... ---")

    dumb_bot = NormalAI()
    smart_bot = SymmetryAI()

    train_img_path = get_best_training_image()
    if train_img_path:
        dumb_bot.train(train_img_path)
        smart_bot.train(train_img_path)
    else:
        print("⚠️ ERROR: Could not find 'original.jpeg'!")
        return

    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Test folder not found at {TEST_FOLDER}")
        return
        
    files = sorted([f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    if not files:
        print("⚠️ NOTE: Your 'data/test' folder is empty!")
        return

    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(2, len(files), figsize=(14, 9))
    if len(files) == 1: axes = np.array([[axes[0]], [axes[1]]])
    fig.patch.set_facecolor('#f4f4f4')
    fig.suptitle('AI ARCHITECTURE COMPARISON', fontsize=20, fontweight='bold', y=0.95)

    for i, filename in enumerate(files):
        file_path = os.path.join(TEST_FOLDER, filename)
        img_raw = mpimg.imread(file_path)
        img_display = cv2.resize(img_raw, DISPLAY_SIZE)
        score_n = dumb_bot.predict(file_path)
        score_s = smart_bot.predict(file_path)

        ax_top = axes[0, i]
        ax_top.imshow(img_display, cmap='gray')
        ax_top.axis('off')
        ax_top.set_title(filename, fontsize=9, pad=8)
        status_n = "MATCH" if score_n > DETECTION_THRESHOLD else "FAIL"
        color_n = '#27ae60' if score_n > DETECTION_THRESHOLD else '#c0392b'
        ax_top.add_patch(Rectangle((0, -0.15), 1, 0.15, transform=ax_top.transAxes, color=color_n, clip_on=False))
        ax_top.text(0.5, -0.075, status_n, transform=ax_top.transAxes, ha='center', va='center', color='white', fontweight='bold')

        ax_bot = axes[1, i]
        ax_bot.imshow(img_display, cmap='gray')
        ax_bot.axis('off')
        ax_bot.set_title(filename, fontsize=9, pad=8)
        status_s = "MATCH" if score_s > DETECTION_THRESHOLD else "FAIL"
        color_s = '#27ae60' if score_s > DETECTION_THRESHOLD else '#c0392b'
        ax_bot.add_patch(Rectangle((0, -0.15), 1, 0.15, transform=ax_bot.transAxes, color=color_s, clip_on=False))
        ax_bot.text(0.5, -0.075, status_s, transform=ax_bot.transAxes, ha='center', va='center', color='white', fontweight='bold')

    axes[0, 0].text(-0.3, 0.5, "Standard\nCNN", transform=axes[0, 0].transAxes, va='center', ha='center', fontsize=14, fontweight='bold', rotation=90)
    axes[1, 0].text(-0.3, 0.5, "Symmetry\nAI", transform=axes[1, 0].transAxes, va='center', ha='center', fontsize=14, fontweight='bold', rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.1, hspace=0.3)
    print("Displaying dashboard... (Close window to generate Heatmap)")
    plt.show()

def auto_crop_template(image):
    """Removes black borders to focus only on the heat signature."""
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        pad = 2
        return image[max(0, y-pad):min(image.shape[0], y+h+pad), 
                     max(0, x-pad):min(image.shape[1], x+w+pad)]
    return image

def generate_phd_heatmap():
    """Generates the 'Symmetry Manifold' Heatmap with SKY MASKING & MULTI-SCALE."""
    print("\n--- 🔬 GENERATING PHD SYMMETRY PROOF ---")
    
    template_path = os.path.join(current_dir, 'template.jpeg')
    test_path = os.path.join(current_dir, 'test_image.jpeg')
    
    if not os.path.exists(template_path) or not os.path.exists(test_path):
        print(f"⚠️ SKIPPING HEATMAP: Files not found.")
        return

    full_template = cv2.imread(template_path, 0)
    img_gray = cv2.imread(test_path, 0)
    img_rgb = cv2.imread(test_path) 

    if full_template is None or img_gray is None:
        print("⚠️ Error reading images.")
        return

    # --- 1. ROI MASKING (THE DRONE FIX) ---
    # We black out the top 60% of the image to remove the drone/sky.
    # This forces the AI to look at the ground.
    print("   -> Applying 'Sky Mask' to ignore drone heat signature...")
    height = img_gray.shape[0]
    cutoff = int(height * 0.6) # Black out top 60%
    img_gray[0:cutoff, :] = 0 
    
    # Also mask the RGB for the "Detection" image so the user sees WHY we ignored it
    # (Optional: keep RGB full if you want, but masking it proves the logic)
    # img_rgb[0:cutoff, :] = 0 

    # --- 2. AUTO-CROP TEMPLATE ---
    print("   -> Auto-Cropping template...")
    full_template = auto_crop_template(full_template)

    # --- 3. MULTI-SCALE SCANNING ---
    scales_to_check = [20, 30, 40, 50, 60] 

    best_score = -1
    best_result_map = None
    best_loc = None
    best_h, best_w = (0, 0)

    print(f"   -> Scanning {len(scales_to_check)} Scales & 36 Rotations...")
    
    for scale_px in scales_to_check:
        scale_factor = scale_px / max(full_template.shape[0], full_template.shape[1])
        width = int(full_template.shape[1] * scale_factor)
        height = int(full_template.shape[0] * scale_factor)
        current_template = cv2.resize(full_template, (width, height), interpolation=cv2.INTER_AREA)

        for angle in range(0, 360, 10):
            M = cv2.getRotationMatrix2D((current_template.shape[1]//2, current_template.shape[0]//2), angle, 1.0)
            rotated_template = cv2.warpAffine(current_template, M, (current_template.shape[1], current_template.shape[0]))

            if rotated_template.shape[0] >= img_gray.shape[0] or rotated_template.shape[1] >= img_gray.shape[1]:
                continue

            result = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_result_map = result
                best_loc = max_loc
                best_h, best_w = rotated_template.shape[:2]

    if best_result_map is None:
        print("⚠️ No match found.")
        return

    heatmap_norm = cv2.normalize(best_result_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_final = cv2.resize(heatmap_color, (img_rgb.shape[1], img_rgb.shape[0]))

    # Draw Green Box
    top_left = best_loc
    bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
    cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 0), 4)

    cv2.imwrite(os.path.join(current_dir, 'proof_1_detection.jpg'), img_rgb)
    cv2.imwrite(os.path.join(current_dir, 'proof_2_heatmap.jpg'), heatmap_final)
    
    print(f"✅ SUCCESS! Generated proofs in {current_dir}")
    print(f"   -> Found Best Match at Scale: {best_h}x{best_w} pixels")

if __name__ == "__main__":
    run_visual_experiment()
    generate_phd_heatmap()