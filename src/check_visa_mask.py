import numpy as np
import os
import glob
from PIL import Image
from pathlib import Path

# SET YOUR PATH: Based on your terminal, this is the relative path
visa_path = Path("data/visa") 

# The categories that gave you 0.0 scores
problem_categories = ["candle", "capsules", "macaroni1", "pcb4"]

print("--- VISA MASK DIAGNOSTIC START ---")

for cat in problem_categories:
    cat_dir = visa_path / cat
    # VisA uses 'GroundTruth' (case sensitive) and subfolders like /001/
    mask_pattern = str(cat_dir / "GroundTruth" / "**" / "*.png")
    mask_files = glob.glob(mask_pattern, recursive=True)
    
    print(f"\nCATEGORY: {cat.upper()}")
    print(f"  Path searched: {mask_pattern}")
    print(f"  Mask files found: {len(mask_files)}")
    
    if len(mask_files) > 0:
        # Check the first mask to see if it's actually blank
        test_mask = np.array(Image.open(mask_files[0]))
        unique_vals = np.unique(test_mask)
        print(f"  First mask resolution: {test_mask.shape}")
        print(f"  Pixel values found: {unique_vals}")
        
        if len(unique_vals) == 1 and unique_vals[0] == 0:
            print("  ⚠️ ALERT: This mask is completely BLACK (all zeros).")
        elif 255 in unique_vals:
            print("  ✅ SUCCESS: Found binary mask data (0 and 255).")
    else:
        print("  ❌ ERROR: No mask files found. Check your folder names!")

print("\n--- DIAGNOSTIC COMPLETE ---")