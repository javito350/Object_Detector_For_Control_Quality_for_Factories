import cv2
import numpy as np
import os

class NormalAI:
    def __init__(self):
        self.memory = None # "Weights" of the model

    def load_and_prep(self, path):
        """Helper to load image in grayscale and force square size"""
        if not os.path.exists(path): return None
        img = cv2.imread(path)
        if img is None: return None
        
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- FIX: FORCE SQUARE SIZE ---
        # We resize everything to 200x200. This ensures that 
        # 90-degree rotations are valid and don't crash OpenCV.
        return cv2.resize(gray, (200, 200))

    def train(self, image_path):
        """Learns the features of the object in ONE specific orientation"""
        print(f"[{self.__class__.__name__}] Training on: {os.path.basename(image_path)}")
        self.memory = self.load_and_prep(image_path)
        if self.memory is None:
            raise ValueError(f"Could not load training image at {image_path}")

    def predict(self, unknown_img_path):
        """Tries to match features strictly (Brute Force)"""
        unknown_img = self.load_and_prep(unknown_img_path)
        if unknown_img is None: return 0.0
        
        # Resize to match memory (redundant now, but safe)
        if unknown_img.shape != self.memory.shape:
            unknown_img = cv2.resize(unknown_img, (self.memory.shape[1], self.memory.shape[0]))
        
        # Template Matching
        result = cv2.matchTemplate(unknown_img, self.memory, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max_val * 100