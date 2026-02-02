import cv2
import numpy as np
from models.normal_ai import NormalAI 

class SymmetryAI(NormalAI):
    """
    My custom AI that uses Group Theory (Dihedral Group D8) 
    to be 'Invariant' to rotation and reflection.
    """

    def preprocess(self, img):
        """
        Standardizes the image so math works better.
        1. Resizes to 64x64 (removes zoom differences).
        2. Adds blur (ignores noise like stars or pixel artifacts).
        """
        if img is None: return None
        # Force square 64x64 for easier calculation
        resized = cv2.resize(img, (64, 64))
        # Add slight blur to help match shapes, not just pixels
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def rotate_custom(self, image, angle):
        """Helper function: Rotates an image by a specific angle (Group Action)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        # Create the Affine Matrix for rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def train(self, image_path):
        """Memorizes the object (The 'Canonical' state)"""
        # We use the same loader as NormalAI, but we apply our special preprocess
        raw_img = self.load_and_prep(image_path)
        self.memory = self.preprocess(raw_img)
        print("[SymmetryAI] Memorized object with Group Theory logic.")

    def predict(self, unknown_img_path):
        """
        The 'Equivariant' Prediction.
        Instead of checking 1 image, we generate the 'Orbit' of the image 
        (all 16 possible positions) and check them all.
        """
        # 1. Load and Standardize the unknown image
        raw_img = self.load_and_prep(unknown_img_path)
        img = self.preprocess(raw_img)
        
        if img is None: return 0.0
        
        # --- THE ALGEBRA PART: Generating the Group Orbit (G . x) ---
        orbit_images = []
        
        # SUBGROUP C8: The Rotational Group (0, 45, 90... 315)
        # We manually generate every element of the cyclic group
        img_0   = img
        img_45  = self.rotate_custom(img, 45)
        img_90  = self.rotate_custom(img, 90)
        img_135 = self.rotate_custom(img, 135)
        img_180 = self.rotate_custom(img, 180)
        img_225 = self.rotate_custom(img, 225)
        img_270 = self.rotate_custom(img, 270)
        img_315 = self.rotate_custom(img, 315)
        
        orbit_images.extend([img_0, img_45, img_90, img_135, img_180, img_225, img_270, img_315])

        # SUBGROUP D8: Adding Reflection (The 'Flip' element)
        # A mirror image is just the original multiplied by the reflection element 's'
        img_flip = cv2.flip(img, 1) 
        
        # Now we rotate the reflection too (s*r, s*r^2...)
        flip_45  = self.rotate_custom(img_flip, 45)
        flip_90  = self.rotate_custom(img_flip, 90)
        flip_135 = self.rotate_custom(img_flip, 135)
        flip_180 = self.rotate_custom(img_flip, 180)
        flip_225 = self.rotate_custom(img_flip, 225)
        flip_270 = self.rotate_custom(img_flip, 270)
        flip_315 = self.rotate_custom(img_flip, 315)
        
        orbit_images.extend([img_flip, flip_45, flip_90, flip_135, flip_180, flip_225, flip_270, flip_315])
        
        # --- THE DECISION: Group Max Pooling ---
        # We look for the best match across the entire Orbit.
        # This makes the result "Invariant" (the score doesn't change if you rotate the input).
        best_score = 0.0
        
        for transformed_img in orbit_images:
            # Check similarity for this specific rotation
            result = cv2.matchTemplate(transformed_img, self.memory, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # Keep the highest score found in the group
            if (max_val * 100) > best_score:
                best_score = max_val * 100
                
        return best_score