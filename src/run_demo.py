"""
G-CNN Defect Detection System - Presentation Demo & Feature Factory
Refactored for Research Sweeps and Few-Shot Edge AI Benchmarks
"""
import argparse
import io
import os
import sys
import logging
from pathlib import Path
import cv2
import matplotlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Force a non-interactive backend for server/CLI compatibility
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import model classes
from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SquarePad

# Ensure proper encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

class PresentationDemo:
    def __init__(self, model_path=None):
        print("=" * 70)
        print("FEW-SHOT DEFECT DETECTION SYSTEM - EVALUATION MODE")
        print("=" * 70)
        
        if model_path is None:
            calibrated_path = WEIGHTS_DIR / "calibrated_inspector.pth"
            sensitive_path = WEIGHTS_DIR / "sensitive_inspector.pth"
            model_path = calibrated_path if calibrated_path.exists() else sensitive_path
            
            if not model_path.exists():
                print("\n[WARNING] No serialized inspector found in weights/. Running in uncalibrated mode.")
                print("You must run a training/fit script to build the FAISS bank before doing inference.")
                self.inspector = EnhancedAnomalyInspector()
            else:
                print(f"\n[1/3] Loading model from: {model_path.name}")
                try:
                    self.inspector = torch.load(str(model_path), map_location='cpu', weights_only=False)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load model at {model_path}: {exc}") from exc
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"\n[1/3] Loading model from: {model_path.name}")
            try:
                self.inspector = torch.load(str(model_path), map_location='cpu', weights_only=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to load model at {model_path}: {exc}") from exc
                
        # The Academic Transform Pipeline (Preserves geometry via SquarePad)
        self.transform = transforms.Compose([
            SquarePad(),                      # MUST be first to prevent stretch distortion
            transforms.Resize((256, 256)),    # Standard WRN50 resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.output_dir = PROJECT_ROOT / "presentation_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("[2/3] Transform pipeline and output directories configured.")
        print("[3/3] System ready.\n")

    def extract_and_cache_features(self, dataset_name, output_path):
        """
        The Feature Factory: Converts raw images into .npy patch arrays.
        Crucial for accelerating the grid sweep and AUROC calculations.
        """
        print(f"Starting Feature Extraction for: {dataset_name}")
        
        base_data_path = PROJECT_ROOT / "data" / dataset_name
        train_path = base_data_path / "train" / "good"
        
        if not train_path.exists():
            print(f"ERROR: Could not find training data at {train_path}")
            return

        image_files = list(train_path.glob("*.png")) + list(train_path.glob("*.jpg")) + list(train_path.glob("*.jpeg"))
        if not image_files:
            raise FileNotFoundError(f"No training images found in {train_path}")
        all_features = []
        
        with torch.no_grad():
            for img_path in tqdm(image_files, desc="Processing Support Set"):
                img_pil = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img_pil).unsqueeze(0)
                
                # Extract features (apply_p4m=True because this is the normal training set)
                patch_features = self.inspector.feature_extractor.extract_patch_features(img_tensor, apply_p4m=True)
                all_features.append(patch_features)

        final_output = Path(output_path) if output_path else (PROJECT_ROOT / "data" / "features")
        final_output.mkdir(parents=True, exist_ok=True)
        
        train_array = np.vstack(all_features)
        np.save(final_output / "train_features.npy", train_array)
        print(f"Successfully cached {len(all_features)} samples (Shape: {train_array.shape}) to {final_output}")

    def inspect_image(self, image_path, save_visualization=True):
        """
        Runs full inference on a single image and generates the interpretable heatmap.
        """
        # 1. Validate that the file actually exists before trying to open it
        if not os.path.exists(image_path):
            print(f"\n[ERROR] File not found: '{image_path}'")
            print("Please check the path and ensure the dataset is downloaded correctly.")
            sys.exit(1) # Exit gracefully with an error code

        try:
            # 2. Try to load the image
            img_pil = Image.open(image_path).convert('RGB')
            
            # 3. Validate that the image was loaded correctly
            if img_pil is None:
                raise ValueError(f"Unable to read the image file at '{image_path}'. It may be corrupted or an unsupported format.")
                
            print(f"Inspecting: {Path(image_path).name}...")
            img_tensor = self.transform(img_pil).unsqueeze(0)
            
            # 4. Predict (apply_p4m=False during inference!)
            # predict() returns a list of results (one per batch item). We take the first [0].
            result = self.inspector.predict(img_tensor, apply_p4m=False)[0]

            # 5. Console Output
            status = "DEFECT DETECTED" if result.is_defective else "NOMINAL (PASS)"
            print(f" -> Result: {status}")
            print(f" -> Anomaly Score: {result.image_score:.3f} (Threshold: {self.inspector.image_threshold:.3f})")
            print(f" -> Latency: {result.inference_time_ms:.1f} ms")

            # 6. Visualization
            if save_visualization:
                self.save_visualization(Path(image_path), img_pil, result)

        except Exception as e:
            # Catch any other unexpected errors during processing
            print(f"\n[ERROR] An unexpected error occurred while processing the image:")
            print(f"Details: {str(e)}")
            sys.exit(1)

    def save_visualization(self, img_path, img_pil, result):
        """
        Generates the 3-panel matplotlib figure for the presentation/paper.
        """
        # Apply the same SquarePad to the original image just for visual alignment
        display_img = SquarePad()(img_pil)
        display_img = display_img.resize((256, 256))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Inspection Report: {img_path.name} | Score: {result.image_score:.2f} | {result.defect_type.value.upper()}", fontsize=14)
        
        # Panel 1: Original Image
        axes[0].imshow(display_img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Panel 2: Continuous Heatmap
        axes[1].imshow(display_img, alpha=0.5)
        im = axes[1].imshow(result.anomaly_map, cmap='jet', alpha=0.5, vmin=0, vmax=1.0)
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Panel 3: Binary Decision Mask
        axes[2].imshow(display_img, alpha=0.5)
        axes[2].imshow(result.binary_mask, cmap='Reds', alpha=0.5)
        axes[2].set_title(f"Binary Defect Mask (tau={self.inspector.pixel_threshold:.2f})")
        axes[2].axis('off')
        
        save_path = self.output_dir / f"result_{img_path.name}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" -> Visualization saved to {save_path}")

    def batch_inspect(self, image_dir):
        """
        Runs inference on an entire folder of test images.
        """
        dir_path = Path(image_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"ERROR: Directory not found: {dir_path}")
            return

        image_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg"))
        if not image_files:
            print(f"ERROR: No supported image files found in {dir_path}")
            return
        
        print(f"\nStarting Batch Inspection on {len(image_files)} images in {dir_path.name}...")
        for img_path in tqdm(image_files):
            self.inspect_image(img_path)

def main():
    parser = argparse.ArgumentParser(description="Few-Shot Edge AI Defect Detection")
    parser.add_argument("path", nargs="?", default=None, help="Path to image or directory")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "extract_only"])
    parser.add_argument("--dataset", type=str, default="mvtec_toothbrush", help="Dataset name for feature extraction")
    parser.add_argument("--output", type=str, default=None, help="Output directory for cached features")
    parser.add_argument("--model_path", type=str, default=None, help="Optional serialized inspector path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    args = parser.parse_args()

    configure_logging(args.verbose)

    try:
        demo = PresentationDemo(model_path=args.model_path)
    except Exception as exc:
        LOGGER.error("Initialization failed: %s", exc)
        sys.exit(1)

    if args.mode == "extract_only":
        demo.extract_and_cache_features(args.dataset, args.output)
    else:
        if args.path and os.path.isdir(args.path):
            demo.batch_inspect(args.path)
        elif args.path:
            demo.inspect_image(args.path)
        else:
            print("\n[INFO] No path provided. Usage examples:")
            print("  python run_demo.py path/to/image.png")
            print("  python run_demo.py path/to/test_folder/")
            print("  python run_demo.py --mode extract_only --dataset mvtec_toothbrush")

if __name__ == "__main__":
    main()