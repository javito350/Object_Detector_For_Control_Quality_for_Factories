"""
G-CNN Defect Detection System - Presentation Demo
High-quality anomaly detection for water bottle inspection
"""
# Load the argument parsing module so we can read command-line options.
import argparse
# Load the io module to control text encoding for console output.
import io
# Load os to check paths and open folders.
import os
# Load sys to change stdout encoding safely.
import sys
# Load Path so we can build OS-safe file paths.
from pathlib import Path

# Load OpenCV for image resizing and drawing.
import cv2
# Load matplotlib to generate and save charts.
import matplotlib
# Load numpy to convert images into arrays.
import numpy as np
# Load torch to run the model and read weights.
import torch
# Load torchvision transforms for preprocessing input images.
import torchvision.transforms as transforms
# Load PIL to open images from disk.
from PIL import Image

# Import model class so the serialized weights can be deserialized.
from models.anomaly_inspector import EnhancedAnomalyInspector

# Force a non-interactive backend so image saves work without a GUI.
matplotlib.use('Agg')
# Import pyplot after setting the backend.
import matplotlib.pyplot as plt

# Ensure proper encoding for Windows console output.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Allow torch to load custom model classes safely.
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

# Resolve the project root directory once.
ROOT_DIR = Path(__file__).resolve().parent
# Point to the folder where model weights are stored.
WEIGHTS_DIR = ROOT_DIR / "weights"

# Define a class that runs the demo end-to-end.
class PresentationDemo:
    # Build the demo object and load model weights.
    def __init__(self, model_path=None):
        # Print a banner so the user knows the demo started.
        print("=" * 70)
        # Print the demo title.
        print("G-CNN DEFECT DETECTION SYSTEM")
        # Print the subtitle.
        print("Automated Quality Control for Manufacturing")
        # Print a line under the header.
        print("=" * 70)
        
        # Show progress step 1: load the model.
        print("\n[1/3] Loading trained model...")
        # If no model path is provided, choose one automatically.
        if model_path is None:
            # Prefer the calibrated model for balanced detection.
            calibrated_path = WEIGHTS_DIR / "calibrated_inspector.pth"
            # Use the sensitive model as a fallback.
            sensitive_path = WEIGHTS_DIR / "sensitive_inspector.pth"

            # If the calibrated model exists, use it.
            if calibrated_path.exists():
                model_path = calibrated_path
            # Otherwise use the sensitive model if it exists.
            elif sensitive_path.exists():
                model_path = sensitive_path
            # If neither exists, stop with an error.
            else:
                raise FileNotFoundError("No inspector weights found in the weights/ directory.")

        # Load the model weights onto the CPU for inference.
        self.inspector = torch.load(str(model_path), map_location='cpu', weights_only=False)
        # Confirm that the model loaded correctly.
        print("      Model loaded successfully!")
        
        # Build the preprocessing steps the model expects.
        self.transform = transforms.Compose([
            # Resize to the input size expected by the backbone.
            transforms.Resize((224, 224)),
            # Convert image pixels to a torch tensor.
            transforms.ToTensor(),
            # Normalize with ImageNet mean and std.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Read the model threshold used to decide defect vs good.
        self.threshold = self.inspector.image_threshold
        # Convert torch scalar to Python float if needed.
        if hasattr(self.threshold, 'item'):
            self.threshold = self.threshold.item()

        # Store how the threshold is applied (less or greater).
        self.decision_rule = getattr(self.inspector, 'decision_rule', 'greater')
        
        # Show progress step 2: model configured.
        print("[2/3] System configured")
        # Print the active threshold.
        print(f"      Threshold: {self.threshold:.4f}")
        # Print the decision rule.
        print(f"      Decision rule: {self.decision_rule}")
        # Show progress step 3: ready for inspection.
        print("[3/3] Ready for inspection!")
        # Print a line under the header.
        print("=" * 70)

        # Create the output folder for images and summaries.
        self.output_dir = ROOT_DIR / "presentation_results"
        # Make sure the output folder exists.
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inspect a single image and return the result.
    def inspect_image(self, image_path, save_visualization=True):
        """Inspect a single image and return results"""
        # If the file does not exist, show an error.
        if not os.path.exists(image_path):
            # Print a helpful error message.
            print(f"\nERROR: Image not found: {image_path}")
            # Stop and return nothing.
            return None
        
        # Open the image and force RGB format.
        img_pil = Image.open(image_path).convert('RGB')
        # Apply preprocessing and add a batch dimension.
        img_tensor = self.transform(img_pil).unsqueeze(0)
        
        # Run the model prediction on the image.
        results = self.inspector.predict(img_tensor)
        # Extract the first (and only) result.
        result = results[0]
        
        # Read the anomaly score from the result.
        score = result.image_score
        # Decide if the score is defective or good.
        is_defective = self._is_defective(score)
        
        # Choose a human-readable status label.
        status = "DEFECTIVE" if is_defective else "GOOD"
        # Print a short status line for this image.
        print(f"{Path(image_path).name}: {status} | Score: {score:.4f}")
        
        # If visualization is enabled, save plots.
        if save_visualization:
            # Generate and save the plots.
            self.save_visualization(image_path, img_pil, result, is_defective)
        
        # Return the result object and defect flag.
        return result, is_defective

    # Decide whether a score means defective.
    def _is_defective(self, score: float) -> bool:
        # If the rule is "less", lower scores are defects.
        if self.decision_rule == 'less':
            # Return True when the score is below the threshold.
            return score < self.threshold
        # Otherwise, higher scores are defects.
        return score > self.threshold
    
    # Save images that visualize the anomaly map.
    def save_visualization(self, image_path, img_pil, result, is_defective):
        """Create and save visualization with heatmap"""
        # Use the file stem as the output name base.
        img_name = Path(image_path).stem
        
        # Convert the PIL image to a numpy array.
        img_np = np.array(img_pil)
        
        # Read height and width so we can resize the heatmap.
        h, w = img_np.shape[:2]
        # Resize the anomaly map to match the image size.
        anomaly_map = cv2.resize(result.anomaly_map, (w, h))
        
        # Create a figure with three panels.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Show the original image on the left.
        axes[0].imshow(img_np)
        # Add a title to the original image.
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        # Hide axis ticks for a cleaner view.
        axes[0].axis('off')
        
        # Show the anomaly heatmap in the middle.
        im1 = axes[1].imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
        # Add a title to the heatmap.
        axes[1].set_title('Anomaly Heatmap', fontsize=14, fontweight='bold')
        # Hide axis ticks for the heatmap.
        axes[1].axis('off')
        # Add a color bar to show intensity scale.
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Make a copy of the image for overlay.
        overlay = img_np.copy()
        # Convert the heatmap values to colors.
        heatmap_colored = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # Convert BGR to RGB so colors are correct.
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        # Blend the heatmap with the original image.
        overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
        
        # If a defect is predicted, draw a bounding box.
        if is_defective and result.bbox:
            # Unpack the bounding box coordinates.
            x1, y1, x2, y2 = result.bbox
            # Draw a red rectangle around the defect area.
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Show the overlay image on the right.
        axes[2].imshow(overlay)
        # Pick the correct status label.
        status_text = "DEFECTIVE" if is_defective else "GOOD"
        # Choose the title color based on status.
        color = 'red' if is_defective else 'green'
        # Add a title for the overlay panel.
        axes[2].set_title(f'Result: {status_text}', fontsize=14, fontweight='bold', color=color)
        # Hide axis ticks for the overlay.
        axes[2].axis('off')
        
        # Create a text footer with score and threshold.
        score_text = f"Score: {result.image_score:.4f}\nThreshold: {self.threshold:.4f}"
        # Draw the footer text under the plots.
        fig.text(0.5, 0.02, score_text, ha='center', fontsize=10, family='monospace')
        
        # Fit plots neatly in the figure.
        plt.tight_layout()
        
        # Build the output file path.
        output_path = self.output_dir / f"{img_name}_analysis.png"
        # Save the figure as a PNG image.
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        # Close the figure to free memory.
        plt.close()
        
        # Intentionally keep output quiet for clean demo logs.
    
    # Inspect every image in a folder.
    def batch_inspect(self, image_dir):
        """Inspect all images in a directory"""
        # Print a header for batch mode.
        print("\n\nBATCH INSPECTION MODE")
        # Print the target directory name.
        print(f"Directory: {image_dir}")
        # Print a separator line.
        print("=" * 70)
        
        # Build a list of image file paths.
        image_files = []
        # Check common image extensions.
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            # Add all matching files to the list.
            image_files.extend(Path(image_dir).glob(ext))
        
        # If no images were found, stop early.
        if not image_files:
            # Tell the user nothing was found.
            print("No images found!")
            # Exit the function.
            return
        
        # Print how many images were found.
        print(f"Found {len(image_files)} images")
        
        # Create a list to store summary rows.
        results_summary = []
        # Process images in sorted order.
        for img_path in sorted(image_files):
            # Inspect the image and get results.
            result, is_defective = self.inspect_image(str(img_path), save_visualization=True)
            # Add a row to the summary table.
            results_summary.append({
                'image': img_path.name,
                'status': 'DEFECTIVE' if is_defective else 'GOOD',
                'score': result.image_score,
                'defect_type': result.defect_type.value if is_defective else 'N/A'
            })
        
        # Print the summary header.
        print(f"\n\n{'='*70}")
        # Print the title for the summary.
        print("INSPECTION SUMMARY")
        # Print a separator line.
        print(f"{'='*70}")
        # Print the summary column headers.
        print(f"{'Image':<30} {'Status':<12} {'Score':<10} {'Defect Type'}")
        # Print a line under the headers.
        print("-" * 70)
        
        # Initialize the count of good images.
        good_count = 0
        # Initialize the count of defect images.
        defect_count = 0
        
        # Print each summary row.
        for r in results_summary:
            # Choose a symbol for quick scanning.
            status_symbol = "[X]" if r['status'] == 'DEFECTIVE' else "[OK]"
            # Print the formatted summary line.
            print(f"{r['image']:<30} {status_symbol} {r['status']:<8} {r['score']:<10.4f} {r['defect_type']}")
            # Count goods and defects.
            if r['status'] == 'GOOD':
                # Increment good count.
                good_count += 1
            else:
                # Increment defect count.
                defect_count += 1
        
        # Print a line under the table.
        print("-" * 70)
        # Print total count.
        print(f"Total: {len(results_summary)} images")
        # Print number of good images.
        print(f"Good: {good_count} ({good_count/len(results_summary)*100:.1f}%)")
        # Print number of defective images.
        print(f"Defective: {defect_count} ({defect_count/len(results_summary)*100:.1f}%)")
        # Print the end separator line.
        print(f"{'='*70}")
        
        # Write the summary to a text file.
        summary_path = self.output_dir / "inspection_summary.txt"
        # Open the summary file for writing.
        with open(summary_path, 'w') as f:
            # Write a header line.
            f.write("G-CNN DEFECT DETECTION - INSPECTION SUMMARY\n")
            # Write a separator line.
            f.write("=" * 70 + "\n\n")
            # Write column headers.
            f.write(f"{'Image':<30} {'Status':<12} {'Score':<10} {'Defect Type'}\n")
            # Write a line under headers.
            f.write("-" * 70 + "\n")
            # Write each summary row.
            for r in results_summary:
                # Write the row values.
                f.write(f"{r['image']:<30} {r['status']:<12} {r['score']:<10.4f} {r['defect_type']}\n")
            # Write a line under the rows.
            f.write("-" * 70 + "\n")
            # Write total count.
            f.write(f"Total: {len(results_summary)}\n")
            # Write good count and percent.
            f.write(f"Good: {good_count} ({good_count/len(results_summary)*100:.1f}%)\n")
            # Write defect count and percent.
            f.write(f"Defective: {defect_count} ({defect_count/len(results_summary)*100:.1f}%)\n")
        
        # Tell the user where the summary was saved.
        print(f"\nSummary saved to: {summary_path}")

    # Remove old outputs to keep the folder clean.
    def clean_outputs(self) -> None:
        """Remove demo-generated files to keep the results folder tidy"""
        # Define the filename patterns to delete.
        for pattern in ["*_analysis.png", "inspection_summary.txt"]:
            # Find files that match the pattern.
            for path in self.output_dir.glob(pattern):
                # Try to delete the file.
                try:
                    # Delete the file from disk.
                    path.unlink()
                # Ignore errors from locked files.
                except OSError:
                    # Skip files that cannot be deleted.
                    pass

    # Open the output folder in the system file explorer.
    def open_results_dir(self) -> None:
        """Open the results folder in the system file explorer"""
        # Try to open the folder on Windows.
        try:
            # Open the folder in the file explorer.
            os.startfile(str(self.output_dir))
        # If that fails, print the path.
        except OSError:
            # Tell the user to open it manually.
            print(f"Open this folder manually: {self.output_dir}")

# Main entry point for the script.
def main():
    # Create the CLI parser.
    parser = argparse.ArgumentParser(description="Presentation demo for inspection results")
    # Optional path to a single image or a directory.
    parser.add_argument("path", nargs="?", default=None, help="Image path or directory")
    # Optional number of repeated runs.
    parser.add_argument("--runs", type=int, default=1, help="Run the demo N times")
    # Optional flag to skip opening the results folder.
    parser.add_argument("--no-open", dest="open_results", action="store_false",
                        help="Do not open the results folder after the run")
    # Optional flag to keep old outputs.
    parser.add_argument("--no-clean", dest="clean_results", action="store_false",
                        help="Do not delete previous demo outputs before the run")
    # Set default values for flags.
    parser.set_defaults(open_results=True, clean_results=True)
    # Parse all CLI arguments.
    args = parser.parse_args()

    # Create the demo object.
    demo = PresentationDemo()

    # Optionally clear older outputs.
    if args.clean_results:
        # Delete old outputs in presentation_results.
        demo.clean_outputs()

    # Ensure runs is at least 1.
    runs = max(1, args.runs)
    # Loop for the requested number of runs.
    for run_idx in range(runs):
        # Show the run number if more than one run.
        if runs > 1:
            # Print the run counter.
            print(f"\n=== RUN {run_idx + 1} / {runs} ===")

        # If a path is provided, use it.
        if args.path:
            # If the path is a directory, run in batch mode.
            if os.path.isdir(args.path):
                # Run batch inspection on the folder.
                demo.batch_inspect(args.path)
            # Otherwise treat it as a single image path.
            else:
                # Inspect the single image.
                demo.inspect_image(args.path, save_visualization=True)
        # If no path was provided, use the default test set.
        else:
            # Build the default test directory path.
            test_dir = ROOT_DIR / "data" / "water_bottles" / "test"
            # If the test directory exists, run batch inspection.
            if test_dir.exists():
                # Inspect all test images.
                demo.batch_inspect(str(test_dir))
            # Otherwise print how to run the script.
            else:
                # Print usage instructions.
                print("\nUsage:")
                # Print example for a single image.
                print("  python run_demo.py <image_path>         # Single image")
                # Print example for a folder.
                print("  python run_demo.py <directory_path>     # Batch mode")
                # Print example for default test set.
                print("  python run_demo.py                      # Default test set")

    # Optionally open the results folder after the run.
    if args.open_results:
        # Open the output folder in the file explorer.
        demo.open_results_dir()

# Run main() only when executed directly.
if __name__ == "__main__":
    # Call the main function.
    main()
