import os
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"

sys.path.append(str(ROOT_DIR))
from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SquarePad

def main():
    print("=" * 60)
    print("FEW-SHOT SYSTEM CALIBRATION (N=1)")
    print("=" * 60)

    # 1. Path to your single MVTec reference image
    # CHANGE THIS to point to a good toothbrush image
    image_path = ROOT_DIR / "data" / "mvtec_toothbrush" / "train" / "good" / "000.png" 
    
    if not image_path.exists():
        print(f"❌ Cannot find image at {image_path}")
        return

    # 2. The Academic Transform Pipeline
    transform = transforms.Compose([
        SquarePad(),                      
        transforms.Resize((256, 256)),    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load the Image
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)
    print(f"✓ Loaded reference image: {image_path.name}")

    # 4. Initialize Inspector
    print("\nInitializing Enhanced Anomaly Inspector...")
    inspector = EnhancedAnomalyInspector(coreset_percentage=0.1)

    # 5. Dummy Dataloader for fit()
    class OneImageLoader:
        def __iter__(self):
            yield (img_tensor, torch.zeros(1), [str(image_path)])
        def __len__(self):
            return 1

    # 6. Run Fit (Builds FAISS + EVT Thresholds)
    inspector.fit(OneImageLoader(), apply_p4m=True)

    # 7. Save Weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = WEIGHTS_DIR / "calibrated_inspector.pth"
    
    # Save model state as a dictionary instead of the full object
    # (avoids pickling issues with nested hook functions)
    checkpoint = {
        'feature_extractor_state': inspector.feature_extractor.state_dict(),
        'memory_bank_index': inspector.memory_bank.index,
        'memory_bank_dimension': inspector.memory_bank.dimension,
        'image_threshold': inspector.image_threshold,
        'pixel_threshold': inspector.pixel_threshold,
        'coreset_percentage': inspector.coreset_percentage,
        'device': inspector.device,
    }
    torch.save(checkpoint, output_file)
    print(f"\n✅ Calibration Complete! Model saved to {output_file.name}")

if __name__ == "__main__":
    main()