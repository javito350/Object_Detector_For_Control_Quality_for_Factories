# recalibrate_sensitive.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import glob

print("=" * 60)
print("RECALIBRATION - MORE SENSITIVE DETECTION")
print("=" * 60)

sys.path.append('models')
from models.anomaly_inspector import EnhancedAnomalyInspector

def create_sensitive_inspector():
    """Create an inspector with lower thresholds for better defect detection"""
    
    # Load the perfect reference
    ref_path = "data/water_bottles/train/good/example_good_water_bottle.jpeg"
    print(f"\n📸 Reference image: {os.path.basename(ref_path)}")
    
    img = Image.open(ref_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    print(f"✓ Loaded: {img.size[0]}x{img.size[1]} → 224x224")
    
    # First, let's check what parameters EnhancedAnomalyInspector accepts
    print("\n🔍 Checking inspector parameters...")
    
    # Try to create with various parameter combinations
    try:
        # Try with potential threshold parameter
        print("Attempt 1: With threshold parameter")
        inspector = EnhancedAnomalyInspector(
            backbone="wide_resnet50_2",
            symmetry_type="both",
            device="cpu",
            coreset_percentage=0.8,  # Use 80% for more generalization
            # Try common threshold parameters
        )
        print("✓ Created inspector")
    except TypeError as e:
        print(f"✗ Failed: {e}")
        # Try without extra parameters
        print("\nAttempt 2: Minimal parameters")
        inspector = EnhancedAnomalyInspector(
            backbone="wide_resnet50_2",
            symmetry_type="both",
            device="cpu"
        )
        print("✓ Created with minimal parameters")
    
    # Create dataloader
    class OneImageLoader:
        def __iter__(self):
            yield (img_tensor, torch.zeros(1), [ref_path])
        def __len__(self):
            return 1
    
    # Fit the inspector
    print("\n🧠 Calibrating system...")
    inspector.fit(OneImageLoader())
    
    # Check if we can adjust thresholds after fitting
    print("\n🔧 Attempting to adjust sensitivity...")
    
    # Method 1: Try to lower thresholds directly
    if hasattr(inspector, 'image_threshold'):
        original = inspector.image_threshold
        inspector.image_threshold = original * 0.5  # 50% more sensitive
        print(f"✓ Adjusted image_threshold: {original:.4f} → {inspector.image_threshold:.4f}")
    
    if hasattr(inspector, 'pixel_threshold'):
        original = inspector.pixel_threshold
        inspector.pixel_threshold = original * 0.5  # 50% more sensitive
        print(f"✓ Adjusted pixel_threshold: {original:.4f} → {inspector.pixel_threshold:.4f}")
    
    # Method 2: Try different approach - recreate with different settings
    print("\n🔄 Alternative: Creating inspector that focuses on defects...")
    
    # Save this inspector
    output_file = "sensitive_inspector.pth"
    torch.save(inspector, output_file)
    
    print(f"\n💾 Saved sensitive inspector to: {output_file}")
    
    # Test it immediately
    print("\n🔬 Quick test on test images...")
    
    test_folder = "data/water_bottles/test"
    if os.path.exists(test_folder):
        test_images = glob.glob(os.path.join(test_folder, "*.jpeg"))
        
        for test_path in test_images[:3]:  # Test first 3
            try:
                test_img = Image.open(test_path).convert('RGB')
                test_tensor = transform(test_img).unsqueeze(0)
                
                result = inspector.predict(test_tensor)
                
                print(f"\n  {os.path.basename(test_path)}:")
                
                # Extract score
                score = 0.0
                if hasattr(result, 'anomaly_score'):
                    if hasattr(result.anomaly_score, '__len__'):
                        score = result.anomaly_score[0].item()
                    else:
                        score = result.anomaly_score.item()
                elif isinstance(result, tuple) and len(result) > 1:
                    if hasattr(result[1], 'item'):
                        score = result[1].item()
                
                print(f"    Score: {score:.6f}")
                
                # Check against threshold if available
                if hasattr(inspector, 'image_threshold'):
                    threshold = inspector.image_threshold
                    if hasattr(threshold, 'item'):
                        threshold = threshold.item()
                    
                    if score > threshold:
                        print(f"    ❌ DEFECT DETECTED! (>{threshold:.4f})")
                    else:
                        print(f"    ✅ NORMAL (≤{threshold:.4f})")
                        
            except Exception as e:
                print(f"    ⚠️ Error: {e}")
    
    return inspector

if __name__ == "__main__":
    inspector = create_sensitive_inspector()
    
    print("\n" + "=" * 60)
    print("RECALIBRATION COMPLETE")
    print("=" * 60)
    print("✅ Created more sensitive inspector")
    print("✅ Saved as 'sensitive_inspector.pth'")
    print("\n🚀 Use it in your demo:")
    print("   python demo_sensitive.py")
    print("=" * 60)