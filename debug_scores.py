# debug_scores.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from models.anomaly_inspector import EnhancedAnomalyInspector

# Fix PyTorch 2.6+ security
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("=" * 60)
print("DEBUG: CHECKING ANOMALY SCORES & THRESHOLDS")
print("=" * 60)

# Load inspector
inspector = torch.load('calibrated_inspector.pth', 
                      map_location='cpu',
                      weights_only=False)

print("✅ Inspector loaded")
print("\n🔧 INSPECTOR PROPERTIES:")
print("-" * 40)

# List all attributes
attrs = [attr for attr in dir(inspector) if not attr.startswith('_')]
for attr in attrs[:20]:  # Show first 20
    try:
        value = getattr(inspector, attr)
        if not callable(value):
            print(f"  {attr}: {type(value).__name__}")
    except:
        pass

# Check specific thresholds
print("\n📊 THRESHOLDS:")
thresholds_to_check = ['image_threshold', 'pixel_threshold', 'normal_mean', 'normal_std', 
                       'symmetry_threshold', 'threshold', 'thresh']
for t in thresholds_to_check:
    if hasattr(inspector, t):
        value = getattr(inspector, t)
        if hasattr(value, 'item'):
            print(f"  {t}: {value.item():.4f}")
        else:
            print(f"  {t}: {value}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Test images - include the reference too
test_images = [
    ("REFERENCE", "data/water_bottles/train/good/example_good_water_bottle.jpeg"),
    ("TEST 1", "data/water_bottles/test/test.jpeg"),
    ("TEST 2", "data/water_bottles/test/test2.jpeg"),
    ("TEST 3", "data/water_bottles/test/test3.jpeg"), 
    ("TEST 4", "data/water_bottles/test/test4.jpeg"),
    ("TEST 5", "data/water_bottles/test/test5.jpeg")
]

print("\n" + "=" * 60)
print("IMAGE ANALYSIS")
print("=" * 60)

for label, img_path in test_images:
    if not os.path.exists(img_path):
        print(f"❌ Missing: {img_path}")
        continue
    
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        print(f"\n📸 {label}: {os.path.basename(img_path)}")
        print(f"   Size: {img.size[0]}x{img.size[1]}")
        
        # Get prediction
        result = inspector.predict(img_tensor)
        
        print(f"   Result type: {type(result)}")
        
        # Method 1: Check if result has anomaly_score
        if hasattr(result, 'anomaly_score'):
            if hasattr(result.anomaly_score, '__len__'):
                score = result.anomaly_score[0].item()
            else:
                score = result.anomaly_score.item()
            print(f"   🔢 anomaly_score: {score:.6f}")
            
        # Method 2: Check if result has image_score  
        if hasattr(result, 'image_score'):
            if hasattr(result.image_score, '__len__'):
                score = result.image_score[0].item()
            else:
                score = result.image_score.item()
            print(f"   🔢 image_score: {score:.6f}")
            
        # Method 3: Check if result is a tuple
        if isinstance(result, tuple):
            print(f"   📦 Tuple length: {len(result)}")
            for i, item in enumerate(result):
                if hasattr(item, 'shape'):
                    print(f"     [{i}] shape: {item.shape}")
                    if len(item.shape) == 1 and item.shape[0] == 1:
                        print(f"     [{i}] value: {item.item():.6f}")
                elif hasattr(item, 'item'):
                    print(f"     [{i}] value: {item.item():.6f}")
                    
        # Method 4: Check if it's a simple tensor
        if isinstance(result, torch.Tensor):
            if result.numel() == 1:
                print(f"   🔢 Tensor value: {result.item():.6f}")
            else:
                print(f"   🔢 Tensor shape: {result.shape}")
                
        # Check against threshold if available
        if hasattr(inspector, 'image_threshold'):
            threshold = inspector.image_threshold
            if hasattr(threshold, 'item'):
                threshold = threshold.item()
            print(f"   📏 image_threshold: {threshold:.6f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)
print("If all scores are 0.000000:")
print("1. Predict() might return labels (0=normal, 1=defect)")
print("2. Need to check actual distance/scores")
print("3. Might need to call a different method")