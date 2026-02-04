# quick_test.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

print("=" * 60)
print("QUICK TEST - SEE RAW SCORES")
print("=" * 60)

# Load inspector
from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

inspector = torch.load('sensitive_inspector.pth', 
                      map_location='cpu',
                      weights_only=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Test single image
test_path = "data/water_bottles/test/test.jpeg"
img = Image.open(test_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

result = inspector.predict(img_tensor)

print(f"\n🔍 Result for {os.path.basename(test_path)}:")
print(f"Type: {type(result)}")

if isinstance(result, list):
    print(f"List length: {len(result)}")
    
    # Try to extract meaningful information
    for i, item in enumerate(result):
        print(f"\n[{i}]: {type(item)}")
        
        if isinstance(item, torch.Tensor):
            print(f"  Shape: {item.shape}")
            
            # If it's a single value
            if item.numel() == 1:
                value = item.item()
                print(f"  Value: {value:.6f}")
                
                # Try to guess what it is
                if i == 0:
                    print(f"  Likely: Label (0=normal, 1=defect)")
                    if value == 0 or value == 1:
                        print(f"  Interpretation: {'DEFECT' if value == 1 else 'NORMAL'}")
                elif i == 1:
                    print(f"  Likely: Anomaly score")
                    # Check against threshold
                    if hasattr(inspector, 'image_threshold'):
                        thresh = inspector.image_threshold
                        if hasattr(thresh, 'item'):
                            thresh = thresh.item()
                        print(f"  Threshold: {thresh:.4f}")
                        print(f"  Defective? {value > thresh}")
            
            # If it's multiple values
            elif item.numel() > 1 and item.numel() <= 10:
                values = item.flatten().tolist()
                print(f"  Values: {[f'{v:.4f}' for v in values]}")
                
                # If 2 values, might be [label, score]
                if len(values) == 2:
                    print(f"  Possible: [label={int(values[0])}, score={values[1]:.4f}]")

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("Common patterns in anomaly detection output:")
print("1. [label, score] - label=0/1, score=anomaly measure")
print("2. [label, score, heatmap] - with visualization")
print("3. [score] - just the anomaly score")
print("4. Custom object with attributes")
print("=" * 60)