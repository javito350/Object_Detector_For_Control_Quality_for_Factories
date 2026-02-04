# debug_predict_output.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from models.anomaly_inspector import EnhancedAnomalyInspector

# Fix PyTorch 2.6+ security
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("=" * 60)
print("DEBUG: UNDERSTANDING PREDICT() OUTPUT")
print("=" * 60)

# Load inspector
inspector = torch.load('sensitive_inspector.pth', 
                      map_location='cpu',
                      weights_only=False)

print("✅ Inspector loaded")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Test one image
test_path = "data/water_bottles/test/test.jpeg"
if os.path.exists(test_path):
    img = Image.open(test_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    print(f"\n🔍 Testing: {os.path.basename(test_path)}")
    
    # Get prediction
    result = inspector.predict(img_tensor)
    
    print(f"📦 Result type: {type(result)}")
    print(f"📦 Is list: {isinstance(result, list)}")
    
    if isinstance(result, list):
        print(f"📦 List length: {len(result)}")
        
        for i, item in enumerate(result):
            print(f"\n  Item {i}:")
            print(f"    Type: {type(item)}")
            
            if isinstance(item, torch.Tensor):
                print(f"    Tensor shape: {item.shape}")
                print(f"    Tensor dtype: {item.dtype}")
                
                # Try to show values if tensor is small
                if item.numel() <= 10:
                    print(f"    Values: {item}")
                elif item.numel() == 1:
                    print(f"    Value: {item.item():.6f}")
                else:
                    # Show first few values for large tensors
                    flat = item.flatten()
                    print(f"    First 5 values: {flat[:5].tolist()}")
            
            elif isinstance(item, (int, float)):
                print(f"    Value: {item}")
            
            elif hasattr(item, '__dict__'):
                print(f"    Has attributes: {list(item.__dict__.keys())[:5]}...")
            
            # Try to check common attributes
            if hasattr(item, 'anomaly_score'):
                print(f"    Has anomaly_score: {item.anomaly_score}")
            
            if hasattr(item, 'image_label'):
                print(f"    Has image_label: {item.image_label}")

print("\n" + "=" * 60)
print("TRYING DIFFERENT PREDICTION METHODS")
print("=" * 60)

# Try other methods that might exist
methods_to_try = ['predict', 'evaluate', 'score', 'get_anomaly_score', 'infer']
for method_name in methods_to_try:
    if hasattr(inspector, method_name):
        print(f"\n🔍 Trying method: {method_name}()")
        try:
            # Call with different signatures
            if method_name == 'predict':
                result = getattr(inspector, method_name)(img_tensor)
            else:
                result = getattr(inspector, method_name)(img_tensor)
            
            print(f"   Result type: {type(result)}")
            
            if isinstance(result, torch.Tensor):
                print(f"   Shape: {result.shape}")
                if result.numel() <= 5:
                    print(f"   Values: {result}")
            
        except Exception as e:
            print(f"   Error: {e}")

print("\n" + "=" * 60)
print("CHECKING INSPECTOR ATTRIBUTES")
print("=" * 60)

# List all methods
print("\n📋 Available methods:")
methods = [attr for attr in dir(inspector) if not attr.startswith('_') and callable(getattr(inspector, attr))]
for method in sorted(methods)[:15]:  # Show first 15
    print(f"  • {method}()")

print("\n📊 Available attributes:")
attrs = [attr for attr in dir(inspector) if not attr.startswith('_') and not callable(getattr(inspector, attr))]
for attr in sorted(attrs)[:15]:  # Show first 15
    try:
        value = getattr(inspector, attr)
        print(f"  • {attr}: {type(value).__name__}")
    except:
        print(f"  • {attr}: [Error]")