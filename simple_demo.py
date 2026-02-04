"""
Simple G-CNN Defect Detector Demo
Checks if water bottles are good or defective
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

print("=" * 60)
print("🔍 G-CNN DEFECT INSPECTOR")
print("=" * 60)

# Load the trained inspector model
from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("\n📦 Loading trained model...")
inspector = torch.load('sensitive_inspector.pth', 
                      map_location='cpu',
                      weights_only=False)
print("✅ Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def check_image(image_path):
    """Check if an image shows a good or defective product"""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # Run inspection
    results = inspector.predict(img_tensor)
    result = results[0]  # Get first result
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Get the anomaly score
    score = result.image_score
    threshold = inspector.image_threshold
    if hasattr(threshold, 'item'):
        threshold = threshold.item()
    
    is_defective = score > threshold
    
    # Display verdict
    if is_defective:
        print(f"❌ VERDICT: DEFECTIVE")
        print(f"   Defect Type: {result.defect_type.value}")
        print(f"   Severity: {result.severity:.2%}")
    else:
        print(f"✅ VERDICT: GOOD")
        print(f"   Product is ready for sale!")
    
    print(f"\n📊 Details:")
    print(f"   Anomaly Score: {score:.4f}")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Confidence: {result.confidence:.2%}")
    
    if result.metadata:
        print(f"   Anomaly Pixels: {result.metadata.get('num_anomaly_pixels', 0)}")
    
    if result.bbox and is_defective:
        x1, y1, x2, y2 = result.bbox
        print(f"   Defect Location: ({x1}, {y1}) to ({x2}, {y2})")
    
    return result

# Main execution
if __name__ == "__main__":
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "data/water_bottles/test/test.jpeg"
    
    print(f"\n🔍 Inspecting: {image_path}")
    result = check_image(image_path)
    
    print(f"\n{'='*60}")
    print("💡 TIP: To check other images, run:")
    print("   python simple_demo.py <path_to_image>")
    print(f"{'='*60}\n")
