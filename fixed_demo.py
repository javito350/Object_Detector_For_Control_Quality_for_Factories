# fixed_demo.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

# Fix PyTorch 2.6+ security
sys.path.append('models')
from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

def interpret_prediction(result):
    """Interpret the list output from predict()"""
    if not isinstance(result, list):
        return "unknown", 0.0
    
    # Common pattern: [image_labels, anomaly_scores, ...]
    if len(result) >= 2:
        # First element might be labels (0=normal, 1=defect)
        labels = result[0]
        # Second element might be scores
        scores = result[1]
        
        if isinstance(labels, torch.Tensor) and labels.numel() == 1:
            label = labels.item()
            score = 0.0
            
            if isinstance(scores, torch.Tensor) and scores.numel() == 1:
                score = scores.item()
            
            return ("defect" if label == 1 else "normal"), score
    
    # Try to find scores in the list
    for item in result:
        if isinstance(item, torch.Tensor) and item.numel() == 1:
            score = item.item()
            # Assume threshold of 1.0
            return ("defect" if score > 1.0 else "normal"), score
    
    return "unknown", 0.0

def fixed_demo():
    print("\n" + "=" * 60)
    print("AI FACTORY INSPECTION - FIXED VERSION")
    print("=" * 60)
    print("Correctly interpreting predict() output")
    print("=" * 60)
    
    # Try different inspectors
    inspector_files = ["sensitive_inspector.pth", "calibrated_inspector.pth"]
    inspector = None
    
    for file in inspector_files:
        if os.path.exists(file):
            print(f"\n📂 Loading: {file}")
            try:
                inspector = torch.load(file, 
                                      map_location='cpu',
                                      weights_only=False)
                print(f"✅ Loaded successfully")
                break
            except Exception as e:
                print(f"❌ Failed: {e}")
    
    if inspector is None:
        print("❌ No inspector found!")
        return
    
    # Test images
    test_images = [
        ("data/water_bottles/train/good/example_good_water_bottle.jpeg", "REFERENCE (Perfect)"),
        ("data/water_bottles/test/test.jpeg", "Test 1"),
        ("data/water_bottles/test/test2.jpeg", "Test 2"),
        ("data/water_bottles/test/test3.jpeg", "Test 3"),
        ("data/water_bottles/test/test4.jpeg", "Test 4"),
        ("data/water_bottles/test/test5.jpeg", "Test 5")
    ]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "=" * 60)
    print("DEFECT DETECTION ANALYSIS")
    print("=" * 60)
    
    results = []
    
    for img_path, label in test_images:
        if not os.path.exists(img_path):
            print(f"❌ Missing: {label}")
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            print(f"\n🔍 {label}:")
            print(f"   File: {os.path.basename(img_path)}")
            print(f"   Size: {img.size[0]}x{img.size[1]}")
            
            # Get prediction
            result = inspector.predict(img_tensor)
            
            print(f"   📦 predict() returned: {type(result)}")
            
            if isinstance(result, list):
                print(f"   📦 List length: {len(result)}")
                
                # Show each element
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        print(f"     [{i}] Tensor: shape={item.shape}")
                        if item.numel() <= 3:
                            values = item.flatten().tolist()
                            print(f"          values: {values}")
                    else:
                        print(f"     [{i}] {type(item).__name__}: {item}")
            
            # Interpret the result
            status, score = interpret_prediction(result)
            
            if status != "unknown":
                print(f"   📊 Interpreted as: {status.upper()}")
                print(f"   🔢 Score: {score:.6f}")
                
                # Check against inspector threshold if available
                if hasattr(inspector, 'image_threshold'):
                    threshold = inspector.image_threshold
                    if hasattr(threshold, 'item'):
                        threshold = threshold.item()
                    
                    if score > threshold:
                        final_status = "❌ DEFECT"
                        print(f"   🎯 Threshold: {threshold:.4f} → EXCEEDED!")
                    else:
                        final_status = "✅ NORMAL"
                        print(f"   🎯 Threshold: {threshold:.4f} → within limits")
                else:
                    # Default threshold
                    final_status = "❌ DEFECT" if score > 1.0 else "✅ NORMAL"
                
                results.append((label, final_status, score))
            else:
                print(f"   ⚠️  Could not interpret result")
                results.append((label, "⚠️  UNKNOWN", 0.0))
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((label, "❌ ERROR", 0.0))
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Summary
    defect_count = sum(1 for _, status, _ in results if "❌ DEFECT" in status)
    normal_count = sum(1 for _, status, _ in results if "✅ NORMAL" in status)
    total_tests = len([r for r in results if "REFERENCE" not in r[0]])
    
    print(f"\n📊 SUMMARY:")
    print(f"   Defects detected: {defect_count}/{total_tests}")
    print(f"   Normal products: {normal_count}/{total_tests}")
    
    print("\n📋 DETAILED RESULTS:")
    print("-" * 50)
    for label, status, score in results:
        if score > 0:
            print(f"{label:25} → {status:15} (score: {score:.6f})")
        else:
            print(f"{label:25} → {status}")
    
    print("\n" + "=" * 60)
    print("DEMO SCRIPT FOR YOUR SPEECH")
    print("=" * 60)
    print("\n💬 WHAT TO SAY:")
    print("-" * 40)
    print("1. 'This system learned from ONE perfect example'")
    print("2. 'It extracts 784 feature vectors for comparison'")
    print("3. 'Each test gets an anomaly score (shown here)'")
    print(f"4. 'We detected {defect_count} defective products'")
    print("5. 'Traditional AI needs 1000s of defects to learn'")
    print("6. 'Our system: One sample → Infinite detection'")
    print("-" * 40)
    
    # Create a simple visualization
    print("\n📈 VISUALIZATION OF SCORES:")
    print("-" * 40)
    
    # Sort by score (excluding reference)
    test_results = [(label, status, score) for label, status, score in results 
                   if "REFERENCE" not in label and score > 0]
    
    if test_results:
        test_results.sort(key=lambda x: x[2], reverse=True)
        
        print("Most suspicious → Least suspicious:")
        for label, status, score in test_results:
            # Create a simple bar
            bar_length = int(score * 10)
            bar = "█" * min(bar_length, 30)
            print(f"  {label:15} {bar:30} {score:.4f} {status}")

if __name__ == "__main__":
    fixed_demo()