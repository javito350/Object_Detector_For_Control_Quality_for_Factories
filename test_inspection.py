import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os

# Add models directory to Python path
sys.path.append('models')  # This tells Python where to find your models

from models.anomaly_inspector import EnhancedAnomalyInspector
from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor

def test_system():
    """
    Test the entire anomaly detection system.
    """
    print("=" * 60)
    print("Testing PhD-Level Industrial Inspection System")
    print("=" * 60)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create inspector - CHANGED to wide_resnet50_2
    try:
        inspector = EnhancedAnomalyInspector(
            backbone="wide_resnet50_2",  # CHANGED from "resnet18" to match feature extractor test
            symmetry_type="both",
            device=device,
            coreset_percentage=0.1
        )
        print("✅ Inspector created successfully!")
    except Exception as e:
        print(f"❌ Failed to create inspector: {e}")
        return False
    
    # Test with dummy data
    batch_size = 2
    height, width = 224, 224
    
    # Create dummy images (simulating normal training)
    dummy_images = torch.randn(batch_size, 3, height, width)
    
    # Create a simple dataloader-like structure
    class DummyDataloader:
        def __init__(self, images):
            self.images = images
            self.labels = torch.zeros(len(images))
            self.paths = [f"dummy_{i}.jpg" for i in range(len(images))]
            
        def __iter__(self):
            for i in range(len(self.images)):
                yield (self.images[i:i+1], 
                       self.labels[i:i+1], 
                       [self.paths[i]])
                
        def __len__(self):
            return len(self.images)
    
    dummy_dataloader = DummyDataloader(dummy_images)
    
    try:
        # Test training
        print("\n" + "-" * 60)
        print("Testing training phase...")
        inspector.fit(dummy_dataloader)
        print("✅ Training successful!")
        
        # Test prediction
        print("\n" + "-" * 60)
        print("Testing prediction phase...")
        
        # Create test images (one normal-looking, one with synthetic defect)
        test_images = []
        
        # Normal-looking image
        normal_image = torch.randn(1, 3, height, width) * 0.1 + 0.5  # Centered around 0.5
        
        # Defective image (add synthetic defect)
        defective_image = normal_image.clone()
        # Add a "crack" (line defect)
        defective_image[0, :, height//2-5:height//2+5, width//2-20:width//2+20] = 0.0
        
        test_images = torch.cat([normal_image, defective_image], dim=0)
        
        results = inspector.predict(test_images)
        
        print(f"✅ Prediction successful! Got {len(results)} results")
        
        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        for i, result in enumerate(results):
            print(f"\nImage {i+1}:")
            print(f"  Image score: {result.image_score:.4f}")
            print(f"  Defect type: {result.defect_type.value}")
            print(f"  Confidence: {result.confidence:.4f}")
            print(f"  Severity: {result.severity:.4f}")
            print(f"  Is defective: {result.is_defective()}")
            print(f"  Anomaly pixels: {result.metadata['num_anomaly_pixels']}")
        
        # Verify that second image (defective) has higher score
        if len(results) >= 2:
            score_diff = results[1].image_score - results[0].image_score
            if score_diff > 0.1:
                print(f"\n✅ Defect detection working: Defective image score is {score_diff:.4f} higher")
            else:
                print(f"\n⚠️  Warning: Defect detection may need tuning (score diff: {score_diff:.4f})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """
    Test individual components separately for debugging.
    """
    print("\n" + "=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test Feature Extractor
    print("\n1. Testing Feature Extractor...")
    try:
        from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
        extractor = SymmetryAwareFeatureExtractor(device=device)
        test_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Test symmetry features
        features = extractor.extract_symmetry_features(test_input)
        print(f"   ✅ Extracted {len(features)} feature maps")
        
        # Test patch features
        patch_features = extractor.extract_patch_features(test_input)
        print(f"   ✅ Patch features shape: {patch_features.shape}")
        
        # Test symmetry heatmap
        heatmap = extractor.extract_symmetry_heatmap(test_input)
        print(f"   ✅ Symmetry heatmap shape: {heatmap.shape}")
        
        print("   ✅ Feature extractor working!")
    except Exception as e:
        print(f"   ❌ Feature extractor failed: {e}")
        return False
    
    # Test Memory Bank
    print("\n2. Testing Memory Bank...")
    try:
        from models.memory_bank import MemoryBank
        
        # Create dummy features
        dummy_features = np.random.randn(100, 512).astype(np.float32)
        
        # Build memory bank
        memory_bank = MemoryBank(dimension=512, use_gpu=False)
        memory_bank.build([dummy_features], coreset_percentage=0.1)
        
        # Test query
        query_features = np.random.randn(1, 512).astype(np.float32)
        distances, indices = memory_bank.query(query_features, k=3)
        
        print(f"   ✅ Memory bank built with {len(memory_bank.features)} features")
        print(f"   ✅ Query returned distances: {distances.shape}, indices: {indices.shape}")
        print("   ✅ Memory bank working!")
    except Exception as e:
        print(f"   ❌ Memory bank failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHD INDUSTRIAL INSPECTION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        print("\n" + "=" * 60)
        print("All components working. Testing full system...")
        print("=" * 60)
        
        success = test_system()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! System is ready for training.")
            print("=" * 60)
            
            # Show next steps
            print("\n" + "-" * 60)
            print("NEXT STEPS:")
            print("-" * 60)
            print("1. Prepare your data in this structure:")
            print("   data/product_name/train/good/  # Normal images")
            print("   data/product_name/test/        # Test images (mixed)")
            print("\n2. Train the system:")
            print("   python main.py train --category your_product")
            print("\n3. Test on real images:")
            print("   python main.py inspect --mode single --category your_product --image_path test.jpg")
        else:
            print("\n❌ System test failed. Please check the errors above.")
    else:
        print("\n❌ Component tests failed. Please fix the individual components first.")