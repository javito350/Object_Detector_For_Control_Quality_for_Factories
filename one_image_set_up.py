# one_image_setup.py - USING TRAINING IMAGES FOR CALIBRATION
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import glob

print("=" * 60)
print("MOON SYMMETRY EXPERIMENT - SINGLE IMAGE CALIBRATION")
print("=" * 60)
print("This tool learns from perfect reference images")
print("in the TRAIN/GOOD folder for defect detection!")
print("=" * 60)

sys.path.append('models')
from models.anomaly_inspector import EnhancedAnomalyInspector

def setup_with_one_image(perfect_image_path):
    """
    Setup the system with ONE perfect image.
    This learns the 'normal' pattern from just one example.
    """
    
    print(f"\n📸 Using reference image: {perfect_image_path}")
    
    if not os.path.exists(perfect_image_path):
        print(f"❌ Image not found: {perfect_image_path}")
        raise FileNotFoundError(f"Image not found: {perfect_image_path}")
    
    print(f"✓ Selected: {os.path.basename(perfect_image_path)}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load the perfect image
    print("\n🖼️  Loading reference image...")
    try:
        img = Image.open(perfect_image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        print(f"✓ Image loaded: {img.size[0]}x{img.size[1]} → Resized to 224x224")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        raise
    
    # Create inspector
    print("\n🛠️  Creating anomaly inspector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create inspector
    inspector = EnhancedAnomalyInspector(
        backbone="wide_resnet50_2",
        symmetry_type="both",
        device=device,
        coreset_percentage=1.0
    )
    
    # Create a dataloader with just that one image
    class OneImageLoader:
        def __iter__(self):
            # Yield: (images, labels, paths)
            yield (img_tensor, torch.zeros(1), [perfect_image_path])
        def __len__(self):
            return 1
    
    # This is the 'learning' phase
    print("\n🧠 Learning normal pattern from perfect sample...")
    print("   This establishes what a 'good' product looks like.")
    
    print("\n" + "─" * 40)
    print("CALIBRATING SYSTEM...")
    print("─" * 40)
    
    inspector.fit(OneImageLoader())
    
    print("\n" + "=" * 60)
    print("✅ CALIBRATION COMPLETE!")
    print("=" * 60)
    print("System has learned the reference pattern.")
    print("Ready to detect defects in new images!")
    print("=" * 60)
    
    return inspector

def setup_with_training_images(train_folder="data/water_bottles/train/good"):
    """
    Setup the system with ALL good images from training folder.
    """
    
    print(f"\n📁 Using training images from: {train_folder}")
    
    if not os.path.exists(train_folder):
        print(f"❌ Training folder not found: {train_folder}")
        raise FileNotFoundError(f"Training folder not found: {train_folder}")
    
    # Get all training images
    images = []
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        images.extend(glob.glob(os.path.join(train_folder, f"*{ext}")))
    
    if not images:
        raise FileNotFoundError(f"No images found in {train_folder}")
    
    # Remove duplicates (case-insensitive)
    unique_images = []
    seen = set()
    for img in images:
        key = img.lower()
        if key not in seen:
            seen.add(key)
            unique_images.append(img)
    
    print(f"📸 Found {len(unique_images)} training images:")
    for i, img in enumerate(unique_images, 1):
        print(f"   {i:2d}. {os.path.basename(img)}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load all images
    print("\n🖼️  Loading training images...")
    image_tensors = []
    valid_paths = []
    for img_path in unique_images:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            image_tensors.append(img_tensor)
            valid_paths.append(img_path)
            print(f"✓ {os.path.basename(img_path)}")
        except Exception as e:
            print(f"✗ {os.path.basename(img_path)}: {e}")
    
    if not image_tensors:
        raise ValueError("No training images could be loaded!")
    
    # Concatenate all tensors
    all_images = torch.cat(image_tensors, dim=0)
    
    # Create inspector
    print("\n🛠️  Creating anomaly inspector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    inspector = EnhancedAnomalyInspector(
        backbone="wide_resnet50_2",
        symmetry_type="both",
        device=device,
        coreset_percentage=1.0
    )
    
    # Create a dataloader with all images
    class TrainingLoader:
        def __iter__(self):
            yield (all_images, torch.zeros(len(all_images)), valid_paths)
        def __len__(self):
            return 1
    
    # Learning phase
    print(f"\n🧠 Learning from {len(valid_paths)} training samples...")
    inspector.fit(TrainingLoader())
    
    print(f"\n✅ Learned from {len(valid_paths)} perfect examples!")
    return inspector, valid_paths

if __name__ == "__main__":
    print("\nSelect calibration source:")
    print("1. 🎯 Use ONE perfect image from train/good/")
    print("2. 📚 Use ALL good images from train/good/")
    print("3. 🧪 Use test images (for debugging)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip() or "1"
    
    try:
        if choice == "1":
            # SINGLE IMAGE from TRAINING
            print("\n" + "=" * 60)
            print("SINGLE TRAINING IMAGE CALIBRATION")
            print("=" * 60)
            
            train_folder = "data/water_bottles/train/good"
            if os.path.exists(train_folder):
                print(f"\n📁 Perfect reference images in {train_folder}:")
                images = []
                for ext in [".jpeg", ".jpg", ".png"]:
                    found = glob.glob(os.path.join(train_folder, f"*{ext}"))
                    for img in found:
                        img_lower = img.lower()
                        if not any(img_lower == existing.lower() for existing in images):
                            images.append(img)
                
                if images:
                    images.sort()
                    for i, img in enumerate(images, 1):
                        print(f"  {i}. {os.path.basename(img)}")
                    
                    use_first = input(f"\nUse '{os.path.basename(images[0])}'? (Y/n): ").strip().lower()
                    if use_first in ['', 'y', 'yes']:
                        image_path = images[0]
                    else:
                        choice = input(f"Enter image number (1-{len(images)}) or path: ").strip()
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(images):
                                image_path = images[idx]
                            else:
                                print(f"Invalid. Using '{os.path.basename(images[0])}'")
                                image_path = images[0]
                        except:
                            if choice and os.path.exists(choice):
                                image_path = choice
                            else:
                                print(f"Invalid. Using '{os.path.basename(images[0])}'")
                                image_path = images[0]
                    
                    print(f"\n🌟 CALIBRATING WITH PERFECT REFERENCE IMAGE")
                    print(f"   This is what a 'good' {os.path.basename(image_path)} looks like!")
                    
                    inspector = setup_with_one_image(image_path)
                else:
                    print("❌ No training images found!")
                    sys.exit(1)
            else:
                print(f"❌ Training folder not found: {train_folder}")
                sys.exit(1)
            
        elif choice == "2":
            # ALL TRAINING IMAGES
            print("\n" + "=" * 60)
            print("MULTI-IMAGE TRAINING CALIBRATION")
            print("=" * 60)
            
            inspector, used_images = setup_with_training_images("data/water_bottles/train/good")
            print(f"\n✅ Calibrated with {len(used_images)} perfect training examples")
            
        elif choice == "3":
            # TEST IMAGES (for debugging)
            print("\n" + "=" * 60)
            print("TEST IMAGE CALIBRATION (DEBUG MODE)")
            print("=" * 60)
            print("⚠️  Using test images for calibration")
            print("   (Normally use train/good/ images!)")
            
            test_folder = "data/water_bottles/test"
            if os.path.exists(test_folder):
                images = []
                for ext in [".jpeg", ".jpg", ".png"]:
                    found = glob.glob(os.path.join(test_folder, f"*{ext}"))
                    for img in found:
                        img_lower = img.lower()
                        if not any(img_lower == existing.lower() for existing in images):
                            images.append(img)
                
                if images:
                    images.sort()
                    print(f"\n📁 Test images in {test_folder}:")
                    for i, img in enumerate(images[:5], 1):
                        print(f"  {i}. {os.path.basename(img)}")
                    
                    use_first = input(f"\nUse '{os.path.basename(images[0])}'? (Y/n): ").strip().lower()
                    if use_first in ['', 'y', 'yes']:
                        image_path = images[0]
                    else:
                        choice = input(f"Enter image number (1-{len(images)}) or path: ").strip()
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(images):
                                image_path = images[idx]
                            else:
                                image_path = images[0]
                        except:
                            image_path = images[0]
                    
                    inspector = setup_with_one_image(image_path)
                else:
                    print("❌ No test images found!")
                    sys.exit(1)
            else:
                print(f"❌ Test folder not found: {test_folder}")
                sys.exit(1)
        else:
            print("❌ Invalid choice. Using single training image...")
            inspector = setup_with_one_image("data/water_bottles/train/good/example_good_water_bottle.jpeg")
        
        # Save the calibrated inspector
        output_file = "calibrated_inspector.pth"
        torch.save(inspector, output_file)
        
        print(f"\n💾 Saved calibrated inspector to: {output_file}")
        file_size = os.path.getsize(output_file) / 1024 / 1024
        print(f"📏 Model size: {file_size:.2f} MB")
        
        # Show calibration summary
        print("\n" + "=" * 60)
        print("🎯 CALIBRATION COMPLETE - READY FOR DEMO")
        print("=" * 60)
        
        if choice == "1":
            print("✅ Learned from ONE perfect training example")
            print("✅ Established 'good' product reference")
            print("✅ Ready to detect ANY deviations")
        elif choice == "2":
            print("✅ Learned from MULTIPLE perfect examples")  
            print("✅ More robust 'good' product model")
            print("✅ Better generalization for defect detection")
        
        print("\n🚀 Test it with your demo:")
        print("   python demo.py --image data/water_bottles/test/test.jpeg")
        print("\n🔍 The system will compare test images")
        print("   to the learned 'perfect' reference")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n📁 Current directory: {os.getcwd()}")
        
        # Show available training data
        train_path = "data/water_bottles/train/good"
        if os.path.exists(train_path):
            print(f"\n📂 Training images in {train_path}:")
            items = os.listdir(train_path)
            if items:
                for item in items:
                    print(f"  • {item}")
            else:
                print("  (empty folder)")