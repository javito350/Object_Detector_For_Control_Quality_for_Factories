# create_defective_samples.py
from PIL import Image, ImageDraw, ImageFilter
import os
import random

print("=" * 60)
print("CREATE DEFECTIVE SAMPLES FOR DEMO")
print("=" * 60)

# Path to good reference
good_path = "data/water_bottles/train/good/example_good_water_bottle.jpeg"

if not os.path.exists(good_path):
    print(f"❌ Good image not found: {good_path}")
    exit()

print(f"📸 Original: {good_path}")
img = Image.open(good_path).convert('RGB')
print(f"   Size: {img.size[0]}x{img.size[1]}")

# Create output directory
output_dir = "defective_samples"
os.makedirs(output_dir, exist_ok=True)

def create_defect(name, description, image):
    """Create a specific type of defect"""
    print(f"\n🔧 Creating: {name}")
    print(f"   {description}")
    
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    
    if "scratch" in name.lower():
        # Add scratches
        for _ in range(random.randint(3, 7)):
            x1 = random.randint(width//4, 3*width//4)
            y1 = random.randint(height//4, 3*height//4)
            x2 = x1 + random.randint(20, 100)
            y2 = y1 + random.randint(-10, 10)
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=3)
    
    elif "dent" in name.lower():
        # Add a dent (dark ellipse)
        center_x = random.randint(width//3, 2*width//3)
        center_y = random.randint(height//3, 2*height//3)
        draw.ellipse([center_x-30, center_y-20, center_x+30, center_y+20], 
                     fill=(50, 50, 50, 100))
    
    elif "color" in name.lower():
        # Color shift
        r, g, b = result.split()
        r = r.point(lambda x: min(x + 40, 255))  # Increase red
        result = Image.merge("RGB", (r, g, b))
    
    elif "stain" in name.lower():
        # Add a stain
        center_x = random.randint(width//4, 3*width//4)
        center_y = random.randint(height//4, 3*height//4)
        for i in range(5):
            radius = 20 + i*5
            draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                        fill=(150, 100, 50, 50))
    
    elif "label" in name.lower():
        # Missing/wrong label (white rectangle)
        label_x = width // 3
        label_y = height // 4
        label_w = width // 3
        label_h = height // 6
        draw.rectangle([label_x, label_y, label_x+label_w, label_y+label_h], 
                      fill=(255, 255, 255))
    
    elif "blur" in name.lower():
        # Blurred section (out of focus)
        result = result.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Save
    output_path = os.path.join(output_dir, f"defective_{name.replace(' ', '_').lower()}.jpg")
    result.save(output_path)
    print(f"   💾 Saved: {output_path}")
    
    return result, output_path

# Create different defect types
defects = [
    ("Minor Scratch", "Small surface scratches on bottle"),
    ("Major Dent", "Visible dent in bottle body"),
    ("Color Shift", "Incorrect color tint"),
    ("Oil Stain", "Surface contamination/stain"),
    ("Missing Label", "Brand label missing or damaged"),
    ("Blurred", "Out-of-focus manufacturing defect"),
]

created_files = []

print("\n" + "=" * 60)
print("CREATING DEFECTIVE SAMPLES")
print("=" * 60)

for defect_name, description in defects:
    defective_img, filepath = create_defect(defect_name, description, img)
    created_files.append((defect_name, filepath))

print("\n" + "=" * 60)
print("DEFECTIVE SAMPLES CREATED!")
print("=" * 60)
print(f"\n📁 Saved to: {output_dir}/")
print("\n📋 Created files:")
for name, path in created_files:
    print(f"  • {os.path.basename(path)} - {name}")

print("\n🚀 Use these in your demo:")
print(f"   python demo_sensitive.py --image {created_files[0][1]}")
print("\n💡 For your speech: Show BEFORE/AFTER comparison!")
print("=" * 60)