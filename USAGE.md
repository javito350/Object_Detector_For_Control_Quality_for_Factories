# Usage Guide

Comprehensive guide for using the Moon Symmetry Experiment anomaly detection system.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Inference on Single Image](#inference-on-single-image)
3. [Batch Processing](#batch-processing)
4. [Training on New Data](#training-on-new-data)
5. [Visualization](#visualization)
6. [Advanced Configuration](#advanced-configuration)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Basic Usage

### Quick Demo

Run the built-in demonstration:

```bash
# Show help
python run_demo.py --help

# Run demo on a single image
python run_demo.py --image_path path/to/image.jpg

# Save results to directory
python run_demo.py --image_path path/to/image.jpg --output_dir results/

# Process entire folder
python run_demo.py --data_dir data/water_bottles/test/ --verbose

# Use specific model
python run_demo.py --image_path test.jpg --model_path weights/sensitive_inspector.pth
```

### Minimal Python Example

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_inspector import EnhancedAnomalyInspector

# 1. Initialize inspector
inspector = EnhancedAnomalyInspector(device="cuda")

# 2. Load image
image = Image.open("test_image.jpg").convert('RGB')

# 3. Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# 4. Predict
results = inspector.predict(image_tensor)

# 5. Inspect results
result = results[0]
print(f"Defective: {result.is_defective()}")
print(f"Defect Type: {result.defect_type.value}")
print(f"Confidence: {result.confidence:.4f}")
```

---

## Inference on Single Image

### Example 1: Basic Single Image Inference

```python
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_inspector import EnhancedAnomalyInspector

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
inspector = EnhancedAnomalyInspector(device=device)

# Load and preprocess image
image_path = Path("data/water_bottles/test/sample.jpg")
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    results = inspector.predict(image_tensor)

result = results[0]

# Access results
print(f"Image Score (0-1): {result.image_score:.4f}")
print(f"Defect Type: {result.defect_type.value}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Severity: {result.severity:.4f}")
print(f"Is Defective: {result.is_defective(threshold=0.5)}")

if result.bbox:
    x1, y1, x2, y2 = result.bbox
    print(f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")

# Access heatmaps
print(f"Anomaly Map Shape: {result.anomaly_map.shape}")
print(f"Symmetry Map Shape: {result.symmetry_map.shape}")
print(f"Anomaly Pixels: {result.metadata['num_anomaly_pixels']}")
```

### Example 2: Custom Preprocessing

```python
import cv2
import numpy as np
from models.anomaly_inspector import EnhancedAnomalyInspector

# Load image with OpenCV
image = cv2.imread("test_image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Custom preprocessing
image_resized = cv2.resize(image_rgb, (224, 224))
image_normalized = image_resized.astype(np.float32) / 255.0

# Convert to tensor
import torch
image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)

# Normalize (ImageNet mean/std)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
image_tensor = (image_tensor - mean) / std

# Predict
inspector = EnhancedAnomalyInspector(device="cuda")
results = inspector.predict(image_tensor)
print(f"Result: {results[0].defect_type.value}")
```

### Example 3: Real-time Webcam Feed

```python
import cv2
import torch
import numpy as np
from models.anomaly_inspector import EnhancedAnomalyInspector
import torchvision.transforms as transforms

inspector = EnhancedAnomalyInspector(device="cuda")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert and preprocess
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = transforms.ToPILImage()(rgb_frame)
    tensor = transform(pil_image).unsqueeze(0).cuda()
    
    # Predict
    with torch.no_grad():
        results = inspector.predict(tensor)
    
    result = results[0]
    is_defective = result.is_defective()
    
    # Draw on frame
    color = (0, 0, 255) if is_defective else (0, 255, 0)  # Red if defective
    text = f"Defect: {result.defect_type.value} ({result.image_score:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2)
    
    cv2.imshow("Anomaly Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Batch Processing

### Example 1: Process Multiple Images

```python
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_inspector import EnhancedAnomalyInspector
import pandas as pd

def process_image_folder(folder_path, output_csv=None):
    """Process all images in a folder and return results."""
    
    inspector = EnhancedAnomalyInspector(device="cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    results_list = []
    image_files = list(Path(folder_path).glob("*.jpg")) + \
                  list(Path(folder_path).glob("*.png"))
    
    print(f"Processing {len(image_files)} images...")
    
    for idx, image_path in enumerate(image_files):
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).cuda()
        
        # Predict
        with torch.no_grad():
            results = inspector.predict(image_tensor)
        
        result = results[0]
        
        # Store results
        results_list.append({
            'filename': image_path.name,
            'is_defective': result.is_defective(),
            'defect_type': result.defect_type.value,
            'anomaly_score': result.image_score,
            'confidence': result.confidence,
            'severity': result.severity,
            'anomaly_pixels': result.metadata['num_anomaly_pixels']
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)}")
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return df

# Usage
results_df = process_image_folder("data/water_bottles/test/", 
                                  output_csv="results.csv")
print(results_df.head())
print(f"\nSummary:")
print(f"Total images: {len(results_df)}")
print(f"Defective: {results_df['is_defective'].sum()}")
print(f"Mean anomaly score: {results_df['anomaly_score'].mean():.4f}")
```

### Example 2: Batch Inference (GPU Optimized)

```python
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.images = list(Path(image_folder).glob("*.jpg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(self.images[idx].name)

# Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset("data/water_bottles/test/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

inspector = EnhancedAnomalyInspector(device="cuda")

# Process batches
all_results = []
for batch_images, filenames in dataloader:
    batch_images = batch_images.cuda()
    
    with torch.no_grad():
        results = inspector.predict(batch_images)
    
    for filename, result in zip(filenames, results):
        all_results.append({
            'filename': filename,
            'score': result.image_score,
            'defect': result.defect_type.value
        })

print(f"Processed {len(all_results)} images in batches")
```

---

## Training on New Data

### Example: Train Inspector on Custom Dataset

```python
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from models.anomaly_inspector import EnhancedAnomalyInspector
import torchvision.transforms as transforms

class NormalSamplesDataset(Dataset):
    """Dataset of normal/good samples for memory bank training."""
    
    def __init__(self, good_samples_folder, transform=None):
        self.images = list(Path(good_samples_folder).glob("*.jpg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = 0  # Normal sample
        return image, label, str(self.images[idx])

# Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = NormalSamplesDataset("data/water_bottles/train/good/", 
                               transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Initialize and train inspector
inspector = EnhancedAnomalyInspector(
    backbone="wide_resnet50_2",
    symmetry_type="both",
    device="cuda",
    coreset_percentage=0.1  # Keep top 10% of features
)

print("Training inspector on good samples...")
inspector.fit(dataloader)

# Save trained model
torch.save(inspector, "weights/custom_inspector.pth")
print("Model saved!")
```

### Example: Load and Use Custom-Trained Model

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load custom model
inspector = torch.load("weights/custom_inspector.pth")
inspector.device = "cuda"
inspector = inspector.to("cuda")

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0).cuda()

# Predict
with torch.no_grad():
    results = inspector.predict(image_tensor)

print(f"Result: {results[0].defect_type.value}")
```

---

## Visualization

### Example 1: Display Heatmaps

```python
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_inspector import EnhancedAnomalyInspector
import numpy as np

# Setup
inspector = EnhancedAnomalyInspector(device="cuda")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open("test_image.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0).cuda()
results = inspector.predict(image_tensor)
result = results[0]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Anomaly heatmap
im1 = axes[0, 1].imshow(result.anomaly_map, cmap='hot')
axes[0, 1].set_title("Anomaly Map")
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

# Symmetry map
im2 = axes[1, 0].imshow(result.symmetry_map, cmap='cool')
axes[1, 0].set_title("Symmetry Map")
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0])

# Binary mask
axes[1, 1].imshow(result.binary_mask, cmap='gray')
axes[1, 1].set_title("Binary Mask")
axes[1, 1].axis('off')

# Add main title with metadata
fig.suptitle(
    f"Defect Type: {result.defect_type.value} | "
    f"Score: {result.image_score:.4f} | "
    f"Confidence: {result.confidence:.4f}",
    fontsize=14
)

plt.tight_layout()
plt.savefig("result_visualization.png", dpi=150, bbox_inches='tight')
print("Visualization saved to result_visualization.png")
plt.show()
```

### Example 2: Overlay Bounding Box

```python
import cv2
import numpy as np
from PIL import Image

# Draw bounding box on original image
image = cv2.imread("test_image.jpg")
result = ...  # Your prediction result

if result.bbox:
    x1, y1, x2, y2 = result.bbox
    color = (0, 0, 255) if result.is_defective() else (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, result.defect_type.value, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imwrite("result_with_bbox.jpg", image)
```

### Example 3: Confidence Distribution

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load results from batch processing
results_df = pd.read_csv("results.csv")

# Create distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Anomaly score distribution
axes[0].hist(results_df['anomaly_score'], bins=30, edgecolor='black')
axes[0].set_xlabel("Anomaly Score")
axes[0].set_ylabel("Count")
axes[0].set_title("Score Distribution")
axes[0].axvline(x=0.5, color='r', linestyle='--', label='Threshold')
axes[0].legend()

# Defect type count
defect_counts = results_df['defect_type'].value_counts()
axes[1].bar(defect_counts.index, defect_counts.values)
axes[1].set_xlabel("Defect Type")
axes[1].set_ylabel("Count")
axes[1].set_title("Defect Type Distribution")
axes[1].tick_params(axis='x', rotation=45)

# Severity vs Confidence
axes[2].scatter(results_df['severity'], results_df['confidence'], alpha=0.5)
axes[2].set_xlabel("Severity")
axes[2].set_ylabel("Confidence")
axes[2].set_title("Severity vs Confidence")

plt.tight_layout()
plt.savefig("distribution_analysis.png", dpi=150)
print("Analysis plot saved!")
```

---

## Advanced Configuration

### Custom Model Configuration

```python
from models.anomaly_inspector import EnhancedAnomalyInspector

# Different backbone options
inspector1 = EnhancedAnomalyInspector(
    backbone="resnet18",      # Lighter, faster
    symmetry_type="both",      # Horizontal + vertical
    device="cuda",
    coreset_percentage=0.05    # Keep 5% for speed
)

inspector2 = EnhancedAnomalyInspector(
    backbone="wide_resnet50_2", # More parameters, more accurate
    symmetry_type="horizontal",  # Only horizontal symmetry
    device="cuda",
    coreset_percentage=0.2       # Keep 20% for accuracy
)

# Adjust detection thresholds
inspector1.pixel_threshold = 0.4   # Lower = more sensitive
inspector1.image_threshold = 0.3
inspector1.symmetry_threshold = 0.25

# Adjust post-processing
inspector1.gaussian_sigma = 3      # Stronger smoothing
inspector1.min_anomaly_size = 100  # Larger minimum defect size
```

### Threshold Tuning

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

# Collect results on validation set
validation_scores = []
validation_labels = []  # 0 = normal, 1 = defective

# ... run predictions and collect scores ...

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(
    validation_labels, 
    validation_scores
)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"F1-Score: {f1_scores[optimal_idx]:.4f}")

# Use optimal threshold
inspector.image_threshold = optimal_threshold
```

---

## API Reference

### EnhancedAnomalyInspector

Main class for anomaly detection.

**Constructor**:
```python
EnhancedAnomalyInspector(
    backbone="wide_resnet50_2",  # CNN backbone
    symmetry_type="both",         # "both", "horizontal", "vertical", "rotational"
    device="cuda",                # "cuda" or "cpu"
    coreset_percentage=0.1        # Fraction of features to keep [0, 1]
)
```

**Methods**:

- `fit(dataloader)` - Train on normal samples
- `predict(images)` - Predict on batch of images (torch.Tensor)
- Parameters adjustable as attributes

### InspectionResult

Result dataclass for a single prediction.

**Attributes**:
```python
@dataclass
class InspectionResult:
    image_score: float              # 0-1 anomaly score
    anomaly_map: np.ndarray         # H×W heatmap
    symmetry_map: np.ndarray        # H×W symmetry heatmap
    binary_mask: np.ndarray         # H×W binary mask
    defect_type: DefectType         # Enum: CRACK, SCRATCH, etc.
    confidence: float               # 0-1 prediction confidence
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2) or None
    severity: float                 # 0-1 defect severity
    metadata: Dict                  # Additional info
```

**Methods**:
- `is_defective(threshold=0.5)` - Check if defective

---

## Troubleshooting

### Issue: Predictions are all zeros

```python
# Check if model is properly loaded
print(inspector.memory_bank)

# Verify image preprocessing
print(image_tensor.min(), image_tensor.max())  # Should be ~[-2, 2]

# Try on CPU to debug
inspector_cpu = EnhancedAnomalyInspector(device="cpu")
```

### Issue: Inconsistent predictions

```python
# Ensure deterministic behavior
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

### Issue: Out of memory

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=8)  # Instead of 32

# Use CPU for feature extraction
inspector = EnhancedAnomalyInspector(device="cpu")
```

---

**Last Updated**: February 2024
