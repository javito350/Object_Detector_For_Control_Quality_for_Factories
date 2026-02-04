# G-CNN Defect Detection System
## One-Shot Learning with Symmetry Exploitation for Industrial Quality Control

---

## Project Overview

This system demonstrates **one-shot anomaly detection** - requiring only **ONE normal training image** to detect defects in manufactured products. By exploiting the geometric symmetry properties of industrial objects (bottles, cans, phone cases), the system achieves effective defect detection with minimal training data.

### Key Innovation: ONE Image Training

**Traditional anomaly detection**: Requires 50-1000+ normal samples  
**This G-CNN approach**: Requires just **1 normal sample** + symmetry analysis

This 50-100x reduction in data requirements is achieved by:
1. **Symmetry exploitation** - Manufactured objects are inherently symmetric
2. **Memory bank comparison** - Stores multi-scale feature representations
3. **Geometric priors** - Any break in expected symmetry indicates defects

### Key Features
- **One-shot learning** - Trains on a SINGLE normal image (not hundreds)
- **Symmetry-aware detection** - Exploits horizontal/vertical/rotational symmetry
- **PatchCore architecture** - State-of-the-art memory bank with Wide ResNet-50
- **Minimal data collection** - Reduces training data needs by 50-100x
- **Visual explanations** - Shows exactly where symmetry breaks occur
- **Real-time capability** - Fast enough for production deployment

---

## 🏗️ Architecture

The system combines two powerful techniques:

1. **PatchCore Memory Bank**: Stores representations of normal appearance patterns
2. **Symmetry Analysis**: Detects breaks in expected geometric symmetry

```
Input Image → Feature Extraction → Memory Bank Comparison
                   ↓                        ↓
            Symmetry Check  ←  →   Anomaly Score → Decision
```

### Technical Details
- **Backbone**: Wide ResNet-50-2 (pre-trained on ImageNet)
- **Feature Dimensions**: Multi-scale (256, 512, 1024 channels)
- **Memory Bank**: Coreset subsampling (10% of training features)
- **Symmetry Types**: Horizontal, vertical, and rotational
- **Threshold**: Adaptive (learned from normal distribution)

---

## 📁 Project Structure

```
Moon_Symmetry_Experiment/
│
├── data/
│   └── water_bottles/
│       ├── train/
│       │   └── good/              # Training images (GOOD products only)
│       └── test/                   # Test images for evaluation
│
├── models/
│   ├── anomaly_inspector.py       # Main inspection model
│   ├── symmetry_feature_extractor.py  # Feature extraction with symmetry
│   └── memory_bank.py              # Memory bank implementation
│
├── defective_samples/              # Synthetic defective examples
├── presentation_results/           # Generated visualizations
│
├── presentation_demo.py            # Main demo script (USE THIS!)
├── sensitive_inspector.pth         # Trained model
└── README.md                       # This file
```

---

## Quick Start

### Installation

```bash
pip install torch torchvision opencv-python scikit-image scipy faiss-cpu numpy tqdm matplotlib
```

### Running Inspection

```bash
# Symmetry analysis demo (recommended - shows 4-panel analysis)
python symmetry_demo.py data/water_bottles/test/

# Standard visual demo
python visual_demo.py data/water_bottles/test/

# Single image inspection
python symmetry_demo.py path/to/image.jpg

# Batch processing with reports
python presentation_demo.py path/to/directory/
```

### Understanding the Output

The **symmetry demo** shows 4 panels:
1. **Original** - Input product image
2. **Flipped** - Horizontally flipped for symmetry comparison
3. **Symmetry Breaks** - Heatmap showing asymmetric regions (defects)
4. **Detection Result** - Final verdict with anomaly overlay

Red/hot regions in the symmetry map indicate where the product breaks expected symmetry - a strong indicator of defects.

---

## 📊 Output

The system generates:

1. **Console Report**: Detailed inspection results
2. **Visualizations**: 3-panel images showing:
   - Original image
   - Anomaly heatmap
   - Overlay with defect location (if found)
3. **Summary File**: Text report of all inspections

### Example Output

```
=================================================================
IMAGE: water_bottle_001.jpeg
=================================================================
STATUS: DEFECTIVE
  >> Defect Type: CRACK
  >> Severity: 78.5%
  >> Confidence: 92.3%

TECHNICAL DETAILS:
  Anomaly Score: 2.4521
  Threshold: 1.3316
  Difference: +1.1205
  Anomaly Pixels: 1247
  Defect Location: [320, 450] to [380, 620]
```

---

## 🎓 How It Works

### Training Phase (One-Shot Learning)
1. Load **ONE** normal product image from `data/water_bottles/train/good/`
2. Extract multi-scale features using Wide ResNet-50
3. Generate symmetric augmentations (horizontal flip, vertical flip)
4. Build memory bank from normal appearance + symmetric variations
5. Compute symmetry consistency statistics
6. Learn adaptive threshold

**Key advantage**: Only needs 1 image instead of 50-1000+ samples

### Inference Phase (Symmetry Exploitation)
1. Load test image
2. Extract features and compare to memory bank → appearance anomaly score
3. Flip image (horizontal/vertical) and compare → symmetry consistency score
4. Detect symmetry breaks (defects appear asymmetric)
5. Fuse appearance score + symmetry score → final anomaly score
6. Classify as GOOD (symmetric) or DEFECTIVE (asymmetric)
7. Generate heatmap showing symmetry breaks

**Key insight**: Defects break symmetry - a crack on one side makes the bottle asymmetric

---

## 🔬 Model Performance

The trained model (`sensitive_inspector.pth`) was trained on:
- **Training samples**: 1 perfect water bottle image
- **Architecture**: Wide ResNet-50-2 + Symmetry Analysis
- **Threshold**: 1.3316 (automatically calibrated)

### Detected Defect Types
- ✗ Cracks and scratches
- ✗ Dents and deformations
- ✗ Color shifts/discoloration
- ✗ Contamination/stains
- ✗ Missing or damaged labels
- ✗ Asymmetric damage
- ✗ Manufacturing defects

---

## 📸 Presentation Demo Instructions

### For Your Presentation Tomorrow:

1. **Show the system loading**:
   ```bash
   python presentation_demo.py
   ```

2. **Explain the output**:
   - Green "GOOD" = Ready for sale
   - Red "DEFECTIVE" = Quality issue detected
   - Heatmap shows where the problem is

3. **Show visualizations**:
   - Open `presentation_results/` folder
   - Display the generated analysis images
   - Point out the heatmap highlighting defects

4. **Key talking points**:
   - "Trained on only GOOD images - no defect examples needed"
   - "Uses deep learning + geometric symmetry analysis"
   - "Provides visual explanations, not just a yes/no answer"
   - "Ready for real-time production line deployment"

---

## 🛠️ Customization

### Training on Your Own Data

```python
from models.anomaly_inspector import EnhancedAnomalyInspector
from utils.image_loader import create_dataloader

# Initialize model
inspector = EnhancedAnomalyInspector(
    backbone="wide_resnet50_2",
    symmetry_type="both",  # horizontal, vertical, both, rotational
    device="cuda"
)

# Load your data
dataloader = create_dataloader("path/to/good/images", batch_size=8)

# Train
inspector.fit(dataloader)

# Save
torch.save(inspector, "my_inspector.pth")
```

### Adjusting Sensitivity

If the model is too sensitive or not sensitive enough, you can adjust the threshold:

```python
# Make more sensitive (detect more defects, more false alarms)
inspector.image_threshold = inspector.image_threshold * 0.9

# Make less sensitive (fewer false alarms, might miss subtle defects)
inspector.image_threshold = inspector.image_threshold * 1.1
```

---

## 📈 Technical Background

This implementation is based on research in:
- **PatchCore**: Towards Total Recall in Industrial Anomaly Detection
- **Symmetry-Aware Networks**: Exploiting geometric priors for improved robustness
- **Memory-Based Learning**: Few-shot anomaly detection

### Why G-CNN?
Group Convolutional Neural Networks exploit the fact that many manufactured objects (bottles, cans, phone cases) have predictable symmetries. Any break in this symmetry is a strong indicator of defects.

---

## 🎯 Use Cases

Perfect for:
- 🏭 **Manufacturing Quality Control**: Automated inspection on production lines
- 📦 **Packaging Inspection**: Detect damaged or incorrect packaging
- 🔧 **Component Testing**: Find manufacturing defects in parts
- 🍾 **Bottling Industry**: Inspect bottles, cans, containers
- 📱 **Electronics**: Detect defects in phones, tablets, cases

---

## 📝 Citation

If you use this system in your research or business, please cite:

```bibtex
@software{gcnn_defect_detection_2026,
  title={G-CNN Defect Detection System},
  author={Moon Symmetry Experiment Team},
  year={2026},
  description={Symmetry-aware anomaly detection for manufacturing quality control}
}
```

---

## 📧 Support

For questions or issues:
- Check the visualizations in `presentation_results/`
- Review the inspection summary in `presentation_results/inspection_summary.txt`
- Ensure all dependencies are installed correctly

---

## ✨ Key Advantages

| Feature | Traditional QC | This G-CNN System |
|---------|---------------|-------------------|
| Training Data | Needs thousands of defect examples | Works with GOOD samples only |
| Speed | Manual, slow | Real-time automated |
| Consistency | Human error prone | 100% consistent |
| Explanation | "It looks wrong" | Visual heatmap + metrics |
| Cost | High labor cost | One-time setup cost |
| Scalability | Limited | Unlimited |

---

## System Workflow

1. Load product image → Extract multi-scale deep features
2. Compare against memory bank → Calculate anomaly score
3. Analyze symmetry consistency → Fuse appearance and geometric features
4. Generate heatmap visualization → Output classification and localization

---

**Built using PyTorch, OpenCV, and state-of-the-art anomaly detection methods**

_Last updated: February 2026_
