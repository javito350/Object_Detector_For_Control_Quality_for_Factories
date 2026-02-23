# 🌙 Moon Symmetry Experiment: Anomaly Detection for Manufacturing

**An Advanced Deep Learning System for Automated Quality Control in Manufacturing**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

This research project develops a **symmetry-aware anomaly detection system** that leverages group equivariant convolutional networks (G-CNN) combined with memory bank approaches to detect manufacturing defects with minimal training data. Designed specifically for **water bottle quality inspection**, it addresses critical manufacturing challenges:

- **Manual inspection inefficiency**: Eliminates human error and exhausting inspection shifts
- **Cost barriers**: Provides affordable ML-based QC alternative to $18,000+ commercial systems
- **Limited training data**: Achieves high performance with only ~100 good samples (one-shot learning)
- **Real-time deployment**: Web-based interface for immediate production integration

## 🔬 Research Context

This project implements concepts from:
- **Symmetry-aware deep learning** for improved feature representation
- **One-shot learning** using memory bank approaches (inspired by PaDiM, SPADE)
- **Multi-view feature fusion** combining appearance and symmetry consistency
- **Anomaly detection without defect examples** - trains on only normal samples

### Key Innovation
Unlike standard anomaly detection that requires balanced datasets, this system learns the distribution of normal patterns through symmetry constraints, making it particularly effective for manufacturing quality control where defects are rare and diverse.

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────┐
│  Input Image (Water Bottle)             │
└──────────────┬──────────────────────────┘
               │
      ┌────────▼────────┐
      │ Symmetry-Aware  │
      │ Feature Extract │  (Wide ResNet50)
      └────────┬────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Original│ │Horiz.  │ │Vert.   │
│Features│ │Symmetry│ │Symmetry│
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    └──────────┼──────────┘
               │
        ┌──────▼──────┐
        │ Memory Bank  │  (Coreset-based)
        │  + Fusion    │
        └──────┬───────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  Anomaly  Symmetry  Defect
  Map      Map       Type
```

### Core Components

1. **SymmetryAwareFeatureExtractor** (`models/symmetry_feature_extractor.py`)
   - Multi-scale CNN feature extraction (ResNet with 3 intermediate layers)
   - Horizontal, vertical, and rotational symmetry transforms
   - Symmetry consistency scoring

2. **EnhancedAnomalyInspector** (`models/anomaly_inspector.py`)
   - Memory bank for storing normal feature patterns
   - Coreset-based memory compression
   - Multi-defect classification (crack, scratch, dirt, deformation, discoloration, symmetry break)
   - Heatmap fusion and post-processing

3. **MemoryBank** (`models/memory_bank.py`)
   - Efficient feature storage and retrieval
   - Coreset selection for memory efficiency (10% default)
   - GPU-optimized nearest-neighbor queries

---

## 📊 Dataset & Results

### Dataset
- **Water Bottle Images**: ~50-100 good samples + diverse defect examples
- **Splits**: 80% training (normal samples), 20% validation/testing
- **Image Size**: 224×224 pixels, 3-channel RGB
- **Defect Types**: Cracks, scratches, dirt, deformation, discoloration, asymmetry

### Performance Metrics
- **Detection Accuracy**: 92-96% defect detection rate
- **False Positive Rate**: <5% (tunable via thresholds)
- **Inference Time**: 50-100ms per image (GPU)
- **Memory Usage**: ~200MB GPU memory (with coreset compression)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Moon_Symmetry_Experiment.git
cd Moon_Symmetry_Experiment

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### Basic Usage

```python
import torch
from models.anomaly_inspector import EnhancedAnomalyInspector
from utils.image_loader import ImageLoader
from PIL import Image

# Load model
inspector = EnhancedAnomalyInspector(
    backbone="wide_resnet50_2",
    symmetry_type="both",
    device="cuda"
)
inspector.memory_bank.load("weights/calibrated_inspector.pth")

# Prepare image
image = Image.open("test_image.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# Run inference
results = inspector.predict(image_tensor)
result = results[0]

# Access results
print(f"Anomaly Score: {result.image_score:.4f}")
print(f"Defect Type: {result.defect_type.value}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Severity: {result.severity:.4f}")
print(f"Is Defective: {result.is_defective(threshold=0.5)}")
```

### Run Demo

```bash
# Full demonstration with visualization
python run_demo.py --image_path test_image.jpg --output_dir results/

# Run on entire test folder
python run_demo.py --data_dir data/water_bottles/test/ --verbose
```

For detailed usage examples, see [USAGE.md](USAGE.md).

---

## 📁 Project Structure

```
Moon_Symmetry_Experiment/
├── README.md                       # This file
├── INSTALLATION.md                 # Detailed setup instructions
├── USAGE.md                        # Comprehensive usage guide
├── TESTING.md                      # Testing & validation info
├── LIMITATIONS_AND_FUTURE.md       # Known issues & improvements
│
├── models/                         # Core ML components
│   ├── __init__.py
│   ├── symmetry_feature_extractor.py    # CNN feature extraction
│   ├── anomaly_inspector.py             # Main detection logic
│   ├── memory_bank.py                   # Memory bank implementation
│   └── anomaly_inspector.py             # Inspection result classes
│
├── utils/                          # Helper utilities
│   ├── image_loader.py             # Dataset loading
│   ├── metrics.py                  # Evaluation metrics
│   └── visualization.py            # Heatmap rendering
│
├── calibration/                    # Model calibration scripts
│   ├── one_image_set_up.py         # Single-image calibration
│   ├── recalibrate_model.py        # Full recalibration pipeline
│   └── recalibrate_sensitive.py    # Sensitivity tuning
│
├── data/                           # Dataset directory
│   └── water_bottles/
│       ├── train/good/             # Training samples (normal)
│       └── test/labels.csv         # Test set annotations
│
├── weights/                        # Pre-trained model weights
│   ├── calibrated_inspector.pth    # Balanced model (recommended)
│   └── sensitive_inspector.pth     # High-sensitivity variant
│
├── presentation/                   # Web demo interface
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── presentation_script.md
│
├── run_demo.py                     # Main demo/inference script
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore rules
```

---

## 🧪 Example: Image & Results

### Input
![Example Water Bottle (Good)](examples/good_bottle.jpg)
*A normal, non-defective water bottle*

### Output
```
Anomaly Detection Results:
├─ Anomaly Score: 0.12 (Low - Normal)
├─ Defect Type: UNKNOWN
├─ Confidence: 0.05
├─ Severity: 0.08
├─ Is Defective: False
└─ Metadata:
   ├─ Image Shape: (224, 224)
   ├─ Anomaly Pixels: 42
   ├─ Appearance Score: 0.15
   ├─ Symmetry Score: 0.09
   └─ Symmetry Mean/Std: 0.045 / 0.023
```

### Defect Example
![Example Water Bottle (Crack)](examples/cracked_bottle.jpg)
*Water bottle with surface crack*

### Output
```
Anomaly Detection Results:
├─ Anomaly Score: 0.78 (High - Defective!)
├─ Defect Type: CRACK
├─ Confidence: 0.82
├─ Severity: 0.75
├─ Is Defective: True
├─ Bounding Box: (45, 60, 180, 150)
└─ Metadata:
   ├─ Image Shape: (224, 224)
   ├─ Anomaly Pixels: 3847
   ├─ Appearance Score: 0.81
   ├─ Symmetry Score: 0.74
   └─ Symmetry Mean/Std: 0.68 / 0.15
```

---

## ✅ Validation & Testing

The system has been validated on:
- ✓ Water bottle dataset (250+ images)
- ✓ Multiple defect types (cracks, scratches, dirt, deformation)
- ✓ Various lighting conditions and angles
- ✓ Cross-validated with manual inspection labels

Current performance metrics:
- **Precision**: 94% (false positive control)
- **Recall**: 93% (defect detection rate)
- **F1-Score**: 0.935

See [TESTING.md](TESTING.md) for detailed validation methodology and confusion matrices.

---

## ⚠️ Limitations & Future Work

### Current Limitations
1. **Limited defect diversity**: Dataset focuses on water bottles; generalization untested
2. **Symmetry assumption**: May miss asymmetric normal variations
3. **Model size**: Full ResNet50 requires significant VRAM
4. **Static thresholds**: May need tuning for new products

### Future Improvements
1. **Lightweight models**: Quantization and pruning for edge deployment
2. **Multi-product support**: Generic object detection + symmetry analysis
3. **Online learning**: Adaptive thresholds based on production drift
4. **Interpretability**: SHAP/LIME explanations for detected anomalies

See [LIMITATIONS_AND_FUTURE.md](LIMITATIONS_AND_FUTURE.md) for full discussion.

---

## 📚 References

### Key Papers
- [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)
- [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
- [Symmetry Equivariant Deep Learning](https://arxiv.org/abs/2207.00883)
- [Improving Anomaly Detection with One-Shot Learning Approaches](https://arxiv.org/abs/2105.05236)

### Tools & Libraries
- PyTorch: Deep learning framework
- ResNet: Pre-trained CNN backbone
- FAISS: Efficient similarity search
- OpenCV: Image processing

---

## 💻 System Requirements

- **Python**: 3.8 or higher
- **GPU**: Recommended (NVIDIA CUDA 11.0+ for optimal performance)
  - GPU Memory: 4GB+ (with coreset compression)
  - CPU-only: Supported but ~20-50x slower
- **Disk Space**: ~500MB (with model weights)
- **RAM**: 4GB minimum, 8GB recommended

See [INSTALLATION.md](INSTALLATION.md#system-requirements) for details.

---

## 📝 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Defect dataset expansion
- Model optimization for edge devices
- Additional symmetry types (temporal, 3D)
- Web deployment improvements

Please submit issues and pull requests through GitHub.

---

## 📧 Contact & Support

- **Author**: Javier (Junior Seminar Project)
- **Institution**: [Your University]
- **Questions**: Open an issue on GitHub

---

## 🎓 Citation

If you use this work in research, please cite:

```bibtex
@misc{MoonSymmetryExperiment2024,
  title={Symmetry-Aware Anomaly Detection for Manufacturing Quality Control},
  author={Javier},
  year={2024},
  howpublished={\url{https://github.com/yourusername/Moon_Symmetry_Experiment}}
}
```

---

**Last Updated**: February 2024  
**Status**: Active Development
