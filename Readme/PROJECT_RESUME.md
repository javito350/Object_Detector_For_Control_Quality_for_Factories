# Moon Symmetry Experiment - Project Resume

## Executive Summary
A GPU-accelerated **anomaly detection system** for manufacturing quality control with focus on **water bottle inspection**. Uses symmetry-aware deep learning features combined with memory bank approaches to detect defects without requiring large labeled datasets.

---

## Problem Statement
Manufacturing quality control is a bottleneck for small-to-medium factories:
1. **Manual Inspection**: Requires exhausting 8+ hour shifts, prone to human error
2. **Cost Barrier**: Commercial vision systems cost $18,000+ (prohibitive for small factories)
3. **Data Complexity**: Standard AI requires thousands of labeled defect images and weeks of training
4. **Time-to-Market**: Small factories need immediate solutions, not long training pipelines

---

## Solution Overview
**G-CNN Defect Detection System** leveraging:
- **Symmetry-aware feature extraction** (horizontal, vertical, and rotational symmetry)
- **Memory bank architecture** for anomaly detection without large defect training sets
- **FastAPI-based web interface** for production deployment
- **Real-time heatmap visualization** showing exactly where defects are detected

---

## Technical Architecture

### Core Components

#### 1. **Feature Extraction** (`models/symmetry_feature_extractor.py`)
- **Backbone**: Wide ResNet50 or ResNet18 (pretrained on ImageNet)
- **Multi-scale extraction**: 3 intermediate layers (layer1, layer2, layer3)
- **Symmetry Types**:
  - Horizontal flip symmetry
  - Vertical flip symmetry
  - Rotational (90°) symmetry
  - Combined "both" mode (horizontal + vertical)
- **Feature Dimensions**:
  - Layer1: 256 channels
  - Layer2: 512 channels
  - Layer3: 1024 channels
  - Total per-view: 1792 dims
  - Multi-view: 1792 × num_views + 1 (symmetry score)

#### 2. **Anomaly Detection** (`models/anomaly_inspector.py`)
- **InspectionResult dataclass**: Encapsulates all inspection outputs
  - `image_score`: Overall anomaly likelihood (0-1)
  - `anomaly_map`: Heatmap showing anomaly regions
  - `symmetry_map`: Separate map for symmetry violations
  - `binary_mask`: Segmented defect regions
  - `defect_type`: Classification (crack, scratch, dirt, deformation, discoloration, symmetry_break)
  - `confidence`: Defect type confidence
  - `bbox`: Bounding box of primary defect
  - `severity`: Defect severity score
  - `metadata`: Debug/measurement info

- **EnhancedAnomalyInspector Class**:
  - Combines SymmetryAwareFeatureExtractor + MemoryBank
  - **Adaptive Thresholds**:
    - `pixel_threshold`: 0.5 (per-pixel anomaly threshold)
    - `image_threshold`: 0.5 (whole-image defect decision)
    - `symmetry_threshold`: 0.3 (symmetry consistency)
  - **Post-processing**:
    - Gaussian smoothing (sigma=4)
    - Minimum anomaly size filtering (50 pixels)
  - **Training**: Only uses normal (defect-free) samples
  - **Inference**: Compares test images to memory bank of normal features

#### 3. **Memory Bank** (`models/memory_bank.py`)
- Stores coreset of representative normal samples
- GPU-accelerated similarity search using FAISS
- **Coreset percentage**: 0.1 (10% of normal samples retained)
- Enables anomaly detection without defect labels

### Supporting Utilities
- **image_loader.py**: Image I/O and preprocessing
- **metricis.py**: Evaluation metrics (precision, recall, F1, AUC-ROC)
- **visualization.py**: Heatmap rendering and result visualization

---

## Data Structure

### Data Organization
```
data/
├── phone_cases/
│   └── [case images for alternative product line]
└── water_bottles/
    ├── test/
    │   └── labels.csv          # Test set annotations
    └── train/
        └── good/               # Normal (non-defective) samples only
```

### Dataset Components
- **Good/Normal Sample Images**: Used for memory bank training
- **Test Set**: Contains both good and defective samples with labels
- **Defect Types Annotated**: Crack, scratch, dirt, deformation, discoloration, symmetry_break

---

## Key Features & Workflow

### Training (`fit` method)
1. Extract symmetry-aware features from all normal training images
2. Compute symmetry consistency scores (agreement between original/flipped views)
3. Build memory bank from coreset (top 10% representative features)
4. Extract statistical properties (mean, std) of normal feature distributions

### Inference (`inspect` method)
1. Extract test image features and symmetry scores
2. Compute distance to nearest memory bank neighbors
3. Generate anomaly heatmap via weighted distance visualization
4. Apply Gaussian smoothing and morphological cleanup
5. Extract connected components and compute bounding boxes
6. Classify defect type based on heatmap characteristics
7. Return `InspectionResult` with all visualization data

### Defect Type Classification
- **Symmetry-based detection**: Identifies symmetry_break type from symmetry_map
- **Morphological analysis**: Distinguishes cracks (linear), scratches, deformation via shape
- **Intensity analysis**: Identifies dirt (localized intensity) and discoloration (color shifts)
- **Ensemble confidence**: Unknown type falls back if no clear pattern matches

---

## Training Capabilities

### Calibration Scripts (`calibration/`)
- **one_image_set_up.py**: Single-image baseline setup and testing
- **recalibrate_model.py**: Fine-tune thresholds on new product lines
- **recalibrate_sensitive.py**: Increase detection sensitivity for critical defects

### Sample Creation
- **create_defective_samples.py**: Synthetically generate defect examples
- **embed_images.py**: Precompute embeddings for faster inference
- **copy_images.py**: Organize and preprocess image datasets

---

## Demonstration & Deployment

### Demo System (`run_demo.py`)
- **PresentationDemo class**: End-to-end inspection pipeline
- Loads pre-trained weights from `weights/`
- Processes arbitrary test images
- Generates publication-quality visualizations
- Produces detailed inspection reports

### Pre-trained Models (`weights/`)
- `calibrated_inspector.pth`: Standard defect detection
- `sensitive_inspector.pth`: High-sensitivity variant

### Web Interface (`presentation/`)
- **index.html + script.js + style.css**: Interactive Flask/web UI
- **presentation_script.md**: 3-minute pitch explaining problem → solution

---

## Technology Stack

### Deep Learning
- **PyTorch** 1.9.0+: Neural networks and GPU computation
- **Torchvision** 0.10.0+: Pretrained vision models (ResNet, Wide ResNet)

### Numerical Computing
- **NumPy** 1.19.5+: Array operations
- **SciPy** 1.7.1+: Gaussian/median filtering, morphology
- **scikit-image** 0.18.3+: Image morphology and component labeling
- **scikit-learn** 0.24.2+: Machine learning utilities (distance metrics)
- **FAISS** 1.7.2+: CPU-based similarity search for memory bank

### Image Processing
- **OpenCV** 4.5.3+: Image resizing, contour detection, drawing
- **Pillow** (PIL): Image loading and format conversion

### Visualization & I/O
- **Matplotlib** 3.4.2+: Chart generation and heatmap rendering
- **h5py** 3.3.0+: HDF5 file format for model checkpoints
- **tqdm** 4.62.0+: Progress bars for training/inference

---

## Key Innovation Points

1. **Symmetry Awareness**: Leverages manufacturing product symmetries to catch asymmetric defects that standard models miss

2. **No Defect Labels Required**: Memory bank approach trains only on normal samples—no need for thousands of defect images

3. **Explainability**: Heatmaps and defect type predictions provide transparent decision-making for quality assurance teams

4. **Multi-View Consistency**: Symmetry scores provide additional confidence signals beyond standard feature similarity

5. **Production-Ready**: Pre-trained weights, adaptive thresholds, and post-processing ensure robust real-world performance

---

## Presentation Context
The project is framed as the solution to a critical manufacturing pain point:
- **Problem**: Small factories can't afford $18k+ vision systems or 3-6 months of AI development
- **Solution**: Symmetry-aware, few-shot defect detection that works out-of-the-box
- **Impact**: Automate quality control within weeks, not months; cost < $5k total

---

## File Manifest
- `run_demo.py`: Main entry point for demonstration
- `requirements.txt`: Pinned dependency versions
- `models/`: Core neural network components
- `calibration/`: Model tuning and sensitivity adjustment
- `data/`: Image datasets (water bottles, phone cases)
- `weights/`: Pre-trained model checkpoints
- `utils/`: Visualization, metrics, image loading
- `presentation/`: Web UI and pitch materials
- `presentation_results/`: Inspection summaries and logs

---

## Next Steps / Future Work
- Real-time video stream processing
- Multi-product line support (transfer learning)
- Edge deployment (ONNX export, mobile optimization)
- Integration with manufacturing MES systems
- Defect root-cause analysis and feedback loops
