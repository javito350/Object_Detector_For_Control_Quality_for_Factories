# Installation Guide

Complete installation instructions for the Moon Symmetry Experiment anomaly detection system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [GPU Setup](#gpu-setup)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **CPU**: Dual-core 2.0+ GHz processor
- **RAM**: 4GB
- **Disk Space**: 500MB (includes model weights)

### Recommended Setup
- **OS**: Windows 11 or Ubuntu 20.04 LTS
- **Python**: 3.10 or 3.11
- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA compute capability 3.5+)
- **RAM**: 8GB or higher
- **SSD**: For faster data loading

### GPU Support
- **NVIDIA GPUs**: Fully supported (CUDA + cuDNN)
- **AMD GPUs**: Supported via ROCm (experimental)
- **Apple Silicon**: Supported via MPS backend
- **CPU-Only**: Supported but ~20-50x slower

---

## Prerequisites

### 1. Install Python

**Windows**:
```bash
# Download from https://www.python.org/downloads/
# Or use chocolatey:
choco install python
```

**macOS**:
```bash
# Using Homebrew
brew install python@3.11

# Or MacPorts
sudo port install python311 +universal
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

Verify installation:
```bash
python --version  # Should output Python 3.8+
pip --version     # Should output pip version
```

### 2. Install Git

Required for cloning the repository.

**Windows**:
```bash
# Using Chocolatey
choco install git

# Or download from https://git-scm.com/download/win
```

**macOS**:
```bash
brew install git
```

**Linux**:
```bash
sudo apt install git
```

### 3. NVIDIA GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU and want GPU acceleration:

#### Step 1: Verify GPU Compatibility
```bash
# Windows
nvidia-smi

# Linux
nvidia-smi
```

Should display your GPU model and current CUDA version.

#### Step 2: Install CUDA Toolkit

**Windows**: Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

**Linux (Ubuntu 20.04)**:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-1804
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

#### Step 3: Install cuDNN

1. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cuDNN) (requires account)
2. Extract and add to CUDA directory:

**Windows**:
```cmd
:: Extract cuDNN and copy to CUDA installation
copy cudnn\bin\cudnn64_8.dll "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\"
```

**Linux**:
```bash
tar -xzf cudnn-linux-x86_64-8.x.x.x_cuda11.x-archive.tar.xz
sudo cp -r cudnn-archive/include/* /usr/local/cuda/include/
sudo cp -r cudnn-archive/lib/* /usr/local/cuda/lib64/
```

#### Step 4: Verify GPU Setup
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # Shows your GPU
```

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Moon_Symmetry_Experiment.git
cd Moon_Symmetry_Experiment

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import torch; import torchvision; print('Installation successful!')"
```

### Method 2: Conda Installation

```bash
# 1. Download Anaconda from https://www.anaconda.com/download
# Or use Miniconda for lighter installation

# 2. Create conda environment
conda create -n moon_symmetry python=3.11

# 3. Activate environment
conda activate moon_symmetry

# 4. Install dependencies
conda install -c pytorch pytorch torchvision

# 5. Install remaining packages
pip install -r requirements.txt
```

### Method 3: Docker Installation

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_demo.py"]
```

Build and run:
```bash
docker build -t moon_symmetry .
docker run --gpus all -v /path/to/data:/app/data moon_symmetry
```

---

## GPU Setup

### Verify GPU Installation

```python
import torch

# Check if GPU is available
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Check CUDA version
print(f"CUDA Version: {torch.version.cuda}")

# Create a test tensor on GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)  # This runs on GPU
print(f"GPU Test Successful: {z is not None}")
```

### Force GPU Usage

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)
```

### Force CPU Usage

```python
import torch

device = torch.device("cpu")
model = model.to(device)
```

---

## Download Pre-trained Weights

The system requires pre-trained model weights. Download them:

```bash
# Option 1: From GitHub (if hosted)
wget https://github.com/yourusername/Moon_Symmetry_Experiment/releases/download/v1.0/weights.tar.gz
tar -xzf weights.tar.gz

# Option 2: Manual download
# Download calibrated_inspector.pth to weights/ directory
# Download sensitive_inspector.pth to weights/ directory (optional)
```

Verify weights are in the correct location:
```bash
ls -la weights/
# Should show:
# calibrated_inspector.pth
# sensitive_inspector.pth (optional)
```

---

## Verification

### 1. Test Basic Import

```python
# test_import.py
import torch
import torchvision
import cv2
import numpy as np
from models.anomaly_inspector import EnhancedAnomalyInspector

print("✓ All imports successful!")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
```

Run:
```bash
python test_import.py
```

### 2. Test Model Loading

```bash
python -c "from models.anomaly_inspector import EnhancedAnomalyInspector; print('Model import OK')"
```

### 3. Test on Sample Image

```python
# test_sample.py
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from models.anomaly_inspector import EnhancedAnomalyInspector

# Load model
inspector = EnhancedAnomalyInspector(device="cuda")

# Test image path
test_image_path = Path("data/water_bottles/test/sample_image.jpg")

if test_image_path.exists():
    # Load and preprocess image
    image = Image.open(test_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        results = inspector.predict(image_tensor)
    
    print(f"✓ Inference successful!")
    print(f"  Anomaly score: {results[0].image_score:.4f}")
```

Run:
```bash
python test_sample.py
```

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Solution**:
```bash
# Ensure virtual environment is activated
python -m pip install torch torchvision

# Or if using conda:
conda install -c pytorch pytorch torchvision
```

### Issue 2: "CUDA out of memory"

**Solutions**:
```bash
# Option A: Use CPU instead
export CUDA_VISIBLE_DEVICES=""  # Linux/macOS
set CUDA_VISIBLE_DEVICES=""     # Windows

# Option B: Reduce batch size in code
# inspector = EnhancedAnomalyInspector(device="cpu")

# Option C: Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Issue 3: "ImportError: DLL load failed" (Windows GPU)

**Solution**:
```bash
# Ensure CUDA Toolkit and cuDNN are properly installed
# Reinstall PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 4: "FileNotFoundError: weights not found"

**Solution**:
```bash
# Verify weights directory
ls -la weights/

# If missing, download them:
# See "Download Pre-trained Weights" section above
```

### Issue 5: "OpenCV version mismatch"

**Solution**:
```bash
# Reinstall with specific version
pip install opencv-python==4.8.0.76
```

### Issue 6: "numpy compatibility error"

**Solution**:
```bash
# Update numpy
pip install --upgrade numpy

# If still issues, specify compatible version:
pip install numpy==1.24.3
```

### Runtime Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your code
inspector = EnhancedAnomalyInspector(device="cuda")
```

Check system info:
```bash
python -c "import platform; import torch; print(platform.platform()); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Next Steps

After successful installation:

1. **Read the Usage Guide**: See [USAGE.md](USAGE.md) for detailed examples
2. **Run the Demo**: Execute `python run_demo.py --help`
3. **Explore Examples**: Check the `examples/` directory
4. **Review Tests**: Run test suite with `pytest tests/`

---

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review existing [GitHub Issues](https://github.com/yourusername/Moon_Symmetry_Experiment/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Error message and traceback
   - Output of `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

---

**Last Updated**: February 2024  
**Version**: 1.0
