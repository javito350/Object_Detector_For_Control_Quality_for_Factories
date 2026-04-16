# Object Detector for Control Quality in Factories

Research prototype for CMPSC 580 Junior Seminar.

This repository implements a one-shot and few-shot factory quality-control pipeline focused on defect inspection with limited normal samples. The current production prototype uses a symmetry-aware anomaly detector with FAISS retrieval and EVT thresholding. YOLO can be integrated as a front-end detector for object localization, while this system handles defect scoring and pass/fail decisions.

## Why This Prototype Matters

Factory lines often have many normal parts and very few labeled defects. This project addresses that gap by:
- Learning only from normal support images (few-shot setup).
- Retrieving nearest normal patch features from a compressed FAISS memory bank.
- Producing image-level and pixel-level anomaly scores for QC decisions.
- Running in practical latency ranges for edge-style deployment experiments.

## Repository Map

- src/: Core Python pipeline, experiment scripts, evaluation utilities.
- src/models/: Feature extraction, memory bank, threshold calibration, inference logic.
- data/: Datasets (MVTec, water_bottles, VisA format).
- weights/: Serialized model artifacts.
- results/: CSV outputs and benchmark results.
- Readme/: Extended guides (installation, usage, testing, limitations).
- docs/: Technical notes created for reproducibility and grading.

## Prerequisites

- Python 3.10+ (tested with 3.13 in this repo).
- Git.
- Recommended: NVIDIA GPU + CUDA for faster inference.
- Works on CPU, but slower.

Required Python packages are listed in requirements.txt.

## Step-by-Step Installation

1. Clone and enter the repository.

```bash
git clone <your-repo-url>
cd Quality_control_for_Factories
```

2. Create a virtual environment.

```bash
python -m venv .venv
```

3. Activate environment.

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Windows Git Bash:
```bash
source .venv/Scripts/activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

5. Verify installation.

```bash
python -c "import torch, faiss, cv2, numpy; print('OK')"
```

6. Ensure model/data assets exist.

- Model weights in weights/ (for example, calibrated_inspector.pth).
- Dataset available under data/ (for example, data/mvtec or data/water_bottles).

## Usage Guide

### A. Demo Inference on One Image

```bash
python src/run_demo.py data/water_bottles/test/<image>.jpg --verbose
```

Expected behavior:
- Prints anomaly score and pass/fail status.
- Saves visualization into presentation_results/.

### B. Batch Inference on a Folder

```bash
python src/run_demo.py data/water_bottles/test/
```

### C. Reproduce 8-bit Seeded Evaluation (5 seeds)

```bash
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 111 --output-csv results_8bit_seed111.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 333 --output-csv results_8bit_seed333.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 999 --output-csv results_8bit_seed999.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 2026 --output-csv results_8bit_seed2026.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 3407 --output-csv results_8bit_seed3407.csv
```

### D. Aggregate Seeded Results into Markdown Table

```bash
python src/summarize_seeded_bits8_markdown.py results_8bit_seed111.csv results_8bit_seed333.csv results_8bit_seed999.csv results_8bit_seed2026.csv results_8bit_seed3407.csv
```

Expected output columns:
- Category
- Image AUROC (Mean +- SD)
- Pixel AUROC (Mean +- SD)

## Input and Output Formats

Inputs:
- RGB images (.jpg, .jpeg, .png, .bmp, .webp) for src/run_demo.py.
- CSV files containing category, image_auroc, pixel_auroc for src/summarize_seeded_bits8_markdown.py.

Outputs:
- Visualization images in presentation_results/.
- Benchmark CSV files in repository root or results/.
- Console metrics such as AUROC and retrieval latency.

## Troubleshooting

1. ModuleNotFoundError (torch/faiss/opencv)
- Activate virtual environment.
- Re-run pip install -r requirements.txt.

2. CUDA not available or GPU errors
- The code falls back to CPU in multiple paths.
- Verify GPU with python -c "import torch; print(torch.cuda.is_available())".

3. Missing model file
- Ensure weights/calibrated_inspector.pth exists, or pass --model_path explicitly.

4. Missing dataset path
- Ensure data is placed under data/ with expected subfolders.

5. Placeholder path errors in CSV summarizer
- Use real file paths, not path/to/seed1.csv.

## Research Journal and Report Links

- Project Resume / Journal Notes: Readme/PROJECT_RESUME.md
- Extended Project Documentation: Readme/README.md
- Full Presentation Script (report narrative): presentation/presentation_script.md
- Technical alignment and metrics note: docs/PROTOTYPE_TECHNICAL_GUIDE.md
- Commit evidence snapshot: docs/COMMIT_EVIDENCE.md

## Additional Documentation

- Readme/INSTALLATION.md
- Readme/USAGE.md
- Readme/TESTING.md
- Readme/LIMITATIONS_AND_FUTURE.md
- docs/PROTOTYPE_TECHNICAL_GUIDE.md
- docs/EXAMPLE_RUNS.md
