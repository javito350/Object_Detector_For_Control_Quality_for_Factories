# Edge AI Anomaly Detection: Resolving the Trilemma

A fast, memory-efficient, few-shot ($N \le 10$) industrial anomaly detection system optimized for CPU-bound edge gateways. This repository contains the official prototype for the CMPSC 580 Junior Seminar research project: *"One Shot Detection for Control of Quality in Factories."*

## 📌 Overview
Modern industrial visual inspection faces an **Edge AI Trilemma**: balancing geometric robustness, memory footprint, and retrieval latency. This pipeline resolves the trilemma by replacing standard exact-search memory banks with a symmetry-aware, quantized retrieval system.

**Key Architectural Features:**
* **$p4m$ Symmetry Augmentation:** Offline manifold expansion resolving right-angle rotational blindness for directional geometries.
* **8-bit FAISS IVF-PQ Memory Bank:** Compresses the feature index by >10x (450MB → 38MB) while preserving 84.3% of exact-search fidelity.
* **EVT Thresholding:** Calibration-free, automated pass/fail boundaries modeled on Extreme Value Theory (GPD tail fitting).
* **CPU-Optimized:** Achieves a **12.73ms retrieval-stage latency** on standard consumer CPUs.

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.10+ and Git. (Tested on Python 3.13).

1. **Clone the repository:**
	```bash
	git clone [https://github.com/javito350/Object_Detector_For_Control_Quality_for_Factories.git](https://github.com/javito350/Object_Detector_For_Control_Quality_for_Factories.git)
	cd Object_Detector_For_Control_Quality_for_Factories
Create and activate a virtual environment:

Bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Git Bash:
source .venv/Scripts/activate
# macOS/Linux:
source .venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Dataset & Weights:

Model weights should be placed in weights/ (e.g., calibrated_inspector.pth).

Note on Data: The full raw image datasets (MVTec AD, VisA) and generated experimental CSVs for this project are hosted in our dedicated [Insert Your Dedicated Data Repository URL Here]. Extract datasets into the data/ folder in the project root.

🚀 Usage Guide
A. Real-Time Demo Inference
Run the inspection pipeline on a single image. The system will print the anomaly score, pass/fail status, and save a heatmap visualization into presentation_results/.

Bash
python src/run_demo.py data/water_bottles/test/<image>.jpg --verbose
To run batch inference on an entire folder:

Bash
python src/run_demo.py data/water_bottles/test/
B. Reproduce the WACV 8-bit Deployment Evaluation
To replicate the 5-seed support-set variance autopsy reported in the paper:

Bash
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 111 --output-csv final_csv_exports/results_8bit_seed111.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 333 --output-csv final_csv_exports/results_8bit_seed333.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 999 --output-csv final_csv_exports/results_8bit_seed999.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 2026 --output-csv final_csv_exports/results_8bit_seed2026.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 3407 --output-csv final_csv_exports/results_8bit_seed3407.csv
C. Aggregate Seeded Results
Aggregate the multi-seed evaluation into the final Markdown table (reports Mean ± SD for Image/Pixel AUROC):

Bash
python final_csv_exports/summarize_seeded_bits8_markdown.py final_csv_exports/results_8bit_seed111.csv final_csv_exports/results_8bit_seed333.csv final_csv_exports/results_8bit_seed999.csv final_csv_exports/results_8bit_seed2026.csv final_csv_exports/results_8bit_seed3407.csv
📊 Core Empirical Results (N=10)
When evaluated on the MVTec AD benchmark across 5 random support-set seeds under the 8-bit FAISS configuration:

Mean Image AUROC: 0.7792 ± 0.0332

Mean Pixel AUROC: 0.9013 ± 0.0064

Memory Footprint: 38 MB

Retrieval Latency: 12.73 ms

🛠️ Troubleshooting
ModuleNotFoundError (torch/faiss/opencv): Ensure your virtual environment is activated before running pip install -r requirements.txt.

CUDA not available: The code safely falls back to CPU logic for edge simulation. You can verify GPU status via python -c "import torch; print(torch.cuda.is_available())".

Missing model/dataset: Ensure weights/calibrated_inspector.pth exists and that datasets are placed under data/ with the standard category/train/good/ structure.

📚 Project Documentation
Research Report & Journal: [Link to your Report Repo]

Extended Setup/Testing: See Readme/INSTALLATION.md and Readme/TESTING.md.

Technical Metrics Guide: See docs/PROTOTYPE_TECHNICAL_GUIDE.md.

🧑‍💻 Author
Javier Bejarano Jiménez | Allegheny College | Computer Science and Mathematics


***

### 🏁 Final Instructions:
1. Paste this into your repository.
2. Update the **two bracketed links** (the URL for your new Data Repository under Step 4, and the URL for your Research Report Repo under Project Documentation).
3. Commit and push.

Once this is pushed, you have officially satisfied every single rubric item, coding requirement, and writing standard for CMPSC 580. Congratulations on an incredibly successful semester!
