# Prototype Technical Guide

## 1. Research Problem Alignment

Manufacturing quality-control systems are constrained by:
- Sparse defect labels.
- Changing defect morphology across products.
- Need for low-latency pass/fail decisions.

This prototype addresses the problem with a few-shot anomaly detection design:
- Learns from normal support images only.
- Detects out-of-distribution visual patterns as potential defects.
- Produces interpretable anomaly heatmaps for inspection workflows.

Note on YOLO alignment:
- The current branch uses anomaly retrieval and thresholding as the core detector.
- A YOLO front-end can be added to crop object-of-interest regions before anomaly scoring.
- This keeps the research question intact: scalable, low-label factory QC.

## 2. Model Architecture and Training Approach

Pipeline:
1. Symmetry-aware feature extraction (WRN50-family or configured backbone).
2. Orbit-aware reduction + coreset sampling of support features.
3. FAISS IVF-PQ memory bank indexing of compressed normal embeddings.
4. EVT calibration of image-level threshold tau.
5. Inference on test image patches + image/pixel anomaly maps.

Training approach:
- No supervised defect classification training in the main path.
- Few-shot support-set construction with deterministic seeding.
- Threshold calibration from support-distance distribution.

## 3. Data Requirements and I/O Contracts

Input image requirements:
- RGB image files.
- Supported extensions: .jpg, .jpeg, .png, .bmp, .webp.
- Images are transformed to normalized tensors for inference.

Dataset expectations:
- MVTec-style structure for benchmark scripts under data/mvtec.
- Optional water bottle demo data under data/water_bottles.

Key CSV schema requirements:
- category
- image_auroc
- pixel_auroc

Primary output artifacts:
- result images and heatmaps (presentation_results/)
- benchmark CSV tables (results/ or root)
- latency and AUROC metrics in stdout

## 4. Performance Metrics and Evaluation Results

Common metrics tracked:
- Image AUROC: defect/non-defect ranking quality per image.
- Pixel AUROC: localization quality over pixel-level maps.
- Retrieval latency (ms): nearest-neighbor speed from FAISS index.
- Quantization diagnostics: r_max and r_avg in selected runs.

Example seeded 8-bit summary (5 seeds):
- bottle: image AUROC 0.9829 +- 0.0292, pixel AUROC 0.9769 +- 0.0072
- screw: image AUROC 0.5761 +- 0.0401, pixel AUROC 0.8608 +- 0.0087

## 5. Error Handling and Validation Improvements Applied

Implemented improvements in this repository:
- Memory bank input validation for shape, NaN/Inf, and parameter ranges.
- Safer FAISS setup with GPU fallback to CPU when GPU init fails.
- Query-time validation for dimensions and k.
- Inference-time checks for tensor format and trained-state preconditions.
- Robust model loading and image-format validation in demo runner.
- Clearer user-facing error messages for invalid CSV usage.

## 6. Limitations

Current limitations:
- Limited benchmark categories in some scripts (for rapid experimentation).
- Dependency on organized dataset folder structures.
- No integrated YOLO object detector in the main inference script yet.
- Performance can vary by seed due to few-shot support sampling.

## 7. Future Improvements

Recommended next steps:
1. Integrate YOLO object localization before anomaly scoring.
2. Add unit tests for CLI argument validation and schema checks.
3. Add CI workflow to run smoke tests on each commit.
4. Expand cross-dataset benchmarking and class coverage.
5. Add model versioning and checksum validation for weight files.
