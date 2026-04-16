# Example Runs and Demonstrations

## Demo 1: Single-Image Inspection

Command:
```bash
python src/run_demo.py data/water_bottles/test/<image>.jpg --verbose
```

Expected console fields:
- Result: DEFECT DETECTED or NOMINAL (PASS)
- Anomaly Score
- Threshold
- Latency (ms)

Expected artifact:
- Visualization image saved to presentation_results/

## Demo 2: Batch Inspection

Command:
```bash
python src/run_demo.py data/water_bottles/test/
```

Expected behavior:
- Processes all supported images in folder.
- Prints per-image status and score.

## Demo 3: 8-bit Seeded Reproducibility Run

Commands:
```bash
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 111 --output-csv results_8bit_seed111.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 333 --output-csv results_8bit_seed333.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 999 --output-csv results_8bit_seed999.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 2026 --output-csv results_8bit_seed2026.csv
python src/conference_multiclass_eval.py --pq-bits 8 --support-seed 3407 --output-csv results_8bit_seed3407.csv
```

Expected output files:
- results_8bit_seed111.csv
- results_8bit_seed333.csv
- results_8bit_seed999.csv
- results_8bit_seed2026.csv
- results_8bit_seed3407.csv

## Demo 4: Aggregate Result Table for Report

Command:
```bash
python src/summarize_seeded_bits8_markdown.py results_8bit_seed111.csv results_8bit_seed333.csv results_8bit_seed999.csv results_8bit_seed2026.csv results_8bit_seed3407.csv
```

Expected Markdown table format:
```text
| Category | Image AUROC (Mean +- SD) | Pixel AUROC (Mean +- SD) |
|---|---:|---:|
| bottle | ... | ... |
| screw  | ... | ... |
```

## Test Data Options

- Local factory/water bottle examples under data/water_bottles/test
- MVTec benchmark data under data/mvtec
- Existing benchmark CSV outputs under results/
