# G-CNN Defect Detection System - Project Summary

## System Overview

This project implements a Group Convolutional Neural Network (G-CNN) for automated defect detection in manufactured products using one-class learning and symmetry analysis.

## Current Status

### Model Performance
- **Test Set**: 5 water bottle images
- **Ground Truth**: test2.jpeg and test4.jpeg are GOOD; test.jpeg, test3.jpeg, test5.jpeg are DEFECTIVE
- **Current Accuracy**: 60% (3/5 correct after recalibration)
- **Issue**: Defective samples have very subtle defects with scores close to normal samples

### Model Details
- **Architecture**: Wide ResNet-50 with symmetry-aware features
- **Threshold**: 1.0112 (recalibrated from 1.3316)
- **Training**: Single normal water bottle image

## Usage

### Visual Demo (Shows Images)
```bash
python visual_demo.py data/water_bottles/test/test2.jpeg
```

### Model Evaluation
```bash
python evaluate_model.py
```

### Model Recalibration
```bash
python recalibrate_model.py
```

## Files Structure

- `calibrated_inspector.pth` - Recalibrated model
- `visual_demo.py` - Interactive visualization
- `evaluate_model.py` - Performance metrics
- `recalibrate_model.py` - Threshold optimization
- `data/water_bottles/test/labels.csv` - Ground truth

## Ground Truth

- test.jpeg → DEFECTIVE
- test2.jpeg → GOOD
- test3.jpeg → DEFECTIVE
- test4.jpeg → GOOD
- test5.jpeg → DEFECTIVE

## Technical Notes

The system uses PatchCore memory bank approach with symmetry analysis. Current limitation: test defects are very subtle (all scores between 1.008-1.020), making reliable detection challenging. More diverse training data and clearer defect examples recommended for better performance.
