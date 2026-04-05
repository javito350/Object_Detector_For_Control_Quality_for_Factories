#!/bin/bash

# 1. Extract and cache features (The "Feature Factory")
python run_demo.py --mode extract_only --dataset mvtec --output ./data/features

# 2. Run the Bitrate Sweeps
python run_grid_sweep.py --input ./data/features --output ./presentation_results/bitrate_cliff.csv

# 3. Run the Systems Stress Test
python run_concurrency_test.py --streams 1,8,16,32,64 --output ./presentation_results/concurrency_scaling.csv