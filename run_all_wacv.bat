@echo off
echo STARTING WACV BATCH RUN...

echo [1/3] Running 15-Category PatchCore Latency Sweep...
.\.venv\Scripts\python.exe run_missing_patchcore.py

echo [2/3] Running VisA PCA Feature Extraction...
.\.venv\Scripts\python.exe run_wacv_pca.py

echo [3/3] Running EVT-Quantization Stability Test...
.\.venv\Scripts\python.exe run_wacv_evt_stability.py

echo BATCH COMPLETE. SHUTTING DOWN.
pause
