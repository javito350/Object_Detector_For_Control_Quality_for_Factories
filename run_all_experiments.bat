@echo off
setlocal

echo [%date% %time%] Starting compute_intrinsic_dim.py
python compute_intrinsic_dim.py --apply-p4m-support
if errorlevel 1 (
    echo [%date% %time%] compute_intrinsic_dim.py failed
    exit /b 1
)
echo [%date% %time%] Finished compute_intrinsic_dim.py

echo [%date% %time%] Starting evaluate_pca_256.py
python evaluate_pca_256.py
if errorlevel 1 (
    echo [%date% %time%] evaluate_pca_256.py failed
    exit /b 1
)
echo [%date% %time%] Finished evaluate_pca_256.py

echo [%date% %time%] Starting export_onnx_benchmark.py
python export_onnx_benchmark.py
if errorlevel 1 (
    echo [%date% %time%] export_onnx_benchmark.py failed
    exit /b 1
)
echo [%date% %time%] Finished export_onnx_benchmark.py

echo [%date% %time%] All experiments completed successfully
endlocal
