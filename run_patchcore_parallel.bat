@echo off
start /B .venv\Scripts\python.exe src/wacv_fix1_multiseed_patchcore.py --seeds 111 --output-csv results/wacv/patchcore_111.csv --accelerator cpu > patchcore_111.log 2>&1
start /B .venv\Scripts\python.exe src/wacv_fix1_multiseed_patchcore.py --seeds 333 --output-csv results/wacv/patchcore_333.csv --accelerator cpu > patchcore_333.log 2>&1
start /B .venv\Scripts\python.exe src/wacv_fix1_multiseed_patchcore.py --seeds 999 --output-csv results/wacv/patchcore_999.csv --accelerator cpu > patchcore_999.log 2>&1
start /B .venv\Scripts\python.exe src/wacv_fix1_multiseed_patchcore.py --seeds 2026 --output-csv results/wacv/patchcore_2026.csv --accelerator cpu > patchcore_2026.log 2>&1
start /B .venv\Scripts\python.exe src/wacv_fix1_multiseed_patchcore.py --seeds 3407 --output-csv results/wacv/patchcore_3407.csv --accelerator cpu > patchcore_3407.log 2>&1
