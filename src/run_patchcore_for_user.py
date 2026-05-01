import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Remove thread limits for speed
if "LOKY_MAX_CPU_COUNT" in os.environ: del os.environ["LOKY_MAX_CPU_COUNT"]
if "OMP_NUM_THREADS" in os.environ: del os.environ["OMP_NUM_THREADS"]

# Import the original script's main logic but override constants
from src.wacv_fix1_multiseed_patchcore import main

if __name__ == "__main__":
    # We can override sys.argv to pass seeds
    sys.argv = [
        "src/wacv_fix1_multiseed_patchcore.py",
        "--seeds", "111", "333", "999", "2026", "3407",
        "--accelerator", "cpu"
    ]
    main()
