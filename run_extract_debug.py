import traceback
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('extract_example', Path('src/extract_real_evt_calibration_example.py'))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    try:
        mod.main()
    except Exception as e:
        print('Exception during mod.main():', type(e), e)
        traceback.print_exc()
except Exception as e:
    print('Exception during import:', type(e), e)
    traceback.print_exc()
