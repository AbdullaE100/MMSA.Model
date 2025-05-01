#!/usr/bin/env python3
import sys
from pathlib import Path

# Add parent directory to path to import testing_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import testing_utils
    print("Successfully imported testing_utils")
except ImportError as e:
    print(f"Error importing testing_utils: {e}")

try:
    from repositories.MMSA.src.MMSA.models import AMIO
    print("Successfully imported AMIO from MMSA")
except ImportError as e:
    print(f"Error importing AMIO: {e}")

print("Python path:")
for p in sys.path:
    print(f"  {p}") 