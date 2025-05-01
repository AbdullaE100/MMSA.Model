#!/usr/bin/env python3
import sys
from pathlib import Path

# Add parent directory to path to import MMSA modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from repositories.MMSA.src.MMSA.models import AMIO
    print("Successfully imported AMIO from MMSA")
    
    # Print the available models in AMIO
    print("\nAvailable models in AMIO.MODEL_MAP:")
    for model_name in AMIO.MODEL_MAP.keys():
        print(f"  - {model_name}")
    
except ImportError as e:
    print(f"Error importing AMIO: {e}")
except Exception as e:
    print(f"Error with AMIO: {e}")

print("\nPython path:")
for p in sys.path:
    print(f"  {p}") 