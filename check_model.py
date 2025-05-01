#!/usr/bin/env python3
"""
Simple utility to check if a PyTorch model file is valid.
This helps avoid running tests with corrupted model files.
"""

import sys
import os
import torch

def check_model(model_path):
    """Check if a PyTorch model file can be loaded"""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False
        
    try:
        # Check file type
        with open(model_path, 'rb') as f:
            magic = f.read(10)
            if b'PK\x03\x04' not in magic and b'\x80\x02' not in magic:
                print(f"ERROR: {model_path} doesn't appear to be a valid PyTorch model file")
                return False
                
        # Try to load the model
        device = torch.device("cpu")
        state_dict = torch.load(model_path, map_location=device)
        print(f"SUCCESS: Model verified at {model_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    if check_model(model_path):
        sys.exit(0)
    else:
        sys.exit(1) 