#!/bin/bash

# Set paths - modify these as needed
VIDEO_PATH="./test_videos"
OUTPUT_DIR="./mmsa_fet_outputs"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Use the specific Python path
PYTHON_PATH="repositories/MMSA-FET/venv/bin/python3"

# Check if Python is available
if [ ! -x "$PYTHON_PATH" ]; then
    echo "Error: Python not found at $PYTHON_PATH"
    echo "Using system Python instead"
    PYTHON_PATH="python3"
fi

# Run the MMSA-FET feature extraction
$PYTHON_PATH repositories/MMSA-FET/test_repo.py \
  --folder_path $VIDEO_PATH \
  --output_format json

echo "Feature extraction completed!"

# Note: This script uses the default config in MMSA-FET/test_repo.py.
# For a real-world scenario, you might want to provide a custom config file
# with the --config_path argument.

# The features should be saved as:
# - {video_name}_text.npy
# - {video_name}_audio.npy
# - {video_name}_vision.npy
# in the output directory. 