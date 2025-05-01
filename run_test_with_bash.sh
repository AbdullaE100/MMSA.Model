#!/bin/bash

# Set paths - modify these as needed
VIDEO_PATH="./test_videos"
FET_DIR="./mmsa_fet_outputs"
MODEL_PATH="./pretrained_models/self_mm-mosi_fixed.pth"
CONFIG_PATH="./pretrained_models/self_mm-mosi-config_fixed.json"

# Create feature directory if it doesn't exist
mkdir -p $FET_DIR

echo "Step 1: Creating dummy feature files for testing..."
# Create dummy feature files for testing
for video in "$VIDEO_PATH"/*.mp4; do
  base_name=$(basename "$video" .mp4)
  
  # Create dummy numpy arrays and save them
  python3 -c "
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('$FET_DIR', exist_ok=True)

# Create dummy features with appropriate dimensions
text_features = np.random.rand(50, 768)
audio_features = np.random.rand(50, 74)
vision_features = np.random.rand(50, 35)

# Save to files
np.save('$FET_DIR/${base_name}_text.npy', text_features)
np.save('$FET_DIR/${base_name}_audio.npy', audio_features)
np.save('$FET_DIR/${base_name}_vision.npy', vision_features)

print(f'Created feature files for ${base_name}')
"
done

echo "Step 2: Running MMSA sentiment analysis..."
# Run the test_repo.py script with bash
python3 repositories/MMSA/test_repo.py \
  --folder_path $VIDEO_PATH \
  --fet_dir $FET_DIR \
  --model_path $MODEL_PATH \
  --config_path $CONFIG_PATH \
  --output_format json

echo "Test completed!" 