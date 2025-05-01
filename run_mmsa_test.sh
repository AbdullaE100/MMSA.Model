#!/bin/bash
# Simple script to run a test on the MMSA model with the modality shapes debug flag

# Set paths
VIDEO_PATH="${1:-./test_videos/Calm.mp4}"
MODEL_PATH="./pretrained_models/self_mm-mosi_fixed.pth"
CONFIG_PATH="./pretrained_models/self_mm-mosi-config_fixed.json"
FET_DIR="./mmsa_fet_outputs"

# Create output directory if it doesn't exist
mkdir -p "$FET_DIR"

# Print what we're running
echo "Running test with video: $VIDEO_PATH"
echo "Using model path: $MODEL_PATH"
echo "Using config path: $CONFIG_PATH"
echo "Feature directory: $FET_DIR"
echo "--------------------------------------"

# Run the test with the debug flag
python test_self_mm.py --video_path "$VIDEO_PATH" \
                      --model_path "$MODEL_PATH" \
                      --config_path "$CONFIG_PATH" \
                      --fet_dir "$FET_DIR" \
                      --print_modality_shapes \
                      --live

# Allow specifying a folder of videos
if [ -n "$2" ] && [ "$2" == "--folder" ]; then
  FOLDER_PATH="${3:-./test_videos}"
  echo "Processing all videos in: $FOLDER_PATH"
  python test_self_mm.py --folder_path "$FOLDER_PATH" \
                        --model_path "$MODEL_PATH" \
                        --config_path "$CONFIG_PATH" \
                        --fet_dir "$FET_DIR" \
                        --print_modality_shapes \
                        --live 