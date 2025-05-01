#!/bin/bash

# Script to batch score videos using the Self-MM model

# Default values
MODEL_PATH=""
FEATURE_ROOT="./mmsa_fet_outputs"
VIDEO_DIR="../test_videos"
OUTPUT_DIR="./outputs"
SKIP_WEIGHTS="false"
LIVE="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --feature_root)
      FEATURE_ROOT="$2"
      shift 2
      ;;
    --video_dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip_weights)
      SKIP_WEIGHTS="true"
      shift
      ;;
    --no-live)
      LIVE="false"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model_path <path> [--feature_root <path>] [--video_dir <path>] [--output_dir <path>] [--skip_weights] [--no-live]"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] && [ "$SKIP_WEIGHTS" != "true" ]; then
  echo "Error: --model_path is required unless --skip_weights is specified"
  echo "Usage: $0 --model_path <path> [--feature_root <path>] [--video_dir <path>] [--output_dir <path>] [--skip_weights] [--no-live]"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get current timestamp for the output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$OUTPUT_DIR/fusion_results_$TIMESTAMP.md"

# Create the header for the markdown table
echo "# Multimodal Sentiment Analysis Results" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_FILE"
echo "Model: $(basename "$MODEL_PATH")" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| Video | Text Score | Audio Score | Video Score | Fusion Score | Sentiment |" >> "$RESULTS_FILE"
echo "|-------|------------|-------------|------------|--------------|-----------|" >> "$RESULTS_FILE"

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
  echo "Error: Video directory '$VIDEO_DIR' does not exist"
  exit 1
fi

echo "Starting batch scoring of videos in $VIDEO_DIR"
echo "Results will be saved to $RESULTS_FILE"

# Process each video file
for VIDEO_FILE in "$VIDEO_DIR"/*.mp4; do
  if [ ! -f "$VIDEO_FILE" ]; then
    echo "No mp4 files found in $VIDEO_DIR"
    exit 1
  fi
  
  VIDEO_NAME=$(basename "$VIDEO_FILE" .mp4)
  echo "Processing $VIDEO_NAME..."
  
  # Skip weights parameter
  SKIP_ARGS=""
  if [ "$SKIP_WEIGHTS" = "true" ]; then
    SKIP_ARGS="--skip_weights"
  fi
  
  # Live parameter
  LIVE_ARGS=""
  if [ "$LIVE" = "true" ]; then
    LIVE_ARGS="--live"
  fi
  
  # Run test_self_mm.py for this video
  OUTPUT=$(python3 test_self_mm.py \
    --video_path "$VIDEO_FILE" \
    --fet_dir "$FEATURE_ROOT" \
    --model_path "$MODEL_PATH" \
    --config_path "./pretrained_models/self_mm-mosi-config_fixed.json" \
    $SKIP_ARGS $LIVE_ARGS)
  
  # Extract scores from the output
  TEXT_SCORE=$(echo "$OUTPUT" | grep "Text-only score:" | sed -n 's/Text-only score: \([-0-9.]*\).*/\1/p')
  AUDIO_SCORE=$(echo "$OUTPUT" | grep "Audio-only score:" | sed -n 's/Audio-only score: \([-0-9.]*\).*/\1/p')
  VIDEO_SCORE=$(echo "$OUTPUT" | grep "Video-only score:" | sed -n 's/Video-only score: \([-0-9.]*\).*/\1/p')
  FUSION_SCORE=$(echo "$OUTPUT" | grep "Multimodal score:" | sed -n 's/Multimodal score: \([-0-9.]*\).*/\1/p')
  SENTIMENT=$(echo "$OUTPUT" | grep "Multimodal score:" | sed -n 's/.*(\(.*\)).*/\1/p')
  
  # Add the results to the markdown table
  echo "| $VIDEO_NAME | $TEXT_SCORE | $AUDIO_SCORE | $VIDEO_SCORE | $FUSION_SCORE | $SENTIMENT |" >> "$RESULTS_FILE"
done

echo "Batch scoring complete! Results saved to $RESULTS_FILE"
echo "Total videos processed: $(ls -1 "$VIDEO_DIR"/*.mp4 | wc -l | xargs)" 