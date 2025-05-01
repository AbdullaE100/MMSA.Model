#!/bin/bash

# Main testing script for the Self-MM multimodal sentiment analysis

# Default values
TRAIN=false
VIDEO_DIR="../test_videos"
DATA_ROOT="./datasets/MOSI"
MODEL_PATH="./pretrained_models/self_mm-mosi_fixed.pth"
OUTPUT_DIR="./outputs"
FEATURE_DIR="./mmsa_fet_outputs"
LOG_DIR="./logs"
SINGLE_VIDEO=""
OUTPUT_FORMAT="json"
PRINT_MODALITY_SHAPES=false
USE_STRICT_LOADING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --train)
      TRAIN=true
      shift
      ;;
    --video_dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --feature_dir)
      FEATURE_DIR="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --videos=*)
      VIDEO_DIR="${1#*=}"
      shift
      ;;
    --video_path=*)
      SINGLE_VIDEO="${1#*=}"
      shift
      ;;
    --format=*)
      OUTPUT_FORMAT="${1#*=}"
      shift
      ;;
    --print_modality_shapes)
      PRINT_MODALITY_SHAPES=true
      shift
      ;;
    --strict_loading)
      USE_STRICT_LOADING=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--train] [--video_dir <path>] [--data_root <path>] [--model_path <path>] [--output_dir <path>] [--feature_dir <path>] [--log_dir <path>] [--videos=<directory>|--video_path=<file>] [--format=json|csv] [--print_modality_shapes] [--strict_loading]"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR" "$LOG_DIR"

# Get current timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_all_tests_$TIMESTAMP.log"

echo "Starting Self-MM multimodal sentiment analysis test suite" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Training mode: $TRAIN" | tee -a "$LOG_FILE"
echo "  Video directory: $VIDEO_DIR" | tee -a "$LOG_FILE"
echo "  Data root: $DATA_ROOT" | tee -a "$LOG_FILE"
echo "  Model path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Feature directory: $FEATURE_DIR" | tee -a "$LOG_FILE"
echo "  Log directory: $LOG_DIR" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
  echo "Error: Video directory '$VIDEO_DIR' does not exist" | tee -a "$LOG_FILE"
  exit 1
fi

# Set environment variable for dataset path
export MOSI_DATA_DIR="$DATA_ROOT"

# Build the modality shapes argument if needed
MODALITY_SHAPES_ARG=""
if [ "$PRINT_MODALITY_SHAPES" = true ]; then
  MODALITY_SHAPES_ARG="--print_modality_shapes"
  echo "Enabling modality shape printing"
fi

# Build the strict loading argument if needed
STRICT_LOADING_ARG=""
if [ "$USE_STRICT_LOADING" = true ]; then
  STRICT_LOADING_ARG="--strict_loading"
  echo "Enabling strict weight loading"
fi

# Step 1: Train the model or download pretrained weights
if [ "$TRAIN" = true ]; then
  echo "Training Self-MM model..." | tee -a "$LOG_FILE"
  bash scripts/train_self_mm.sh \
    --data_root "$DATA_ROOT" \
    --save_path "$MODEL_PATH" 2>&1 | tee -a "$LOG_FILE"
    
  # Check if training was successful
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Training failed, model file not created at $MODEL_PATH" | tee -a "$LOG_FILE"
    exit 1
  fi
else
  # Check if model exists, otherwise download it
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Pretrained model not found, downloading..." | tee -a "$LOG_FILE"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$MODEL_PATH")"
    
    # Download pretrained model
    # Replace this URL with the actual download URL for your pretrained model
    MODEL_URL="https://github.com/thuiar/Self-MM/raw/main/pretrained/self_mm-mosi.pth"
    curl -L "$MODEL_URL" -o "$MODEL_PATH" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ! -f "$MODEL_PATH" ]; then
      echo "Error: Failed to download pretrained model" | tee -a "$LOG_FILE"
      exit 1
    fi
    
    echo "Downloaded pretrained model to $MODEL_PATH" | tee -a "$LOG_FILE"
  else
    echo "Using existing model at $MODEL_PATH" | tee -a "$LOG_FILE"
  fi
fi

# Validate the model file
echo "Validating model file integrity..." | tee -a "$LOG_FILE"
if ! python3 check_model.py "$MODEL_PATH"; then
  echo "Error: The model file at $MODEL_PATH appears to be corrupted or invalid." | tee -a "$LOG_FILE"
  echo "Please provide a valid PyTorch model file." | tee -a "$LOG_FILE"
  exit 1
fi

# Step 2: Extract MMSA-FET features for each video (if not cached)
echo "Checking for feature extraction requirements..." | tee -a "$LOG_FILE"
VIDEOS_WITHOUT_FEATURES=0

for VIDEO_FILE in "$VIDEO_DIR"/*.mp4; do
  if [ ! -f "$VIDEO_FILE" ]; then
    echo "No mp4 files found in $VIDEO_DIR" | tee -a "$LOG_FILE"
    exit 1
  fi
  
  VIDEO_NAME=$(basename "$VIDEO_FILE" .mp4)
  TEXT_FEATURE="$FEATURE_DIR/${VIDEO_NAME}_text.npy"
  AUDIO_FEATURE="$FEATURE_DIR/${VIDEO_NAME}_audio.npy"
  VISION_FEATURE="$FEATURE_DIR/${VIDEO_NAME}_vision.npy"
  
  if [ ! -f "$TEXT_FEATURE" ] || [ ! -f "$AUDIO_FEATURE" ] || [ ! -f "$VISION_FEATURE" ]; then
    ((VIDEOS_WITHOUT_FEATURES++))
  fi
done

if [ $VIDEOS_WITHOUT_FEATURES -gt 0 ]; then
  echo "Extracting features for $VIDEOS_WITHOUT_FEATURES videos..." | tee -a "$LOG_FILE"
  
  # This would typically call the MMSA-FET feature extractor
  # For now, we'll use our test_self_mm.py which creates dummy features
  for VIDEO_FILE in "$VIDEO_DIR"/*.mp4; do
    VIDEO_NAME=$(basename "$VIDEO_FILE" .mp4)
    TEXT_FEATURE="$FEATURE_DIR/${VIDEO_NAME}_text.npy"
    AUDIO_FEATURE="$FEATURE_DIR/${VIDEO_NAME}_audio.npy"
    VISION_FEATURE="$FEATURE_DIR/${VIDEO_NAME}_vision.npy"
    
    if [ ! -f "$TEXT_FEATURE" ] || [ ! -f "$AUDIO_FEATURE" ] || [ ! -f "$VISION_FEATURE" ]; then
      echo "Extracting features for $VIDEO_NAME..." | tee -a "$LOG_FILE"
      python3 test_self_mm.py --video_path "$VIDEO_FILE" --fet_dir "$FEATURE_DIR" --skip_weights --live 2>&1 | tee -a "$LOG_FILE"
    fi
  done
fi

# Step 3: Run batch scoring on all videos
echo "Running batch scoring on all videos..." | tee -a "$LOG_FILE"
bash scripts/batch_score.sh \
  --model_path "$MODEL_PATH" \
  --feature_root "$FEATURE_DIR" \
  --video_dir "$VIDEO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --skip_weights 2>&1 | tee -a "$LOG_FILE"

# Determine whether to use a single video or a directory of videos
if [ -n "$SINGLE_VIDEO" ]; then
  VIDEO_ARG="--video_path=$SINGLE_VIDEO"
  echo "Processing single video: $SINGLE_VIDEO"
else
  VIDEO_ARG="--folder_path=$VIDEO_DIR"
  echo "Processing videos in directory: $VIDEO_DIR"
fi

# Part 1: Test Self-MM model
echo
echo "===== Testing Self-MM Model ====="
python test_self_mm.py $VIDEO_ARG \
                     --model_path="$MODEL_PATH" \
                     --config_path="./pretrained_models/self_mm-mosi-config_fixed.json" \
                     --fet_dir="$FEATURE_DIR" \
                     --live \
                     $MODALITY_SHAPES_ARG \
                     $STRICT_LOADING_ARG

# Part 2: Test MMSA model from repositories
echo
echo "===== Testing MMSA Repository ====="
cd repositories/MMSA
python test_repo.py $VIDEO_ARG \
                   --model_path="../../$MODEL_PATH" \
                   --config_path="../../pretrained_models/self_mm-mosi-config_fixed.json" \
                   --fet_dir="../../$FEATURE_DIR" \
                   --output_format="$OUTPUT_FORMAT" \
                   $MODALITY_SHAPES_ARG \
                   $STRICT_LOADING_ARG
cd "$ROOT_DIR"

# Part 3: Test Video-Sentiment-Analysis
echo
echo "===== Testing Video-Sentiment-Analysis ====="
cd repositories/Video-Sentiment-Analysis
python test_repo.py $VIDEO_ARG \
                   --output_format="$OUTPUT_FORMAT"
cd "$ROOT_DIR"

# Generate combined report
echo
echo "===== Generating Combined Report ====="
python generate_combined_report.py --output_format="$OUTPUT_FORMAT" --timestamp="$TIMESTAMP"

echo "All tests completed!" | tee -a "$LOG_FILE"
echo "Check results in $OUTPUT_DIR/fusion_results_*.md" | tee -a "$LOG_FILE"
echo "Full logs available at $LOG_FILE" | tee -a "$LOG_FILE" 
echo "Results saved to $OUTPUT_DIR/combined_report_$TIMESTAMP.$OUTPUT_FORMAT" | tee -a "$LOG_FILE" 