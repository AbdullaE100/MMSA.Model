#!/bin/bash

# Script to train the Self-MM model on MOSI dataset

# Default values
DATA_ROOT=""
SAVE_PATH="./saved/self_mm_mosi_model.pth"
EPOCHS=40
BATCH_SIZE=32
CONFIG_PATH="./configs/self_mm_mosi_train.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --save_path)
      SAVE_PATH="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --data_root <path> [--save_path <path>] [--epochs <num>] [--batch <size>] [--config <path>]"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$DATA_ROOT" ]; then
  echo "Error: --data_root is required"
  echo "Usage: $0 --data_root <path> [--save_path <path>] [--epochs <num>] [--batch <size>] [--config <path>]"
  exit 1
fi

# Set environment variable for data path
export MOSI_DATA_DIR="$DATA_ROOT"

# Ensure output directory exists
mkdir -p "$(dirname "$SAVE_PATH")"

echo "Starting Self-MM training:"
echo "  Data Root: $DATA_ROOT"
echo "  Save Path: $SAVE_PATH"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Config: $CONFIG_PATH"

# Run the training
python -m repositories.MMSA.src.MMSA.__main__ \
  --model_name self_mm \
  --dataset mosi \
  --config_file "$CONFIG_PATH" \
  --num_epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --model_save_path "$SAVE_PATH"

echo "Training complete. Model saved to $SAVE_PATH" 