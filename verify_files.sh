#!/bin/bash

# Script to verify all files are in the repository and copy any missing ones

REPO_DIR="MM-Senti-NEW"
echo "Verifying files in $REPO_DIR..."

# Check main files
MAIN_FILES=(
  "extract_features.sh"
  "run_outside_cursor.sh"
  "run_test_with_bash.sh"
  "test_self_mm.py"
  "README.md"
  "requirements.txt"
)

for file in "${MAIN_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/$file" ]; then
    echo "Missing file: $file, copying..."
    cp "$file" "$REPO_DIR/"
  else
    echo "✓ Found file: $file"
  fi
done

# Check configs directory
CONFIG_FILES=(
  "self_mm-mosi-config_fixed.json"
)

for file in "${CONFIG_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/configs/$file" ]; then
    echo "Missing config: $file, copying..."
    cp "pretrained_models/$file" "$REPO_DIR/configs/"
  else
    echo "✓ Found config: $file"
  fi
done

# Check mmsa_scripts directory
SCRIPT_FILES=(
  "test_repo.py"
)

for file in "${SCRIPT_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/mmsa_scripts/$file" ]; then
    echo "Missing script: $file, copying..."
    cp "repositories/MMSA/$file" "$REPO_DIR/mmsa_scripts/"
  else
    echo "✓ Found script: $file"
  fi
done

# Check utils directory
UTIL_FILES=(
  "prepare_environment.py"
)

for file in "${UTIL_FILES[@]}"; do
  if [ ! -f "$REPO_DIR/utils/$file" ]; then
    echo "Missing utility: $file, copying..."
    cp "$REPO_DIR/utils/$file" "$REPO_DIR/utils/"
  else
    echo "✓ Found utility: $file"
  fi
done

echo "Verification complete."
echo "Now push the changes to GitHub with:"
echo "cd $REPO_DIR && git add . && git commit -m 'Add missing files' && git push origin main" 