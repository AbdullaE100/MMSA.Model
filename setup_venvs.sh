#!/bin/bash

# Set working directory to the script's location
cd "$(dirname "$0")"

# Function to create and set up a virtual environment
setup_venv() {
  local repo_name=$1
  local venv_name="venv_${repo_name}"
  local repo_dir="repositories/${repo_name}"
  
  echo "======================================"
  echo "Setting up virtual environment for ${repo_name}"
  echo "======================================"
  
  # Create virtual environment
  python3 -m venv "${venv_name}"
  
  # Activate virtual environment and install dependencies
  source "${venv_name}/bin/activate"
  
  # Install common dependencies
  pip install -U pip
  pip install -r requirements.txt
  
  # Check for repository-specific requirements
  if [ -f "${repo_dir}/requirements.txt" ]; then
    echo "Installing ${repo_name} specific requirements..."
    pip install -r "${repo_dir}/requirements.txt"
  fi
  
  # Add symlink to testing_utils.py in the virtual environment
  ln -sf "$(pwd)/testing_utils.py" "${venv_name}/lib/python3.*/site-packages/"
  
  # Deactivate virtual environment
  deactivate
  
  echo "Virtual environment for ${repo_name} set up successfully"
  echo "======================================"
}

# Create virtual environments for each repository
setup_venv "MMSA-FET"
setup_venv "MMSA"
setup_venv "Video-Sentiment-Analysis"

echo "All virtual environments set up successfully"
echo "To run tests using a specific virtual environment, use:"
echo "  source venv_REPOSITORY_NAME/bin/activate"
echo "  cd repositories/REPOSITORY_NAME"
echo "  python3 test_repo.py --folder_path ../../test_videos --output_format json"
echo "  deactivate"
echo ""
echo "Or run all tests with: ./run_all_tests.sh" 