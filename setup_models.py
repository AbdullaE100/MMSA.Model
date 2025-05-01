#!/usr/bin/env python3
"""
Script to set up the model files from the Video-Sentiment-Analysis repository.
"""

import os
import sys
import subprocess
import shutil
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from a URL to a destination path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_model_files_directly():
    """Download required model files directly from GitHub without cloning"""
    base_url = "https://raw.githubusercontent.com/faizarashid/Video-Sentiment-Analysis/master"
    model_files = {
        "model.json": f"{base_url}/model.json",
        "model.h5": f"{base_url}/model.h5",
        "haarcascade_frontalface_default.xml": f"{base_url}/haarcascade_frontalface_default.xml"
    }
    
    print("Attempting to download model files directly...")
    success = True
    
    for filename, url in model_files.items():
        if download_file(url, filename):
            print(f"Successfully downloaded {filename}")
        else:
            success = False
            print(f"Failed to download {filename}")
    
    return success

def setup_models():
    """Clone the repo and copy model files to current directory"""
    temp_dir = "temp_video_sa_repo"
    
    # First try a shallow clone (faster and less prone to network issues)
    print("Attempting shallow clone of Video-Sentiment-Analysis repository...")
    clone_cmd = [
        "git", "clone",
        "--depth=1",  # Only get the latest commit
        "--single-branch",  # Only get the main branch
        "https://github.com/faizarashid/Video-Sentiment-Analysis.git", 
        temp_dir
    ]
    
    clone_successful = False
    try:
        subprocess.run(clone_cmd, check=True)
        clone_successful = True
    except subprocess.CalledProcessError as e:
        print(f"Shallow clone failed: {e}")
        
        # Try regular clone as fallback
        try:
            print("Trying regular clone...")
            subprocess.run(["git", "clone", "https://github.com/faizarashid/Video-Sentiment-Analysis.git", temp_dir], check=True)
            clone_successful = True
        except subprocess.CalledProcessError as e:
            print(f"Regular clone also failed: {e}")
    
    # If cloning worked, copy the files
    if clone_successful:
        # Copy necessary model files
        model_files = [
            "model.json",
            "model.h5",
            "haarcascade_frontalface_default.xml"
        ]
        
        print("Copying model files...")
        for file_name in model_files:
            src_path = os.path.join(temp_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, ".")
                print(f"Copied {file_name}")
            else:
                print(f"Warning: {file_name} not found in repository")
        
        # Clean up
        print("Cleaning up...")
        shutil.rmtree(temp_dir)
    
    # If cloning failed or files are missing, try direct download
    if not clone_successful or not all(os.path.exists(file) for file in ["model.json", "model.h5", "haarcascade_frontalface_default.xml"]):
        print("\nAttempting direct download of model files...")
        download_model_files_directly()
    
    # Verify model files are available
    missing_files = []
    for file_name in ["model.json", "model.h5", "haarcascade_frontalface_default.xml"]:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\nWarning: The following model files are still missing: {', '.join(missing_files)}")
        print("\nManual download instructions:")
        print("1. Visit https://github.com/faizarashid/Video-Sentiment-Analysis")
        print("2. Download the following files manually:")
        for file in missing_files:
            print(f"   - {file}")
        print("3. Place them in the current directory")
        return False
    else:
        print("\nAll necessary model files have been set up successfully!")
        print("You can now run weighted_sentiment_analysis.py to analyze videos.")
        return True

if __name__ == "__main__":
    setup_models() 