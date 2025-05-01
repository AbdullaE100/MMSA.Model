#!/usr/bin/env python3
"""
Direct download script for required model files from the Video-Sentiment-Analysis repository.
This script downloads the necessary files without requiring git.
"""

import os
import sys
import requests
from tqdm import tqdm

# URLs for the necessary files
FILES = {
    "model.json": "https://raw.githubusercontent.com/faizarashid/Video-Sentiment-Analysis/master/model.json",
    "model.h5": "https://raw.githubusercontent.com/faizarashid/Video-Sentiment-Analysis/master/model.h5",
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/faizarashid/Video-Sentiment-Analysis/master/haarcascade_frontalface_default.xml"
}

def download_file(url, destination, description=None):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        desc = description or os.path.basename(destination)
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    """Download all required model files"""
    print("Downloading model files for Video Emotion Analysis...")
    
    success = True
    for filename, url in FILES.items():
        print(f"Downloading {filename}...")
        if not download_file(url, filename):
            success = False
            print(f"Failed to download {filename}")
    
    if success:
        print("\nAll files downloaded successfully!")
        print("You can now run the video emotion analysis with:")
        print("python3 run_video_emotion_analysis.py")
    else:
        print("\nSome files could not be downloaded.")
        print("Please try again or download them manually from:")
        print("https://github.com/faizarashid/Video-Sentiment-Analysis")

if __name__ == "__main__":
    main() 