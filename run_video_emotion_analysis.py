#!/usr/bin/env python3
"""
Main script to run the Video Emotion Analysis with weighted sentiment scores.
This script guides users through the setup and analysis process.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "tensorflow",
        "opencv-python",
        "matplotlib",
        "tabulate",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        
        install = input("Would you like to install them now? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                subprocess.run([sys.executable, "-m", "pip", "install", package])
            print("Dependencies installed successfully.")
        else:
            print("Please install the required packages and run this script again.")
            sys.exit(1)

def setup_models():
    """Set up the model files from the repository"""
    if (not os.path.exists("model.json") or 
        not os.path.exists("model.h5") or 
        not os.path.exists("haarcascade_frontalface_default.xml")):
        
        print("Model files are missing. Setting up...")
        setup_process = subprocess.run([sys.executable, "setup_models.py"], capture_output=True, text=True)
        
        # Print the output from the setup process
        if setup_process.stdout:
            print(setup_process.stdout)
        if setup_process.stderr:
            print(setup_process.stderr)
            
        # Check if all required files are now available
        if (not os.path.exists("model.json") or 
            not os.path.exists("model.h5") or 
            not os.path.exists("haarcascade_frontalface_default.xml")):
            
            print("\nSome model files could not be downloaded automatically.")
            print("Please follow the manual download instructions above.")
            
            # Ask if the user has manually downloaded the files
            while True:
                response = input("Have you manually downloaded and placed the files? (yes/no/skip): ")
                if response.lower() == 'yes':
                    # Verify files again
                    if (os.path.exists("model.json") and 
                        os.path.exists("model.h5") and 
                        os.path.exists("haarcascade_frontalface_default.xml")):
                        print("All files found. Proceeding with analysis.")
                        break
                    else:
                        print("Still missing some files. Please try again.")
                elif response.lower() == 'skip':
                    print("Warning: Proceeding without complete model files may cause errors.")
                    break
                elif response.lower() == 'no':
                    print("Please download the required files and try again.")
                    sys.exit(1)
    else:
        print("Model files already exist. Skipping setup.")

def analyze_videos(video_dir, output_dir):
    """Run the video emotion analysis"""
    # Make sure the model files are present
    if (not os.path.exists("model.json") or 
        not os.path.exists("model.h5") or 
        not os.path.exists("haarcascade_frontalface_default.xml")):
        print("Error: Model files are missing. Please run setup first.")
        sys.exit(1)
    
    # Make sure the video directory exists
    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' does not exist.")
        sys.exit(1)
    
    # Make sure there are videos in the directory
    video_files = list(Path(video_dir).glob("*.mp4"))
    if not video_files:
        print(f"Error: No video files found in '{video_dir}'.")
        sys.exit(1)
    
    print(f"Found {len(video_files)} videos to analyze.")
    
    # Run the analysis
    cmd = [
        sys.executable, 
        "weighted_sentiment_analysis.py",
        "--video_dir", video_dir,
        "--output_dir", output_dir
    ]
    
    print("Starting video emotion analysis...")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Video Emotion Analysis with weighted sentiment scores')
    parser.add_argument('--video_dir', type=str, default='./test_videos', 
                        help='Directory containing the videos to analyze')
    parser.add_argument('--output_dir', type=str, default='./sentiment_results', 
                        help='Directory for output files')
    parser.add_argument('--skip_setup', action='store_true',
                        help='Skip checking and setting up model files')
    args = parser.parse_args()
    
    print("=========================================")
    print("Video Emotion Analysis with Sentiment Scoring")
    print("=========================================")
    print("This tool analyzes videos for emotions and assigns weighted sentiment scores.")
    print("The analysis is based on the Video-Sentiment-Analysis repository by Faiza Rashid.")
    print("Reference: https://github.com/faizarashid/Video-Sentiment-Analysis")
    print("=========================================")
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    check_dependencies()
    
    # Setup models if needed
    if not args.skip_setup:
        print("\n2. Setting up model files...")
        setup_models()
    
    # Run the analysis
    print("\n3. Running video emotion analysis...")
    analyze_videos(args.video_dir, args.output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results have been saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 