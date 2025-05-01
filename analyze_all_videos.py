#!/usr/bin/env python3
"""
Script to analyze all test videos and calculate overall accuracy.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Find all test videos
    video_dir = './test_videos'
    video_paths = list(Path(video_dir).glob('*.mp4'))
    
    if not video_paths:
        print(f"No videos found in {video_dir}")
        return
    
    # Initialize results tracking
    results = []
    correct_count = 0
    total_count = 0
    
    # Output file
    output_file = 'all_video_results.txt'
    with open(output_file, 'w') as f:
        f.write("Video Emotion Analysis Results\n")
        f.write("============================\n\n")
    
    # Analyze each video
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        print(f"\nAnalyzing {video_name}...")
        
        # Run the simple analyzer
        try:
            result = subprocess.run(
                [sys.executable, 'simple_emotion_analyzer.py', video_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get the output
            output = result.stdout
            
            # Check if the analysis was correct
            if "Accuracy: ✓ Correct" in output:
                correct_count += 1
                
            if "Expected Sentiment" in output:
                total_count += 1
            
            # Save output to file
            with open(output_file, 'a') as f:
                f.write(f"Analysis for {video_name}\n")
                f.write("=" * (12 + len(video_name)) + "\n\n")
                f.write(output)
                f.write("\n\n" + "-" * 50 + "\n\n")
            
            # Print a summary
            for line in output.split('\n'):
                if any(s in line for s in ["Dominant Emotion", "Sentiment Score", "Sentiment:", "Accuracy"]):
                    print(line)
            
        except subprocess.CalledProcessError as e:
            print(f"Error analyzing {video_name}: {e}")
            print(e.stdout)
            print(e.stderr)
    
    # Calculate accuracy
    accuracy = 0
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
    
    # Print summary
    summary = f"\nOverall Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)"
    print(summary)
    
    # Add summary to output file
    with open(output_file, 'a') as f:
        f.write("\nSummary\n")
        f.write("=======\n\n")
        f.write(summary + "\n")
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main() 