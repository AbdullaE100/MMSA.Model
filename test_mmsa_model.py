#!/usr/bin/env python3
"""
Test script to evaluate the MMSA model on test videos
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from video_sentiment_app import SentimentAnalyzer

def main():
    # Define test videos directory
    test_dir = "test_videos"
    
    # Get list of video files (filter out non-video files)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(test_dir) 
                  if os.path.isfile(os.path.join(test_dir, f)) and 
                  any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print("No video files found in test_videos directory")
        return
    
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Store results for comparison
    results = []
    
    print("Testing MMSA model on videos in test_videos directory:")
    print("-" * 60)
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(test_dir, video_file)
        print(f"Processing {video_file}...")
        
        # Run MMSA analysis
        mmsa_score = analyzer.analyze_mmsa(video_path)
        
        # Get sentiment label
        if mmsa_score > 0.3:
            sentiment = "POSITIVE"
        elif mmsa_score < -0.3:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
            
        # Store result
        results.append((video_file, mmsa_score, sentiment))
        
        # Print result
        print(f"  MMSA Score: {mmsa_score:.2f} ({sentiment})")
    
    print("-" * 60)
    
    # Create a bar chart to visualize results
    plt.figure(figsize=(12, 6))
    
    # Sort videos by their scores
    results.sort(key=lambda x: x[1])
    
    video_names = [name.split('.')[0] for name, _, _ in results]
    scores = [score for _, score, _ in results]
    colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in scores]
    
    plt.bar(video_names, scores, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Video')
    plt.ylabel('MMSA Sentiment Score')
    plt.title('MMSA Model Sentiment Scores for Test Videos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('mmsa_test_results.png')
    print("Results visualization saved to mmsa_test_results.png")

if __name__ == "__main__":
    main() 