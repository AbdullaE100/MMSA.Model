#!/usr/bin/env python3
"""
Simple script to run our improved emotion analyzer on test videos and save results to a file.
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path

def run_analyzer_on_video(video_path):
    """Run the improved emotion analyzer on a single video"""
    try:
        from improved_emotion_analyzer import load_model, analyze_video, determine_sentiment_label
        
        # Load model
        model = load_model('./model.json', './model.h5')
        if model is None:
            return f"Failed to load model for {video_path}"
        
        # Analyze video
        sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
            model, 
            video_path, 
            './haarcascade_frontalface_default.xml', 
            threshold=0.4  # Lower threshold for better detection
        )
        
        if sentiment_score is None:
            return f"Analysis failed for {video_path}"
        
        # Determine sentiment label
        sentiment_label = determine_sentiment_label(sentiment_score)
        
        # Format results
        result = f"Video: {os.path.basename(video_path)}\n"
        result += f"Dominant Emotion: {dominant_emotion}\n"
        result += f"Sentiment Score: {sentiment_score:.4f}\n"
        result += f"Sentiment Label: {sentiment_label}\n"
        result += "Emotion Distribution:\n"
        
        for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0:
                result += f"  {emotion}: {percentage:.1f}%\n"
        
        # Get expected sentiment from labels if available
        expected_sentiment = None
        labels_path = os.path.join(os.path.dirname(video_path), 'labels.csv')
        if os.path.exists(labels_path):
            import csv
            with open(labels_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if os.path.basename(video_path) == row['file']:
                        expected_sentiment = float(row['expected_sentiment'])
                        break
        
        if expected_sentiment is not None:
            expected_label = determine_sentiment_label(expected_sentiment)
            result += f"Expected Sentiment: {expected_sentiment:.4f} ({expected_label})\n"
            
            # Check if our analysis is correct
            is_correct = (
                sentiment_label == expected_label or 
                abs(sentiment_score - expected_sentiment) < 0.2
            )
            result += f"Accuracy: {'✓ Correct' if is_correct else '✗ Incorrect'}\n"
        
        return result
    
    except Exception as e:
        traceback.print_exc()
        return f"Error analyzing {video_path}: {str(e)}"

def main():
    # Find test videos
    video_dir = './test_videos'
    output_file = 'emotion_analysis_results.txt'
    
    # Get video files
    video_paths = list(Path(video_dir).glob('*.mp4'))
    if not video_paths:
        print(f"No videos found in {video_dir}")
        return
    
    print(f"Found {len(video_paths)} videos to analyze")
    
    # Initialize results list
    results = []
    correct_count = 0
    total_count = 0
    
    # Process each video
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        print(f"\nAnalyzing {video_name}...")
        result = run_analyzer_on_video(str(video_path))
        results.append(result)
        
        # Count accuracy
        if "✓ Correct" in result:
            correct_count += 1
        if "Expected Sentiment" in result:
            total_count += 1
    
    # Calculate accuracy
    accuracy = 0
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
    
    # Add summary
    summary = f"\n{'='*50}\n"
    summary += f"Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)\n"
    summary += f"{'='*50}\n"
    results.append(summary)
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write('\n\n'.join(results))
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)")

if __name__ == "__main__":
    main() 