#!/usr/bin/env python3
"""
Test script specifically for lol.mp4
"""

import os
from improved_emotion_analyzer_final import analyze_video, determine_sentiment_label

def main():
    # Test lol.mp4
    video_name = "lol.mp4"
    video_path = f"./test_videos/{video_name}"
    cascade_path = "./haarcascade_frontalface_default.xml"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found")
        return
    
    # Analyze video
    print(f"Analyzing {video_name}...")
    sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
        video_path, cascade_path, video_name
    )
    
    if sentiment_score is None:
        print(f"Analysis failed for {video_name}")
        return
    
    # Determine sentiment
    sentiment = determine_sentiment_label(sentiment_score)
    
    # Print results
    print("\nResults:")
    print(f"Dominant Emotion: {dominant_emotion}")
    print(f"Sentiment Score: {sentiment_score:.4f}")
    print(f"Sentiment: {sentiment}")
    
    print("\nEmotion Distribution:")
    for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
        if percentage > 0:
            print(f"  {emotion}: {percentage:.1f}%")
    
    # Check if classification is correct
    expected = "POSITIVE"
    is_correct = sentiment == expected
    print(f"\nExpected: {expected}")
    print(f"{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

if __name__ == "__main__":
    main() 