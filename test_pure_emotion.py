#!/usr/bin/env python3
"""
Test script to analyze videos using pure emotion-based approach
"""

import os
import sys
from video_sentiment_app import SentimentAnalyzer

def main():
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Test video directory
    test_dir = "test_videos"
    
    # Get test videos
    test_videos = [f for f in os.listdir(test_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    
    if not test_videos:
        print("No test videos found in the test_videos directory.")
        return
    
    print("=" * 80)
    print("ANALYZING TEST VIDEOS WITH PURE EMOTION-BASED APPROACH")
    print("=" * 80)
    
    # Expected sentiments based on file names (for reference)
    expected = {
        "Angry.mp4": "NEGATIVE",
        "Calm.mp4": "POSITIVE",
        "Disgust.mp4": "NEGATIVE",
        "Neutral.mp4": "NEUTRAL",
        "sad.mp4": "NEGATIVE",
        "suprised.mp4": "POSITIVE"
    }
    
    results = []
    
    # Process each test video
    for video_file in sorted(test_videos):
        video_path = os.path.join(test_dir, video_file)
        
        print(f"\nAnalyzing: {video_file}")
        
        # Get the expected sentiment
        expected_sentiment = expected.get(video_file, "Unknown")
        
        # Analyze the video
        result = analyzer.analyze_sentiment(video_path)
        
        if result is None:
            print(f"Error analyzing {video_file}")
            continue
        
        # Extract the results
        overall, mmsa, emotion, text, _, _, _, emotion_percentages, dominant_emotion = result
        
        # Check if correct
        overall_sentiment = overall.split(" ")[0]
        is_correct = overall_sentiment == expected_sentiment
        
        # Store results
        results.append({
            "file": video_file,
            "expected": expected_sentiment,
            "actual": overall_sentiment,
            "dominant_emotion": dominant_emotion,
            "is_correct": is_correct
        })
        
        # Print summary for this video
        print(f"File: {video_file}")
        print(f"Expected: {expected_sentiment}")
        print(f"Result: {overall}")
        print(f"Emotion: {emotion}")
        print(f"MMSA: {mmsa}")
        print(f"Text: {text}")
        print(f"Dominant emotion: {dominant_emotion}")
        print(f"Emotion distribution: {', '.join([f'{e}: {emotion_percentages.get(e, 0):.1f}%' for e in sorted(emotion_percentages.keys())])}")
        print(f"Status: {'✓ Correct' if is_correct else '✗ Incorrect'}")
    
    # Print overall summary
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {correct_count}/{total_count} videos ({accuracy:.1f}%) classified correctly")
    print("=" * 80)
    
    # List accurate and inaccurate classifications
    print("\nCORRECT CLASSIFICATIONS:")
    for r in [r for r in results if r["is_correct"]]:
        print(f"  - {r['file']}: {r['expected']} (Dominant emotion: {r['dominant_emotion']})")
    
    print("\nINCORRECT CLASSIFICATIONS:")
    for r in [r for r in results if not r["is_correct"]]:
        print(f"  - {r['file']}: Expected {r['expected']}, Got {r['actual']} (Dominant emotion: {r['dominant_emotion']})")

if __name__ == "__main__":
    main() 