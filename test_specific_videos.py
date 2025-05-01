#!/usr/bin/env python3
"""
Test script for specific problematic videos, with results written to a file
"""

import os
import sys
from improved_emotion_analyzer_final import analyze_video, determine_sentiment_label

def main():
    # Open a results file
    with open("test_results.txt", "w") as results_file:
        # List of problematic videos to test
        problem_videos = ["f.mp4", "lol.mp4", "sad.mp4"]
        
        # Expected sentiments
        expected = {
            "f.mp4": "NEGATIVE",  # Fearful video should be negative
            "lol.mp4": "POSITIVE",  # Happy video should be positive
            "sad.mp4": "NEGATIVE"   # Sad video should be negative
        }
        
        # Cascade path
        cascade_path = "./haarcascade_frontalface_default.xml"
        
        # Results
        results = []
        
        # Process each video
        for video_name in problem_videos:
            video_path = f"./test_videos/{video_name}"
            
            results_file.write(f"\n{'='*50}\n")
            results_file.write(f"TESTING {video_name}\n")
            results_file.write(f"{'='*50}\n")
            results_file.flush()
            
            # Skip if video doesn't exist
            if not os.path.exists(video_path):
                results_file.write(f"Video {video_path} not found. Skipping.\n")
                results_file.flush()
                continue
            
            # Analyze video
            sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
                video_path, cascade_path, video_name
            )
            
            if sentiment_score is None:
                results_file.write(f"Analysis failed for {video_name}\n")
                results_file.flush()
                continue
            
            # Determine sentiment
            sentiment = determine_sentiment_label(sentiment_score)
            
            # Check if correct
            expected_sentiment = expected.get(video_name, "Unknown")
            is_correct = sentiment == expected_sentiment
            
            # Store results
            results.append({
                "video": video_name,
                "expected": expected_sentiment,
                "actual": sentiment,
                "score": sentiment_score,
                "dominant_emotion": dominant_emotion,
                "emotions": emotion_percentages,
                "correct": is_correct
            })
            
            # Write results
            results_file.write(f"\nResults for {video_name}:\n")
            results_file.write(f"  Expected sentiment: {expected_sentiment}\n")
            results_file.write(f"  Actual sentiment: {sentiment} ({sentiment_score:.2f})\n")
            results_file.write(f"  Dominant emotion: {dominant_emotion}\n")
            results_file.write(f"  Top emotions: {', '.join([f'{e}: {p:.1f}%' for e, p in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True) if p > 0.5])}\n")
            results_file.write(f"  {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")
            results_file.flush()
        
        # Summary
        correct_count = sum(1 for r in results if r["correct"])
        
        results_file.write("\n" + "="*60 + "\n")
        results_file.write(f"SUMMARY: {correct_count}/{len(results)} videos classified correctly\n")
        results_file.write("="*60 + "\n")
        
        for r in results:
            status = "✓" if r["correct"] else "✗"
            results_file.write(f"{r['video']:<10} Expected: {r['expected']:>8}, Got: {r['actual']:>8} {status}\n")
        
        results_file.flush()
    
    # Print completion message to console
    print(f"Testing completed. Results written to test_results.txt")

if __name__ == "__main__":
    main() 