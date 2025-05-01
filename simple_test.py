#!/usr/bin/env python3
"""
Simplified test script for pure emotion-based analysis
"""

import os
import sys
import re
from video_sentiment_app import SentimentAnalyzer

def main():
    # Redirect stdout to suppress long output
    original_stdout = sys.stdout
    null_output = open(os.devnull, 'w')
    sys.stdout = null_output
    
    try:
        # Initialize the analyzer
        analyzer = SentimentAnalyzer()
        
        # Test video directory
        test_dir = "test_videos"
        
        # Expected sentiments based on file names
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
        for video_file in sorted(os.listdir(test_dir)):
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue
                
            video_path = os.path.join(test_dir, video_file)
            
            # Get the expected sentiment
            expected_sentiment = expected.get(video_file, "Unknown")
            
            # Analyze the video
            result = analyzer.analyze_sentiment(video_path)
            
            if result is None:
                continue
            
            # Extract results
            overall_formatted, mmsa_formatted, emotion_formatted, text_formatted, _, _, _, emotion_percentages, dominant_emotion = result
            
            # Extract sentiment label from formatted string
            overall_match = re.match(r"([A-Z]+)", overall_formatted)
            emotion_match = re.match(r"([A-Z]+)", emotion_formatted)
            
            overall_sentiment = overall_match.group(1) if overall_match else "UNKNOWN"
            emotion_sentiment = emotion_match.group(1) if emotion_match else "UNKNOWN"
            
            # Extract scores
            overall_score_match = re.search(r"\(([-+]?[0-9]*\.?[0-9]+)\)", overall_formatted)
            emotion_score_match = re.search(r"\(([-+]?[0-9]*\.?[0-9]+)\)", emotion_formatted)
            
            overall_score = float(overall_score_match.group(1)) if overall_score_match else 0.0
            emotion_score = float(emotion_score_match.group(1)) if emotion_score_match else 0.0
            
            # Store results
            results.append({
                "file": video_file,
                "expected": expected_sentiment,
                "overall": overall_sentiment,
                "emotion": emotion_sentiment,
                "dominant_emotion": dominant_emotion,
                "overall_score": overall_score,
                "emotion_score": emotion_score,
                "top_emotions": sorted([(e, p) for e, p in emotion_percentages.items() if p > 10], key=lambda x: x[1], reverse=True)
            })
    
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        null_output.close()
    
    # Write results to file
    with open("emotion_test_results.txt", "w") as f:
        f.write("PURE EMOTION-BASED SENTIMENT ANALYSIS TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        
        # Count correct predictions
        correct_count = 0
        for r in results:
            if r["expected"] == r["overall"]:
                correct_count += 1
        
        accuracy = (correct_count / len(results) * 100) if results else 0
        f.write(f"Accuracy: {correct_count}/{len(results)} videos ({accuracy:.1f}%)\n\n")
        
        # Write individual test results
        f.write("VIDEO RESULTS:\n")
        f.write("-"*70 + "\n")
        
        for r in results:
            is_correct = r["expected"] == r["overall"]
            status = "✓" if is_correct else "✗"
            
            f.write(f"{r['file']:<15} Expected: {r['expected']:>8}, Got: {r['overall']:>8} {status}\n")
            f.write(f"  Pure emotion: {r['emotion']:>8} ({r['emotion_score']:.2f})\n")
            f.write(f"  Dominant emotion: {r['dominant_emotion']}\n")
            f.write(f"  Top emotions: {', '.join([f'{e}: {p:.1f}%' for e, p in r['top_emotions']])}\n")
            f.write("-"*70 + "\n")
        
        # List incorrect classifications
        incorrect = [r for r in results if r["expected"] != r["overall"]]
        if incorrect:
            f.write("\nINCORRECT CLASSIFICATIONS:\n")
            for r in incorrect:
                f.write(f"  - {r['file']}: Expected {r['expected']}, Got {r['overall']}\n")
                f.write(f"    Top emotions: {', '.join([f'{e}: {p:.1f}%' for e, p in r['top_emotions']])}\n")
    
    # Print simple status to console
    print(f"Analysis complete - {correct_count}/{len(results)} videos ({accuracy:.1f}%) classified correctly.")
    print("Detailed results written to emotion_test_results.txt")

if __name__ == "__main__":
    main() 