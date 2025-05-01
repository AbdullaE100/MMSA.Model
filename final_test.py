#!/usr/bin/env python3
"""
Final test script for problematic videos
"""

import os
from improved_emotion_analyzer_final import analyze_video, determine_sentiment_label

def main():
    # Open results file
    with open("final_results.txt", "w") as f:
        # Test videos
        videos = ["f.mp4", "lol.mp4", "sad.mp4"]
        
        # Expected results
        expected = {
            "f.mp4": "NEGATIVE",
            "lol.mp4": "POSITIVE",
            "sad.mp4": "NEGATIVE"
        }
        
        # Cascade path
        cascade_path = "./haarcascade_frontalface_default.xml"
        
        # Results
        results = []
        
        # Process each video
        for video in videos:
            video_path = f"./test_videos/{video}"
            
            f.write(f"\n{'='*50}\n")
            f.write(f"Testing {video}\n")
            f.write(f"{'='*50}\n")
            f.flush()
            
            # Skip if video doesn't exist
            if not os.path.exists(video_path):
                f.write(f"Video {video_path} not found\n")
                f.flush()
                continue
            
            # Analyze video
            sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
                video_path, cascade_path, video
            )
            
            if sentiment_score is None:
                f.write(f"Analysis failed for {video}\n")
                f.flush()
                continue
            
            # Get sentiment
            sentiment = determine_sentiment_label(sentiment_score)
            
            # Check if correct
            expected_sentiment = expected.get(video)
            is_correct = sentiment == expected_sentiment
            
            # Add to results
            results.append({
                "video": video,
                "expected": expected_sentiment,
                "actual": sentiment,
                "score": sentiment_score,
                "dominant_emotion": dominant_emotion,
                "correct": is_correct
            })
            
            # Write results
            f.write(f"\nResults for {video}:\n")
            f.write(f"  Expected sentiment: {expected_sentiment}\n")
            f.write(f"  Actual sentiment: {sentiment} (score: {sentiment_score:.2f})\n")
            f.write(f"  Dominant emotion: {dominant_emotion}\n")
            f.write(f"  Emotion distribution:\n")
            
            for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
                if percentage > 0.5:
                    f.write(f"    {emotion}: {percentage:.1f}%\n")
            
            f.write(f"  {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")
            f.flush()
        
        # Summary
        correct_count = sum(1 for r in results if r["correct"])
        
        f.write(f"\n{'='*50}\n")
        f.write(f"Summary: {correct_count}/{len(results)} videos classified correctly\n")
        f.write(f"{'='*50}\n\n")
        
        for r in results:
            status = "✓" if r["correct"] else "✗"
            f.write(f"{r['video']:<10} Expected: {r['expected']:>8}, Got: {r['actual']:>8} {status}\n")
    
    # Print completion message
    print(f"Testing completed. Results written to final_results.txt")

if __name__ == "__main__":
    main() 