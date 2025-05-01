#!/usr/bin/env python3
"""
Test script for videos using base weights
"""

import os
from improved_emotion_analyzer_final import analyze_video, determine_sentiment_label, SENTIMENT_WEIGHTS

def main():
    # Test videos
    videos = ["sad.mp4", "f.mp4", "lol.mp4", "Angry.mp4", "Neutral.mp4", "Happy.mp4", "Surprise.mp4"]
    
    # Expected results (just for reference)
    expected = {
        "sad.mp4": "NEGATIVE",
        "f.mp4": "NEGATIVE",
        "lol.mp4": "POSITIVE",
        "Angry.mp4": "NEGATIVE",
        "Neutral.mp4": "NEUTRAL",
        "Happy.mp4": "POSITIVE",
        "Surprise.mp4": "POSITIVE"
    }
    
    # Cascade path
    cascade_path = "./haarcascade_frontalface_default.xml"
    
    # Print current weights
    print("Current sentiment weights:")
    for emotion, weight in SENTIMENT_WEIGHTS.items():
        print(f"  {emotion}: {weight}")
    print("\n")
    
    # Process each video
    results = []
    for video in videos:
        video_path = f"./test_videos/{video}"
        
        print(f"\n{'='*50}")
        print(f"Testing {video}")
        print(f"{'='*50}")
        
        # Skip if video doesn't exist
        if not os.path.exists(video_path):
            print(f"Video {video_path} not found, skipping")
            continue
        
        # Analyze video
        sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
            video_path, cascade_path, video
        )
        
        if sentiment_score is None:
            print(f"Analysis failed for {video}")
            continue
        
        # Determine sentiment
        sentiment = determine_sentiment_label(sentiment_score)
        
        # Check if correct based on expected values
        expected_sentiment = expected.get(video, "Unknown")
        is_correct = sentiment == expected_sentiment
        
        # Add to results
        results.append({
            "video": video,
            "expected": expected_sentiment,
            "actual": sentiment,
            "score": sentiment_score,
            "dominant_emotion": dominant_emotion,
            "emotions": emotion_percentages,
            "correct": is_correct
        })
        
        # Print results
        print(f"\nResults for {video}:")
        print(f"  Expected sentiment: {expected_sentiment}")
        print(f"  Actual sentiment: {sentiment} (score: {sentiment_score:.2f})")
        print(f"  Dominant emotion: {dominant_emotion}")
        print(f"  Emotion distribution:")
        
        for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0.5:
                print(f"    {emotion}: {percentage:.1f}%")
        
        print(f"  {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    
    print("\n" + "="*60)
    print(f"SUMMARY: {correct_count}/{len(results)} videos classified correctly")
    print("="*60 + "\n")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"{r['video']:<12} Expected: {r['expected']:>8}, Got: {r['actual']:>8} {status}")
    
    # Save results to file
    with open("base_weights_results.txt", "w") as f:
        f.write("TEST RESULTS WITH BASE WEIGHTS\n\n")
        f.write("Current sentiment weights:\n")
        for emotion, weight in SENTIMENT_WEIGHTS.items():
            f.write(f"  {emotion}: {weight}\n")
        f.write("\n")
        
        for r in results:
            f.write(f"\n{'='*50}\n")
            f.write(f"Results for {r['video']}:\n")
            f.write(f"  Expected sentiment: {r['expected']}\n")
            f.write(f"  Actual sentiment: {r['actual']} (score: {r['score']:.2f})\n")
            f.write(f"  Dominant emotion: {r['dominant_emotion']}\n")
            f.write(f"  Emotion distribution:\n")
            
            for emotion, percentage in sorted(r['emotions'].items(), key=lambda x: x[1], reverse=True):
                if percentage > 0.5:
                    f.write(f"    {emotion}: {percentage:.1f}%\n")
            
            f.write(f"  {'✓ CORRECT' if r['correct'] else '✗ INCORRECT'}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"SUMMARY: {correct_count}/{len(results)} videos classified correctly\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            status = "✓" if r["correct"] else "✗"
            f.write(f"{r['video']:<12} Expected: {r['expected']:>8}, Got: {r['actual']:>8} {status}\n")

if __name__ == "__main__":
    main() 