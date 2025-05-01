#!/usr/bin/env python3
"""
Test script to verify the sentiment analysis on the surprised video specifically
"""

from video_sentiment_app import SentimentAnalyzer
import os

def main():
    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Test the surprised video
    video_file = 'suprised.mp4'
    expected_score = 0.60
    video_path = os.path.join('test_videos', video_file)
    
    print(f'Testing {video_file}...')
    
    # Run the analysis
    result = analyzer.analyze_sentiment(video_path)
    
    if result is None:
        print(f'Error: Analysis failed for {video_file}')
        return
    
    # Extract overall sentiment
    overall_formatted, mmsa_formatted, emotion_formatted, text_formatted, viz_path, transcript, emotion_html, emotion_percentages, dominant_emotion = result
    
    # Parse the overall sentiment score
    import re
    match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
    if match:
        actual_score = float(match.group(1))
    else:
        print(f'Error: Could not parse sentiment score from {overall_formatted}')
        return
    
    # Extract the sentiment label
    label_match = re.match(r'([A-Z]+)', overall_formatted)
    actual_label = label_match.group(1) if label_match else 'UNKNOWN'
    
    # Determine expected label
    expected_label = 'POSITIVE' if expected_score >= 0.1 else 'NEGATIVE' if expected_score <= -0.1 else 'NEUTRAL'
    
    # Check if results match expectations
    score_diff = abs(actual_score - expected_score)
    is_correct_label = actual_label == expected_label
    
    # Print the results
    print(f'\nResults:')
    print(f'Expected: {expected_label} ({expected_score:.2f}), Actual: {actual_label} ({actual_score:.2f})')
    print(f'Correct label: {is_correct_label}, Score difference: {score_diff:.2f}')
    print(f'Test {"PASSED" if is_correct_label else "FAILED"}')

if __name__ == "__main__":
    main() 