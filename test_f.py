#!/usr/bin/env python3
"""
Test script to verify the sentiment analysis on f.mp4
"""

from video_sentiment_app import SentimentAnalyzer
import os

def main():
    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Test the f.mp4 video
    video_file = 'f.mp4'
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
    
    # Print the detailed results without comparing to any expected value
    # since we don't know what the expected value should be
    print(f'\nResults for f.mp4:')
    print(f'Sentiment: {actual_label} ({actual_score:.2f})')
    print(f'MMSA sentiment: {mmsa_formatted}')
    print(f'Primary emotion sentiment: {emotion_formatted}')
    print(f'Text sentiment: {text_formatted}')
    print(f'Dominant emotion: {dominant_emotion}')
    
    # Print emotion percentages
    print('\nEmotion Percentages:')
    for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
        if percentage > 0:
            print(f'{emotion}: {percentage:.1f}%')
    
    print(f'\nTranscript: {transcript}')

if __name__ == "__main__":
    main() 