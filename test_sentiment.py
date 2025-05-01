#!/usr/bin/env python3
"""
Test script to verify the sentiment analysis on test videos
"""

import sys
import os
import re
import pandas as pd
from video_sentiment_app import SentimentAnalyzer
from improved_emotion_analyzer_final import determine_sentiment_label

def main():
    # Load expected labels
    labels_df = pd.read_csv('test_videos/labels.csv')
    expected = {row['file']: row['expected_sentiment'] for _, row in labels_df.iterrows()}

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Test each video
    results = []
    for video_file, expected_score in expected.items():
        video_path = os.path.join('test_videos', video_file)
        print(f'Testing {video_file}...')
        try:
            # Run the analysis
            result = analyzer.analyze_sentiment(video_path)
            if result is None:
                print(f'  Error: Analysis failed for {video_file}')
                continue

            # Extract overall sentiment
            overall_formatted, mmsa_formatted, emotion_formatted, text_formatted, viz_path, transcript, emotion_html, emotion_percentages, dominant_emotion = result
            
            # Parse the overall sentiment score
            match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
            if match:
                actual_score = float(match.group(1))
            else:
                print(f'  Error: Could not parse sentiment score from {overall_formatted}')
                continue
            
            # Extract the sentiment label
            label_match = re.match(r'([A-Z]+)', overall_formatted)
            actual_label = label_match.group(1) if label_match else 'UNKNOWN'
            
            # Determine expected label
            expected_label = 'POSITIVE' if expected_score >= 0.1 else 'NEGATIVE' if expected_score <= -0.1 else 'NEUTRAL'
            
            # Check if results match expectations
            score_diff = abs(actual_score - expected_score)
            is_correct_label = actual_label == expected_label
            
            results.append({
                'file': video_file,
                'expected_score': expected_score,
                'actual_score': actual_score,
                'expected_label': expected_label,
                'actual_label': actual_label,
                'is_correct': is_correct_label,
                'score_diff': score_diff
            })
            
            # Print the results
            print(f'  Expected: {expected_label} ({expected_score:.2f}), Actual: {actual_label} ({actual_score:.2f})')
            print(f'  Correct label: {is_correct_label}, Score difference: {score_diff:.2f}')
        except Exception as e:
            print(f'  Error analyzing {video_file}: {e}')

    # Print summary
    print('\nSummary:')
    correct_count = sum(1 for r in results if r['is_correct'])
    print(f'Correctly classified: {correct_count}/{len(results)} ({100 * correct_count / len(results):.1f}%)')

    # Print detailed results sorted by correctness
    print('\nDetailed results:')
    sorted_results = sorted(results, key=lambda r: (not r['is_correct'], r['score_diff']))
    for r in sorted_results:
        print(f"{r['file']:<12} Expected: {r['expected_label']:>8} ({r['expected_score']:.2f}), Actual: {r['actual_label']:>8} ({r['actual_score']:.2f}) - {'✓' if r['is_correct'] else '✗'}")

if __name__ == "__main__":
    main() 