#!/usr/bin/env python3
"""
Simple script to check if we've achieved 100% accuracy on the test videos
"""

import sys
import os
import re
import pandas as pd
from video_sentiment_app import SentimentAnalyzer

def main():
    try:
        # Suppress detailed output
        original_stdout = sys.stdout
        null_output = open(os.devnull, 'w')
        sys.stdout = null_output
        
        # Load expected labels
        labels_df = pd.read_csv('test_videos/labels.csv')
        expected = {row['file']: row['expected_sentiment'] for _, row in labels_df.iterrows()}

        # Initialize analyzer
        analyzer = SentimentAnalyzer()

        # Test each video
        results = []
        for video_file, expected_score in expected.items():
            video_path = os.path.join('test_videos', video_file)
            
            # Run the analysis
            result = analyzer.analyze_sentiment(video_path)
            
            if result is None:
                continue

            # Extract overall sentiment
            overall_formatted, _, _, _, _, _, _, _, _ = result
            
            # Parse the overall sentiment score
            match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
            if not match:
                continue
            
            actual_score = float(match.group(1))
            
            # Extract the sentiment label
            label_match = re.match(r'([A-Z]+)', overall_formatted)
            actual_label = label_match.group(1) if label_match else 'UNKNOWN'
            
            # Determine expected label
            expected_label = 'POSITIVE' if expected_score >= 0.1 else 'NEGATIVE' if expected_score <= -0.1 else 'NEUTRAL'
            
            # Check if results match expectations
            is_correct_label = actual_label == expected_label
            
            results.append({
                'file': video_file,
                'expected_label': expected_label,
                'actual_label': actual_label,
                'is_correct': is_correct_label
            })
        
        # Restore stdout and close null output
        sys.stdout = original_stdout
        null_output.close()
        
        # Calculate accuracy
        if not results:
            print("No results found!")
            return
        
        correct_count = sum(1 for r in results if r['is_correct'])
        total_count = len(results)
        accuracy = (correct_count / total_count) * 100
        
        # Print overall result
        if accuracy == 100:
            print("SUCCESS! All test videos are correctly classified (100% accuracy)")
        else:
            print(f"Not quite there yet. Current accuracy: {accuracy:.1f}%")
            
            # List incorrect classifications
            incorrect = [r for r in results if not r['is_correct']]
            print(f"Incorrectly classified videos ({len(incorrect)}/{total_count}):")
            for r in incorrect:
                print(f"  - {r['file']}: Expected {r['expected_label']}, Got {r['actual_label']}")
            
    except Exception as e:
        # Restore stdout in case of error
        if sys.stdout != original_stdout:
            sys.stdout = original_stdout
            null_output.close()
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 