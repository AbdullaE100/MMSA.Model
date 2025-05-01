#!/usr/bin/env python3
"""
Simplified test to check for 100% accuracy 
"""

import os
import re
import sys
import pandas as pd
from video_sentiment_app import SentimentAnalyzer

# Suppress output
class NullOutput:
    def write(self, *args, **kwargs):
        pass
    def flush(self):
        pass

def main():
    # Test if we have 100% accuracy
    original_stdout = sys.stdout
    sys.stdout = NullOutput()

    try:
        # Load expected labels
        labels_df = pd.read_csv('test_videos/labels.csv')
        expected = {row['file']: row['expected_sentiment'] for _, row in labels_df.iterrows()}

        # Initialize analyzer
        analyzer = SentimentAnalyzer()

        # Test each video
        incorrect = []
        for video_file, expected_score in expected.items():
            video_path = os.path.join('test_videos', video_file)
            
            # Run the analysis
            result = analyzer.analyze_sentiment(video_path)
            
            if result is None:
                incorrect.append(f"{video_file}: Analysis failed")
                continue

            # Extract overall sentiment
            overall_formatted = result[0]
            
            # Parse the overall sentiment score
            match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
            if not match:
                incorrect.append(f"{video_file}: Could not parse score")
                continue
            
            actual_score = float(match.group(1))
            
            # Extract the sentiment label
            label_match = re.match(r'([A-Z]+)', overall_formatted)
            actual_label = label_match.group(1) if label_match else 'UNKNOWN'
            
            # Determine expected label
            expected_label = 'POSITIVE' if expected_score >= 0.1 else 'NEGATIVE' if expected_score <= -0.1 else 'NEUTRAL'
            
            # Check if results match expectations
            if actual_label != expected_label:
                incorrect.append(f"{video_file}: Expected {expected_label}, Got {actual_label}")
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Print success or list incorrect videos
        if not incorrect:
            print("SUCCESS! All test videos are correctly classified")
        else:
            print(f"Not quite there yet. {len(incorrect)}/{len(expected)} videos incorrectly classified:")
            for error in incorrect:
                print(f"  - {error}")
                
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = original_stdout
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 