#!/usr/bin/env python3
"""
Script to check accuracy and write results to a file
"""

import sys
import os
import re
import pandas as pd
from video_sentiment_app import SentimentAnalyzer

def main():
    # File to write results to
    result_file = 'accuracy_results.txt'
    
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
        video_results = []
        
        for video_file, expected_score in expected.items():
            video_path = os.path.join('test_videos', video_file)
            
            # Run the analysis
            result = analyzer.analyze_sentiment(video_path)
            
            if result is None:
                video_results.append(f"{video_file}: Analysis failed")
                continue

            # Extract overall sentiment
            overall_formatted, _, _, _, _, _, _, emotion_percentages, dominant_emotion = result
            
            # Parse the overall sentiment score
            match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
            if not match:
                video_results.append(f"{video_file}: Could not parse sentiment score")
                continue
            
            actual_score = float(match.group(1))
            
            # Extract the sentiment label
            label_match = re.match(r'([A-Z]+)', overall_formatted)
            actual_label = label_match.group(1) if label_match else 'UNKNOWN'
            
            # Determine expected label
            expected_label = 'POSITIVE' if expected_score >= 0.1 else 'NEGATIVE' if expected_score <= -0.1 else 'NEUTRAL'
            
            # Check if results match expectations
            is_correct_label = actual_label == expected_label
            
            # Format result for this video
            status = "✓" if is_correct_label else "✗"
            result_text = (f"{video_file:<12} Expected: {expected_label:>8} ({expected_score:.2f}), "
                        f"Actual: {actual_label:>8} ({actual_score:.2f}) - {status}")
            
            # Get top emotions
            top_emotions = ", ".join([f"{e}: {p:.1f}%" for e, p in 
                                   sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True) 
                                   if p > 5])
            
            video_results.append(f"{result_text} | Dominant: {dominant_emotion} | {top_emotions}")
            
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
            with open(result_file, 'w') as f:
                f.write("No results found!\n")
            return
        
        correct_count = sum(1 for r in results if r['is_correct'])
        total_count = len(results)
        accuracy = (correct_count / total_count) * 100
        
        # Write results to file
        with open(result_file, 'w') as f:
            # Write overall accuracy
            if accuracy == 100:
                f.write("SUCCESS! All test videos are correctly classified (100% accuracy)\n\n")
            else:
                f.write(f"Current accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)\n\n")
            
            # Write detailed results for each video
            f.write("DETAILED RESULTS:\n")
            for result in video_results:
                f.write(f"{result}\n")
            
            # List incorrect classifications separately
            incorrect = [r for r in results if not r['is_correct']]
            if incorrect:
                f.write(f"\nINCORRECTLY CLASSIFIED VIDEOS ({len(incorrect)}/{total_count}):\n")
                for r in incorrect:
                    f.write(f"  - {r['file']}: Expected {r['expected_label']}, Got {r['actual_label']}\n")
        
        # Also print to console
        print(f"Test results written to {result_file}")
        
    except Exception as e:
        # Restore stdout in case of error
        if sys.stdout != original_stdout:
            sys.stdout = original_stdout
            null_output.close()
        
        # Write error to file
        with open(result_file, 'w') as f:
            f.write(f"Error during testing: {e}\n")
        
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 