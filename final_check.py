#!/usr/bin/env python3
"""
Final check script that writes results to a file
"""

import sys
import os
import re
import pandas as pd
from video_sentiment_app import SentimentAnalyzer

def main():
    # Redirect output to null
    original_stdout = sys.stdout
    null_output = open(os.devnull, 'w')
    sys.stdout = null_output
    
    try:
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
                'expected_score': expected_score,
                'actual_score': actual_score,
                'is_correct': is_correct_label
            })
    
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = original_stdout
        null_output.close()
        
        # Write error to file
        with open('final_check_results.txt', 'w') as f:
            f.write(f"Error: {str(e)}\n")
        return
    
    # Restore stdout
    sys.stdout = original_stdout
    null_output.close()
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100
    
    # Write results to file
    with open('final_check_results.txt', 'w') as f:
        if accuracy == 100:
            f.write(f"SUCCESS: 100% accuracy achieved ({correct_count}/{total_count} videos)\n\n")
        else:
            f.write(f"FAILURE: {correct_count}/{total_count} videos ({accuracy:.1f}%) classified correctly\n\n")
        
        # Write details for all videos
        f.write("DETAILS:\n")
        for r in results:
            status = "✓" if r['is_correct'] else "✗"
            f.write(f"{r['file']:<15} Expected: {r['expected_label']:>8} ({r['expected_score']:.2f}), " +
                   f"Got: {r['actual_label']:>8} ({r['actual_score']:.2f}) {status}\n")
        
        # List incorrect classifications separately
        incorrect = [r for r in results if not r['is_correct']]
        if incorrect:
            f.write("\nINCORRECT CLASSIFICATIONS:\n")
            for r in incorrect:
                f.write(f"  - {r['file']}: Expected {r['expected_label']}, Got {r['actual_label']}\n")
    
    # Print simple status
    if accuracy == 100:
        print("SUCCESS! All videos classified correctly. See final_check_results.txt for details.")
    else:
        print(f"FAILURE: {accuracy:.1f}% accuracy. See final_check_results.txt for details.")

if __name__ == "__main__":
    main() 