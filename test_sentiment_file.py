#!/usr/bin/env python3
"""
Test script to verify the sentiment analysis on test videos and write results to a file
"""

import sys
import os
import re
import pandas as pd
from video_sentiment_app import SentimentAnalyzer
from improved_emotion_analyzer_final import determine_sentiment_label

def main():
    # Redirect all output to a file
    output_file = 'test_results.txt'
    with open(output_file, 'w') as f_out:
        try:
            # Function to write to both file and screen
            def write_output(text):
                f_out.write(text + '\n')
                
            # Load expected labels
            labels_df = pd.read_csv('test_videos/labels.csv')
            expected = {row['file']: row['expected_sentiment'] for _, row in labels_df.iterrows()}
            write_output('Loaded expected labels for the following videos:')
            for video, score in expected.items():
                write_output(f'  - {video}: {score}')
            write_output('')

            # Initialize analyzer
            analyzer = SentimentAnalyzer()

            # Test each video
            results = []
            for video_file, expected_score in expected.items():
                video_path = os.path.join('test_videos', video_file)
                write_output(f'Testing {video_file}...')
                
                # Redirect stdout to capture the detailed analysis
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                try:
                    # Run the analysis
                    result = analyzer.analyze_sentiment(video_path)
                    
                    # Restore stdout
                    sys.stdout.close()
                    sys.stdout = original_stdout
                    
                    if result is None:
                        write_output(f'  Error: Analysis failed for {video_file}')
                        continue

                    # Extract overall sentiment
                    overall_formatted, mmsa_formatted, emotion_formatted, text_formatted, viz_path, transcript, emotion_html, emotion_percentages, dominant_emotion = result
                    
                    # Parse the overall sentiment score
                    match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
                    if match:
                        actual_score = float(match.group(1))
                    else:
                        write_output(f'  Error: Could not parse sentiment score from {overall_formatted}')
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
                    
                    # Write video results
                    write_output(f'  Result: {video_file}')
                    write_output(f'  Expected: {expected_label} ({expected_score:.2f}), Actual: {actual_label} ({actual_score:.2f})')
                    write_output(f'  Correct label: {is_correct_label}, Score difference: {score_diff:.2f}')
                    write_output(f'  Dominant emotion: {dominant_emotion}')
                    
                    # Write top emotions
                    write_output('  Top emotions:')
                    for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
                        if percentage > 5:  # Only show emotions with > 5% presence
                            write_output(f'    - {emotion}: {percentage:.1f}%')
                    
                    write_output('')  # Empty line between videos
                    
                except Exception as e:
                    # Restore stdout in case of error
                    if sys.stdout != original_stdout:
                        sys.stdout.close()
                        sys.stdout = original_stdout
                    write_output(f'  Error analyzing {video_file}: {e}')

            # Print summary
            write_output('\nSummary:')
            correct_count = sum(1 for r in results if r['is_correct'])
            write_output(f'Correctly classified: {correct_count}/{len(results)} ({100 * correct_count / len(results):.1f}%)')

            # Print detailed results sorted by correctness
            write_output('\nDetailed results:')
            sorted_results = sorted(results, key=lambda r: (not r['is_correct'], r['score_diff']))
            for r in sorted_results:
                correct_mark = '✓' if r['is_correct'] else '✗'
                write_output(f"{r['file']:<12} Expected: {r['expected_label']:>8} ({r['expected_score']:.2f}), Actual: {r['actual_label']:>8} ({r['actual_score']:.2f}) - {correct_mark}")
                
            # Screen output of completion
            print(f"Test complete! Results written to {output_file}")
            
        except Exception as e:
            f_out.write(f"Error during testing: {e}\n")
            print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 