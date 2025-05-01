#!/usr/bin/env python3
"""
Minimal test script that writes results to a file
"""

from video_sentiment_app import SentimentAnalyzer
import os
import sys
import re

def main():
    # Open a file for writing results
    with open('f_mp4_result.txt', 'w') as f:
        try:
            # Initialize analyzer
            analyzer = SentimentAnalyzer()

            # Test the f.mp4 video
            video_file = 'f.mp4'
            video_path = os.path.join('test_videos', video_file)
            
            f.write(f'Analyzing {video_file}...\n')
            
            # Redirect stdout to suppress detailed output
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            # Run the analysis
            result = analyzer.analyze_sentiment(video_path)
            
            # Restore stdout
            sys.stdout.close()
            sys.stdout = old_stdout
            
            if result is None:
                f.write(f'Error: Analysis failed for {video_file}\n')
                return
            
            # Extract overall sentiment
            overall_formatted, mmsa_formatted, emotion_formatted, text_formatted, viz_path, transcript, emotion_html, emotion_percentages, dominant_emotion = result
            
            # Parse the overall sentiment score
            match = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)', overall_formatted)
            if match:
                actual_score = float(match.group(1))
            else:
                f.write(f'Error: Could not parse sentiment score from {overall_formatted}\n')
                return
            
            # Extract the sentiment label
            label_match = re.match(r'([A-Z]+)', overall_formatted)
            actual_label = label_match.group(1) if label_match else 'UNKNOWN'
            
            # Write the detailed results
            f.write(f'\nResults for f.mp4:\n')
            f.write(f'Sentiment: {actual_label} ({actual_score:.2f})\n')
            f.write(f'MMSA sentiment: {mmsa_formatted}\n')
            f.write(f'Primary emotion sentiment: {emotion_formatted}\n')
            f.write(f'Text sentiment: {text_formatted}\n')
            f.write(f'Dominant emotion: {dominant_emotion}\n')
            
            # Write emotion percentages
            f.write('\nEmotion Percentages:\n')
            for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
                if percentage > 0:
                    f.write(f'{emotion}: {percentage:.1f}%\n')
            
            f.write(f'\nTranscript: {transcript}\n')
            
            # Also print to console that the analysis is complete
            print(f"Analysis complete. Results written to f_mp4_result.txt")
        
        except Exception as e:
            f.write(f"Error during analysis: {e}\n")
            print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 