#!/usr/bin/env python3
"""
Simplified test script to verify the sentiment analysis on f.mp4
"""

from video_sentiment_app import SentimentAnalyzer
import os
import sys

# Redirect stdout to suppress detailed output
class DummyOutput:
    def write(self, text):
        pass
    def flush(self):
        pass

def main():
    # Suppress output during analysis
    original_stdout = sys.stdout
    sys.stdout = DummyOutput()
    
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer()

        # Test the f.mp4 video
        video_file = 'f.mp4'
        video_path = os.path.join('test_videos', video_file)
        
        # Run the analysis
        result = analyzer.analyze_sentiment(video_path)
        
        # Restore stdout for results output
        sys.stdout = original_stdout
        
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
        
        # Print just the key results
        print(f'F.MP4 ANALYSIS RESULT:')
        print(f'Sentiment: {actual_label} ({actual_score:.2f})')
        print(f'Dominant emotion: {dominant_emotion}')
        print(f'Top emotions: {", ".join([f"{e}: {p:.1f}%" for e, p in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True) if p > 5])}')
    
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = original_stdout
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 