#!/usr/bin/env python3
"""
Simple script to launch the Gradio interface
"""

import gradio as gr
from improved_emotion_analyzer_final import analyze_video, determine_sentiment_label
import os

class SimpleAnalyzer:
    def __init__(self):
        self.cascade_path = "./haarcascade_frontalface_default.xml"
        
    def analyze_sentiment(self, video_path):
        """Analyze video for sentiment"""
        if not video_path:
            return "Please upload a video file.", "", ""
            
        # Get the video name
        video_name = os.path.basename(video_path)
        
        # Run analysis
        sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
            video_path, self.cascade_path, video_name
        )
        
        if sentiment_score is None:
            return "Analysis failed. Please try a different video.", "", ""
        
        # Determine sentiment
        sentiment = determine_sentiment_label(sentiment_score)
        
        # Format emotion distribution
        emotion_html = "<ul style='list-style-type:none; padding:0;'>"
        for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0.5:
                emotion_html += f"<li><b>{emotion}</b>: {percentage:.1f}%</li>"
        emotion_html += "</ul>"
        
        return f"{sentiment} (Score: {sentiment_score:.2f})", f"Dominant emotion: {dominant_emotion}", emotion_html

# Create a simple interface
def create_interface():
    analyzer = SimpleAnalyzer()
    
    demo = gr.Interface(
        fn=analyzer.analyze_sentiment,
        inputs=gr.Video(label="Upload Video"),
        outputs=[
            gr.Textbox(label="Sentiment"),
            gr.Textbox(label="Dominant Emotion"),
            gr.HTML(label="Emotion Distribution")
        ],
        title="Video Emotion Analyzer",
        description="Upload a video to analyze its emotional content.",
        examples=[
            ["./test_videos/sad.mp4"],
            ["./test_videos/f.mp4"],
            ["./test_videos/lol.mp4"]
        ],
        allow_flagging="never",
        theme=gr.themes.Soft()
    )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    public_url = demo.launch(share=True)
    print(f"Gradio public URL: {public_url}") 