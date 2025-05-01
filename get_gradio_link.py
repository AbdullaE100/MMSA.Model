#!/usr/bin/env python3
"""
Script to get the Gradio public link
"""

import time
import sys
from video_sentiment_app import demo

if __name__ == "__main__":
    # Save original stdout to restore later
    original_stdout = sys.stdout
    
    # Redirect stdout to capture the Gradio link
    with open('gradio_link.txt', 'w') as f:
        sys.stdout = f
        # Start Gradio in a separate thread
        demo.queue().launch(share=True, prevent_thread_lock=True)
        # Wait a moment for the link to be generated
        time.sleep(10)
    
    # Restore original stdout
    sys.stdout = original_stdout
    
    # Print the link for the user
    with open('gradio_link.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'gradio.live' in line:
                print(line.strip()) 