#!/bin/bash
# Run the quick check and display results
python3 quick_check.py

cd /Users/abdullaehsan/Desktop/FINALproject
python3 repositories/Video-Sentiment-Analysis/test_repo.py --video_path test_videos/Calm.mp4 --output_format json
ls -la outputs/ 