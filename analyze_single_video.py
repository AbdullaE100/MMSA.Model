#!/usr/bin/env python3
"""
Simple script to analyze a single video's emotions and sentiment.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Weighted sentiment mapping
SENTIMENT_WEIGHTS = {
    "Angry": -0.8,     # Strong negative
    "Disgust": -0.7,   # Strong negative
    "Fear": -0.6,      # Moderate negative
    "Happy": 0.8,      # Strong positive
    "Sad": -0.5,       # Moderate negative
    "Surprise": 0.3,   # Mild positive
    "Neutral": 0.0     # Neutral
}

def main():
    # Load model
    try:
        json_file = open('./model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("./model.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Load face cascade
    try:
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        print("Face cascade loaded successfully")
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        sys.exit(1)
    
    # Load video
    video_path = './test_videos/Angry.mp4'  # You can change this to any video path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    print(f"Analyzing video: {video_path}")
    
    # Store emotion counts
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    frame_count = 0
    processed_frames = 0
    
    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Process every 5th frame for speed
        if frame_count % 5 != 0:
            continue
            
        processed_frames += 1
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            
            # Predict emotion
            emotion_probs = model.predict(reshaped_face, verbose=0)[0]
            emotion_index = np.argmax(emotion_probs)
            emotion = EMOTIONS[emotion_index]
            confidence = emotion_probs[emotion_index]
            
            # Store result
            emotion_counts[emotion] += 1
            
            # Print result for debugging
            if processed_frames % 10 == 0:
                print(f"Frame {frame_count}, Face at {x},{y}: {emotion} ({confidence:.2f})")
    
    cap.release()
    
    # Calculate results
    total_emotions = sum(emotion_counts.values())
    if total_emotions == 0:
        print("No faces or emotions detected in the video")
        return
        
    print(f"\nProcessed {processed_frames} frames, detected {total_emotions} faces/emotions")
    
    # Calculate emotion percentages
    emotion_percentages = {emotion: count/total_emotions * 100 for emotion, count in emotion_counts.items()}
    
    # Print emotion distribution
    print("\nEmotion Distribution:")
    for emotion, percentage in emotion_percentages.items():
        if percentage > 0:
            print(f"{emotion}: {percentage:.1f}%")
    
    # Calculate weighted sentiment score
    overall_sentiment_score = sum(SENTIMENT_WEIGHTS[emotion] * (count/total_emotions) 
                                 for emotion, count in emotion_counts.items())
    
    # Determine overall sentiment label
    if overall_sentiment_score > 0.2:
        overall_sentiment = "POSITIVE"
    elif overall_sentiment_score < -0.2:
        overall_sentiment = "NEGATIVE"
    else:
        overall_sentiment = "NEUTRAL"
    
    print(f"\nOverall Sentiment Score: {overall_sentiment_score:.4f}")
    print(f"Overall Sentiment: {overall_sentiment}")

if __name__ == "__main__":
    main() 