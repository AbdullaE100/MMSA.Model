#!/usr/bin/env python3
"""
Simplified emotion analyzer for analyzing a single test video.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Sentiment weights based on psychological research
SENTIMENT_WEIGHTS = {
    "Angry": -0.8,     # Strong negative
    "Disgust": -0.7,   # Strong negative
    "Fear": -0.6,      # Moderate negative
    "Happy": 0.8,      # Strong positive
    "Sad": -0.5,       # Moderate negative
    "Surprise": 0.3,   # Mild positive
    "Neutral": 0.0     # Neutral
}

def create_model():
    """Create a compatible emotion recognition model"""
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(7, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def analyze_video(video_path, cascade_path):
    """Analyze a video for emotions and calculate sentiment"""
    print(f"Analyzing video: {video_path}")
    print(f"Using cascade file: {cascade_path}")
    
    # Create model
    model = create_model()
    print("Created emotion recognition model")
    
    # Check if model weights exist
    if os.path.exists("./model.h5"):
        try:
            model.load_weights("./model.h5")
            print("Loaded model weights")
        except:
            print("Could not load weights, using untrained model")
    else:
        print("Model weights not found, using untrained model")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load cascade from {cascade_path}")
        return None, None, None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None
    
    # Initialize variables
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    frame_count = 0
    processed_frames = 0
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 5th frame
        if frame_count % 5 != 0:
            continue
            
        processed_frames += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 (expected by model)
            resized_face = cv2.resize(face_roi, (48, 48))
            
            # Normalize
            normalized_face = resized_face / 255.0
            
            # Reshape for model input
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            
            # Predict emotion
            predictions = model.predict(reshaped_face, verbose=0)[0]
            emotion_index = np.argmax(predictions)
            emotion = EMOTIONS[emotion_index]
            confidence = float(predictions[emotion_index])
            
            # Count emotion
            emotion_counts[emotion] += 1
            
            # Print occasionally
            if processed_frames % 10 == 0:
                print(f"Frame {frame_count}, Face at {x},{y}: {emotion} ({confidence:.2f})")
    
    # Release video
    cap.release()
    
    # Check if emotions were detected
    total_emotions = sum(emotion_counts.values())
    if total_emotions == 0:
        print("No emotions detected in the video")
        return 0.0, {}, None
    
    print(f"Processed {processed_frames} frames, detected {total_emotions} emotions")
    
    # Calculate percentages
    emotion_percentages = {emotion: count/total_emotions * 100 for emotion, count in emotion_counts.items()}
    
    # Calculate weighted sentiment score
    sentiment_score = sum(SENTIMENT_WEIGHTS[emotion] * (percentage / 100) 
                          for emotion, percentage in emotion_percentages.items())
    
    # Find dominant emotion
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    return sentiment_score, emotion_percentages, dominant_emotion

def determine_sentiment_label(score):
    """Convert score to sentiment label"""
    if score > 0.2:
        return "POSITIVE"
    elif score < -0.2:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def get_expected_sentiment(video_name):
    """Get expected sentiment from labels file"""
    labels_path = "./test_videos/labels.csv"
    
    if not os.path.exists(labels_path):
        return None
    
    try:
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if video_name == row['file']:
                    return float(row['expected_sentiment'])
    except:
        pass
        
    return None

def main():
    # Default video to analyze
    video_name = "Angry.mp4"
    
    # Check if a video name was provided
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    
    # Paths
    video_path = f"./test_videos/{video_name}"
    cascade_path = "./haarcascade_frontalface_default.xml"
    
    # Check files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
        
    if not os.path.exists(cascade_path):
        print(f"Error: Cascade file {cascade_path} not found")
        return
    
    # Analyze video
    sentiment_score, emotion_percentages, dominant_emotion = analyze_video(video_path, cascade_path)
    
    if sentiment_score is None:
        print("Analysis failed")
        return
    
    # Determine sentiment label
    sentiment_label = determine_sentiment_label(sentiment_score)
    
    # Print results
    print("\nResults:")
    print(f"Dominant Emotion: {dominant_emotion}")
    print(f"Sentiment Score: {sentiment_score:.4f}")
    print(f"Sentiment: {sentiment_label}")
    
    print("\nEmotion Distribution:")
    for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
        if percentage > 0:
            print(f"  {emotion}: {percentage:.1f}%")
    
    # Get expected sentiment
    expected_sentiment = get_expected_sentiment(video_name)
    
    if expected_sentiment is not None:
        expected_label = determine_sentiment_label(expected_sentiment)
        print(f"\nExpected Sentiment: {expected_sentiment:.4f} ({expected_label})")
        
        # Check accuracy
        is_correct = (
            sentiment_label == expected_label or 
            abs(sentiment_score - expected_sentiment) < 0.2
        )
        
        print(f"Accuracy: {'✓ Correct' if is_correct else '✗ Incorrect'}")

if __name__ == "__main__":
    main() 