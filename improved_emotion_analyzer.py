#!/usr/bin/env python3
"""
Improved emotion analyzer that fixes model compatibility issues and improves accuracy.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import argparse
from pathlib import Path
import csv

# Register any custom objects
tf.keras.utils.get_custom_objects().update({})

# Emotion labels - ensure these match the model's output order
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Weighted sentiment mapping based on psychological research
SENTIMENT_WEIGHTS = {
    "Angry": -0.8,     # Strong negative
    "Disgust": -0.7,   # Strong negative
    "Fear": -0.6,      # Moderate negative
    "Happy": 0.8,      # Strong positive
    "Sad": -0.5,       # Moderate negative
    "Surprise": 0.3,   # Mild positive
    "Neutral": 0.0     # Neutral
}

def load_model(model_json_path, model_weights_path):
    """Load the model with proper error handling and version compatibility"""
    try:
        # Set memory growth to avoid OOM errors
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        print(f"TensorFlow version: {tf.__version__}")
        
        # Load the model architecture
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
        
        # Try different loading strategies
        try:
            # First try normal loading
            model = model_from_json(loaded_model_json)
        except:
            # If that fails, try the keras.saving method
            try:
                print("Standard loading failed, trying alternative method...")
                model = keras.models.model_from_json(loaded_model_json, 
                                                   custom_objects={'tf': tf})
            except:
                # If all else fails, build a compatible model
                print("Creating a compatible model architecture...")
                model = create_compatible_model()
        
        # Load weights
        model.load_weights(model_weights_path)
        print("Model loaded successfully")
        
        # Compile the model (required for some TF versions)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_compatible_model():
    """Create a model with the same architecture as the one we're trying to load"""
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
    return model

def preprocess_face(face):
    """Preprocess face for emotion detection"""
    # Convert to grayscale if not already
    if len(face.shape) == 3 and face.shape[2] == 3:
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face
        
    # Resize to the model's expected input size
    resized_face = cv2.resize(gray_face, (48, 48))
    
    # Apply histogram equalization to enhance contrast
    equalized_face = cv2.equalizeHist(resized_face)
    
    # Normalize pixel values
    normalized_face = equalized_face / 255.0
    
    # Reshape for model input
    reshaped_face = normalized_face.reshape(1, 48, 48, 1)
    
    return reshaped_face

def analyze_video(model, video_path, face_cascade_path, threshold=0.5):
    """
    Analyze video emotions with improved accuracy
    
    Args:
        model: The loaded Keras model
        video_path: Path to the video file
        face_cascade_path: Path to the Haar cascade file
        threshold: Confidence threshold for emotion detection
        
    Returns:
        overall_sentiment_score, emotion_percentages, dominant_emotion
    """
    try:
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            print(f"Error: Could not load face cascade from {face_cascade_path}")
            return None, None, None
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None, None
            
        # Initialize variables
        emotion_counts = {emotion: 0 for emotion in EMOTIONS}
        emotion_confidences = {emotion: [] for emotion in EMOTIONS}
        frame_count = 0
        processed_frames = 0
        
        print(f"Analyzing video: {video_path}")
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 3rd frame (more frequent than before for better accuracy)
            if frame_count % 3 != 0:
                continue
                
            processed_frames += 1
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no faces detected in this frame, try more aggressive detection
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            # If still no faces, process the entire frame as one face
            if len(faces) == 0 and processed_frames % 10 == 0:  # Do this occasionally
                # Resize frame to a reasonable size
                resized_frame = cv2.resize(frame, (120, 120))
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                
                # Use center region as face approximation
                h, w = gray_frame.shape
                center_y, center_x = h // 2, w // 2
                face_size = min(h, w) // 2
                x1, y1 = max(0, center_x - face_size), max(0, center_y - face_size)
                x2, y2 = min(w, center_x + face_size), min(h, center_y + face_size)
                
                face_roi = gray_frame[y1:y2, x1:x2]
                
                # Only proceed if we have a valid ROI
                if face_roi.size > 0:
                    # Preprocess and predict
                    reshaped_face = preprocess_face(face_roi)
                    predictions = model.predict(reshaped_face, verbose=0)[0]
                    
                    # Get dominant emotion
                    emotion_index = np.argmax(predictions)
                    emotion = EMOTIONS[emotion_index]
                    confidence = float(predictions[emotion_index])
                    
                    # Only count if confidence is above threshold
                    if confidence > threshold:
                        emotion_counts[emotion] += 1
                        emotion_confidences[emotion].append(confidence)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Only process if we have a valid ROI
                if face_roi.size == 0:
                    continue
                
                # Preprocess face
                reshaped_face = preprocess_face(face_roi)
                
                # Predict emotion
                predictions = model.predict(reshaped_face, verbose=0)[0]
                
                # Get emotion with highest probability
                emotion_index = np.argmax(predictions)
                emotion = EMOTIONS[emotion_index]
                confidence = float(predictions[emotion_index])
                
                # Only count emotions with confidence above threshold
                if confidence > threshold:
                    emotion_counts[emotion] += 1
                    emotion_confidences[emotion].append(confidence)
                    
                    # Print progress occasionally
                    if processed_frames % 10 == 0 and len(faces) > 0:
                        print(f"Frame {frame_count}, Face at {x},{y}: {emotion} ({confidence:.2f})")
        
        # Release resources
        cap.release()
        
        # Check if emotions were detected
        total_emotions = sum(emotion_counts.values())
        if total_emotions == 0:
            print("No emotions detected with sufficient confidence")
            return 0.0, {}, "Unknown"
            
        print(f"\nProcessed {processed_frames} frames, detected {total_emotions} faces/emotions")
        
        # Calculate emotion percentages with confidence weighting
        emotion_percentages = {}
        for emotion in EMOTIONS:
            # Use both count and average confidence for better accuracy
            count = emotion_counts[emotion]
            if count > 0:
                avg_confidence = sum(emotion_confidences[emotion]) / count
                # Weight by both frequency and confidence
                emotion_percentages[emotion] = (count / total_emotions) * avg_confidence * 100
            else:
                emotion_percentages[emotion] = 0.0
        
        # Normalize percentages to sum to 100%
        total_percentage = sum(emotion_percentages.values())
        if total_percentage > 0:
            for emotion in emotion_percentages:
                emotion_percentages[emotion] = (emotion_percentages[emotion] / total_percentage) * 100
        
        # Calculate weighted sentiment score using the improved percentages
        overall_sentiment_score = sum(SENTIMENT_WEIGHTS[emotion] * (emotion_percentages[emotion] / 100)
                                     for emotion in EMOTIONS)
        
        # Find dominant emotion
        dominant_emotion = max(emotion_percentages.items(), key=lambda x: x[1])[0]
        
        return overall_sentiment_score, emotion_percentages, dominant_emotion
        
    except Exception as e:
        print(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def get_expected_sentiment(labels_path, video_name):
    """Get the expected sentiment from labels file"""
    try:
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if os.path.basename(video_name) == row['file']:
                    return float(row['expected_sentiment'])
        return None
    except Exception as e:
        print(f"Error reading labels: {e}")
        return None

def determine_sentiment_label(score):
    """Convert sentiment score to label"""
    if score > 0.2:
        return "POSITIVE"
    elif score < -0.2:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Improved video emotion analysis')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--video_dir', type=str, default='./test_videos', 
                       help='Directory containing videos to analyze')
    parser.add_argument('--model_json', type=str, default='./model.json',
                       help='Path to model JSON file')
    parser.add_argument('--model_weights', type=str, default='./model.h5',
                       help='Path to model weights file')
    parser.add_argument('--cascade', type=str, default='./haarcascade_frontalface_default.xml',
                       help='Path to Haar cascade file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for emotion detection')
    parser.add_argument('--all', action='store_true', help='Analyze all videos in directory')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_json, args.model_weights)
    if model is None:
        sys.exit(1)
    
    # Labels file for expected sentiment
    labels_path = os.path.join(args.video_dir, 'labels.csv')
    
    # Analyze single video or all videos
    if args.all or args.video is None:
        video_paths = list(Path(args.video_dir).glob('*.mp4'))
        if not video_paths:
            print(f"No videos found in {args.video_dir}")
            sys.exit(1)
    else:
        video_paths = [args.video]
    
    results = []
    correct_count = 0
    total_count = 0
    
    # Analyze each video
    for video_path in video_paths:
        video_path_str = str(video_path)
        print(f"\n{'='*50}")
        print(f"Analyzing {os.path.basename(video_path_str)}")
        print(f"{'='*50}")
        
        # Get expected sentiment if available
        expected_sentiment = get_expected_sentiment(labels_path, os.path.basename(video_path_str))
        
        # Analyze video
        sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
            model, video_path_str, args.cascade, args.threshold
        )
        
        if sentiment_score is None:
            print(f"Analysis failed for {video_path_str}")
            continue
        
        # Determine sentiment label
        sentiment_label = determine_sentiment_label(sentiment_score)
        
        # Print emotion distribution
        print("\nEmotion Distribution:")
        for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0:
                print(f"{emotion}: {percentage:.1f}%")
        
        # Print sentiment results
        print(f"\nDominant Emotion: {dominant_emotion}")
        print(f"Overall Sentiment Score: {sentiment_score:.4f}")
        print(f"Overall Sentiment: {sentiment_label}")
        
        # Check against expected sentiment if available
        if expected_sentiment is not None:
            expected_label = determine_sentiment_label(expected_sentiment)
            is_correct = (
                # Either the sentiment labels match
                sentiment_label == expected_label or
                # Or the scores are within 0.2 of each other
                abs(sentiment_score - expected_sentiment) < 0.2
            )
            
            print(f"Expected Sentiment: {expected_sentiment:.4f} ({expected_label})")
            print(f"Accuracy: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
        # Store result
        results.append({
            'video': os.path.basename(video_path_str),
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'dominant_emotion': dominant_emotion,
            'emotion_percentages': emotion_percentages,
            'expected_sentiment': expected_sentiment
        })
    
    # Print summary of results
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\n{'='*50}")
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)")
        print(f"{'='*50}")

if __name__ == "__main__":
    main() 