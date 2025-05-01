#!/usr/bin/env python3
"""
Self-contained script for video emotion analysis.
This combines the model loading, video analysis, and evaluation in one file.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from pathlib import Path
import csv

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

class VideoEmotionAnalyzer:
    def __init__(self, model_json_path='./model.json', model_weights_path='./model.h5',
                 cascade_path='./haarcascade_frontalface_default.xml'):
        self.model = self.load_model(model_json_path, model_weights_path)
        self.cascade_path = cascade_path
        
    def load_model(self, model_json_path, model_weights_path):
        """Load emotion recognition model"""
        try:
            # GPU memory growth
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            
            # Print TF version for debugging
            print(f"TensorFlow version: {tf.__version__}")
            
            # Try to load the model architecture from JSON
            try:
                with open(model_json_path, 'r') as json_file:
                    loaded_model_json = json_file.read()
                
                # Try different loading methods
                try:
                    # Method 1: Standard loading
                    model = model_from_json(loaded_model_json)
                except:
                    # Method 2: Try with custom objects
                    try:
                        model = keras.models.model_from_json(loaded_model_json, 
                                                           custom_objects={'tf': tf})
                    except:
                        # Method 3: Create compatible model manually
                        print("Creating a compatible model architecture...")
                        model = self.create_compatible_model()
            except Exception as e:
                print(f"Error loading model JSON: {e}")
                print("Creating a compatible model architecture...")
                model = self.create_compatible_model()
            
            # Load weights if they exist
            if os.path.exists(model_weights_path):
                try:
                    model.load_weights(model_weights_path)
                    print("Model weights loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load weights: {e}")
            else:
                print(f"Warning: Model weights file {model_weights_path} not found")
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model ready")
            return model
            
        except Exception as e:
            print(f"Error setting up model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_compatible_model(self):
        """Create a compatible model architecture"""
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
    
    def preprocess_face(self, face):
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
    
    def analyze_video(self, video_path, threshold=0.4):
        """Analyze video for emotions and calculate sentiment"""
        try:
            # Check if model is loaded
            if self.model is None:
                print("Error: Model not loaded")
                return None, None, None
                
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if face_cascade.empty():
                print(f"Error: Could not load face cascade from {self.cascade_path}")
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
                        reshaped_face = self.preprocess_face(face_roi)
                        predictions = self.model.predict(reshaped_face, verbose=0)[0]
                        
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
                    reshaped_face = self.preprocess_face(face_roi)
                    
                    # Predict emotion
                    predictions = self.model.predict(reshaped_face, verbose=0)[0]
                    
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
        if not os.path.exists(labels_path):
            return None
            
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

def analyze_and_save_results():
    """Analyze all test videos and save results to a file"""
    # Initialize analyzer
    analyzer = VideoEmotionAnalyzer()
    
    # Define video directory and output file
    video_dir = './test_videos'
    output_file = 'improved_emotion_results.txt'
    labels_path = os.path.join(video_dir, 'labels.csv')
    
    # Get all video files
    video_paths = list(Path(video_dir).glob('*.mp4'))
    if not video_paths:
        print(f"No videos found in {video_dir}")
        return
        
    print(f"Found {len(video_paths)} videos to analyze")
    
    # Initialize results
    results = []
    correct_count = 0
    total_count = 0
    
    # Analyze each video
    for video_path in video_paths:
        video_path_str = str(video_path)
        video_name = os.path.basename(video_path_str)
        
        print(f"\n{'='*50}")
        print(f"Analyzing {video_name}")
        print(f"{'='*50}")
        
        # Get expected sentiment if available
        expected_sentiment = get_expected_sentiment(labels_path, video_name)
        
        # Analyze video
        sentiment_score, emotion_percentages, dominant_emotion = analyzer.analyze_video(video_path_str)
        
        if sentiment_score is None:
            result = f"Analysis failed for {video_name}"
            results.append(result)
            continue
            
        # Determine sentiment label
        sentiment_label = determine_sentiment_label(sentiment_score)
        
        # Format result
        result = f"Video: {video_name}\n"
        result += f"Dominant Emotion: {dominant_emotion}\n"
        result += f"Sentiment Score: {sentiment_score:.4f}\n"
        result += f"Sentiment Label: {sentiment_label}\n"
        result += "Emotion Distribution:\n"
        
        # Add emotion percentages
        for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0:
                result += f"  {emotion}: {percentage:.1f}%\n"
        
        # Add expected sentiment if available
        if expected_sentiment is not None:
            expected_label = determine_sentiment_label(expected_sentiment)
            result += f"Expected Sentiment: {expected_sentiment:.4f} ({expected_label})\n"
            
            # Check if our analysis is correct
            is_correct = (
                sentiment_label == expected_label or 
                abs(sentiment_score - expected_sentiment) < 0.2
            )
            result += f"Accuracy: {'✓ Correct' if is_correct else '✗ Incorrect'}\n"
            
            # Count for accuracy calculation
            if is_correct:
                correct_count += 1
            total_count += 1
        
        # Print result to console and add to results list
        print(result)
        results.append(result)
    
    # Calculate overall accuracy
    accuracy = 0
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
    
    # Add summary
    summary = f"\n{'='*50}\n"
    summary += f"Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)\n"
    summary += f"{'='*50}\n"
    
    # Print summary
    print(summary)
    results.append(summary)
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write('\n\n'.join(results))
        
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    analyze_and_save_results() 