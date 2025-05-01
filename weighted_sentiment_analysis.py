#!/usr/bin/env python3
"""
Sentiment Analysis using Video-Sentiment-Analysis repository with weighted emotion scores.
This script analyzes videos for emotions and converts them to weighted sentiment scores.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from collections import Counter

# Emotion labels from the Video-Sentiment-Analysis repository
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Weighted sentiment mapping (can be adjusted based on psychological research)
SENTIMENT_WEIGHTS = {
    "Angry": -0.8,     # Strong negative
    "Disgust": -0.7,   # Strong negative
    "Fear": -0.6,      # Moderate negative
    "Happy": 0.8,      # Strong positive
    "Sad": -0.5,       # Moderate negative
    "Surprise": 0.3,   # Mild positive (surprise can be positive or negative, but slightly more positive)
    "Neutral": 0.0     # Neutral
}

def load_emotion_model(model_json='model.json', model_weights='model.h5'):
    """Load the facial emotion recognition model from the repository"""
    try:
        # Load model from json file
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        
        # Load weights into the model
        model.load_weights(model_weights)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def preprocess_face(face):
    """Preprocess the face image for the model"""
    # Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize to 48x48 which is what the model expects
    resized = cv2.resize(gray, (48, 48))
    
    # Normalize the image
    normalized = resized / 255.0
    
    # Reshape for model input
    reshaped = normalized.reshape(1, 48, 48, 1)
    
    return reshaped

def analyze_video(video_path, model, face_cascade_path='haarcascade_frontalface_default.xml'):
    """
    Analyze video for facial emotions and calculate weighted sentiment scores
    Returns a dictionary with overall scores and frame-by-frame analysis
    """
    # Check if the face cascade file exists
    if os.path.exists(face_cascade_path):
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        use_cascade = True
    else:
        print(f"Warning: Face cascade file {face_cascade_path} not found.")
        print("Using simple frame-based analysis without face detection.")
        use_cascade = False
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    frame_results = []
    all_emotions = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame to improve performance
            continue
        
        frame_emotions = []
        
        if use_cascade:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    continue
                    
                # Preprocess face for emotion recognition
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion
                emotion_predictions = model.predict(processed_face)[0]
                emotion_index = np.argmax(emotion_predictions)
                emotion_label = EMOTIONS[emotion_index]
                confidence = emotion_predictions[emotion_index]
                
                # Calculate sentiment score
                sentiment_score = SENTIMENT_WEIGHTS[emotion_label] * confidence
                
                # Store result for this face
                face_result = {
                    "emotion": emotion_label,
                    "confidence": float(confidence),
                    "sentiment_score": float(sentiment_score),
                    "position": (x, y, w, h)
                }
                
                frame_emotions.append(face_result)
                all_emotions.append(emotion_label)
        else:
            # Fallback: Process the entire frame as a single face
            # Resize the frame to a reasonable size
            resized_frame = cv2.resize(frame, (224, 224))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to 48x48 which is what the model expects
            resized = cv2.resize(gray, (48, 48))
            
            # Normalize and reshape
            normalized = resized / 255.0
            processed_input = normalized.reshape(1, 48, 48, 1)
            
            # Predict emotion
            emotion_predictions = model.predict(processed_input)[0]
            emotion_index = np.argmax(emotion_predictions)
            emotion_label = EMOTIONS[emotion_index]
            confidence = emotion_predictions[emotion_index]
            
            # Calculate sentiment score
            sentiment_score = SENTIMENT_WEIGHTS[emotion_label] * confidence
            
            # Store result for this frame
            face_result = {
                "emotion": emotion_label,
                "confidence": float(confidence),
                "sentiment_score": float(sentiment_score),
                "position": (0, 0, resized_frame.shape[1], resized_frame.shape[0])
            }
            
            frame_emotions.append(face_result)
            all_emotions.append(emotion_label)
        
        # If emotions found in this frame, store the results
        if frame_emotions:
            frame_result = {
                "frame_number": frame_count,
                "faces": frame_emotions,
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # time in seconds
            }
            frame_results.append(frame_result)
    
    cap.release()
    
    # If no emotions detected, return None
    if not all_emotions:
        print(f"No faces/emotions detected in {video_path}")
        return None
    
    # Calculate overall statistics
    emotion_counts = Counter(all_emotions)
    total_emotions = len(all_emotions)
    
    # Calculate weighted emotion scores
    emotion_percentages = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}
    
    # Calculate overall sentiment score (weighted average)
    overall_sentiment_score = sum(SENTIMENT_WEIGHTS[emotion] * percentage 
                                 for emotion, percentage in emotion_percentages.items())
    
    # Determine overall sentiment label
    if overall_sentiment_score > 0.2:
        overall_sentiment = "POSITIVE"
    elif overall_sentiment_score < -0.2:
        overall_sentiment = "NEGATIVE"
    else:
        overall_sentiment = "NEUTRAL"
    
    # Format results
    result = {
        "video_name": Path(video_path).stem,
        "frame_count": frame_count,
        "faces_detected": sum(len(frame["faces"]) for frame in frame_results),
        "emotion_distribution": {emotion: {"count": count, 
                                          "percentage": count/total_emotions * 100} 
                                for emotion, count in emotion_counts.items()},
        "emotion_weighted_scores": {emotion: SENTIMENT_WEIGHTS[emotion] * emotion_percentages.get(emotion, 0) 
                                    for emotion in EMOTIONS},
        "overall_sentiment_score": overall_sentiment_score,
        "overall_sentiment": overall_sentiment,
        "frame_results": frame_results
    }
    
    return result

def plot_emotion_distribution(result):
    """Generate a pie chart of emotion distribution"""
    emotions = []
    counts = []
    
    for emotion, data in result["emotion_distribution"].items():
        emotions.append(emotion)
        counts.append(data["count"])
    
    plt.figure(figsize=(10, 6))
    colors = ['red', 'purple', 'darkblue', 'green', 'gray', 'orange', 'lightblue']
    
    plt.pie(counts, labels=emotions, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Emotion Distribution for {result["video_name"]}')
    
    output_file = f'{result["video_name"]}_emotion_distribution.png'
    plt.savefig(output_file)
    plt.close()
    print(f"Emotion distribution saved to {output_file}")
    
    return output_file

def plot_sentiment_timeline(result):
    """Generate a timeline of sentiment scores across the video"""
    if not result["frame_results"]:
        return None
        
    timestamps = []
    sentiment_scores = []
    
    for frame in result["frame_results"]:
        if not frame["faces"]:
            continue
            
        # Average the sentiment scores for all faces in this frame
        avg_score = sum(face["sentiment_score"] for face in frame["faces"]) / len(frame["faces"])
        timestamps.append(frame["timestamp"])
        sentiment_scores.append(avg_score)
    
    if not timestamps:
        return None
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, sentiment_scores, '-o', color='blue', alpha=0.6)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.fill_between(timestamps, sentiment_scores, 0, where=[s > 0 for s in sentiment_scores], 
                     color='green', alpha=0.3, interpolate=True)
    plt.fill_between(timestamps, sentiment_scores, 0, where=[s <= 0 for s in sentiment_scores], 
                     color='red', alpha=0.3, interpolate=True)
    
    plt.title(f'Sentiment Timeline for {result["video_name"]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sentiment Score')
    plt.grid(True, alpha=0.3)
    
    output_file = f'{result["video_name"]}_sentiment_timeline.png'
    plt.savefig(output_file)
    plt.close()
    print(f"Sentiment timeline saved to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Video sentiment analysis with weighted emotion scores')
    parser.add_argument('--video_dir', type=str, default='./test_videos', 
                        help='Directory containing the videos')
    parser.add_argument('--output_dir', type=str, default='./sentiment_results', 
                        help='Directory for output files')
    parser.add_argument('--model_json', type=str, default='model.json',
                        help='Path to model architecture JSON file')
    parser.add_argument('--model_weights', type=str, default='model.h5',
                        help='Path to model weights HDF5 file')
    parser.add_argument('--face_cascade', type=str, default='haarcascade_frontalface_default.xml',
                        help='Path to face detection cascade file')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model files exist
    if not os.path.exists(args.model_json) or not os.path.exists(args.model_weights):
        print(f"Error: Model files not found at {args.model_json} and/or {args.model_weights}")
        print("Please ensure you have run setup_models.py or manually downloaded the required files.")
        return
    
    # Save current directory to restore later
    original_dir = os.getcwd()
    
    # Change to output directory for saving results
    os.chdir(args.output_dir)
    
    try:
        # Load emotion recognition model
        model = load_emotion_model(args.model_json, args.model_weights)
        
        # Find all videos
        video_files = list(Path(args.video_dir).glob('*.mp4'))
        if not video_files:
            print(f"No videos found in {args.video_dir}")
            return
        
        print(f"Found {len(video_files)} videos. Running emotion analysis...")
        
        # Analyze each video
        results = []
        for video_path in video_files:
            print(f"Processing {video_path.name}...")
            try:
                result = analyze_video(str(video_path), model, args.face_cascade)
                if result:
                    results.append(result)
                    
                    # Generate visualizations
                    plot_emotion_distribution(result)
                    plot_sentiment_timeline(result)
                    
                    # Save detailed results to JSON file
                    import json
                    output_file = f'{result["video_name"]}_analysis.json'
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Detailed analysis saved to {output_file}")
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
        
        # Display summary results in a table
        if results:
            from tabulate import tabulate
            
            table_data = []
            headers = ["Video", "Dominant Emotion", "Score", "Sentiment"]
            
            for result in results:
                # Find dominant emotion (highest percentage)
                dominant_emotion = max(result["emotion_distribution"].items(), 
                                      key=lambda x: x[1]["percentage"])[0]
                dominant_percentage = result["emotion_distribution"][dominant_emotion]["percentage"]
                
                video_name = result["video_name"]
                score = f"{result['overall_sentiment_score']:.4f}"
                sentiment = result["overall_sentiment"]
                
                table_data.append([
                    video_name, 
                    f"{dominant_emotion} ({dominant_percentage:.1f}%)", 
                    score, 
                    sentiment
                ])
            
            print("\nSentiment Analysis Results:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Calculate accuracy (if video names indicate emotions)
            positive_emotions = ['happy', 'joy', 'surprise', 'surprised']
            negative_emotions = ['sad', 'angry', 'fear', 'disgust']
            neutral_emotions = ['neutral', 'calm']
            
            correct = 0
            total_with_expected = 0
            
            for result in results:
                video_name = result["video_name"].lower()
                prediction = result["overall_sentiment"]
                
                expected = None
                if any(emotion in video_name for emotion in positive_emotions):
                    expected = "POSITIVE"
                elif any(emotion in video_name for emotion in negative_emotions):
                    expected = "NEGATIVE"
                elif any(emotion in video_name for emotion in neutral_emotions):
                    expected = "NEUTRAL"
                
                if expected:
                    total_with_expected += 1
                    if prediction == expected:
                        correct += 1
            
            if total_with_expected > 0:
                accuracy = (correct / total_with_expected) * 100
                print(f"\nEstimated accuracy: {accuracy:.2f}% ({correct}/{total_with_expected} correct)")
        else:
            print("No results to display")
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 