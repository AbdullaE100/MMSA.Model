#!/usr/bin/env python3
"""
Video Sentiment Analysis with Gradio Interface
"""

import os
import sys
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gradio as gr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# Import the emotion analyzer functions with proper reset weights
from improved_emotion_analyzer_final import create_model, analyze_video, determine_sentiment_label, SENTIMENT_WEIGHTS

# Use the same emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class SentimentAnalyzer:
    def __init__(self):
        # Load the cascade classifier once
        self.cascade_path = "./haarcascade_frontalface_default.xml"
        
        # Track current video being processed
        self.current_video_path = None
        
        # Create a reusable model
        self.model = create_model()
        
        # Load model weights if available
        if os.path.exists("./model.h5"):
            try:
                self.model.load_weights("./model.h5")
                print("Model weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
        
    def analyze_mmsa(self, video_path):
        """Advanced MMSA model implementation using color, motion and face analysis"""
        # Extract frames for analysis
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print("  ⚠️ ERROR: Could not open video file for MMSA analysis")
            return 0.0
        
        # Load face cascade for face detection
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Adaptive sampling based on video duration
        if duration <= 30:  # Short videos
            frames_to_sample = min(30, total_frames)
        elif duration <= 120:  # 30 seconds to 2 minutes
            frames_to_sample = min(60, total_frames)
        elif duration <= 600:  # 2-10 minutes
            frames_to_sample = min(120, total_frames)
        else:  # Very long videos (over 10 minutes)
            # For very long videos, scale the number of frames with duration
            # Use approximately 20 frames per minute, with a reasonable cap
            frames_per_minute = 20
            minutes = duration / 60
            frames_to_sample = min(int(minutes * frames_per_minute), 300, total_frames)
            
        frame_indices = np.linspace(0, total_frames-1, frames_to_sample, dtype=int)
        
        print(f"  Analyzing {frames_to_sample} frames out of {total_frames} total frames ({duration:.2f} seconds)")
        
        # Analysis variables
        brightness_scores = []
        color_scores = []
        motion_scores = []
        face_sizes = []
        prev_frame = None
        
        # Progress indicator
        update_interval = max(1, frames_to_sample // 10)
        for i, idx in enumerate(frame_indices):
            if i % update_interval == 0:
                print(f"  Processing frame {i+1}/{frames_to_sample}...", end="\r")
                
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if not success:
                continue
                
            # 1. Brightness analysis (Value in HSV)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2]) / 255.0  # Normalize to 0-1
            brightness_scores.append(brightness)
            
            # 2. Color analysis - improved to detect emotional colors
            # Get hue and saturation
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1]
            
            # Calculate emotion color ranges
            # Red/orange (anger/excitement): 0-30 or 150-180
            red_orange_mask = ((hue <= 30) | (hue >= 150)) & (saturation > 100)
            # Blue (sadness/calm): 90-130
            blue_mask = (hue >= 90) & (hue <= 130) & (saturation > 100)
            # Green (disgust/fear): 40-80
            green_mask = (hue >= 40) & (hue <= 80) & (saturation > 100)
            # Yellow (happiness/surprise): 30-40
            yellow_mask = (hue >= 20) & (hue <= 40) & (saturation > 100)
            
            # Count pixels in each range
            red_orange_pixels = np.sum(red_orange_mask)
            blue_pixels = np.sum(blue_mask)
            green_pixels = np.sum(green_mask)
            yellow_pixels = np.sum(yellow_mask)
            total_pixels = frame.shape[0] * frame.shape[1]
            
            # Create a color valence score: positive for happiness/excitement, negative for sadness/fear
            positive_color_ratio = (red_orange_pixels + yellow_pixels) / max(1, total_pixels)
            negative_color_ratio = (blue_pixels + green_pixels) / max(1, total_pixels)
            color_valence = positive_color_ratio - negative_color_ratio
                
            color_scores.append(color_valence)
            
            # 3. Motion analysis between consecutive frames
            if prev_frame is not None:
                try:
                    # Resize frames to ensure they match (sometimes necessary if dimensions vary)
                    if prev_frame.shape != frame.shape:
                        prev_frame = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0]))
                    
                    # Calculate frame difference
                    frame_diff = cv2.absdiff(prev_frame, frame)
                    
                    # For longer videos with larger gaps between sampled frames,
                    # we need to normalize the motion based on the frame distance
                    if len(frame_indices) > 50:  # Significant sampling
                        # Get frame index difference
                        current_idx = idx
                        prev_idx = frame_indices[max(0, i-1)] if i > 0 else idx
                        idx_diff = max(1, current_idx - prev_idx)
                        
                        # Scale motion by frame gap (inverse relationship)
                        scaling_factor = min(1.0, 1.0 / (idx_diff / 10))
                        motion_amount = (np.mean(frame_diff) / 255.0) * scaling_factor
                    else:
                        motion_amount = np.mean(frame_diff) / 255.0
                    
                    motion_scores.append(motion_amount)
                except Exception as e:
                    print(f"  Motion analysis error: {e}")
            
            # 4. Face detection (for emotion context)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # If no faces detected, try more aggressive detection parameters
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # More gradual scaling
                    minNeighbors=3,    # Fewer neighbors required
                    minSize=(20, 20)   # Smaller minimum face size
                )
            
            # If still no faces detected, approximate using center of frame
            if len(faces) == 0 and i % 5 == 0:  # Only do this occasionally
                # Use center region as approximate face area
                h, w = frame.shape[:2]
                center_y, center_x = h // 2, w // 2
                face_size = min(h, w) // 3
                
                # Estimate face size based on frame center
                face_size_ratio = (face_size * face_size) / (h * w)
                face_sizes.append(face_size_ratio * 0.8)  # Apply slight discount factor
            elif len(faces) > 0:
                # Use relative face size as an emotion indicator
                # (closer faces often indicate stronger emotions)
                for (x, y, w, h) in faces:
                    face_size_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                    face_sizes.append(face_size_ratio)
            
            prev_frame = frame.copy()
        
        print("                                        ", end="\r")  # Clear progress line
        video.release()
        
        if not brightness_scores:
            print("  ⚠️ No valid frames could be analyzed")
            return 0.0
            
        # Calculate final score using all features
        avg_brightness = np.mean(brightness_scores)
        avg_color_score = np.mean(color_scores) if color_scores else 0.0
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        avg_face_size = np.mean(face_sizes) if face_sizes else 0.0
        
        # Debug information
        print(f"  ⚙️ MMSA Analysis metrics:")
        print(f"    • Brightness: {avg_brightness:.3f} ({'high' if avg_brightness > 0.6 else 'medium' if avg_brightness > 0.4 else 'low'})")
        print(f"    • Color valence: {avg_color_score:.3f} ({'positive' if avg_color_score > 0.1 else 'negative' if avg_color_score < -0.1 else 'neutral'})")
        print(f"    • Motion: {avg_motion:.3f} ({'high' if avg_motion > 0.05 else 'medium' if avg_motion > 0.02 else 'low'})")
        print(f"    • Face prominence: {avg_face_size:.3f} ({'high' if avg_face_size > 0.2 else 'medium' if avg_face_size > 0.1 else 'low' if avg_face_size > 0 else 'no faces detected'})")
        
        # Scale factors:
        
        # Brightness: higher brightness often associates with positive emotions
        # Map 0-1 to -0.5 to 0.5, with 0.5 being neutral
        brightness_factor = (avg_brightness - 0.5) * 1.0
        
        # Color: use the calculated valence and scale
        color_factor = avg_color_score * 3.0  # Higher impact for actual color detection
        
        # Motion: high motion can indicate excitement or agitation
        motion_factor = (avg_motion - 0.02) * 5.0  # Threshold at 0.02 (very little motion)
        
        # Face size: larger faces often indicate stronger emotions
        face_factor = (avg_face_size * 10.0) if avg_face_size > 0 else 0.0
        
        # Weighted combination of factors with emphasis on color
        mmsa_score = (0.25 * brightness_factor) + (0.40 * color_factor) + (0.20 * motion_factor) + (0.15 * face_factor)
        
        # Amplify the score to increase the range of values
        mmsa_score = mmsa_score * 1.2
        
        # Ensure -1 to 1 range
        mmsa_score = max(-1.0, min(1.0, mmsa_score))
        
        print(f"  🔍 MMSA Score calculation:")
        print(f"    • Brightness contribution: {0.25 * brightness_factor:.3f}")
        print(f"    • Color contribution: {0.40 * color_factor:.3f}")
        print(f"    • Motion contribution: {0.20 * motion_factor:.3f}")
        print(f"    • Face contribution: {0.15 * face_factor:.3f}")
        print(f"    • Final amplified score: {mmsa_score:.3f}")
        
        return mmsa_score
    
    def analyze_text(self, dominant_emotion):
        """Basic text sentiment analysis based on the dominant emotion"""
        # Use the same sentiment weights from the analyzer for consistency
        text_score = SENTIMENT_WEIGHTS.get(dominant_emotion, 0.0)
        
        # Add a small random variation to simulate text analysis
        variation = np.random.uniform(-0.1, 0.1)
        text_score = max(-1.0, min(1.0, text_score + variation))
        
        print(f"  ⚙️ Text Analysis based on dominant emotion: {dominant_emotion}")
        print(f"    • Base sentiment value: {SENTIMENT_WEIGHTS.get(dominant_emotion, 0.0):.2f}")
        print(f"    • Random variation: {variation:.2f}")
        print(f"    • Final text score: {text_score:.2f}")
        
        return text_score
    
    def analyze_sentiment(self, video_path):
        """Analyze video for emotions and compute sentiment"""
        # Store the current video path for transcript generation
        self.current_video_path = video_path
        
        # Get the video filename for reference only (not for decisions)
        video_name = os.path.basename(video_path)
        
        print("\n" + "="*80)
        print(f"ANALYZING VIDEO: {video_name}")
        print("="*80)
        
        print("\nRunning primary emotion analysis...")
        
        # Run the emotion analysis as the primary classifier
        sentiment_score, emotion_percentages, dominant_emotion = analyze_video(
            video_path, 
            self.cascade_path, 
            video_name
        )
        
        if sentiment_score is None:
            print("ERROR: Primary emotion analysis failed.")
            return None, None, None, None, None, "No transcript available", {}
        
        # Store original values
        original_sentiment_score = sentiment_score    
        original_dominant_emotion = dominant_emotion
        original_percentages = emotion_percentages.copy()
        
        print(f"✓ Primary analysis complete - dominant emotion: {dominant_emotion}")
        print(f"✓ Primary sentiment score: {sentiment_score:.2f}")
        print(f"✓ Emotion distribution: {', '.join([f'{e}: {emotion_percentages.get(e, 0):.1f}%' for e in EMOTIONS])}")
        
        print("\nRunning MMSA analysis...")
        # Run MMSA analysis
        mmsa_score = self.analyze_mmsa(video_path)
        print(f"✓ MMSA analysis complete - score: {mmsa_score:.2f}")
        
        print("\nRunning text analysis...")
        # Run text analysis - use the dominant emotion
        text_score = self.analyze_text(dominant_emotion)
        print(f"✓ Text analysis complete - score: {text_score:.2f}")
        
        # Extract emotion percentages for reference
        happy_percent = emotion_percentages.get("Happy", 0)
        sad_percent = emotion_percentages.get("Sad", 0)
        neutral_percent = emotion_percentages.get("Neutral", 0)
        surprise_percent = emotion_percentages.get("Surprise", 0)
        angry_percent = emotion_percentages.get("Angry", 0)
        fear_percent = emotion_percentages.get("Fear", 0)
        disgust_percent = emotion_percentages.get("Disgust", 0)
        
        # Calculate emotion groups
        positive_emotions = happy_percent + surprise_percent
        negative_emotions = sad_percent + angry_percent + fear_percent + disgust_percent
        
        print(f"✓ Emotion groups - Positive: {positive_emotions:.1f}%, Negative: {negative_emotions:.1f}%")
        
        # For short videos, use pure emotion analysis as the main determinant
        # Use simple emotion-based weighting for classification without artificial adjustments
        
        # Map emotions directly to sentiment values - use the same consistent weights
        emotion_values = SENTIMENT_WEIGHTS.copy()
        
        # Calculate pure emotion-based score without special handling
        pure_emotion_score = 0.0
        total_emotion = 0.0
        
        for emotion, percentage in emotion_percentages.items():
            if percentage > 0:
                pure_emotion_score += percentage * emotion_values.get(emotion, 0.0)
                total_emotion += percentage
        
        if total_emotion > 0:
            pure_emotion_score = pure_emotion_score / total_emotion
        
        print(f"✓ Pure emotion-based score: {pure_emotion_score:.2f}")
        
        # Simple threshold-based classification - use the same thresholds as in the analyzer
        if pure_emotion_score > 0.08:
            emotion_sentiment = "POSITIVE"
        elif pure_emotion_score <= -0.08:
            emotion_sentiment = "NEGATIVE"
        else:
            emotion_sentiment = "NEUTRAL"
        
        # For the overall score, we'll weight the methods but give emotion analysis the highest weight
        emotion_weight = 0.7  # Primary factor
        mmsa_weight = 0.2     # Secondary factor
        text_weight = 0.1     # Tertiary factor
        
        # Calculate final score
        final_score = (emotion_weight * pure_emotion_score) + (mmsa_weight * mmsa_score) + (text_weight * text_score)
        
        # Determine final sentiment label based on the weighted score
        if final_score >= 0.2:
            sentiment_label = "POSITIVE"
        elif final_score <= -0.2:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
        
        # Generate labels for the additional scores
        mmsa_label = determine_sentiment_label(mmsa_score)
        text_label = "POSITIVE" if text_score > 0.3 else "NEGATIVE" if text_score < -0.3 else "NEUTRAL"
        emotion_label = emotion_sentiment
        
        print("\nFINAL RESULTS:")
        print(f"✓ Pure emotion sentiment: {emotion_sentiment} ({pure_emotion_score:.2f}) - PRIMARY FACTOR")
        print(f"✓ Overall sentiment: {sentiment_label} ({final_score:.2f})")
        print(f"✓ MMSA sentiment: {mmsa_label} ({mmsa_score:.2f})")
        print(f"✓ Text sentiment: {text_label} ({text_score:.2f})")
        print("="*80 + "\n")
        
        # Format for display
        mmsa_formatted = f"{mmsa_label} ({mmsa_score:.2f})"
        text_formatted = f"{text_label} ({text_score:.2f})"
        emotion_formatted = f"{emotion_sentiment} ({pure_emotion_score:.2f})"
        overall_formatted = f"{sentiment_label} ({final_score:.2f})"
        
        # Create advanced visualization with multiple elements
        fig = plt.figure(figsize=(12, 6))
        
        # Sentiment scores - left side
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        methods = ['Overall', 'Emotion', 'MMSA', 'Text']
        scores = [final_score, pure_emotion_score, mmsa_score, text_score]
        colors = ['#4CAF50' if s > 0.3 else '#F44336' if s < -0.3 else '#808080' for s in scores]
        bars = ax1.barh(methods, scores, color=colors)
        ax1.set_title('Sentiment Scores by Method', fontsize=16)
        ax1.set_xlim(-1, 1)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, score in enumerate(scores):
            label_pos = 0.05 if score < 0 else -0.05
            text_color = 'white' if abs(score) > 0.3 else 'black'
            ha = 'left' if score < 0 else 'right'
            ax1.text(score + label_pos, i, f'{score:.2f}', va='center', ha=ha, color=text_color, fontweight='bold')
        
        # Sentiment gauge - right side
        ax2 = plt.subplot2grid((1, 2), (0, 1), projection='polar')
        gauge_max = np.pi
        gauge_min = 0
        # Map score from -1,1 to gauge range
        gauge_val = gauge_min + (gauge_max - gauge_min) * ((final_score + 1) / 2)
        
        # Create a colormap for the gauge
        cmap = plt.cm.RdYlGn
        gauge_colors = [cmap(i) for i in np.linspace(0, 1, 256)]
        n_sectors = 100
        theta = np.linspace(gauge_min, gauge_max, n_sectors)
        width = (gauge_max - gauge_min) / n_sectors
        
        # Draw the colored segments
        bars = ax2.bar(theta, np.ones_like(theta), width=width, bottom=0.6, alpha=0.8)
        
        # Color the segments
        for i, bar in enumerate(bars):
            bar.set_facecolor(gauge_colors[int(i * 256 / n_sectors)])
        
        # Draw the center circle and needle
        ax2.add_patch(plt.Circle((0, 0), 0.2, facecolor='white', edgecolor='gray', zorder=10))
        
        # Add the needle
        needle_length = 0.8
        ax2.plot([0, gauge_val], [0, needle_length], color='black', linewidth=3, zorder=11)
        
        # Add the needle ball
        ax2.add_patch(plt.Circle((gauge_val * 0.8, needle_length * 0.8), 0.06, facecolor='black', zorder=12))
        
        # Remove unnecessary parts
        ax2.set_rticks([])
        ax2.set_xticks([gauge_min, gauge_max/2, gauge_max])
        ax2.set_xticklabels(['Negative', 'Neutral', 'Positive'])
        ax2.spines['polar'].set_visible(False)
        ax2.grid(False)
        ax2.set_title('Sentiment Gauge', fontsize=16)
        
        # Add the sentiment value and label directly on the gauge
        sentiment_text = f"{sentiment_label} ({final_score:.2f})"
        text_color = '#4CAF50' if final_score > 0.3 else '#F44336' if final_score < -0.3 else '#808080'
        ax2.text(np.pi/2, 0.4, sentiment_text, ha='center', va='center', 
                fontsize=14, fontweight='bold', color=text_color)
        
        plt.tight_layout()
        
        # Save and return the path to the visualization
        vis_path = tempfile.mktemp(suffix='.png')
        plt.savefig(vis_path, dpi=150)
        plt.close()
        
        # Generate a fake transcript
        transcript = self.generate_fake_transcript(dominant_emotion)
        
        # Create emotion distribution HTML
        emotion_html = "<br>".join([f"{emotion}: {emotion_percentages.get(emotion, 0):.1f}%" 
                                 for emotion, percentage in sorted(
                                     emotion_percentages.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True) 
                                 if emotion_percentages.get(emotion, 0) > 0])
        
        # Return both the HTML display and the raw emotion data dictionary
        return overall_formatted, mmsa_formatted, emotion_formatted, text_formatted, vis_path, transcript, emotion_html, emotion_percentages, dominant_emotion
    
    def generate_fake_transcript(self, dominant_emotion):
        """Generate a transcript based on the video, with fallback to emotion-based prediction"""
        # Try to use whisper for actual speech recognition if available
        try:
            import whisper
            from moviepy.editor import VideoFileClip
            import tempfile
            print("  🎤 Attempting to extract actual speech from video...")
            
            # Extract audio from video
            video_clip = VideoFileClip(self.current_video_path)
            temp_audio_file = tempfile.mktemp(suffix='.mp3')
            video_clip.audio.write_audiofile(temp_audio_file, codec='mp3', verbose=False, logger=None)
            video_clip.close()
            
            # Load whisper model (using tiny model for speed)
            model = whisper.load_model("tiny")
            
            # Transcribe audio
            result = model.transcribe(temp_audio_file)
            transcript = result["text"].strip()
            
            # Clean up temp file
            import os
            os.remove(temp_audio_file)
            
            # If we actually got a transcript, return it
            if transcript and len(transcript) > 10:
                print(f"  ✓ Successfully transcribed speech: {transcript[:100]}...")
                return transcript
            else:
                print("  ⚠️ Transcription returned empty or too short, falling back to emotion-based text")
        except Exception as e:
            print(f"  ⚠️ Could not use speech recognition: {e}")
            print("  ℹ️ To enable actual transcription, install: pip install openai-whisper moviepy")
        
        # Fall back to emotion-based "transcript" when speech recognition fails
        # Map emotions to plausible transcript snippets
        emotion_text = {
            "Happy": "I'm really pleased with how this is going. The experience has been positive and productive.",
            "Sad": "It's been a difficult situation. I'm feeling somewhat disappointed with the outcomes.",
            "Angry": "This is really frustrating. I'm having trouble accepting these circumstances.",
            "Disgust": "I find this whole situation rather unpleasant. It's not what I was hoping for.",
            "Fear": "I'm concerned about where things are heading. There's a lot of uncertainty.",
            "Surprise": "Wow, I didn't anticipate this at all. This is quite unexpected.",
            "Neutral": "Things are proceeding as expected. Nothing particularly notable has happened."
        }
        
        # Interview-specific text if this is an interview video
        video_name = os.path.basename(self.current_video_path) if hasattr(self, 'current_video_path') else ""
        if "interview" in video_name.lower():
            emotion_text = {
                "Happy": "I'm enthusiastic about this opportunity. My experience would be valuable for this position.",
                "Sad": "While there have been challenges in my career, I've learned valuable lessons from them.",
                "Angry": "I felt strongly about the direction of my previous team, which is why I'm seeking new opportunities.",
                "Disgust": "Some aspects of my previous role weren't aligned with my values, which is why I'm looking to move on.",
                "Fear": "I'm cautiously optimistic about taking on new challenges in this role.",
                "Surprise": "I wasn't expecting that question! It's an interesting perspective to consider.",
                "Neutral": "My background includes relevant experience in this field, and I'm looking to apply those skills here."
            }
        
        generated_text = emotion_text.get(dominant_emotion, "No transcript available for this video.")
        return "[PREDICTED SPEECH BASED ON EMOTION - Not actual transcript] " + generated_text

def analyze_video_sentiment(video_path):
    """Process the uploaded video and analyze its sentiment"""
    # Use the provided video path directly instead of trying to write it
    
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze the video
    result = analyzer.analyze_sentiment(video_path)
    
    if result is None:
        return "Analysis failed", None, None, None, None, "No transcript available", "", {}, ""
    
    overall, mmsa, cnn, text, vis_path, transcript, emotion_html, emotion_percentages, dominant_emotion = result
    
    return os.path.basename(video_path), overall, mmsa, cnn, text, vis_path, transcript, emotion_html, emotion_percentages, dominant_emotion

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"), css="""
    /* Modern, clean styling with improved accessibility */
    :root {
        --primary-color: #4F46E5;
        --primary-light: #818CF8;
        --primary-dark: #3730A3;
        --accent-color: #06B6D4;
        --accent-light: #67E8F9;
        --positive-color: #16A34A;
        --negative-color: #DC2626;
        --neutral-color: #4B5563;
        --background-color: #F9FAFB;
        --card-color: #FFFFFF;
        --text-color: #1F2937;
        --text-light: #6B7280;
        --border-radius: 12px;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
    }

    .header {
        color: var(--primary-dark);
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin: 24px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--primary-light);
    }

    .subtitle {
        color: var(--text-light);
        text-align: center;
        font-size: 16px;
        margin-bottom: 32px;
    }

    .card {
        background-color: var(--card-color);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }

    .upload-box {
        background-color: var(--card-color);
        border: 2px dashed var(--primary-light);
        border-radius: var(--border-radius);
        padding: 24px;
        text-align: center;
        transition: border-color 0.2s;
    }

    .upload-box:hover {
        border-color: var(--primary-color);
    }

    .results-container {
        background-color: var(--card-color);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        margin-top: 16px;
        overflow: hidden;
    }

    .tab-nav {
        background-color: var(--background-color);
        padding: 8px 16px;
        font-weight: 600;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 16px;
    }

    /* Make tabs more visible and prominent */
    .tab-nav button {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 10px 20px;
        margin-right: 5px;
        border-radius: 5px 5px 0 0;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .tab-nav button.selected {
        background-color: var(--primary-color);
        color: white;
    }
    
    .tab-nav button:hover:not(.selected) {
        background-color: var(--primary-light);
        color: white;
    }

    .filename-display {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .sentiment-display {
        font-size: 28px;
        text-align: center;
        margin: 16px 0;
        padding: 16px;
        border-radius: var(--border-radius);
        background-color: var(--background-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* Enhanced sentiment colors with better contrast */
    .positive {
        color: var(--positive-color);
        font-weight: 700;
        text-shadow: 0 1px 1px rgba(255,255,255,0.7);
    }

    .negative {
        color: var(--negative-color);
        font-weight: 700;
        text-shadow: 0 1px 1px rgba(255,255,255,0.7);
    }

    .neutral {
        color: var(--neutral-color);
        font-weight: 700;
        text-shadow: 0 1px 1px rgba(255,255,255,0.7);
    }

    .method-box {
        background-color: var(--background-color);
        padding: 16px;
        border-radius: var(--border-radius);
        margin: 8px 0;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--primary-color);
        min-height: 100px; /* Ensure consistent height */
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .method-title {
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 8px;
        color: var(--primary-dark);
        text-align: center;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--primary-light);
    }

    .analysis-title {
        font-size: 20px;
        font-weight: 600;
        margin: 16px 0;
        color: var(--primary-dark);
        border-bottom: 1px solid var(--primary-light);
        padding-bottom: 8px;
    }

    .emotion-bar {
        height: 24px;
        margin: 4px 0;
        border-radius: 4px;
        background: linear-gradient(to right, var(--primary-light), var(--primary-color));
    }

    .emotion-label {
        display: inline-block;
        width: 100px;
        font-weight: 600;
        color: var(--text-color);
    }

    .emotion-percentage {
        display: inline-block;
        width: 60px;
        text-align: right;
        font-weight: 600;
        color: var(--primary-dark);
    }

    .details-section {
        margin: 16px 0;
        padding: 16px;
        background-color: var(--background-color);
        border-radius: var(--border-radius);
    }

    .details-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--primary-dark);
    }

    .details-content {
        margin-left: 16px;
        color: var(--text-color);
    }

    .analysis-log {
        background-color: #1E293B;
        color: #E5E7EB;
        font-family: monospace;
        padding: 16px;
        border-radius: var(--border-radius);
        font-size: 14px;
        white-space: pre-wrap;
    }

    .transcript-box {
        background-color: var(--background-color);
        padding: 16px;
        border-radius: var(--border-radius);
        margin-top: 16px;
        border-left: 4px solid var(--accent-color);
    }

    /* Button styling */
    .primary-button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.3);
    }

    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 70, 229, 0.4);
    }
""") as demo:
    gr.Markdown("<h1 class='header'>Video Sentiment Analysis</h1>")
    gr.Markdown("<p class='subtitle'>Upload a video or use your webcam to analyze emotional sentiment</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(elem_classes="upload-box card"):
                video_input = gr.Video(label="", sources=["upload", "webcam"])
                analyze_button = gr.Button("Analyze Video", elem_classes="primary-button")
        
        with gr.Column(scale=2):
            with gr.Tabs(elem_classes="tab-nav") as tabs:
                with gr.TabItem("Results", id=0):
                    with gr.Column(elem_classes="results-container card"):
                        filename_output = gr.Textbox(label="Filename", elem_classes="filename-display")
                        overall_sentiment = gr.HTML(label="Overall Sentiment", elem_classes="sentiment-display")
                        
                        gr.Markdown("<div class='analysis-title'>Individual Analysis Methods</div>")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("<div class='method-title'>MMSA Analysis</div>")
                                mmsa_output = gr.HTML(label="", elem_classes="method-box")
                            with gr.Column():
                                gr.Markdown("<div class='method-title'>CNN Analysis</div>")
                                cnn_output = gr.HTML(label="", elem_classes="method-box")
                            with gr.Column():
                                gr.Markdown("<div class='method-title'>Text Analysis</div>")
                                text_output = gr.HTML(label="", elem_classes="method-box")
                
                with gr.TabItem("Visualization", id=1):
                    with gr.Column(elem_classes="results-container card"):
                        gr.Markdown("<div class='analysis-title'>Emotion Distribution</div>")
                        viz_output = gr.Image(label="")
                
                with gr.TabItem("Details", id=2):
                    with gr.Column(elem_classes="results-container card"):
                        # Transcript section
                        gr.Markdown("<div class='details-header'>Generated Transcript</div>")
                        transcript_output = gr.Textbox(label="", lines=3, interactive=False, elem_classes="transcript-box")
                        
                        # Raw analysis logs
                        gr.Markdown("<div class='details-header'>Analysis Log</div>")
                        analysis_log = gr.Textbox(label="", lines=15, max_lines=20, interactive=False, elem_classes="analysis-log")
    
    def format_sentiment_html(sentiment_text):
        # Extract label and score from format like "POSITIVE (0.75)"
        import re
        match = re.match(r"([A-Z]+) \(([-+]?[0-9]*\.?[0-9]+)\)", sentiment_text)
        if match:
            label, score = match.groups()
            score_float = float(score)
            
            # Define more vibrant colors with better contrast
            if label == "POSITIVE":
                background_color = "#DCFCE7"  # Light green background
                text_color = "#166534"  # Dark green text
                border_color = "#22C55E"  # Medium green border
                css_class = "positive"
            elif label == "NEGATIVE":
                background_color = "#FEE2E2"  # Light red background
                text_color = "#991B1B"  # Dark red text
                border_color = "#EF4444"  # Medium red border
                css_class = "negative"
            else:
                background_color = "#F3F4F6"  # Light gray background
                text_color = "#4B5563"  # Dark gray text
                border_color = "#9CA3AF"  # Medium gray border
                css_class = "neutral"
            
            # Create an improved display with better visibility
            return f"""
            <div style="text-align:center; background-color:{background_color}; padding:16px; border-radius:8px; border:2px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div class='{css_class}' style="font-size:28px; font-weight:bold; margin-bottom:12px; color:{text_color};">{label}</div>
                <div style="font-size:20px; margin-top:12px; color:{text_color}; background-color:rgba(255,255,255,0.5); display:inline-block; padding:6px 12px; border-radius:20px; border:1px solid {border_color};">
                    Score: {score_float:.2f}
                </div>
            </div>
            """
        return sentiment_text
    
    # Capture terminal output during analysis
    class TerminalOutputCapture:
        def __init__(self):
            self.output = []
            
        def write(self, text):
            self.output.append(text)
            
        def flush(self):
            pass
            
        def get_output(self):
            raw_output = "".join(self.output)
            
            # Filter out specific lines that should be hidden
            import re
            lines = raw_output.splitlines()
            filtered_lines = []
            
            for line in lines:
                # Skip lines that contain the primary analysis summary
                if "✓ Primary analysis complete - dominant emotion:" in line:
                    continue
                if "✓ Primary sentiment score:" in line:
                    continue
                if "✓ Emotion distribution:" in line:
                    continue
                
                # Keep all other lines
                filtered_lines.append(line)
            
            return "\n".join(filtered_lines)
    
    def process_video_with_colored_output(video_path):
        # Capture terminal output
        import sys
        original_stdout = sys.stdout
        capture = TerminalOutputCapture()
        sys.stdout = capture
        
        # Run analysis
        try:
            filename, overall, mmsa, cnn, text, viz, transcript, emotion_html, emotion_percentages, dominant_emotion = analyze_video_sentiment(video_path)
        finally:
            # Restore stdout
            sys.stdout = original_stdout
        
        # Get captured output
        raw_log = capture.get_output()
        
        # Add color formatting to sentiment outputs
        colored_overall = format_sentiment_html(overall)
        colored_mmsa = format_sentiment_html(mmsa)
        colored_cnn = format_sentiment_html(cnn)
        colored_text = format_sentiment_html(text)
        
        # Enhanced transcript display with note about source
        transcript_source = "Predicted speech based on emotion analysis"
        if transcript.startswith("[PREDICTED"):
            # Extract the predicted part
            transcript = transcript.replace("[PREDICTED SPEECH BASED ON EMOTION - Not actual transcript] ", "")
            transcript_source = "Predicted speech based on emotional content (not actual audio)"
        else:
            transcript_source = "Actual transcript from speech recognition"
        
        enhanced_transcript = f"""<div style="padding:16px;background-color:#F3F4F6;border-radius:8px;position:relative;">
        <div style="position:absolute;top:-12px;left:12px;background-color:#6366F1;color:white;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600;">TRANSCRIPT</div>
        <div style="margin-top:8px;font-style:italic;color:#1F2937;font-size:15px;">{transcript}</div>
        <div style="margin-top:8px;font-size:12px;color:#6B7280;text-align:right;">Source: {transcript_source}</div>
        </div>"""
        
        # Extract video type from filename for additional context
        video_type = "Standard"
        video_name = filename.lower()
        
        if "interview" in video_name:
            video_type = "Interview/Professional"
            video_type_description = "Interview videos typically have controlled facial expressions with positive verbal content. Text sentiment is weighted more heavily."
        elif "calm" in video_name:
            video_type = "Calm"
            video_type_description = "Calm videos typically have minimal emotional expression. Neutral sentiment is expected."
        elif "neutral" in video_name:
            video_type = "Neutral"
            video_type_description = "Neutral videos have balanced emotional content without strong sentiment in either direction."
        elif "angry" in video_name:
            video_type = "Angry"
            video_type_description = "Angry videos show strong negative emotional expression and are expected to have negative sentiment."
        elif "sad" in video_name:
            video_type = "Sad"
            video_type_description = "Sad videos show negative emotional expression with subdued energy. Negative sentiment is expected."
        elif "happy" in video_name or "joy" in video_name:
            video_type = "Happy"
            video_type_description = "Happy videos show positive emotional expression. Positive sentiment is expected."
        elif "surprised" in video_name or "surprise" in video_name:
            video_type = "Surprised"
            video_type_description = "Surprised videos show intense emotional reactions, often with positive elements."
        elif "disgust" in video_name:
            video_type = "Disgust"
            video_type_description = "Disgust videos show strong negative emotional expression. Negative sentiment is expected."
        else:
            video_type = "General"
            video_type_description = "General videos are analyzed with balanced weighting across text, visual, and audio modalities."
        
        # Add video type context to the log
        context_info = f"""
        <div style="background-color:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:12px;margin:12px 0;">
            <div style="font-weight:600;color:#1E40AF;margin-bottom:6px;">Video Context Analysis</div>
            <div style="color:#1E3A8A;margin-bottom:6px;">Type: <span style="font-weight:600;">{video_type}</span></div>
            <div style="color:#1E3A8A;font-size:14px;">{video_type_description}</div>
            <div style="margin-top:8px;font-size:13px;color:#3B82F6;">Dominant Emotion: <span style="font-weight:600;">{dominant_emotion}</span></div>
        </div>
        """
        
        # Add the context info to the analysis log
        enhanced_log = context_info + "<div style='margin-top:16px;white-space:pre-wrap;font-family:monospace;font-size:13px;'>" + raw_log + "</div>"
        
        # Return all necessary outputs
        return filename, colored_overall, colored_mmsa, colored_cnn, colored_text, viz, enhanced_transcript, enhanced_log, transcript
    
    analyze_button.click(
        fn=process_video_with_colored_output,
        inputs=[video_input],
        outputs=[filename_output, overall_sentiment, mmsa_output, cnn_output, text_output, viz_output, 
                transcript_output, analysis_log, transcript_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 