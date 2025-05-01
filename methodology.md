# Methodology: Multimodal Sentiment Analysis System

## 1. System Architecture Overview

The system implements a layered multimodal approach to video sentiment analysis:

1. **Input Layer**: Accepts video files in standard formats (.mp4, .avi, etc.)
2. **Processing Layer**: Extracts and analyzes multiple data modalities
   - Visual processing (facial expressions, scene analysis)
   - Audio processing (when available)
3. **Analysis Layer**: Combines modality-specific results using weighted fusion
4. **Output Layer**: Delivers sentiment scores and visualizations through a Gradio interface

## 2. Data Acquisition and Preprocessing

### Video Processing Pipeline
1. **Frame Extraction**: 
   - Adaptive sampling based on video duration:
     - Short videos (≤30s): Every 5th frame, max 100 frames
     - Medium videos (30s-2min): 1 frame per second, max 200 frames
     - Long videos (2-10min): 1 frame every 2 seconds, max 300 frames
     - Very long videos (>10min): 1 frame every 5 seconds, max 400 frames

2. **Face Detection**:
   - Primary: Haar Cascade classifier with standard parameters
   - Fallback 1: Adjusted parameters (scaleFactor=1.05, minNeighbors=3, minSize=(20,20))
   - Fallback 2: Center-region approximation when faces cannot be detected

3. **Face Preprocessing**:
   - Grayscale conversion
   - Resize to 48×48 pixels (model input dimensions)
   - Histogram equalization for feature enhancement
   - Normalization (pixel values to 0-1 range)

## 3. Multimodal Feature Extraction

### Visual Features
1. **Facial Emotion Recognition**:
   - CNN model architecture:
     - Input: 48×48×1 grayscale images
     - 3 convolutional blocks (Conv2D + MaxPooling)
     - 2 fully connected layers
     - 7-class softmax output (emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

2. **Scene Analysis**:
   - Brightness analysis: Mean value in HSV color space
   - Color emotion mapping:
     - Red/orange (anger/excitement): Hue 0-30° or 150-180°
     - Blue (sadness/calm): Hue 90-130°
     - Green (disgust/fear): Hue 40-80°
     - Yellow (happiness/surprise): Hue 20-40°
   - Motion analysis: Frame difference measurement with temporal scaling

### Contextual Features
- Face size ratio (proportion of frame occupied by face)
- Temporal emotion distribution (consistency/changes over time)
- Confidence scores for each detected emotion

## 4. Sentiment Analysis Algorithm

### Emotion-to-Sentiment Mapping
Emotions are mapped to sentiment values using psychologically-informed weights:
- Angry: -0.8 (Strong negative)
- Disgust: -0.7 (Strong negative)
- Fear: -0.6 (Moderate negative)
- Happy: 0.8 (Strong positive)
- Sad: -0.5 (Moderate negative)
- Surprise: 0.3 (Mild positive)
- Neutral: 0.0 (Neutral)

### Sentiment Score Calculation
1. **Per-frame emotion confidence vector**: [c₁, c₂, ..., c₇]
2. **Emotion-weighted sentiment**: S = Σ(cᵢ × wᵢ) for each emotion i
3. **Temporal aggregation**: Weighted average of frame sentiments, giving higher weights to:
   - Frames with higher emotion confidence
   - Frames with larger faces (presumed more significant)
   - Frames with higher motion (potentially more emotional content)

### Output Normalization
- Raw scores mapped to [-1, 1] range
- Classification thresholds:
  - Negative: score < -0.2
  - Neutral: -0.2 ≤ score ≤ 0.2
  - Positive: score > 0.2

## 5. Implementation and Deployment

### Technology Stack
- **Core Processing**: Python 3.8+, OpenCV, NumPy, TensorFlow
- **Model Development**: Keras with TensorFlow backend
- **User Interface**: Gradio web interface
- **Integration**: Shell scripts for testing across repositories

### Deployment Architecture
1. **Local Development**: Command-line interface for testing and development
2. **Web Interface**: Gradio-based UI running on local server (http://127.0.0.1:7860)
3. **Testing Framework**: Unified harness for comparing different implementations:
   - MMSA-FET: Feature extraction tools
   - MMSA: Multimodal sentiment analysis
   - Video-Sentiment-Analysis: Visual sentiment analysis

## 6. Validation and Testing

### Testing Methodology
- **Comparative Analysis**: Testing across multiple repositories using identical test videos
- **Output Formats**: Standardized JSON and CSV outputs for consistent comparison
- **Visualization**: Matplotlib-based emotion distribution charts and sentiment timelines

### Performance Evaluation
- **Accuracy**: Compared against human-annotated ground truth
- **Processing Efficiency**: Adaptive sampling to maintain reasonable processing times
- **Robustness**: Multiple fallback mechanisms to handle edge cases 