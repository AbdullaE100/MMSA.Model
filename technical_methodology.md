# Technical Methodology: Multimodal Sentiment Analysis System

## 1. Technical Architecture

The system employs a multi-tier architecture implemented with the following exact technologies:

1. **Input Processing**: 
   - `cv2.VideoCapture()` (OpenCV 4.5.5+) for video ingestion
   - `cv2.CAP_PROP_FRAME_COUNT` and `cv2.CAP_PROP_FPS` for metadata extraction

2. **Processing Frameworks**:
   - TensorFlow 2.8.0 with Keras 2.8.0 integration
   - NumPy 1.20.0 for array manipulation and vectorized operations
   - OpenCV 4.5.5 for image processing and feature extraction

3. **Frontend Interface**:
   - Gradio 3.50.2 serving on http://127.0.0.1:7860
   - Matplotlib 3.5.0 for visualization components

## 2. Video Processing and Feature Extraction

### Video Decoding
1. **Frame Extraction**: 
   - `cv2.VideoCapture` class with `read()` method
   - `cv2.set(cv2.CAP_PROP_POS_FRAMES, idx)` for targeted frame access
   - `np.linspace(0, total_frames-1, frames_to_sample, dtype=int)` for sampling strategy

2. **Face Detection Pipeline**:
   - Primary: `cv2.CascadeClassifier('haarcascade_frontalface_default.xml')` with `detectMultiScale()` function
   - Parameters: `scaleFactor=1.1, minNeighbors=5, minSize=(30,30)`
   - Fallback: `detectMultiScale(scaleFactor=1.05, minNeighbors=3, minSize=(20,20))`

3. **Image Preprocessing Functions**:
   - `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` for grayscale conversion
   - `cv2.resize(face_roi, (48, 48))` for dimension standardization
   - `cv2.equalizeHist()` for contrast enhancement
   - Normalization: `normalized_face = equalized_face / 255.0`

### Visual Feature Extraction

1. **CNN Architecture**:
   ```python
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
   ```
   - Optimizer: `'adam'`
   - Loss: `'categorical_crossentropy'`
   - Metrics: `['accuracy']`

2. **HSV-based Color Analysis**:
   - `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)` for colorspace conversion
   - Channel extraction: `hue = hsv[:, :, 0]`, `saturation = hsv[:, :, 1]`
   - Statistical functions: `np.mean(hsv[:, :, 2])` for brightness
   - Boolean masks: `red_orange_mask = ((hue <= 30) | (hue >= 150)) & (saturation > 100)`

3. **Motion Analysis**:
   - `cv2.absdiff(prev_frame, frame)` for inter-frame difference
   - `np.mean(frame_diff) / 255.0` for motion quantification

## 3. Deep Learning Model Implementation

### Model Loading and Prediction

1. **Model Persistence**:
   - Model architecture: `model.json` (3.5KB)
   - Weights: `model.h5` (2.7MB)
   - Loading: `model.load_weights("./model.h5")`

2. **Inference Pipeline**:
   - Reshaping: `reshaped_face = normalized_face.reshape(1, 48, 48, 1)`
   - Prediction: `predictions = model.predict(reshaped_face, verbose=0)[0]`
   - Emotion extraction: `emotion_index = np.argmax(predictions)`
   - Confidence score: `confidence = float(predictions[emotion_index])`

3. **Emotion Label Mapping**:
   ```python
   EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
   ```

## 4. Algorithmic Sentiment Calculation

### Weighted Sentiment Algorithm

1. **SENTIMENT_WEIGHTS Dictionary**:
   ```python
   SENTIMENT_WEIGHTS = {
       "Angry": -0.8,  # Strong negative
       "Disgust": -0.7,  # Strong negative
       "Fear": -0.6,  # Moderate negative
       "Happy": 0.8,  # Strong positive
       "Sad": -0.5,  # Moderate negative
       "Surprise": 0.3,  # Mild positive
       "Neutral": 0.0  # Neutral
   }
   ```

2. **Mathematical Formula**:
   - Per-frame sentiment: `S_frame = Σ(confidence_i × weight_i)` where i ∈ EMOTIONS
   - Temporal weighting: `weighted_sum / total_weight` with `weight = confidence × face_size_ratio × (1 + motion_score)`

3. **Threshold Classification**:
   ```python
   def determine_sentiment_label(score):
       if score < -0.2:
           return "NEGATIVE"
       elif score > 0.2:
           return "POSITIVE"
       else:
           return "NEUTRAL"
   ```

## 5. External Framework Integration

### MMSA Framework Integration

1. **MMSA-FET**:
   - Feature extraction interface: `run_mmsa_test.sh` and `extract_features.sh`
   - Data format: MOSI dataset structure with mandatory fields

2. **MMSA Core**:
   - Model configurations: `/configs/config.json`
   - Feature vectors stored in NumPy arrays through `np.save()`

3. **Self-MM Implementation**:
   - Repository: `Self-MM/`
   - Test harness: `test_self_mm.py` with `ArgumentParser` configuration

## 6. UI and Deployment Architecture

### Gradio Implementation

1. **Interface Components**:
   ```python
   interface = gr.Interface(
       fn=process_video_with_colored_output,
       inputs=gr.Video(),
       outputs=[
           gr.Textbox(label="Analysis Results"),
           gr.HTML(label="Detailed Analysis")
       ],
       title="Multimodal Video Sentiment Analysis",
       description="Upload a video to analyze sentiment and emotions."
   )
   interface.launch(share=False)  # Local deployment on http://127.0.0.1:7860
   ```

2. **Visualization Tools**:
   - `matplotlib.pyplot.figure(figsize=(10, 6))`
   - `plt.bar()` for emotion distribution
   - `plt.savefig()` for exporting visualizations

### Testing and Validation Framework

1. **Batch Testing Interface**:
   - `run_all_tests.sh` with ArgParse parameters
   - `setup_venvs.sh` for environment isolation

2. **Output Formats**:
   - JSON serialization: `json.dump(results, outfile, indent=2)`
   - CSV export: `csv.writer(csvfile).writerows()`

3. **Statistical Metrics**:
   - Accuracy: `(true_positives + true_negatives) / total_samples`
   - Precision: `true_positives / (true_positives + false_positives)`
   - Recall: `true_positives / (true_positives + false_negatives)` 