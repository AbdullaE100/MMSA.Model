# Video Sentiment Analysis

Multimodal sentiment analysis system with text, audio, and video processing capabilities.

## Requirements

- Python 3.8+
- Git
- 4GB free disk space
- Optional: webcam for live analysis

## Setup

1. Clone repositories
```bash
mkdir MMSA_Project
cd MMSA_Project
git clone https://github.com/AbdullaE100/MMSA.Model.git
cd MMSA.Model

mkdir repositories && cd repositories
git clone https://github.com/qiuqiangkong/MMSA.git
git clone https://github.com/TmacMai/MMSA-FET.git
git clone https://github.com/AbdullaE100/Video-Sentiment-Analysis.git
cd ..
```

2. Create environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

3. Download base models
```bash
python download_models.py
```

4. Download large files from [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk)

   Required directory structure:
   ```
   MMSA_Project/MMSA.Model/
   ├── pretrained_models/  # Copy from CH-SIMS v2
   ├── MOSI/              # Copy from CMU-MOSI
   ├── model.h5           # Downloaded by script
   └── haarcascade_frontalface_default.xml  # Downloaded by script
   ```

5. Run application
```bash
python video_sentiment_app.py
```
Access via local URL (http://127.0.0.1:7860) or temporary public URL.

## Troubleshooting

- Python not found: Add to PATH
- Package errors: `python -m pip install --upgrade pip`
- CUDA errors: App works on CPU
- Webcam issues: Check browser permissions
- Permission errors: Run terminal as admin

## Interface Features

### Main Interface
![Main Interface](./images/ui_main.png)

### Analysis Results
![Analysis Results](./images/analysis_results.png)

### Emotion Distribution
![Emotion Distribution](./images/emotion_distribution.png)

### XAI: Feature Importance (SHAP)
![SHAP Visualization](./images/shap_visualization.png)

### Analysis Log
![Analysis Log](./images/analysis_log.png) 