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
git clone https://github.com/thuiar/MMSA.git
git clone https://github.com/thuiar/MMSA-FET.git
git clone https://github.com/faizarashid/Video-Sentiment-Analysis.git
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

## Features

- Upload videos for analysis
- Use webcam for live analysis
- View text, audio, and video sentiment scores
- See SHAP explanations for model predictions
- Visualize emotion distribution

## Interface Screenshots

### Main UI
![Main Interface](./imagez/MAINUI.png)

### Analysis Results
![Analysis Results](./imagez/ANALYSIS%20.png)

### Individual Scores
![Individual Scores](./imagez/INDUVIDUAL%20SCORES.png)

### Visualizations
![Visualizations](./imagez/VIS1.png)

### SHAP Visualization
![SHAP Visualization](./imagez/SHAP%20VIS%202.png) 