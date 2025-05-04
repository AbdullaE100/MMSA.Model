# Video Sentiment Analysis with Gradio Interface

Hey! 👋 This is my video sentiment analysis project that analyzes emotions and sentiment from videos using multimodal analysis (text, audio, and video). I've made it super easy to use with a Gradio interface.

## Quick Setup Guide

### What You'll Need
- Python 3.8 or newer
- Git
- About 4GB free space for models and datasets
- A webcam (optional - for live analysis)

### Setup Steps

1. **Clone the Repository**
```bash
# Create a project folder and clone the main repo
mkdir MMSA_Project
cd MMSA_Project
git clone https://github.com/AbdullaE100/MMSA.Model.git
cd MMSA.Model

# Get the other needed repositories
mkdir repositories && cd repositories
git clone https://github.com/qiuqiangkong/MMSA.git
git clone https://github.com/TmacMai/MMSA-FET.git
git clone https://github.com/AbdullaE100/Video-Sentiment-Analysis.git
cd ..
```

2. **Set Up Your Environment**
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

3. **Download Required Models**
```bash
# This will get the basic models we need
python download_models.py
```

4. **Get the Large Files**
You'll need to download some larger files that couldn't be included in Git. Get them from my Google Drive folder here: [https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk)

The drive contains these important folders:
- `CMU-MOSI/` - The MOSI dataset files
- `CMU-MOSEI/` - Additional dataset (optional)
- `CH-SIMS/` - Pretrained models and weights
- `CH-SIMS v2/` - Updated models (recommended)

Download and place them in these locations:
```
MMSA_Project/MMSA.Model/
├── pretrained_models/  <- Copy contents from CH-SIMS v2
├── MOSI/              <- Copy the CMU-MOSI folder here
├── model.h5           <- Should be downloaded by the script
└── haarcascade_frontalface_default.xml  <- Should be downloaded by the script
```

5. **Run the Gradio Interface**
```bash
python video_sentiment_app.py
```

The app will start and show you two URLs:
- A local URL (like http://127.0.0.1:7860)
- A public URL (temporary, lasts 72 hours)

Just open either URL in your browser and you're good to go! 🚀

### Troubleshooting Tips

If you run into any issues:

- **Python not found?** Make sure Python is added to your PATH
- **Package installation errors?** Try updating pip: `python -m pip install --upgrade pip`
- **CUDA errors?** Don't worry - the app works on CPU too, just a bit slower
- **Webcam not working?** Check your browser's camera permissions
- **Permission errors?** Try running your terminal as administrator
- **Dataset files not found?** Double check you've copied the files from Google Drive to the correct folders

### What You Can Do

Once it's running, you can:
- Upload video files for analysis
- Use your webcam for live analysis
- Get detailed emotion and sentiment breakdowns
- See SHAP explanations for the predictions

### Need Help?

If you run into any problems, just open an issue on the repo or reach out to me directly. Happy analyzing! 😊 