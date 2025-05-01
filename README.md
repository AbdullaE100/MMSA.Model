# Multimodal Sentiment Analysis Testing Framework

This repository contains tools for testing different multimodal sentiment analysis repositories:

1. **MMSA-FET**: Feature extraction tool for multimodal sentiment analysis
2. **MMSA**: Multimodal sentiment analysis model
3. **Video-Sentiment-Analysis**: Computer vision-based sentiment analysis

## Quick Start

To run all tests on the video files in the `test_videos` directory:

```bash
# Make scripts executable (if not already)
chmod +x run_all_tests.sh setup_venvs.sh

# Set up virtual environments for each repository
./setup_venvs.sh

# Run all tests with default settings
./run_all_tests.sh
```

## Directory Structure

```
.
├── repositories/             # Contains all sentiment analysis repositories
│   ├── MMSA-FET/             # Feature extraction tools
│   ├── MMSA/                 # Multimodal sentiment analysis
│   └── Video-Sentiment-Analysis/ # Visual sentiment analysis
├── test_videos/              # Test video files (.mp4)
├── outputs/                  # Results will be saved here
├── venv_*/                   # Virtual environments for each repository
├── testing_utils.py          # Common testing utilities
├── requirements.txt          # Common requirements
├── run_all_tests.sh          # Main testing script
└── setup_venvs.sh            # Script to set up virtual environments
```

## Using Custom Test Videos

You can specify a different directory for test videos:

```bash
./run_all_tests.sh --videos=path/to/custom/videos
```

## Output Formats

Results can be saved in JSON or CSV format:

```bash
./run_all_tests.sh --format=json  # Default
./run_all_tests.sh --format=csv   # CSV format
```

## Running Tests for Individual Repositories

You can run tests for individual repositories using their specific virtual environments:

```bash
# For MMSA-FET
source venv_MMSA-FET/bin/activate
cd repositories/MMSA-FET/
python3 test_repo.py --folder_path ../../test_videos --output_format json
deactivate

# For MMSA
source venv_MMSA/bin/activate
cd repositories/MMSA/
python3 test_repo.py --folder_path ../../test_videos --output_format json
deactivate

# For Video-Sentiment-Analysis
source venv_Video-Sentiment-Analysis/bin/activate
cd repositories/Video-Sentiment-Analysis/
python3 test_repo.py --folder_path ../../test_videos --output_format json
deactivate
```

## Command Line Arguments

Each repository's test script supports the following arguments:

- `--video_path`: Path to a single video file
- `--folder_path`: Path to folder containing .mp4 video files
- `--output_format`: Output format (json or csv)

Additional repository-specific arguments:
- MMSA-FET: `--config_path` for custom feature extraction configuration
- MMSA: `--model_path` and `--config_path` for custom model and configuration
- Video-Sentiment-Analysis: `--model_path` for custom sentiment model

## Output Files

For each run, the following files are created in the `outputs/` directory:

1. **Individual results**: One file per video with detailed analysis:
   - `video_name_result.json` or `video_name_result.csv`

2. **Repository batch results**: One file per repository with all videos' results:
   - `fet_results_TIMESTAMP.json`
   - `mmsa_results_TIMESTAMP.json`
   - `video_sentiment_results_TIMESTAMP.json`

3. **Combined report**: A single file with results from all repositories:
   - `combined_report_TIMESTAMP.json` or `combined_report_TIMESTAMP.csv`

## Notes

- If a dependency is missing (like `espnet2` for MMSA-FET), the system will use dummy features to ensure the tests can still run.
- Temporary folders are automatically cleaned up after processing.
- Failed videos are tracked and reported separately.
- The tests are modular, so new videos can be added to the test_videos directory without code changes. 