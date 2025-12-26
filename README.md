# ForensicsUI - Object Finder

A computer vision application that finds matching objects in video footage by comparing against a target image. Built with YOLOv8 for detection, ByteTrack for tracking, and deep learning models for feature matching.

## Features

- **Target Matching**: Upload an image of an object to find all similar objects in a video
- **Real-time Tracking**: Uses ByteTrack to maintain object identity across frames
- **Feature Comparison**: Deep learning-based similarity matching using Inception/ResNet
- **Interactive UI**: Clean Streamlit web interface with progress tracking
- **Processed Output**: Download annotated video with bounding boxes and similarity scores

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

Place your YOLO model in the `models/` directory:
- `models/trained_yolo.pt` - YOLOv8 model

For feature extraction (optional, uses pretrained by default):
- `models/Inception.pth` - Inception-v3 weights
- `models/ResNet50_v3.pth` - ResNet50 weights

### 3. Run the App

```bash
cd src
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Target Image**: Upload an image containing the object you want to find
2. **Analyze Target**: Click to detect and extract features from the target
3. **Upload Video**: Upload the video to search through
4. **Process**: Start processing to find matching objects
5. **View Results**: See detected matches with similarity scores
6. **Download**: Download the processed video with annotations

## Project Structure

```
forensicsUI/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── config.py           # Configuration settings
│   ├── detector.py         # YOLOv8 detection
│   ├── features.py         # Feature extraction & comparison
│   ├── video_processor.py  # Main processing pipeline
│   ├── visualizer.py       # Draw annotations
│   └── database.py         # SQLite storage
├── models/                 # Model weights
├── database/               # SQLite databases
├── outputs/                # Processed videos
└── requirements.txt
```

## Configuration

Adjust settings in the Streamlit sidebar:
- **Similarity Threshold**: Minimum score to consider a match (0.5-1.0)
- **Feature Model**: Choose between Inception (default) or ResNet
- **Performance Mode**: Extract features less frequently for faster processing

## Requirements

- Python 3.9+
- macOS (MPS), Linux (CUDA), or Windows (CPU/CUDA)
- ~2GB RAM for feature extraction

## License

MIT License
