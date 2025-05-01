# Forensics Video Analysis

A Python application for analyzing video footage using object detection and feature extraction.

## Features

- Object detection using YOLO
- Feature extraction using pre-trained models (ResNet/Inception)
- Feature comparison and matching
- Visualization of detected objects
- Database storage of objects and features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forensics-video-analysis.git
cd forensics-video-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application with a video file:
```bash
python src/main.py path/to/video.mp4
```

Optional arguments:
- `--db-name`: Specify database name (default: "forensics")
- `--no-vis`: Disable visualization

## Project Structure

```
forensics-video-analysis/
├── src/
│   ├── core/
│   │   ├── detection/
│   │   │   └── detector.py
│   │   ├── features/
│   │   │   ├── extractor.py
│   │   │   └── comparator.py
│   │   └── application.py
│   ├── database/
│   │   ├── manager.py
│   │   └── models.py
│   ├── visualization/
│   │   └── visualizer.py
│   ├── exceptions.py
│   ├── logging_config.py
│   └── main.py
├── database/
├── logs/
├── requirements.txt
└── README.md
```

## License

MIT License

