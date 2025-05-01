import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, dest_path: str):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=Path(dest_path).name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    """Download required models"""
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs
    models = {
        "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "resnet50.pth": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "inception_v3.pth": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
    }
    
    # Download models
    for model_name, url in models.items():
        dest_path = models_dir / model_name
        if not dest_path.exists():
            print(f"Downloading {model_name}...")
            download_file(url, str(dest_path))
        else:
            print(f"{model_name} already exists, skipping...")

if __name__ == "__main__":
    main() 