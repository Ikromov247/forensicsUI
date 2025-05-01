import sys
from pathlib import Path

from scripts.download_models import main as download_main

def main():
    """Download required models"""
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download models
    download_main()

if __name__ == "__main__":
    main() 