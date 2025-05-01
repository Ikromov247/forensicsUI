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
    """Download a sample video"""
    # Sample video URL (replace with your own)
    sample_url = "https://example.com/sample.mp4"  # Replace with actual URL
    
    # Download video
    print("Downloading sample video...")
    download_file(sample_url, "sample.mp4")
    
    print("Download complete!")

if __name__ == "__main__":
    main() 