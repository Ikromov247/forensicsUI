import sys
import subprocess
from pathlib import Path

def main():
    """Run the application with a sample video"""
    # Check if sample video exists
    sample_video = Path("sample.mp4")
    if not sample_video.exists():
        print("Error: Sample video not found")
        print("Please download a sample video and save it as 'sample.mp4'")
        sys.exit(1)
    
    # Run application
    print("Running application with sample video...")
    subprocess.run([sys.executable, "run.py", str(sample_video)])

if __name__ == "__main__":
    main() 