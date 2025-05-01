import sys
import subprocess
from pathlib import Path

def main():
    """Run the application with a specific video"""
    # Check if video path is provided
    if len(sys.argv) < 2:
        print("Error: Video path not provided")
        print("Usage: python run_video.py path/to/video.mp4 [--db-name NAME] [--no-vis]")
        sys.exit(1)
    
    # Get video path
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Get additional arguments
    args = sys.argv[2:]
    
    # Run application
    print(f"Running application with video: {video_path}")
    subprocess.run([sys.executable, "run.py", str(video_path), *args])

if __name__ == "__main__":
    main() 