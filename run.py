import sys
import argparse
from pathlib import Path

from src.main import main as app_main

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Forensics Video Analysis")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--db-name", default="forensics", help="Database name")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    return parser.parse_args()

def main():
    """Run the application"""
    # Parse arguments
    args = parse_args()
    
    # Check video file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run application
    sys.argv = [
        sys.argv[0],
        str(video_path),
        f"--db-name={args.db_name}",
        *(["--no-vis"] if args.no_vis else [])
    ]
    app_main()

if __name__ == "__main__":
    main() 