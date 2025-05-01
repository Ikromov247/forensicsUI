import os
import shutil
from pathlib import Path

def main():
    """Clean up generated files and directories"""
    # Directories to clean
    dirs_to_clean = [
        "database",
        "logs",
        "models",
        "__pycache__",
        ".pytest_cache",
        "*.egg-info"
    ]
    
    # Files to clean
    files_to_clean = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.bak",
        "*.log"
    ]
    
    # Clean directories
    for dir_pattern in dirs_to_clean:
        for path in Path(".").glob(dir_pattern):
            if path.is_dir():
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
    
    # Clean files
    for file_pattern in files_to_clean:
        for path in Path(".").rglob(file_pattern):
            if path.is_file():
                print(f"Removing file: {path}")
                os.remove(path)
    
    print("Cleanup complete!")

if __name__ == "__main__":
    main() 