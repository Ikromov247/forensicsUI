import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Optional, Tuple

from ..exceptions import InputError
from ..logging_config import get_logger

logger = get_logger(__name__)

class InputHandler:
    """
    Handles user input for file selection.
    """
    
    def __init__(self):
        """Initialize input handler"""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
    
    def get_image_input(self, title: str = "Select an image") -> Path:
        """
        Get image file path from user.
        
        Args:
            title: Dialog title
            
        Returns:
            Path to selected image file
        """
        try:
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                raise InputError("No image file selected")
                
            path = Path(file_path)
            if not path.exists():
                raise InputError(f"Image file not found: {file_path}")
                
            return path
            
        except Exception as e:
            logger.error(f"Failed to get image input: {str(e)}")
            raise InputError(f"Image selection failed: {str(e)}")
    
    def get_video_input(self, title: str = "Select a video") -> Path:
        """
        Get video file path from user.
        
        Args:
            title: Dialog title
            
        Returns:
            Path to selected video file
        """
        try:
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                raise InputError("No video file selected")
                
            path = Path(file_path)
            if not path.exists():
                raise InputError(f"Video file not found: {file_path}")
                
            return path
            
        except Exception as e:
            logger.error(f"Failed to get video input: {str(e)}")
            raise InputError(f"Video selection failed: {str(e)}")
    
    def get_input_files(self) -> Tuple[Path, Path]:
        """
        Get both image and video file paths from user.
        
        Returns:
            Tuple of (image_path, video_path)
        """
        try:
            image_path = self.get_image_input()
            video_path = self.get_video_input()
            return image_path, video_path
            
        except Exception as e:
            logger.error(f"Failed to get input files: {str(e)}")
            raise InputError(f"Input file selection failed: {str(e)}")
    
    def __del__(self):
        """Clean up resources"""
        try:
            self.root.destroy()
        except:
            pass 