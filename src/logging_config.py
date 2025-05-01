import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    # Create log directory
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        log_file = log_path / "forensics.log"
    else:
        log_file = None
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 