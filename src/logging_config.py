import logging
from pathlib import Path
from typing import Optional
import sys

def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    if log_dir and log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create handlers
    handlers = []
    
    if log_to_file and log_dir:
        file_handler = logging.FileHandler(log_dir / 'app.log')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 