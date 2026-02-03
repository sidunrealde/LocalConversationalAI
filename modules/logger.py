"""
Logging configuration for the Local Conversational AI Agent.
Provides consistent logging across all modules with file and console output.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setup and return a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory to store log files
    
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # File handler (DEBUG level, more verbose)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"agent_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (INFO level, less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.debug(f"Logger initialized: {name}")
    return logger
