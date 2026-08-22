"""
Logging configuration and utilities
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration
    
    Args:
        config: Logging configuration dictionary
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.get("level", "INFO")),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "ai_vision.log")
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)