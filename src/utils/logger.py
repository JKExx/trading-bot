"""Logging utilities."""
import logging
from pathlib import Path


def create_logger(name: str, log_dir: str = "data/logs") -> logging.Logger:
    """Create a basic rotating file logger."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
