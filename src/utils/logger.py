"""Logging utilities."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def create_logger(
    name: str,
    log_dir: str = "data/logs",
    *,
    max_bytes: int = 1_048_576,
    backup_count: int = 5,
) -> logging.Logger:
    """Create a basic rotating file logger.

    Parameters
    ----------
    name:
        Name of the logger to create or fetch.
    log_dir:
        Directory where log files should be written.
    max_bytes:
        Maximum size in bytes before the log file is rotated. Defaults to
        1MB.
    backup_count:
        Number of rotated log files to keep. Defaults to ``5``.
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RotatingFileHandler(
            Path(log_dir) / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
