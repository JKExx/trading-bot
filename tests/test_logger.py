import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Ensure the project root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.logger import create_logger


def test_create_logger_uses_rotating_file_handler(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    logger = create_logger(
        "test_logger", log_dir=str(log_dir), max_bytes=100, backup_count=2
    )

    # Ensure a rotating file handler is configured with given parameters
    assert logger.handlers, "Logger should have at least one handler"
    handler = logger.handlers[0]
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes == 100
    assert handler.backupCount == 2

    # Writing to the logger should create the log file in the specified directory
    logger.info("hello")
    assert (log_dir / "test_logger.log").exists()

    # Calling create_logger again should not add duplicate handlers
    logger_again = create_logger("test_logger", log_dir=str(log_dir))
    assert logger_again is logger
    assert len(logger_again.handlers) == 1
