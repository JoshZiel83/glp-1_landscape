"""
Centralized logging configuration.

Provides:
- Daily rotating log files
- Configurable log levels
- Optional console output
- Structured log formatting
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    log_to_console: Optional[bool] = None,
    retention_days: int = 30
) -> None:
    """
    Configure application-wide logging with daily rotation.

    Args:
        log_dir: Directory for log files (default: ./logs)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_console: Whether to also log to console (default: True)
        retention_days: Number of days to retain logs (default: 30)
    """
    # Get configuration from environment or use defaults
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "./logs")

    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if log_to_console is None:
        log_to_console = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"

    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create log file path
    log_file = log_path / "glp-1-landscape.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler with daily rotation
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler (optional)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Log initial message
    root_logger.info("=" * 80)
    root_logger.info("Logging initialized")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log directory: {log_dir}")
    root_logger.info(f"Console logging: {log_to_console}")
    root_logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Initialize logging when module is imported
# This ensures logging is set up before any other modules use it
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s"
    )
    logging.error(f"Failed to initialize logging configuration: {e}")
