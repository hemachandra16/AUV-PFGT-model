"""Logging utilities for PFGT-UIE training.

Sets up a structured console + file logger with timestamps. Ensures that
logs from every run are preserved in logs/ for post-hoc analysis.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    log_dir: str | Path = "logs",
    log_file: str = "train.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the root logger with console and file handlers.

    Args:
        log_dir:  Directory to write the log file. Created if it does not exist.
        log_file: Name of the log file inside log_dir.
        level:    Logging level (default INFO).

    Returns:
        The configured root logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. during resuming)
    if root_logger.handlers:
        root_logger.handlers.clear()

    import io
    # Wrap stdout in UTF-8 for Windows compatibility
    safe_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout

    # Console handler
    console_handler = logging.StreamHandler(safe_stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return root_logger
