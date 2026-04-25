"""
VeriFace AI — Centralized Logging
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str = "verifaceai", level: str = "INFO") -> logging.Logger:
    """Configure and return a named logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (optional)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / "verifaceai.log")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# Root app logger
logger = setup_logger()
