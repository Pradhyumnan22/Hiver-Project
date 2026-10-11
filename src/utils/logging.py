"""Structured logging configuration."""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "hiver_agent",
    level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a structured console logger."""
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
