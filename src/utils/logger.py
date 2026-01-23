import logging
import sys


def setup_logger(name: str = "detr", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with console output.

    Args:
        name: Name of the logger.
        level: Logging level.

    Returns:
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
