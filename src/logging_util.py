"""
**Author**:
    Fitz Koch
**Created**:
    2025-11-25
**Description**:
    Sets up logging
"""

# library imports
import logging
from datetime import datetime
from pathlib import Path


def set_up_logging(
    save_path: str | Path = "logs", name="lion_gazelle_log"
) -> logging.Logger:
    timestamp = datetime.now().strftime("%d-%m%-Y_%H:%M")
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{name}_{timestamp}"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%d-%m-%Y_%H:%M"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
