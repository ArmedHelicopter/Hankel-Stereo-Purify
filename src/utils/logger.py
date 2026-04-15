import logging
from pathlib import Path
from typing import Optional

import colorlog
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm to avoid terminal tearing."""

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def _ensure_log_dir(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    stream_handler = TqdmLoggingHandler()
    stream_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    file_target = Path(log_file or "logs/purify.log")
    _ensure_log_dir(file_target)
    file_handler = logging.FileHandler(file_target, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
