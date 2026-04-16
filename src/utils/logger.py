import logging
import os
import warnings
from pathlib import Path

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
            # Delegate to logging: must not raise from emit (avoid handler recursion).
            self.handleError(record)


def _ensure_log_dir(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)


def _resolved_log_file_path(
    log_file: str | None,
) -> Path | None:
    """Return path for file logging, or ``None`` to skip the file handler."""
    env = os.environ.get("HSP_LOG_FILE")
    if env is not None:
        key = env.strip().lower()
        if key in ("", "none", "0", "false", "off"):
            return None
        return Path(env)
    if log_file is not None:
        if log_file.strip() == "":
            return None
        return Path(log_file)
    return Path("logs/purify.log")


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
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

    file_target = _resolved_log_file_path(log_file)
    if file_target is not None:
        try:
            _ensure_log_dir(file_target)
            file_handler = logging.FileHandler(file_target, encoding="utf-8")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            warnings.warn(
                f"Skipping file log ({file_target}): {exc}",
                UserWarning,
                stacklevel=1,
            )

    return logger
