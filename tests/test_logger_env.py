"""Logger file handler toggles via ``HSP_LOG_FILE``."""

import logging
import uuid

import pytest

from src.utils.logger import get_logger


def test_hsp_log_file_none_skips_file_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSP_LOG_FILE", "none")
    name = f"hsp_log_none_{uuid.uuid4().hex}"
    logger = get_logger(name)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert logger.handlers
