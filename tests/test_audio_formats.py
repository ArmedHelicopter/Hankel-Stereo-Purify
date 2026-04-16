"""Unit tests for audio format whitelist (libsndfile-backed)."""

from pathlib import Path

import pytest

from src.core.exceptions import ConfigurationError
from src.io.audio_formats import (
    soundfile_write_kwargs,
    validate_io_paths,
)


def test_validate_accepts_flac_wav_pairs(tmp_path: Path) -> None:
    inp = tmp_path / "a.flac"
    out = tmp_path / "b.wav"
    inp.touch()
    validate_io_paths(inp, out)


def test_validate_rejects_bad_input_suffix(tmp_path: Path) -> None:
    inp = tmp_path / "a.mp3"
    out = tmp_path / "b.wav"
    inp.touch()
    with pytest.raises(ConfigurationError, match="Unsupported input"):
        validate_io_paths(inp, out)


def test_validate_rejects_bad_output_suffix(tmp_path: Path) -> None:
    inp = tmp_path / "a.wav"
    out = tmp_path / "b.mp3"
    inp.touch()
    with pytest.raises(ConfigurationError, match="Unsupported output"):
        validate_io_paths(inp, out)


def test_soundfile_write_kwargs_flac_and_wav(tmp_path: Path) -> None:
    assert soundfile_write_kwargs(tmp_path / "x.flac") == {
        "format": "FLAC",
        "subtype": "PCM_24",
    }
    assert soundfile_write_kwargs(str(tmp_path / "y.WAV")) == {
        "format": "WAV",
        "subtype": "PCM_24",
    }


def test_soundfile_write_kwargs_rejects_missing_suffix() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported output"):
        soundfile_write_kwargs("no-extension")
