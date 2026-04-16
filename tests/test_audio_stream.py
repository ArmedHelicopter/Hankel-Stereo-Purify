"""Tests for block-based FLAC reading (F-01)."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError
from src.io.audio_stream import AudioStream, read_audio_metadata


def test_read_blocks_matches_full_read(tmp_path: Path) -> None:
    path = tmp_path / "block.flac"
    rng = np.random.default_rng(0)
    full = rng.standard_normal((5000, 2)).astype(np.float64)
    sf.write(path, full, 48_000, format="FLAC", subtype="PCM_24")

    meta = read_audio_metadata(path)
    assert meta["frames"] == 5000
    assert meta["channels"] == 2
    assert meta["samplerate"] == 48_000

    ref, _ = sf.read(path, dtype="float64", always_2d=True)
    chunks: list[NDArray[np.float64]] = []
    stream = AudioStream(str(path), block_size=1000)
    for block in stream.read_blocks():
        chunks.append(np.asarray(block, dtype=np.float64))
    stacked = np.vstack(chunks)
    np.testing.assert_allclose(stacked, ref)


def test_read_audio_metadata_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioIOError):
        read_audio_metadata(tmp_path / "nope.flac")


def test_read_audio_metadata_io_trace_logs_when_enabled(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "st.flac"
    stereo = np.zeros((50, 2), dtype=np.float64)
    sf.write(path, stereo, 48_000, format="FLAC", subtype="PCM_24")
    caplog.set_level(logging.INFO)
    with patch.dict(os.environ, {"HSP_LOG_IO_TRACE": "1"}):
        read_audio_metadata(path)
    assert "read_audio_metadata opening" in caplog.text


def test_audiostream_rejects_non_positive_block_size(tmp_path: Path) -> None:
    with pytest.raises(AudioIOError, match="block_size must be positive"):
        AudioStream(str(tmp_path / "x.flac"), block_size=0)


def test_read_blocks_requires_stereo(tmp_path: Path) -> None:
    path = tmp_path / "mono.flac"
    mono = np.zeros((100, 1), dtype=np.float64)
    sf.write(path, mono, 48_000, format="FLAC", subtype="PCM_24")
    stream = AudioStream(str(path), block_size=50)
    with pytest.raises(AudioIOError, match="MSSA"):
        list(stream.read_blocks())
