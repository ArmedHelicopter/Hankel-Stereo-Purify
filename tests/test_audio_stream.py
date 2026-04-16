"""Tests for block-based FLAC reading (F-01)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from src.io.audio_stream import AudioStream, read_flac_metadata


def test_read_blocks_matches_full_read(tmp_path: Path) -> None:
    path = tmp_path / "block.flac"
    rng = np.random.default_rng(0)
    full = rng.standard_normal((5000, 2)).astype(np.float64)
    sf.write(path, full, 48_000, format="FLAC", subtype="PCM_24")

    meta = read_flac_metadata(path)
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


def test_read_flac_metadata_missing_file(tmp_path: Path) -> None:
    from src.core.exceptions import AudioIOError

    with pytest.raises(AudioIOError):
        read_flac_metadata(tmp_path / "nope.flac")
