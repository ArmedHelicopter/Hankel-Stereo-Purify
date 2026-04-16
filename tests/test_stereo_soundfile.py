"""Stereo-only ``soundfile`` wrapper."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.core.exceptions import AudioIOError
from src.io.stereo_soundfile import open_stereo_soundfile


def test_open_stereo_accepts_stereo(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    z = np.zeros((10, 2), dtype=np.float64)
    sf.write(p, z, 48_000, format="WAV", subtype="PCM_24")
    with open_stereo_soundfile(p) as snd:
        assert snd.channels == 2


def test_open_stereo_rejects_mono(tmp_path: Path) -> None:
    p = tmp_path / "m.wav"
    z = np.zeros((10, 1), dtype=np.float64)
    sf.write(p, z, 48_000, format="WAV", subtype="PCM_24")
    with pytest.raises(AudioIOError, match="MSSA"):
        with open_stereo_soundfile(p):
            pass
