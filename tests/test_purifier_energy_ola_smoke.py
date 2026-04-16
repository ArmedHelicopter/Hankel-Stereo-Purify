"""Integration smoke: energy truncation + OLA multi-frame ``process_file``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.facade.purifier import MSSAPurifierBuilder


def test_process_file_energy_fraction_ola_short_stereo(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    rng = np.random.default_rng(2026)
    n = 2000
    stereo = (0.02 * rng.standard_normal((n, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")
    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(32)
        .set_energy_fraction(0.95)
        .set_frame_size(128)
        .set_hop_size(64)
        .set_max_working_memory_bytes(50 * 1024 * 1024)
        .build()
    )
    purifier.process_file(str(inp), str(out))
    assert out.is_file()
    with sf.SoundFile(out) as f:
        assert int(f.frames) == n
        assert f.channels == 2
