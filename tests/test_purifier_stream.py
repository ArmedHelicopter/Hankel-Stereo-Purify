"""Facade process_file with OLA + MSSA (week 3)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.core.exceptions import ConfigurationError
from src.facade.purifier import AudioPurifier, MSSAPurifierBuilder


def test_builder_requires_window_length() -> None:
    with pytest.raises(ConfigurationError, match="window_length"):
        MSSAPurifierBuilder().set_truncation_rank(8).build()


def test_builder_requires_truncation_mode() -> None:
    with pytest.raises(ConfigurationError, match="truncation"):
        MSSAPurifierBuilder().set_window_length(32).build()


def test_builder_rejects_both_energy_and_rank() -> None:
    b = (
        MSSAPurifierBuilder()
        .set_window_length(32)
        .set_truncation_rank(8)
        .set_energy_fraction(0.95)
    )
    with pytest.raises(ConfigurationError, match="not both"):
        b.build()


def test_process_file_roundtrip_shape_and_finite(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    rng = np.random.default_rng(9)
    n = 400
    stereo = rng.standard_normal((n, 2)).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    wl = 16
    fs = 64
    hop = 32
    k_h = fs - wl + 1
    rank = min(wl, 2 * k_h)

    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(wl)
        .set_truncation_rank(rank)
        .set_frame_size(fs)
        .set_hop_size(hop)
        .build()
    )
    purifier.process_file(str(inp), str(out))

    y, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == 48_000
    assert y.shape == (n, 2)
    assert np.all(np.isfinite(y))


def test_process_file_uses_memmap_when_budget_tight(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    n = 800
    stereo = np.zeros((n, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    wl = 16
    fs = 64
    hop = 32
    k_h = fs - wl + 1
    rank = min(wl, 2 * k_h)

    purifier = AudioPurifier(
        wl,
        rank,
        frame_size=fs,
        hop_size=hop,
        max_working_memory_bytes=1000,
    )
    purifier.process_file(str(inp), str(out))
    y, _ = sf.read(out, dtype="float64", always_2d=True)
    assert y.shape == (n, 2)
