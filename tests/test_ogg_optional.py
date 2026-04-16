"""OGG/Vorbis roundtrip when libsndfile supports encoding (optional in CI)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.facade.purifier import MSSAPurifierBuilder
from src.io.sndfile_capabilities import can_write_ogg_vorbis


@pytest.fixture
def require_ogg_write() -> None:
    if not can_write_ogg_vorbis():
        pytest.skip("OGG/Vorbis write not supported by this libsndfile build")


def test_can_write_ogg_vorbis_is_bool() -> None:
    assert isinstance(can_write_ogg_vorbis(), bool)


def test_flac_to_ogg_e2e(tmp_path: Path, require_ogg_write: None) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.ogg"
    rng = np.random.default_rng(2028)
    stereo = (0.01 * rng.standard_normal((300, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(16)
        .set_truncation_rank(8)
        .set_frame_size(64)
        .set_hop_size(32)
        .build()
    )
    purifier.process_file(str(inp), str(out))

    assert out.is_file()
    y, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == 48_000
    assert y.shape[0] == stereo.shape[0]
    assert np.all(np.isfinite(y))
