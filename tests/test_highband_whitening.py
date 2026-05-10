import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import ConfigurationError
from src.core.stages.whitening import (
    estimate_noise_profile,
    roundtrip_whiten_signal,
    snr_db,
    whiten_signal,
)
from src.facade.purifier import AudioPurifier


def test_whiten_unwhiten_roundtrip_shape_finite_and_high_snr() -> None:
    rng = np.random.default_rng(20260510)
    x = (0.05 * rng.standard_normal((4096, 2))).astype(np.float64)
    profile = estimate_noise_profile(x, 48_000)

    y = roundtrip_whiten_signal(x, profile)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert snr_db(x, y) > 80.0


def test_whitening_profile_silence_is_positive_and_finite() -> None:
    x = np.zeros((512, 2), dtype=np.float64)
    profile = estimate_noise_profile(x, 8_000)
    y = whiten_signal(x, profile)
    z = roundtrip_whiten_signal(x, profile)

    assert np.all(profile.scale > 0.0)
    assert np.all(np.isfinite(profile.scale))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(z))
    assert np.allclose(z, x)


def test_fractional_alpha_roundtrip_is_reversible() -> None:
    rng = np.random.default_rng(20260511)
    x = (0.02 * rng.standard_normal((4096, 2))).astype(np.float64)
    profile = estimate_noise_profile(x, 44_100)

    z = roundtrip_whiten_signal(x, profile, alpha=0.5)

    assert z.shape == x.shape
    assert np.all(np.isfinite(z))
    assert snr_db(x, z) > 80.0


def test_highband_whiten_requires_bypass_freq() -> None:
    purifier = AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        bypass_freq=None,
        highband_whiten=True,
    )
    with pytest.raises(ConfigurationError, match="bypass_freq"):
        purifier._validate_configuration()


def test_audio_purifier_defaults_to_bpw_candidate() -> None:
    purifier = AudioPurifier(16, truncation_rank=8, frame_size=64)

    assert purifier.bypass_freq == 2_000.0
    assert purifier.highband_whiten is True
    assert purifier.whiten_alpha == 0.75


def test_audio_purifier_explicit_none_restores_fullband() -> None:
    purifier = AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        bypass_freq=None,
        highband_whiten=False,
    )

    assert purifier.bypass_freq is None
    assert purifier.highband_whiten is False


def test_highband_whiten_alpha_must_be_unit_interval() -> None:
    purifier = AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        bypass_freq=1_000.0,
        highband_whiten=True,
        whiten_alpha=1.5,
    )
    with pytest.raises(ConfigurationError, match="whiten_alpha"):
        purifier._validate_configuration()


def test_bandpass_highband_tempfiles_use_float_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.wav"
    out_plain = tmp_path / "out_plain.wav"
    out_whiten = tmp_path / "out_whiten.wav"
    artifact_dir = tmp_path / "artifacts"

    sr = 8_000
    t = np.arange(256, dtype=np.float64) / sr
    stereo = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 440.0 * t),
            0.1 * np.sin(2.0 * np.pi * 660.0 * t),
        )
    ).astype(np.float64)
    sf.write(inp, stereo, sr, format="WAV", subtype="FLOAT")

    calls: list[bool] = []

    def fake_process_high_band_tempfile(
        self: AudioPurifier,
        high_band: NDArray[np.float64],
        samplerate: int,
        denoise_frame: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        f_size: int,
        hop: int,
        w_sqrt: NDArray[np.float64],
        w_sq: NDArray[np.float64],
        *,
        write_float: bool,
    ) -> NDArray[np.float64]:
        calls.append(write_float)
        return np.asarray(high_band, dtype=np.float64)

    monkeypatch.setattr(
        AudioPurifier,
        "_process_high_band_tempfile",
        fake_process_high_band_tempfile,
    )

    AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        bypass_freq=1_000.0,
        highband_whiten=False,
    ).process_file(str(inp), str(out_plain))
    assert calls == [True]
    plain, plain_sr = sf.read(out_plain, dtype="float64", always_2d=True)
    assert plain_sr == sr
    assert plain.shape == stereo.shape
    assert np.all(np.isfinite(plain))

    calls.clear()
    AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        bypass_freq=1_000.0,
        highband_whiten=True,
        whiten_alpha=0.5,
        whitening_artifact_dir=artifact_dir,
    ).process_file(str(inp), str(out_whiten))
    assert calls == [True, True]
    whitened, whitened_sr = sf.read(out_whiten, dtype="float64", always_2d=True)
    assert whitened_sr == sr
    assert whitened.shape == stereo.shape
    assert np.all(np.isfinite(whitened))


@pytest.mark.parametrize(
    ("suffix", "write_kwargs"),
    [
        (".wav", {"format": "WAV", "subtype": "FLOAT"}),
        (".flac", {"format": "FLAC"}),
    ],
)
def test_process_file_highband_whiten_writes_artifacts(
    tmp_path: Path,
    suffix: str,
    write_kwargs: dict[str, str],
) -> None:
    inp = tmp_path / f"in{suffix}"
    out = tmp_path / f"out{suffix}"
    artifact_dir = tmp_path / f"artifacts{suffix}"

    sr = 8_000
    t = np.arange(640, dtype=np.float64) / sr
    left = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    right = 0.1 * np.sin(2.0 * np.pi * 660.0 * t)
    stereo = np.column_stack((left, right)).astype(np.float64)
    sf.write(inp, stereo, sr, **write_kwargs)

    purifier = AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        bypass_freq=1_000.0,
        highband_whiten=True,
        whiten_alpha=0.5,
        whitening_artifact_dir=artifact_dir,
    )
    purifier.process_file(str(inp), str(out))

    y, out_sr = sf.read(out, dtype="float64", always_2d=True)
    assert out_sr == sr
    assert y.shape == stereo.shape
    assert np.all(np.isfinite(y))

    expected = [
        "roundtrip.wav",
        "baseline_no_whiten.wav",
        "whitened_output.wav",
        "diff_baseline_vs_whiten.wav",
        "diff_original_vs_whiten.wav",
        "diff_original_vs_roundtrip.wav",
    ]
    for name in expected:
        p = artifact_dir / name
        assert p.is_file()
        data, artifact_sr = sf.read(p, dtype="float64", always_2d=True)
        assert artifact_sr == sr
        assert data.shape == stereo.shape
        assert np.all(np.isfinite(data))

    metrics = json.loads((artifact_dir / "metrics.json").read_text())
    assert metrics["samplerate"] == sr
    assert metrics["bypass_freq"] == 1_000.0
    assert metrics["whiten_alpha"] == 0.5
    assert metrics["roundtrip_snr_db"] > 80.0
