"""Facade process_file with OLA + MSSA (week 3)."""

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from src.core.exceptions import AudioIOError, ConfigurationError, ProcessingError
from src.facade.purifier import AudioPurifier


def _purifier_std() -> AudioPurifier:
    return AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )


def test_shutdown_pcm_producer_warns_when_thread_still_alive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )

    class _FakeThread:
        name = "HSP-AudioProducer"

        def join(self, timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return True

    fake = _FakeThread()
    pcm_queue: queue.Queue[Any] = queue.Queue()
    abort_event = threading.Event()
    purifier._ola_engine._shutdown_pcm_producer(
        abort_event, pcm_queue, cast(threading.Thread, fake)
    )
    assert "did not finish" in caplog.text


def test_process_file_rejects_input_longer_than_max_samples(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    sf.write(
        inp,
        np.zeros((300, 2), dtype=np.float64),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
        max_input_samples=100,
    )
    with pytest.raises(ConfigurationError, match="limit is 100"):
        purifier.process_file(str(inp), str(out))


def test_process_file_valueerror_maps_to_processing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")
    purifier = _purifier_std()

    def _boom(_i: str, _o: str) -> None:
        raise ValueError("mock stage failure")

    monkeypatch.setattr(purifier, "_run_processing", _boom)
    with pytest.raises(ProcessingError, match="constraint") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert exc_info.value.origin_exception_type == "builtins.ValueError"


def test_purifier_rejects_nonpositive_window_length() -> None:
    with pytest.raises(ConfigurationError, match="window_length"):
        AudioPurifier(0, truncation_rank=8)


def test_purifier_requires_truncation_mode() -> None:
    with pytest.raises(ConfigurationError, match="truncation"):
        AudioPurifier(32)


def test_process_file_rejects_same_path_as_output(tmp_path: Path) -> None:
    p = tmp_path / "same.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(p, stereo, 48_000, format="FLAC", subtype="PCM_24")
    purifier = _purifier_std()
    with pytest.raises(ConfigurationError, match="differ"):
        purifier.process_file(str(p), str(p))


def test_process_file_rejects_when_samefile_unavailable(tmp_path: Path) -> None:
    src = tmp_path / "a.flac"
    dst = tmp_path / "b.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(src, stereo, 48_000, format="FLAC", subtype="PCM_24")
    sf.write(dst, stereo, 48_000, format="FLAC", subtype="PCM_24")
    purifier = _purifier_std()
    with patch("src.facade.purifier.os.path.samefile", side_effect=OSError("mock")):
        with pytest.raises(ConfigurationError, match="Cannot verify"):
            purifier.process_file(str(src), str(dst))


def test_process_file_rejects_hardlinked_output(tmp_path: Path) -> None:
    src = tmp_path / "a.flac"
    link = tmp_path / "b.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(src, stereo, 48_000, format="FLAC", subtype="PCM_24")
    os.link(src, link)
    purifier = _purifier_std()
    with pytest.raises(ConfigurationError, match="same file"):
        purifier.process_file(str(src), str(link))


def test_process_file_rejects_unsupported_input_suffix(tmp_path: Path) -> None:
    purifier = _purifier_std()
    with pytest.raises(ConfigurationError, match="Unsupported input"):
        purifier.process_file(
            str(tmp_path / "in.mp3"),
            str(tmp_path / "out.wav"),
        )


def test_process_file_linalg_failure_maps_to_processing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _boom(*_a: object, **_k: object) -> None:
        raise np.linalg.LinAlgError("mock SVD failure")

    # SVD step uses ``svds`` / ``scipy.linalg.svd``, not ``numpy.linalg.svd``.
    monkeypatch.setattr("src.core.stages.c_svd.svds", _boom)
    monkeypatch.setattr("scipy.linalg.svd", _boom)
    purifier = _purifier_std()
    with pytest.raises(ProcessingError, match="numerical") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert exc_info.value.origin_exception_type == "numpy.linalg.LinAlgError"


def test_process_file_arpack_failure_maps_to_processing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scipy.sparse.linalg import ArpackError  # type: ignore[import-untyped]

    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _arpack_boom(*_a: object, **_k: object) -> None:
        raise ArpackError(0, {0: "mock ARPACK failure"})

    monkeypatch.setattr("src.core.stages.c_svd.svds", _arpack_boom)
    monkeypatch.setattr("scipy.linalg.svd", _arpack_boom)
    purifier = _purifier_std()
    with pytest.raises(ProcessingError, match="numerical") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert (
        exc_info.value.origin_exception_type
        == "scipy.sparse.linalg._eigen.arpack.arpack.ArpackError"
    )


def test_process_file_corrupt_input_raises_audio_io_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a valid wav")
    out = tmp_path / "out.wav"
    purifier = _purifier_std()
    with pytest.raises(AudioIOError, match="Audio I/O failed"):
        purifier.process_file(str(bad), str(out))


def test_process_file_wav_to_wav_roundtrip(tmp_path: Path) -> None:
    inp = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    rng = np.random.default_rng(11)
    n = 400
    stereo = (0.01 * rng.standard_normal((n, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="WAV", subtype="PCM_24")

    wl = 16
    fs = 64
    hop = 32
    k_h = fs - wl + 1
    rank = min(wl, 2 * k_h)

    purifier = AudioPurifier(
        window_length=wl,
        truncation_rank=rank,
        frame_size=fs,
        hop_size=hop,
    )
    purifier.process_file(str(inp), str(out))

    y, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == 48_000
    assert y.shape == (n, 2)
    assert np.all(np.isfinite(y))


def test_purifier_rejects_both_energy_and_rank() -> None:
    with pytest.raises(ConfigurationError, match="not both"):
        AudioPurifier(
            32,
            truncation_rank=8,
            energy_fraction=0.95,
        )


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

    purifier = AudioPurifier(
        window_length=wl,
        truncation_rank=rank,
        frame_size=fs,
        hop_size=hop,
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
        truncation_rank=rank,
        frame_size=fs,
        hop_size=hop,
        max_working_memory_bytes=1000,
    )
    purifier.process_file(str(inp), str(out))
    y, _ = sf.read(out, dtype="float64", always_2d=True)
    assert y.shape == (n, 2)


def test_process_file_unexpected_error_maps_to_processing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _boom(self: AudioPurifier, _i: str, _o: str) -> None:
        raise KeyError("mock internal bug")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _boom)
    purifier = _purifier_std()
    with pytest.raises(ProcessingError, match="Unexpected error") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert isinstance(exc_info.value.__cause__, KeyError)
    assert exc_info.value.origin_exception_type == "builtins.KeyError"


def test_memmap_allocation_failure_maps_to_audio_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    n = 800
    stereo = np.zeros((n, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fail_memmap(*_a: object, **_k: object) -> None:
        raise OSError("mock memmap allocation failure")

    monkeypatch.setattr("src.facade.purifier.np.memmap", _fail_memmap)

    purifier = AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
        max_working_memory_bytes=1000,
    )
    with pytest.raises(AudioIOError, match="temp buffers"):
        purifier.process_file(str(inp), str(out))


def test_memmap_second_allocation_failure_maps_to_audio_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    n = 800
    stereo = np.zeros((n, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    real_memmap = np.memmap
    calls = {"n": 0}

    def _memmap_maybe_fail(*a: object, **k: object) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("mock second memmap failure")
        return real_memmap(*a, **k)  # type: ignore[call-overload]

    monkeypatch.setattr("src.facade.purifier.np.memmap", _memmap_maybe_fail)

    purifier = AudioPurifier(
        16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
        max_working_memory_bytes=1000,
    )
    with pytest.raises(AudioIOError, match="temp buffers"):
        purifier.process_file(str(inp), str(out))


def test_process_file_sf_write_failure_maps_to_audio_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fail_write(*_a: object, **_k: object) -> None:
        raise OSError("mock write failure")

    monkeypatch.setattr("src.facade.purifier.sf.write", _fail_write)

    purifier = _purifier_std()
    with pytest.raises(AudioIOError, match="Audio I/O failed") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert isinstance(exc_info.value.__cause__, OSError)


def test_process_file_energy_fraction_roundtrip(tmp_path: Path) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    rng = np.random.default_rng(2027)
    stereo = (0.02 * rng.standard_normal((600, 2))).astype(np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    purifier = AudioPurifier(
        window_length=32,
        energy_fraction=0.95,
        frame_size=128,
        hop_size=64,
        max_working_memory_bytes=500_000_000,
    )
    purifier.process_file(str(inp), str(out))
    y, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == 48_000
    assert y.shape == stereo.shape
    assert np.all(np.isfinite(y))
