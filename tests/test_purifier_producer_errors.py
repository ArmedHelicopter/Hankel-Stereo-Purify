"""Tests for producer queue protocol and ``process_file`` exception mapping."""

from __future__ import annotations

from pathlib import Path
from queue import Queue

import numpy as np
import pytest
import scipy.linalg  # type: ignore[import-untyped]
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError, ProcessingError
from src.facade.purifier import AudioPurifier
from src.facade.soundfile_ola import SoundfileOlaEngine


def test_raise_if_producer_failed_raises_first_audio_io_error() -> None:
    err = AudioIOError("first failure")
    with pytest.raises(AudioIOError, match="first failure"):
        SoundfileOlaEngine._raise_if_producer_failed([err])


def test_mocked_producer_fill_sets_error_and_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer thread only records I/O error and sends poison pill (no PCM blocks)."""
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_producer_fill_queue(
        _input_path: str,
        _block_size: int,
        pcm_queue: Queue[NDArray[np.float64] | None],
        producer_error: list[BaseException],
        _abort_event: object,
    ) -> None:
        producer_error.append(AudioIOError("mock producer stream failure"))
        pcm_queue.put(None)

    monkeypatch.setattr(
        "src.facade.purifier._producer_fill_queue",
        _fake_producer_fill_queue,
    )
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )
    with pytest.raises(AudioIOError, match="mock producer stream failure"):
        purifier.process_file(str(inp), str(out))


def test_process_file_numpy_linalg_error_maps_to_processing_numerical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        raise np.linalg.LinAlgError("mock numpy linalg")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )
    with pytest.raises(ProcessingError, match="numerical") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert exc_info.value.origin_exception_type == "numpy.linalg.LinAlgError"


def test_process_file_scipy_linalg_error_maps_to_processing_numerical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        raise scipy.linalg.LinAlgError("mock scipy linalg")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )
    with pytest.raises(ProcessingError, match="numerical") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert exc_info.value.origin_exception_type == "numpy.linalg.LinAlgError"


def test_process_file_plain_runtime_error_maps_to_unexpected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        raise RuntimeError("non-linalg runtime")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )
    with pytest.raises(ProcessingError, match="Unexpected error") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert exc_info.value.origin_exception_type == "builtins.RuntimeError"


def test_process_file_numpy_core_runtime_error_maps_to_unexpected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        exc_type = type(
            "NumpyCoreRuntime",
            (RuntimeError,),
            {"__module__": "numpy.core.umath"},
        )
        raise exc_type("mock numpy.core runtime")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = AudioPurifier(
        window_length=16,
        truncation_rank=8,
        frame_size=64,
        hop_size=32,
    )
    with pytest.raises(ProcessingError, match="Unexpected error") as exc_info:
        purifier.process_file(str(inp), str(out))
    assert exc_info.value.origin_exception_type == "numpy.core.umath.NumpyCoreRuntime"
