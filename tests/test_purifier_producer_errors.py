"""Tests for producer queue protocol and ``process_file`` exception mapping."""

from __future__ import annotations

from pathlib import Path
from queue import Queue

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError, ProcessingError
from src.facade.purifier import AudioPurifier, MSSAPurifierBuilder


def test_raise_if_producer_failed_raises_first_audio_io_error() -> None:
    err = AudioIOError("first failure")
    with pytest.raises(AudioIOError, match="first failure"):
        AudioPurifier._raise_if_producer_failed([err])


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
    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(16)
        .set_truncation_rank(8)
        .set_frame_size(64)
        .set_hop_size(32)
        .build()
    )
    with pytest.raises(AudioIOError, match="mock producer stream failure"):
        purifier.process_file(str(inp), str(out))


def test_process_file_numpy_runtime_error_maps_to_processing_numerical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        exc_type = type(
            "NumpyLikeRuntime",
            (RuntimeError,),
            {"__module__": "numpy.linalg"},
        )
        raise exc_type("mock numpy runtime")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(16)
        .set_truncation_rank(8)
        .set_frame_size(64)
        .set_hop_size(32)
        .build()
    )
    with pytest.raises(ProcessingError, match="numerical"):
        purifier.process_file(str(inp), str(out))


def test_process_file_scipy_runtime_error_maps_to_processing_numerical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        exc_type = type(
            "ScipyLikeRuntime",
            (RuntimeError,),
            {"__module__": "scipy.linalg"},
        )
        raise exc_type("mock scipy runtime")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(16)
        .set_truncation_rank(8)
        .set_frame_size(64)
        .set_hop_size(32)
        .build()
    )
    with pytest.raises(ProcessingError, match="numerical"):
        purifier.process_file(str(inp), str(out))


def test_process_file_plain_runtime_error_maps_to_constraint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = tmp_path / "in.flac"
    out = tmp_path / "out.flac"
    stereo = np.zeros((200, 2), dtype=np.float64)
    sf.write(inp, stereo, 48_000, format="FLAC", subtype="PCM_24")

    def _fake_run_processing(self: AudioPurifier, _i: str, _o: str) -> None:
        raise RuntimeError("non-numpy runtime")

    monkeypatch.setattr(AudioPurifier, "_run_processing", _fake_run_processing)
    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(16)
        .set_truncation_rank(8)
        .set_frame_size(64)
        .set_hop_size(32)
        .build()
    )
    with pytest.raises(ProcessingError, match="outside numpy/scipy"):
        purifier.process_file(str(inp), str(out))


def test_process_file_numpy_submodule_runtime_error_maps_to_processing_numerical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``numpy.core.*`` etc. must still match the numpy numerical prefix rule."""
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
    purifier = (
        MSSAPurifierBuilder()
        .set_window_length(16)
        .set_truncation_rank(8)
        .set_frame_size(64)
        .set_hop_size(32)
        .build()
    )
    with pytest.raises(ProcessingError, match="numerical"):
        purifier.process_file(str(inp), str(out))
