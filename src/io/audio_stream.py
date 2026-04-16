"""Block-based streaming read via libsndfile (PRD F-01 sequential decode).

`read_audio_metadata` opens the file only for header fields; `read_blocks` yields
float64 chunks for callers that iterate without holding the whole file.

`AudioStream.read_blocks` opens with ``soundfile.SoundFile`` and enforces exactly
two channels: MSSA denoising requires stereo cross-covariance structure; mono or
surround inputs are rejected at I/O with ``AudioIOError``.

There is no portable timeout around libsndfile blocking reads: slow or stuck
network filesystems may hang the process. Set ``HSP_LOG_IO_TRACE`` (see README)
for one-shot debug lines when metadata/block iteration opens a path.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError
from src.io.io_messages import (
    block_size_must_be_positive,
    failed_to_open_audio_file,
    failed_to_read_audio_stream,
)
from src.io.stereo_soundfile import require_stereo_channels

_LOG = logging.getLogger(__name__)


def _io_trace_log(msg: str, *args: object) -> None:
    if os.environ.get("HSP_LOG_IO_TRACE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    _LOG.info(msg, *args)


class AudioStream:
    """Stereo block stream (2 ch.) for MSSA / ``AudioPurifier``."""

    def __init__(self, path: str | Path, block_size: int = 4096) -> None:
        self.path = str(path)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise AudioIOError(block_size_must_be_positive())

    def read_blocks(self) -> Iterator[NDArray[np.float64]]:
        """Yield contiguous PCM blocks as float64, shape (frames, 2)."""
        _io_trace_log(
            "AudioStream.read_blocks opening path=%r block_size=%s",
            self.path,
            self.block_size,
        )
        try:
            with sf.SoundFile(self.path) as snd:
                require_stereo_channels(
                    snd.channels,
                    context=f"AudioStream={self.path!r}",
                )
                for block in snd.blocks(blocksize=self.block_size):
                    x = np.asarray(block, dtype=np.float64, order="C")
                    if x.ndim == 1:
                        x = np.ascontiguousarray(x[:, np.newaxis])
                    yield x
        except AudioIOError:
            raise
        except (OSError, sf.LibsndfileError) as exc:
            raise AudioIOError(failed_to_read_audio_stream(self.path)) from exc


def read_audio_metadata(path: str | Path) -> dict[str, Any]:
    """Return frames, samplerate, channels without loading PCM into RAM.

    Works for any format libsndfile can open (see `audio_formats` whitelist for CLI).
    """
    p = Path(path)
    _io_trace_log("read_audio_metadata opening path=%r", str(p))
    try:
        with sf.SoundFile(p) as snd:
            return {
                "frames": int(snd.frames),
                "samplerate": int(snd.samplerate),
                "channels": int(snd.channels),
            }
    except (OSError, sf.LibsndfileError) as exc:
        raise AudioIOError(failed_to_open_audio_file(str(p))) from exc
