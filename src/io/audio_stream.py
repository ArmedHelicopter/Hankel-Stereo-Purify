"""Block-based streaming read for FLAC/PCM via libsndfile (PRD F-01 sequential decode).

`read_flac_metadata` avoids loading PCM; `read_blocks` yields float64 chunks for callers
that iterate without holding the whole file.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from src.core.exceptions import AudioIOError


class AudioStream:
    """Audio streaming helper for block-based file reading (F-01)."""

    def __init__(self, path: str | Path, block_size: int = 4096) -> None:
        self.path = str(path)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise AudioIOError("block_size must be positive.")

    def read_blocks(self) -> Iterator[NDArray[np.float64]]:
        """Yield contiguous PCM blocks as float64, shape (frames, channels)."""
        try:
            with sf.SoundFile(self.path) as snd:
                for block in snd.blocks(blocksize=self.block_size):
                    x = np.asarray(block, dtype=np.float64, order="C")
                    if x.ndim == 1:
                        x = np.ascontiguousarray(x[:, np.newaxis])
                    yield x
        except OSError as exc:
            raise AudioIOError(f"Failed to read audio stream: {self.path}") from exc


def read_flac_metadata(path: str | Path) -> dict[str, Any]:
    """Return frames, samplerate, channels without loading PCM into RAM."""
    p = Path(path)
    try:
        with sf.SoundFile(p) as snd:
            return {
                "frames": int(snd.frames),
                "samplerate": int(snd.samplerate),
                "channels": int(snd.channels),
            }
    except Exception as exc:
        raise AudioIOError(f"Failed to open audio file: {p}") from exc
