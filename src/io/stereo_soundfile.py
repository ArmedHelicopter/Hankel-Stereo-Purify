"""Shared ``soundfile`` open path with stereo channel check.

Used by the facade; keeps one place for the 2-channel rule.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf

from src.core.exceptions import AudioIOError

# Single user-visible message for non-stereo inputs (MSSA requires stereo covariance).
STEREO_REQUIRED_MSG = "MSSA 算法当前严格限定为双声道(Stereo)音频，检测到 {n} 声道"


def require_stereo_channels(channels: int, *, context: str | None = None) -> None:
    """Raise ``AudioIOError`` if ``channels`` is not exactly 2."""
    n = int(channels)
    if n != 2:
        msg = STEREO_REQUIRED_MSG.format(n=n)
        if context:
            raise AudioIOError(f"{msg} ({context})") from None
        raise AudioIOError(msg) from None


@contextmanager
def open_stereo_soundfile(path: str | Path) -> Iterator[sf.SoundFile]:
    """Open ``path`` for reading; ``AudioIOError`` if not exactly 2 channels."""
    snd: sf.SoundFile | None = None
    try:
        snd = sf.SoundFile(path)
        require_stereo_channels(int(snd.channels), context=f"path={path!r}")
        yield snd
    finally:
        if snd is not None:
            snd.close()
