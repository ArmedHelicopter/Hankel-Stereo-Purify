"""Audio I/O: format allowlist, streaming reads, stereo wiring (Phase0 placeholder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_stereo_pcm_head(
    path: Path,
    *,
    max_frames: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a bounded prefix of stereo PCM (placeholder API)."""
    _ = (path, max_frames)
    raise NotImplementedError("Full I/O implementation lives on branch ``tutorial``.")
