"""Generic overlap-add helpers (Phase0 placeholder)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def overlap_add_merge(
    frames: NDArray[np.float64],
    *,
    hop: int,
    frame_length: int,
) -> NDArray[np.float64]:
    """Merge windowed frames (OLA). Full implementation on ``tutorial``."""
    _ = (frames, hop, frame_length)
    raise NotImplementedError("Full OLA merge lives on branch ``tutorial``.")
