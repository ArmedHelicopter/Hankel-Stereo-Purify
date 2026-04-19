"""Windowing strategies for OLA (Phase0 placeholder)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_analysis_window(
    frame: NDArray[np.float64],
    *,
    window_length: int,
) -> NDArray[np.float64]:
    """Apply analysis window to a frame. Full implementation on ``tutorial``."""
    raise NotImplementedError(
        "Full windowing implementation lives on branch ``tutorial``."
    )
