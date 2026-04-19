"""Single-frame MSSA chain orchestration (Phase0 placeholder)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def process_frame(
    window_length: int,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    svd_step: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run MSSA for one stereo frame. Full implementation on ``tutorial``."""
    _ = (window_length, left, right, svd_step)
    raise NotImplementedError(
        "Full process_frame implementation lives on branch ``tutorial``."
    )
