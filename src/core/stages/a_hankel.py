"""Hankel embedding (module A)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hankel_embed(
    series: NDArray[np.float64],
    window_length: int,
) -> NDArray[np.float64]:
    """Build Hankel matrix from a 1D series. Full implementation on ``tutorial``."""
    raise NotImplementedError("Full Hankel embedding lives on branch ``tutorial``.")
