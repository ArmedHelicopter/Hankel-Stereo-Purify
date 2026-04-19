"""SVD step factory (module C)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.core.strategies.truncation import TruncationStrategy


def make_svd_step(
    strategy: TruncationStrategy,
    *,
    window_length: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Build per-frame SVD + truncation callable (``tutorial``)."""
    raise NotImplementedError("Full SVD step lives on branch ``tutorial``.")
