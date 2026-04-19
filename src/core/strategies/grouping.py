"""W-correlation grouping helpers (Phase0 placeholder)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def group_components_by_w_correlation(
    components: NDArray[np.float64],
    *,
    window_length: int,
) -> NDArray[np.intp]:
    """Group SSA components using W-correlation. Full implementation on ``tutorial``."""
    raise NotImplementedError(
        "Full grouping implementation lives on branch ``tutorial``."
    )
