"""Diagonal averaging (module D)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def diagonal_reconstruct(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map denoised block matrix back to 1D via diagonal averaging."""
    raise NotImplementedError(
        "Full diagonal reconstruction lives on branch ``tutorial``."
    )
