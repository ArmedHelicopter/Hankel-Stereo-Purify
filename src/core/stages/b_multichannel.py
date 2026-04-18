import numpy as np

from ..array_types import FloatArray


def combine_hankel_blocks(
    h_l: FloatArray,
    h_r: FloatArray,
) -> FloatArray:
    """Stack left/right Hankel blocks into joint matrix (L, 2K)."""
    if h_l.shape != h_r.shape:
        raise ValueError("Left and right Hankel matrices must have the same shape.")
    return np.ascontiguousarray(np.hstack((h_l, h_r)))
