import numpy as np
from numpy.typing import NDArray

from ..array_types import FloatArray

_IntIdx = NDArray[np.intp]


def _t_flat_and_cnt(m: int, n: int) -> tuple[_IntIdx, NDArray[np.float64], int]:
    """Anti-diagonal index ``i+j`` per matrix cell (row-major) and counts per ``t``."""
    minlength = m + n - 1
    ri, rj = np.arange(m, dtype=np.intp), np.arange(n, dtype=np.intp)
    t_flat = np.add.outer(ri, rj).ravel()
    cnt = np.bincount(t_flat, minlength=minlength).astype(np.float64)
    return t_flat, cnt, minlength


def _plane_diagonal_average(
    plane: NDArray[np.float64],
    t_flat: _IntIdx,
    cnt: NDArray[np.float64],
    minlength: int,
) -> NDArray[np.float64]:
    s = np.bincount(t_flat, weights=plane.ravel(), minlength=minlength)
    return s / cnt


def batched_diagonal_average(mats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Batch anti-diagonal average (same rule as ``fast_diagonal_average``).

    For each batch slice ``A`` of shape ``(m, n)`` and each ``t`` in ``0..m+n-2``,
    averages ``A[i, j]`` over all ``(i, j)`` with ``i + j == t`` (Hankel
    anti-diagonals). Per-slice aggregation uses ``numpy.bincount`` over the fixed
    ``t_flat = i+j`` map; only the batch dimension is iterated in Python (``B``
    is small, e.g. stereo ``B=2``).

    Parameters
    ----------
    mats
        Shape ``(B, m, n)``, C-contiguous recommended.

    Returns
    -------
    Array of shape ``(B, m + n - 1)``.
    """
    a = np.asarray(mats, dtype=np.float64, order="C")
    if a.ndim != 3:
        raise ValueError("batched_diagonal_average expects a 3D array (B, m, n).")
    b, m, n = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    if m == 0 or n == 0:
        return np.zeros((b, 0), dtype=np.float64)
    t_flat, cnt, minlength = _t_flat_and_cnt(m, n)
    out = np.empty((b, minlength), dtype=np.float64)
    for bi in range(b):
        out[bi] = _plane_diagonal_average(a[bi], t_flat, cnt, minlength)
    return out


def fast_diagonal_average(matrix: FloatArray) -> FloatArray:
    """Anti-diagonal average: map L x K Hankel block to length-(L+K-1) series."""
    a = np.asarray(matrix, dtype=np.float64, order="C")
    if a.ndim != 2:
        raise ValueError("fast_diagonal_average expects a 2D matrix.")
    m, n = int(a.shape[0]), int(a.shape[1])
    if m == 0 or n == 0:
        return np.zeros(0, dtype=np.float64)
    t_flat, cnt, minlength = _t_flat_and_cnt(m, n)
    out = _plane_diagonal_average(a, t_flat, cnt, minlength)
    return np.asarray(out, dtype=np.float64)


def diagonal_reconstruct(data: FloatArray) -> FloatArray:
    """Reconstruct the denoised time series from the truncated joint matrix.

    Left/right Hankel blocks share one ``batched_diagonal_average`` call (batch
    dim ``B=2``).
    """
    _, cols = data.shape
    if cols % 2 != 0:
        raise ValueError("Joint matrix must have an even number of columns.")
    k_dim = cols // 2
    h_l = data[:, :k_dim]
    h_r = data[:, k_dim:]
    both = np.stack((h_l, h_r), axis=0)
    avg2 = batched_diagonal_average(both)
    left = np.asarray(avg2[0], dtype=np.float64)
    right = np.asarray(avg2[1], dtype=np.float64)
    return np.column_stack((left, right))
