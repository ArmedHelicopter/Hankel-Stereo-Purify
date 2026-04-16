import numpy as np
from numpy.typing import NDArray

from ..pipeline import FloatArray, MSSAStage


def _diagonal_average_channel(h: FloatArray) -> NDArray[np.float64]:
    """Map an L x K matrix to a length-(L+K-1) series by anti-diagonal averaging."""
    l_row, k_dim = int(h.shape[0]), int(h.shape[1])
    n_out = l_row + k_dim - 1
    sums = np.zeros(n_out, dtype=np.float64)
    counts = np.zeros(n_out, dtype=np.float64)
    for i in range(l_row):
        for j in range(k_dim):
            idx = i + j
            sums[idx] += h[i, j]
            counts[idx] += 1.0
    return sums / counts


class DDiagonalStage(MSSAStage[FloatArray, FloatArray]):
    """Diagonal averaging reconstruction stage for MSSA.

    Input: FloatArray, shape (L, 2K)
    Output: FloatArray, shape (N, 2)
    """

    def execute(self, data: FloatArray) -> FloatArray:
        """Reconstruct the denoised time series from the truncated matrix."""
        _, cols = data.shape
        if cols % 2 != 0:
            raise ValueError("Joint matrix must have an even number of columns.")
        k_dim = cols // 2
        h_l = data[:, :k_dim]
        h_r = data[:, k_dim:]
        left = _diagonal_average_channel(h_l)
        right = _diagonal_average_channel(h_r)
        return np.column_stack((left, right))
