import numpy as np

from ..pipeline import FloatArray, MSSAStage


class DDiagonalStage(MSSAStage[FloatArray, FloatArray]):
    """Diagonal averaging reconstruction stage for MSSA.

    Input: FloatArray, shape (L, 2K)
    Output: FloatArray, shape (N, 2)
    """

    @staticmethod
    def fast_diagonal_average(matrix: FloatArray) -> FloatArray:
        """Anti-diagonal average: map L x K Hankel block to length-(L+K-1) series."""
        a = np.asarray(matrix, dtype=np.float64, order="C")
        if a.ndim != 2:
            raise ValueError("fast_diagonal_average expects a 2D matrix.")
        m, n = int(a.shape[0]), int(a.shape[1])
        if m == 0 or n == 0:
            return np.zeros(0, dtype=np.float64)
        # Anti-diagonal index t = i + j for each (i,j). Avoid full np.indices((m,n))
        # (two large int64 grids); accumulate per t in O(m*n) time with lower peak RSS.
        minlength = m + n - 1
        sums = np.zeros(minlength, dtype=np.float64)
        counts = np.zeros(minlength, dtype=np.float64)
        for t in range(minlength):
            i0 = max(0, t - (n - 1))
            i1 = min(t, m - 1)
            if i0 > i1:
                continue
            i = np.arange(i0, i1 + 1, dtype=np.intp)
            j = t - i
            sums[t] = np.sum(a[i, j])
            counts[t] = float(i1 - i0 + 1)
        return sums / counts

    def execute(self, data: FloatArray) -> FloatArray:
        """Reconstruct the denoised time series from the truncated matrix."""
        _, cols = data.shape
        if cols % 2 != 0:
            raise ValueError("Joint matrix must have an even number of columns.")
        k_dim = cols // 2
        h_l = data[:, :k_dim]
        h_r = data[:, k_dim:]
        left = self.fast_diagonal_average(h_l)
        right = self.fast_diagonal_average(h_r)
        return np.column_stack((left, right))
