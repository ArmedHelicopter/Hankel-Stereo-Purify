from ..pipeline import FloatArray, MSSAStage


class DDiagonalStage(MSSAStage[FloatArray, FloatArray]):
    """Diagonal averaging reconstruction stage for MSSA.

    Input: FloatArray, shape (L, 2K)
    Output: FloatArray, shape (N, 2)
    """

    def execute(self, data: FloatArray) -> FloatArray:
        """Reconstruct the denoised time series from the truncated matrix."""
        raise NotImplementedError
