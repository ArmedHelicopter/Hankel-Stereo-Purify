from ..pipeline import FloatArray, MSSAStage


class BMultichannelStage(MSSAStage[tuple[FloatArray, FloatArray], FloatArray]):
    """Multi-channel block construction stage for MSSA.

    Input: Tuple[FloatArray, FloatArray], shape (L, K)
    Output: FloatArray, shape (L, 2K)
    """

    def execute(self, data: tuple[FloatArray, FloatArray]) -> FloatArray:
        """Combine multiple channel Hankel matrices into a joint block matrix."""
        raise NotImplementedError
