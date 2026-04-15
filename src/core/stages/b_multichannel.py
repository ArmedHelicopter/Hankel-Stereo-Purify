from typing import Any, Tuple

from ..pipeline import FloatArray, MSSAStage


class BMultichannelStage(MSSAStage[Tuple[FloatArray, FloatArray], FloatArray]):
    """Multi-channel block construction stage for MSSA.

    Input: Tuple[FloatArray, FloatArray], shape (L, K)
    Output: FloatArray, shape (L, 2K)
    """

    def execute(self, data: Tuple[FloatArray, FloatArray]) -> FloatArray:
        """Combine multiple channel Hankel matrices into a joint block matrix."""
        raise NotImplementedError
