from ..pipeline import FloatArray, MSSAStage


class CSVDStage(MSSAStage[FloatArray, FloatArray]):
    """SVD decomposition and truncation stage for MSSA.

    Input: FloatArray, shape (L, 2K)
    Output: FloatArray, shape (L, 2K)
    """

    def execute(self, data: FloatArray) -> FloatArray:
        """Perform SVD on the block matrix and truncate noise components."""
        raise NotImplementedError
