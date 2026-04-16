import numpy as np

from ..pipeline import FloatArray, MSSAStage
from ..strategies.truncation import TruncationStrategy


class CSVDStage(MSSAStage[FloatArray, FloatArray]):
    """SVD decomposition and truncation stage for MSSA.

    Input: FloatArray, shape (L, 2K)
    Output: FloatArray, shape (L, 2K)
    """

    def __init__(self, truncation_strategy: TruncationStrategy) -> None:
        self.truncation_strategy = truncation_strategy

    def execute(self, data: FloatArray) -> FloatArray:
        """Perform SVD on the block matrix and truncate noise components."""
        u, singular_values, vh = np.linalg.svd(data, full_matrices=False)
        k = int(self.truncation_strategy.get_k(singular_values))
        if k <= 0:
            raise ValueError("Truncation rank must be positive.")
        max_rank = min(int(data.shape[0]), int(data.shape[1]), len(singular_values))
        rank = min(k, max_rank)
        u_r = u[:, :rank]
        s_r = singular_values[:rank]
        vh_r = vh[:rank, :]
        reconstructed: FloatArray = ((u_r * s_r) @ vh_r).astype(np.float64, copy=False)
        return reconstructed
