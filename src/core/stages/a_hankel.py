from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..pipeline import FloatArray, MSSAStage


class AHankelStage(MSSAStage[FloatArray, Tuple[FloatArray, FloatArray]]):
    """Hankel embedding stage for MSSA.

    Input: FloatArray, shape (N, 2)
    Output: Tuple[FloatArray, FloatArray], shape (L, K)
    """

    def __init__(self, window_length: int) -> None:
        self.window_length = window_length

    def execute(self, data: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Embed a time series into its Hankel matrix representation."""
        n = data.shape[0]
        k = n - self.window_length + 1

        def hankel_view(channel: NDArray[np.float64]) -> FloatArray:
            return np.lib.stride_tricks.as_strided(
                channel,
                shape=(self.window_length, k),
                strides=(channel.strides[0], channel.strides[0]),
            )

        h_l = hankel_view(data[:, 0])
        h_r = hankel_view(data[:, 1])
        return h_l, h_r
