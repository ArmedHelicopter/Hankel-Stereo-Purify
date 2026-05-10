import numpy as np
from numpy.typing import NDArray

from ..array_types import FloatArray


def hankel_embed(
    window_length: int,
    data: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Embed stereo channels into Hankel matrices (L, K) each."""
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)

    n = data.shape[0]
    if n < window_length:
        raise ValueError(
            "Hankel embedding requires at least "
            f"window_length={window_length} samples per channel (got {n})."
        )
    k = n - window_length + 1

    def hankel_view(channel: NDArray[np.float64]) -> FloatArray:
        return np.lib.stride_tricks.as_strided(
            channel,
            shape=(window_length, k),
            strides=(channel.strides[0], channel.strides[0]),
        )

    h_l = hankel_view(data[:, 0])
    h_r = hankel_view(data[:, 1])
    return h_l, h_r
