import numpy as np
from numpy.typing import NDArray

from src.core.stages.a_hankel import AHankelStage


def _hankel_view(channel: NDArray[np.float64], window_length: int) -> NDArray[np.float64]:
    n = channel.shape[0]
    shape = (window_length, n - window_length + 1)
    strides = (channel.strides[0], channel.strides[0])
    return np.lib.stride_tricks.as_strided(channel, shape=shape, strides=strides)


def test_hankel_embedding_math_and_memory() -> None:
    input_signal: NDArray[np.float64] = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
        ],
        dtype=np.float64,
    )

    stage = AHankelStage(window_length=3)
    h_l, h_r = stage.execute(input_signal)

    assert h_l.shape == (3, 3)
    assert h_r.shape == (3, 3)

    expected_h_l = _hankel_view(input_signal[:, 0], window_length=3)
    expected_h_r = _hankel_view(input_signal[:, 1], window_length=3)

    np.testing.assert_array_equal(h_l, expected_h_l)
    np.testing.assert_array_equal(h_r, expected_h_r)

    assert np.shares_memory(input_signal, h_l)
    assert np.shares_memory(input_signal, h_r)
