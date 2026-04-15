import numpy as np
from numpy.typing import NDArray

from src.core.stages.a_hankel import AHankelStage


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

    expected_h_l = np.array(
        [
            [0.0, 2.0, 4.0],
            [2.0, 4.0, 6.0],
            [4.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    expected_h_r = np.array(
        [
            [1.0, 3.0, 5.0],
            [3.0, 5.0, 7.0],
            [5.0, 7.0, 9.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(h_l, expected_h_l)
    np.testing.assert_array_equal(h_r, expected_h_r)

    assert np.shares_memory(input_signal, h_l)
    assert np.shares_memory(input_signal, h_r)
