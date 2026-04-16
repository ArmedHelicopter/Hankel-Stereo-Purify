import numpy as np
from numpy.typing import NDArray

from src.core.stages.a_hankel import AHankelStage
from src.core.stages.b_multichannel import BMultichannelStage


def test_multichannel_hstack_shape_and_values() -> None:
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
    window_length = 3
    stage_a = AHankelStage(window_length=window_length)
    h_l, h_r = stage_a.execute(input_signal)
    out = BMultichannelStage().execute((h_l, h_r))

    k = h_l.shape[1]
    assert out.shape == (window_length, 2 * k)
    np.testing.assert_array_equal(out, np.hstack((h_l, h_r)))
