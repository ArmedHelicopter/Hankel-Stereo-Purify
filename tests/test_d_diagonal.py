import numpy as np

from src.core.stages.a_hankel import AHankelStage
from src.core.stages.d_diagonal import DDiagonalStage, _diagonal_average_channel


def test_diagonal_average_recovers_hankel_time_series() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    stereo = np.column_stack((x, x))
    window_length = 3
    h_l, _ = AHankelStage(window_length=window_length).execute(stereo)
    recovered = _diagonal_average_channel(h_l)
    np.testing.assert_allclose(recovered, x)


def test_diagonal_stage_joint_matrix_output_length() -> None:
    rng = np.random.default_rng(3)
    n_rows = 12
    stereo = rng.standard_normal((n_rows, 2))
    window_length = 4
    h_l, h_r = AHankelStage(window_length=window_length).execute(stereo)
    k_dim = h_l.shape[1]
    joint = np.hstack((h_l, h_r))
    out = DDiagonalStage().execute(joint)
    assert out.shape == (n_rows, 2)
    assert joint.shape[1] == 2 * k_dim
    assert n_rows == k_dim + window_length - 1
