import numpy as np

from src.core.strategies.windowing import HanningWindowStrategy


def test_hanning_1d_endpoints_and_shape() -> None:
    strat = HanningWindowStrategy()
    x = np.ones(11, dtype=np.float64)
    y = strat.apply(x)
    assert y.shape == (11,)
    assert y[0] == 0.0
    assert y[-1] == 0.0
    assert np.all(np.isfinite(y))


def test_hanning_2d_applies_along_time() -> None:
    strat = HanningWindowStrategy()
    x = np.ones((7, 2), dtype=np.float64)
    y = strat.apply(x)
    assert y.shape == (7, 2)
    win = np.hanning(7)
    np.testing.assert_allclose(y, x * win[:, np.newaxis])
