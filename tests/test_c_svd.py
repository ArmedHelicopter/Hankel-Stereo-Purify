import numpy as np

from src.core.stages.c_svd import CSVDStage
from src.core.strategies.truncation import FixedRankStrategy


def test_svd_full_rank_recover_matrix() -> None:
    rng = np.random.default_rng(0)
    l_row, cols = 5, 8
    x = rng.standard_normal((l_row, cols))
    k = min(l_row, cols)
    y = CSVDStage(FixedRankStrategy(k)).execute(x)
    np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-10)


def test_svd_truncated_rank_reduces_information() -> None:
    rng = np.random.default_rng(1)
    l_row, cols = 6, 10
    x = rng.standard_normal((l_row, cols))
    y = CSVDStage(FixedRankStrategy(2)).execute(x)
    assert y.shape == x.shape
    err = np.linalg.norm(y - x, ord="fro")
    assert err > 1e-6
