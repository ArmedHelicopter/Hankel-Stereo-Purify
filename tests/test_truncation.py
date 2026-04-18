import numpy as np
import pytest

from src.core.stages.c_svd import make_svd_step
from src.core.strategies.truncation import FixedRankStrategy


def test_fixed_rank_zero_raises_on_execute() -> None:
    x = np.ones((4, 6), dtype=np.float64)
    step = make_svd_step(FixedRankStrategy(0))
    with pytest.raises(ValueError, match="positive"):
        step(x)
