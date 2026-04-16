"""Truncation strategy edge cases."""

import numpy as np
import pytest

from src.core.stages.c_svd import CSVDStage
from src.core.strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
)


def test_energy_threshold_empty_singular_values() -> None:
    strat = EnergyThresholdStrategy(0.9)
    assert strat.get_k(np.array([], dtype=np.float64)) == 1


def test_energy_threshold_zero_total_energy() -> None:
    strat = EnergyThresholdStrategy(0.9)
    assert strat.get_k(np.array([0.0, 0.0], dtype=np.float64)) == 1


def test_energy_threshold_typical_rank() -> None:
    strat = EnergyThresholdStrategy(0.95)
    s = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    k = strat.get_k(s)
    assert 1 <= k <= 3


def test_c_svd_rejects_nonpositive_truncation_rank() -> None:
    data = np.ones((4, 4), dtype=np.float64)
    stage = CSVDStage(FixedRankStrategy(0))
    with pytest.raises(ValueError, match="positive"):
        stage.execute(data)
