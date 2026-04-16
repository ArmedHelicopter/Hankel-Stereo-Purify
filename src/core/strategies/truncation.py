from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class TruncationStrategy(ABC):
    """Interface for selecting the MSSA truncation rank."""

    @abstractmethod
    def get_k(self, singular_values: Any) -> int:
        raise NotImplementedError


class FixedRankStrategy(TruncationStrategy):
    """Fixed-rank truncation strategy."""

    def __init__(self, k: int) -> None:
        self.k = k

    def get_k(self, singular_values: Any) -> int:
        return self.k


class EnergyThresholdStrategy(TruncationStrategy):
    """Smallest k whose cumulative singular-value energy reaches the threshold."""

    def __init__(self, energy_fraction: float) -> None:
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError("energy_fraction must be in (0, 1].")
        self.threshold = energy_fraction

    def get_k(self, singular_values: Any) -> int:
        s = np.asarray(singular_values, dtype=np.float64).ravel()
        n = int(s.size)
        if n == 0:
            return 1
        e: NDArray[np.float64] = s * s
        total = float(np.sum(e))
        if total <= 0.0:
            return 1
        target = self.threshold * total
        cum = np.cumsum(e)
        idx = int(np.searchsorted(cum, target, side="left"))
        return max(1, min(idx + 1, n))
