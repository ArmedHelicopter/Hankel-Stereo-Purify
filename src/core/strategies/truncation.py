from abc import ABC, abstractmethod
from typing import Any


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
    """Energy threshold truncation strategy."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def get_k(self, singular_values: Any) -> int:
        raise NotImplementedError
