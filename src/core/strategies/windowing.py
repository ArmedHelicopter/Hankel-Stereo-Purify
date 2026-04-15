from abc import ABC, abstractmethod
from typing import Any


class WindowingStrategy(ABC):
    """Interface for windowing behavior in MSSA preprocessing."""

    @abstractmethod
    def apply(self, frame: Any) -> Any:
        raise NotImplementedError


class HanningWindowStrategy(WindowingStrategy):
    """Hanning window implementation placeholder."""

    def apply(self, frame: Any) -> Any:
        raise NotImplementedError
