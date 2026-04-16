from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class WindowingStrategy(ABC):
    """Interface for windowing behavior in MSSA preprocessing."""

    @abstractmethod
    def apply(self, frame: Any) -> Any:
        raise NotImplementedError


class HanningWindowStrategy(WindowingStrategy):
    """Hanning window applied along the time axis."""

    @staticmethod
    def sqrt_hanning_1d(num_samples: int) -> NDArray[np.float64]:
        """Sqrt-Hanning for COLA (analysis/synthesis product is Hanning)."""
        w = np.sqrt(np.hanning(num_samples))
        return np.asarray(w, dtype=np.float64)

    def apply(self, frame: Any) -> NDArray[np.float64]:
        arr = np.asarray(frame, dtype=np.float64)
        if arr.ndim == 1:
            win = np.hanning(arr.shape[0])
            return arr * win
        if arr.ndim == 2:
            win = np.hanning(arr.shape[0])
            return arr * win[:, np.newaxis]
        raise ValueError("Frame must be one- or two-dimensional.")
