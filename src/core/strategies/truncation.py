from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


class FixedRankStrategy:
    """Fixed-rank truncation: use attribute ``k``; no spectrum-based selection."""

    def __init__(self, k: int) -> None:
        self.k = k


class EnergyThresholdStrategy:
    """Smallest k whose cumulative singular-value energy reaches the threshold."""

    def __init__(self, energy_fraction: float) -> None:
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError("energy_fraction must be in (0, 1].")
        self.threshold = energy_fraction

    def get_k(self, singular_values: NDArray[np.float64]) -> int:
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


class WienerStrategy:
    """Wiener soft weighting: per-component continuous weight based on SNR estimate.

    Instead of hard truncation (keep/discard), applies the Wiener gain:
        w_i = max(0, 1 - σ_noise² / σ_i²)

    Noise variance is estimated from the bottom ``noise_fraction`` of singular values.
    """

    def __init__(self, noise_fraction: float = 0.1) -> None:
        if not 0.0 < noise_fraction < 1.0:
            raise ValueError("noise_fraction must be in (0, 1).")
        self.noise_fraction = noise_fraction


# Union of supported configs; ``make_svd_step`` dispatches once at construction.
TruncationStrategy: TypeAlias = FixedRankStrategy | EnergyThresholdStrategy | WienerStrategy
