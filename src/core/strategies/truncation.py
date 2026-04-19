"""Truncation configuration types (concrete dataclasses, not abstract Strategy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class FixedRankStrategy:
    """Keep the first *rank* singular values."""

    rank: int


@dataclass(frozen=True, slots=True)
class EnergyThresholdStrategy:
    """Truncate using cumulative energy fraction."""

    energy_fraction: float


TruncationStrategy: TypeAlias = FixedRankStrategy | EnergyThresholdStrategy
