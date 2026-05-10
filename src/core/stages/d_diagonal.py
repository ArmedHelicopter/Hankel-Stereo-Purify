"""Compatibility wrapper for legacy ``d_diagonal`` imports."""

from src.core.stages.diagonal import (
    batched_diagonal_average,
    diagonal_reconstruct,
    fast_diagonal_average,
)

__all__ = [
    "batched_diagonal_average",
    "diagonal_reconstruct",
    "fast_diagonal_average",
]
