"""Strategy implementations for MSSA configuration.

This package contains truncation and windowing strategies used by the core
MSSA pipeline.
"""

from .grouping import compute_w_correlation_matrix
from .truncation import TruncationStrategy
from .windowing import WindowingStrategy

__all__ = [
    "TruncationStrategy",
    "WindowingStrategy",
    "compute_w_correlation_matrix",
]
