"""Strategy implementations for MSSA configuration.

This package contains truncation and windowing strategies used by the core
MSSA pipeline.
"""

from .truncation import TruncationStrategy
from .windowing import WindowingStrategy

__all__ = ["TruncationStrategy", "WindowingStrategy"]
