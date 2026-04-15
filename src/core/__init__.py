"""Core package for MSSA algorithm components.

This package contains the core mathematical pipeline, stage implementations,
strategy implementations, and related helpers.
"""

from .pipeline import MSSAStage, Pipeline

__all__ = ["MSSAStage", "Pipeline"]
