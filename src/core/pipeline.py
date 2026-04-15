from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")

FloatArray = NDArray[np.float64]


class MSSAStage(ABC, Generic[T_in, T_out]):
    """Abstract base class for a single MSSA pipeline stage."""

    @abstractmethod
    def execute(self, data: T_in) -> T_out:
        """Execute the stage and return transformed data."""
        raise NotImplementedError


class Pipeline:
    """Pipeline coordinator that sequences MSSA stages."""

    def __init__(self, stages: list[MSSAStage[Any, Any]] | None = None) -> None:
        self.stages = stages or []

    def add_stage(self, stage: MSSAStage[Any, Any]) -> None:
        self.stages.append(stage)

    def execute(self, data: Any) -> Any:
        result: Any = data
        for stage in self.stages:
            result = stage.execute(result)
        return result
