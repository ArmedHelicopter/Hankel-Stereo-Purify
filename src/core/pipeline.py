from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")

FloatArray = NDArray[np.float64]


class MSSAStage(ABC, Generic[T_in, T_out]):
    """One A→B→C→D step; `execute` runs once per overlap-add frame."""

    @abstractmethod
    def execute(self, data: T_in) -> T_out:
        """Transform `data`; input/output shapes are defined on each concrete stage."""
        raise NotImplementedError


class Pipeline:
    """Ordered stages; `AudioPurifier` feeds one frame per `execute` call."""

    def __init__(self, stages: list[MSSAStage[Any, Any]] | None = None) -> None:
        self.stages = stages or []

    def add_stage(self, stage: MSSAStage[Any, Any]) -> None:
        self.stages.append(stage)

    def execute(self, data: Any) -> Any:
        """Forward `data` through each stage (used inside the inner denoise loop)."""
        result: Any = data
        for stage in self.stages:
            result = stage.execute(result)
        return result
