from abc import ABC, abstractmethod
from typing import Any, List


class MSSAStage(ABC):
    """Abstract base class for a single MSSA pipeline stage."""

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute the stage and return transformed data."""
        raise NotImplementedError


class Pipeline:
    """Pipeline coordinator that sequences MSSA stages."""

    def __init__(self, stages: List[MSSAStage] = None) -> None:
        self.stages = stages or []

    def add_stage(self, stage: MSSAStage) -> None:
        self.stages.append(stage)

    def execute(self, data: Any) -> Any:
        result = data
        for stage in self.stages:
            result = stage.execute(result)
        return result
