from typing import Any


class AHankelStage:
    """Hankel embedding stage for MSSA."""

    def execute(self, data: Any) -> Any:
        """Embed a time series into its Hankel matrix representation."""
        raise NotImplementedError
