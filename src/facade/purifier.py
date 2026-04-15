from typing import Any, Optional


class AudioPurifier:
    """Facade class exposing a simple audio purification API."""

    def process_file(self, input_path: str, output_path: str) -> None:
        """Process one audio file and write the denoised result."""
        raise NotImplementedError


class MSSAPurifierBuilder:
    """Builder for configuring AudioPurifier instances."""

    def __init__(self) -> None:
        self.params = {}

    def set_window_length(self, value: int) -> "MSSAPurifierBuilder":
        self.params["window_length"] = value
        return self

    def set_truncation_rank(self, value: int) -> "MSSAPurifierBuilder":
        self.params["truncation_rank"] = value
        return self

    def build(self) -> AudioPurifier:
        raise NotImplementedError
