"""High-level AudioPurifier facade (Phase0 placeholder)."""

from __future__ import annotations

from pathlib import Path


class AudioPurifier:
    """Stream-level denoiser entry point. Full implementation on ``tutorial``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Full AudioPurifier lives on branch ``tutorial``.")

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Run denoise on a file path."""
        _ = (input_path, output_path)
        raise NotImplementedError("Full process_file lives on branch ``tutorial``.")
