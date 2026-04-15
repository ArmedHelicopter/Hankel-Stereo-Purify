from typing import Any


class AudioStream:
    """Audio streaming helper for block-based file reading."""

    def __init__(self, path: str, block_size: int = 4096) -> None:
        self.path = path
        self.block_size = block_size

    def read_blocks(self) -> Any:
        """Yield audio blocks from the source file."""
        raise NotImplementedError
