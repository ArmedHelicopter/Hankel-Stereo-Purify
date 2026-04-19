"""Soundfile-driven overlap-add engine (Phase0 placeholder)."""

from __future__ import annotations


class SoundfileOlaEngine:
    """Overlap-add engine using libsndfile-backed reads."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "Full SoundfileOlaEngine lives on branch ``tutorial``."
        )

    def run(self) -> None:
        """Drive producer/consumer loop."""
        raise NotImplementedError("Full OLA engine run lives on branch ``tutorial``.")
