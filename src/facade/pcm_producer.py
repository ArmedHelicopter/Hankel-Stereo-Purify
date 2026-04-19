"""PCM producer for bounded queues (Phase0 placeholder)."""

from __future__ import annotations


class PcmProducer:
    """Feed PCM blocks into a thread-safe queue."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Full PcmProducer lives on branch ``tutorial``.")

    def start(self) -> None:
        """Start producer thread or iterator."""
        raise NotImplementedError(
            "Full PcmProducer.start lives on branch ``tutorial``."
        )
