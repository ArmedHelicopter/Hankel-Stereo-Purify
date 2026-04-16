"""Overlap-add frame indexing (PRD F-02).

Sqrt-Hanning keeps overlap sums consistent; `AudioPurifier` divides by squared weights.
"""

import numpy as np
from numpy.typing import NDArray

from src.core.strategies.windowing import HanningWindowStrategy


def sqrt_hanning_weights(frame_size: int) -> NDArray[np.float64]:
    """Per-sample sqrt(Hanning) for analysis/synthesis; product is Hanning."""
    return HanningWindowStrategy.sqrt_hanning_1d(frame_size)


def list_frame_starts(num_samples: int, frame_size: int, hop_size: int) -> list[int]:
    """Frame start indices that cover [0, num_samples) with hop, last frame padded."""
    if num_samples <= 0 or frame_size <= 0 or hop_size <= 0:
        return []
    if hop_size >= frame_size:
        raise ValueError("hop_size must be smaller than frame_size for overlap.")
    span = num_samples - frame_size + 1
    if span <= 0:
        return [0]
    starts = list(range(0, span, hop_size))
    last = starts[-1]
    if last + frame_size < num_samples:
        extra = num_samples - frame_size
        if extra not in starts:
            starts.append(extra)
    return starts
