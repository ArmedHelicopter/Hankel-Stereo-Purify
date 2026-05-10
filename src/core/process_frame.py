"""Single-frame stereo MSSA: filter → Hankel → joint block → SVD step → diagonal reconstruct."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .array_types import FloatArray
from .stages.hankel import hankel_embed
from .stages.multichannel import combine_hankel_blocks
from .stages.diagonal import diagonal_reconstruct
from .stages.filter import split_frame_bands, recombine_bands


def process_frame(
    frame: FloatArray,
    *,
    window_length: int,
    svd_step: Callable[[FloatArray], FloatArray],
    sample_rate: int | None = None,
    bypass_freq: float | None = None,
) -> FloatArray:
    """Run one OLA frame through the MSSA pipeline.

    Pipeline: [optional bandpass split] → Hankel → SVD → diagonal reconstruct.

    Parameters
    ----------
    frame
        Stereo samples, shape ``(F, 2)`` with ``F >= window_length``.
    window_length
        Hankel window length ``L``.
    svd_step
        Truncation (see ``make_svd_step`` in ``svd``).
    sample_rate
        Audio sample rate in Hz. Required when ``bypass_freq`` is set.
    bypass_freq
        If set, split frame at this frequency:
        - low band (< cutoff): bypass SVD, pass through directly
        - high band (> cutoff): go through Hankel → SVD → diagonal reconstruct
        Then recombine. Reduces SVD dimensionality and protects clean low frequencies.

    Returns
    -------
    Denoised stereo frame, same shape as ``frame``.
    """
    if bypass_freq is not None and sample_rate is not None:
        # Compute FFT bin for the cutoff frequency
        F = frame.shape[0]
        nyq = sample_rate / 2.0
        cutoff_bin = max(1, int(round(bypass_freq / nyq * (F // 2 + 1))))

        # Split into low (bypass) and high (SVD) bands
        low_band, high_band = split_frame_bands(frame, cutoff_bin)

        # Process only the high band through MSSA
        h_l, h_r = hankel_embed(window_length, high_band)
        joint = combine_hankel_blocks(h_l, h_r)
        truncated = svd_step(joint)
        denoised_high = diagonal_reconstruct(truncated)

        # Recombine: low band (untouched) + high band (denoised)
        return recombine_bands(low_band, denoised_high)

    # No bandpass: process full frame
    h_l, h_r = hankel_embed(window_length, frame)
    joint = combine_hankel_blocks(h_l, h_r)
    truncated = svd_step(joint)
    return diagonal_reconstruct(truncated)
