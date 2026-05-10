"""Bandpass filter: split full signal into low (bypass) and high (SVD) bands.

Uses zero-phase Butterworth IIR filter (sosfiltfilt) on full signal.
Perfect reconstruction: high_band = signal - low_band (guarantees low + high == original).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


def split_signal(
    signal: NDArray[np.float64],
    cutoff_hz: float,
    sample_rate: int,
    order: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split signal into low and high frequency bands.

    Uses zero-phase filtering (sosfiltfilt) to avoid phase distortion.
    Perfect reconstruction: low_band + high_band == signal (exactly).

    Parameters
    ----------
    signal : NDArray, shape (N,) or (N, C)
        Input audio signal.
    cutoff_hz : float
        Cutoff frequency in Hz.
    sample_rate : int
        Audio sample rate in Hz.
    order : int
        Filter order (default 4 = 24 dB/octave rolloff).

    Returns
    -------
    low_band, high_band : NDArray
        Low and high frequency components. low_band + high_band == signal.
    """
    nyq = sample_rate / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be in (0, {nyq}) for sr={sample_rate}"
        )
    normalized = cutoff_hz / nyq
    lp_sos = butter(order, normalized, btype="low", output="sos")

    # Zero-phase filtering (forward + backward, no phase distortion)
    low_band = sosfiltfilt(lp_sos, signal, axis=0).astype(signal.dtype, copy=False)

    # Perfect reconstruction: high = original - low
    high_band = signal - low_band

    return low_band, high_band
