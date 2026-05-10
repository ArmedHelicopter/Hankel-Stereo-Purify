"""Bandpass filter: split full signal into low (bypass) and high (SVD) bands.

Uses causal Butterworth IIR filter (sosfilt) which works on full signals.
The low band skips SVD entirely; only the high band goes through MSSA.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt


def design_filters(
    cutoff_hz: float,
    sample_rate: int,
    order: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design lowpass and highpass Butterworth filters.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    sample_rate : int
        Audio sample rate in Hz.
    order : int
        Filter order (default 4 = 24 dB/octave rolloff).

    Returns
    -------
    lowpass_sos, highpass_sos : NDArray
        Second-order sections for each filter.
    """
    nyq = sample_rate / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be in (0, {nyq}) for sr={sample_rate}"
        )
    normalized = cutoff_hz / nyq
    lowpass_sos = butter(order, normalized, btype="low", output="sos")
    highpass_sos = butter(order, normalized, btype="high", output="sos")
    return lowpass_sos, highpass_sos


def split_signal(
    signal: NDArray[np.float64],
    lowpass_sos: NDArray[np.float64],
    highpass_sos: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split signal into low and high frequency bands.

    Parameters
    ----------
    signal : NDArray, shape (N,) or (N, C)
        Input audio signal.
    lowpass_sos, highpass_sos : NDArray
        Filter coefficients from design_filters.

    Returns
    -------
    low_band, high_band : NDArray
        Low and high frequency components.
    """
    if signal.ndim == 1:
        low_band = sosfilt(lowpass_sos, signal)
        high_band = sosfilt(highpass_sos, signal)
    else:
        low_band = np.empty_like(signal)
        high_band = np.empty_like(signal)
        for ch in range(signal.shape[1]):
            low_band[:, ch] = sosfilt(lowpass_sos, signal[:, ch])
            high_band[:, ch] = sosfilt(highpass_sos, signal[:, ch])
    return low_band, high_band
