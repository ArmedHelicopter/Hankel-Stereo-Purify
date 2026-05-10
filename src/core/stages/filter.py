"""Bandpass filter stage: FFT-based frequency band splitting.

Splits each frame into low (bypass) and high (SVD) frequency bands.
The low band skips SVD entirely; only the high band goes through
Hankel → SVD → diagonal reconstruct.

Based on Gemini audio-modal noise analysis: noise (tape hiss) is
concentrated in >2kHz, low-mid frequencies are clean.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def split_frame_bands(
    frame: NDArray[np.float64],
    cutoff_bin: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a frame into low and high frequency bands via FFT.

    Parameters
    ----------
    frame : NDArray, shape (F, C)
        Windowed OLA frame, F samples × C channels.
    cutoff_bin : int
        FFT bin index for the cutoff frequency.
        Frequencies below this bin go to low band (bypass).

    Returns
    -------
    low_frame, high_frame : NDArray, shape (F, C)
        Low-band and high-band components in time domain.
    """
    F, C = frame.shape
    low_frame = np.zeros_like(frame)
    high_frame = np.zeros_like(frame)

    for ch in range(C):
        spectrum = np.fft.rfft(frame[:, ch])
        low_spectrum = spectrum.copy()
        high_spectrum = spectrum.copy()

        # Zero out high frequencies in low band
        low_spectrum[cutoff_bin:] = 0
        # Zero out low frequencies in high band
        high_spectrum[:cutoff_bin] = 0

        low_frame[:, ch] = np.fft.irfft(low_spectrum, n=F)
        high_frame[:, ch] = np.fft.irfft(high_spectrum, n=F)

    return low_frame, high_frame


def recombine_bands(
    low_frame: NDArray[np.float64],
    high_frame: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Recombine low (bypass) and high (SVD-processed) bands."""
    return low_frame + high_frame
