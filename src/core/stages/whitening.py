"""Reversible high-band spectral whitening for MSSA preconditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import istft, stft  # type: ignore[import-untyped]

DEFAULT_WHITEN_NPERSEG = 2048
DEFAULT_WHITEN_NOVERLAP = 1024
DEFAULT_PROFILE_PERCENTILE = 20.0
DEFAULT_PROFILE_EPS = 1e-8


@dataclass(frozen=True)
class WhiteningProfile:
    """Frequency-only scale profile for reversible STFT whitening."""

    scale: NDArray[np.float64]
    sample_rate: int
    nperseg: int
    noverlap: int
    percentile: float
    eps: float
    original_length: int


def _as_stereo_array(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr[:, np.newaxis])
    if arr.ndim != 2:
        raise ValueError("Whitening expects a 1D or 2D audio array.")
    return np.ascontiguousarray(arr)


def _effective_stft_sizes(num_samples: int) -> tuple[int, int]:
    if num_samples <= 0:
        raise ValueError("Whitening requires at least one audio sample.")
    nperseg = min(DEFAULT_WHITEN_NPERSEG, num_samples)
    noverlap = min(DEFAULT_WHITEN_NOVERLAP, max(0, nperseg // 2))
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)
    return nperseg, noverlap


def _stft_channel(
    x: NDArray[np.float64],
    profile: WhiteningProfile,
) -> NDArray[np.complex128]:
    _, _, z = stft(
        x,
        fs=profile.sample_rate,
        nperseg=profile.nperseg,
        noverlap=profile.noverlap,
        window="hann",
        boundary="zeros",
        padded=True,
    )
    return np.asarray(z, dtype=np.complex128)


def _istft_channel(
    z: NDArray[np.complex128],
    profile: WhiteningProfile,
) -> NDArray[np.float64]:
    _, y = istft(
        z,
        fs=profile.sample_rate,
        nperseg=profile.nperseg,
        noverlap=profile.noverlap,
        window="hann",
        input_onesided=True,
        boundary=True,
    )
    out = np.asarray(y, dtype=np.float64)
    if out.shape[0] < profile.original_length:
        out = np.pad(out, (0, profile.original_length - out.shape[0]))
    return out[: profile.original_length]


def estimate_noise_profile(
    signal: NDArray[np.float64],
    sample_rate: int,
    *,
    percentile: float = DEFAULT_PROFILE_PERCENTILE,
    eps: float = DEFAULT_PROFILE_EPS,
) -> WhiteningProfile:
    """Estimate a positive frequency-only scale from high-band STFT magnitudes."""
    arr = _as_stereo_array(signal)
    nperseg, noverlap = _effective_stft_sizes(int(arr.shape[0]))
    base_profile = WhiteningProfile(
        scale=np.ones(nperseg // 2 + 1, dtype=np.float64),
        sample_rate=int(sample_rate),
        nperseg=nperseg,
        noverlap=noverlap,
        percentile=float(percentile),
        eps=float(eps),
        original_length=int(arr.shape[0]),
    )
    magnitudes = []
    for ch in range(arr.shape[1]):
        magnitudes.append(np.abs(_stft_channel(arr[:, ch], base_profile)))
    mag = np.concatenate(magnitudes, axis=1)
    raw = np.percentile(mag, percentile, axis=1)
    scale = np.maximum(np.asarray(raw, dtype=np.float64), float(eps))
    return WhiteningProfile(
        scale=scale,
        sample_rate=int(sample_rate),
        nperseg=nperseg,
        noverlap=noverlap,
        percentile=float(percentile),
        eps=float(eps),
        original_length=int(arr.shape[0]),
    )


def whiten_signal(
    signal: NDArray[np.float64],
    profile: WhiteningProfile,
    *,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Apply frequency-only whitening without thresholding or masking."""
    arr = _as_stereo_array(signal)
    out = np.empty_like(arr, dtype=np.float64)
    scale = np.power(profile.scale, float(alpha))[:, np.newaxis]
    for ch in range(arr.shape[1]):
        z = _stft_channel(arr[:, ch], profile)
        out[:, ch] = _istft_channel(z / scale, profile)
    return out


def unwhiten_signal(
    signal: NDArray[np.float64],
    profile: WhiteningProfile,
    *,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Invert :func:`whiten_signal` using the same frequency-only profile."""
    arr = _as_stereo_array(signal)
    out = np.empty_like(arr, dtype=np.float64)
    scale = np.power(profile.scale, float(alpha))[:, np.newaxis]
    for ch in range(arr.shape[1]):
        z = _stft_channel(arr[:, ch], profile)
        out[:, ch] = _istft_channel(z * scale, profile)
    return out


def roundtrip_whiten_signal(
    signal: NDArray[np.float64],
    profile: WhiteningProfile,
    *,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Whiten and unwhiten the same STFT coefficients as a neutral control."""
    arr = _as_stereo_array(signal)
    out = np.empty_like(arr, dtype=np.float64)
    scale = np.power(profile.scale, float(alpha))[:, np.newaxis]
    for ch in range(arr.shape[1]):
        z = _stft_channel(arr[:, ch], profile)
        out[:, ch] = _istft_channel((z / scale) * scale, profile)
    return out


def snr_db(reference: NDArray[np.float64], test: NDArray[np.float64]) -> float:
    """Signal-to-difference ratio in dB for same-shape finite arrays."""
    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)
    if ref.shape != tst.shape:
        raise ValueError("snr_db expects arrays with matching shapes.")
    signal_power = float(np.sum(ref * ref))
    noise = ref - tst
    noise_power = float(np.sum(noise * noise))
    if noise_power <= 1e-30:
        return float("inf")
    if signal_power <= 1e-30:
        return float("inf") if noise_power <= 1e-30 else -float("inf")
    return 10.0 * float(np.log10(signal_power / noise_power))


def rms(x: NDArray[np.float64]) -> float:
    """Root-mean-square amplitude."""
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
