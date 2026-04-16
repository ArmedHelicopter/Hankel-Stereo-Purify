"""Overlap-add frame list and COLA seam sanity (F-02)."""

import numpy as np

from src.facade.ola import list_frame_starts, sqrt_hanning_weights


def test_list_frame_starts_covers_short_signal() -> None:
    assert list_frame_starts(50, 64, 32) == [0]


def test_list_frame_starts_includes_tail() -> None:
    starts = list_frame_starts(100, 64, 32)
    assert 0 in starts
    assert max(s + 64 for s in starts) >= 100


def test_ola_weights_product_is_hanning() -> None:
    f = 128
    w = sqrt_hanning_weights(f)
    np.testing.assert_allclose(w * w, np.hanning(f))


def test_ola_reconstruction_sine_low_seam_ripple() -> None:
    """COLA with sqrt-Hanning: overlap sum of w^2 should be ~flat in interior."""
    n = 10_000
    f_size = 512
    hop = 256
    w2 = sqrt_hanning_weights(f_size) ** 2
    acc = np.zeros(n)
    for start in list_frame_starts(n, f_size, hop):
        end = min(start + f_size, n)
        sl = end - start
        acc[start:end] += w2[:sl]
    mid = acc[n // 4 : 3 * n // 4]
    assert float(np.std(mid) / (np.mean(mid) + 1e-9)) < 0.05
