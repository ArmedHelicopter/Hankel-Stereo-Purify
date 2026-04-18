import numpy as np

from src.core.stages.a_hankel import hankel_embed
from src.core.stages.b_multichannel import combine_hankel_blocks
from src.core.stages.d_diagonal import (
    batched_diagonal_average,
    diagonal_reconstruct,
    fast_diagonal_average,
)


def _diagonal_avg_reference_loops(mats: np.ndarray) -> np.ndarray:
    """Explicit nested loops; order matches row-major anti-diagonal traversal."""
    b, m, n = mats.shape
    out = np.zeros((b, m + n - 1), dtype=np.float64)
    for bi in range(b):
        for t in range(m + n - 1):
            i0 = max(0, t - (n - 1))
            i1 = min(t, m - 1)
            if i0 > i1:
                continue
            s = 0.0
            for i in range(i0, i1 + 1):
                s += float(mats[bi, i, t - i])
            out[bi, t] = s / float(i1 - i0 + 1)
    return out


def test_batched_diagonal_average_matches_reference_shapes() -> None:
    rng = np.random.default_rng(903)
    cases = [
        (1, 1, 1),
        (3, 2, 4),
        (5, 7, 6),
        (2, 1, 8),
        (4, 9, 1),
    ]
    for b, m, n in cases:
        mats = rng.standard_normal((b, m, n))
        ref = _diagonal_avg_reference_loops(mats)
        bat = batched_diagonal_average(mats)
        np.testing.assert_allclose(bat, ref, rtol=1e-14, atol=1e-14)


def test_batched_diagonal_average_matches_fast_per_matrix() -> None:
    rng = np.random.default_rng(502)
    b, m, n = 7, 5, 6
    mats = rng.standard_normal((b, m, n))
    bat = batched_diagonal_average(mats)
    assert bat.shape == (b, m + n - 1)
    for i in range(b):
        np.testing.assert_allclose(
            bat[i],
            fast_diagonal_average(mats[i]),
            rtol=1e-14,
            atol=1e-14,
        )


def test_fast_diagonal_average_recovers_1d_from_hankel() -> None:
    rng = np.random.default_rng(0)
    n = 12
    stereo = rng.standard_normal((n, 2))
    window_length = 4
    h_l, _ = hankel_embed(window_length, stereo)
    recovered = fast_diagonal_average(h_l)
    assert recovered.shape[0] == window_length + h_l.shape[1] - 1


def test_diagonal_reconstruct_stereo() -> None:
    rng = np.random.default_rng(1)
    n = 10
    stereo = rng.standard_normal((n, 2))
    window_length = 3
    h_l, h_r = hankel_embed(window_length, stereo)
    joint = combine_hankel_blocks(h_l, h_r)
    out = diagonal_reconstruct(joint)
    assert out.ndim == 2
    assert out.shape[1] == 2
