import numpy as np
import pytest

from src.core.stages.a_hankel import hankel_embed


def test_hankel_embed_shapes() -> None:
    rng = np.random.default_rng(0)
    n = 20
    stereo = rng.standard_normal((n, 2))
    window_length = 4
    h_l, h_r = hankel_embed(window_length, stereo)
    k = n - window_length + 1
    assert h_l.shape == (window_length, k)
    assert h_r.shape == (window_length, k)


def test_hankel_embed_requires_length() -> None:
    rng = np.random.default_rng(1)
    stereo = rng.standard_normal((2, 2))
    with pytest.raises(ValueError, match="window_length"):
        hankel_embed(3, stereo)
