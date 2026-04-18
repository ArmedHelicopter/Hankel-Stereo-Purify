import numpy as np

from src.core.stages.a_hankel import hankel_embed
from src.core.stages.b_multichannel import combine_hankel_blocks


def test_combine_hankel_blocks_stacks_channels() -> None:
    rng = np.random.default_rng(0)
    n = 12
    stereo = rng.standard_normal((n, 2))
    window_length = 4
    h_l, h_r = hankel_embed(window_length, stereo)
    out = combine_hankel_blocks(h_l, h_r)
    k = n - window_length + 1
    assert out.shape == (window_length, 2 * k)
