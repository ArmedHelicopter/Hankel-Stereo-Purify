"""``process_frame`` matches explicit MSSA stage composition."""

from functools import partial

import numpy as np

from src.core.process_frame import process_frame
from src.core.stages.diagonal import diagonal_reconstruct
from src.core.stages.hankel import hankel_embed
from src.core.stages.multichannel import combine_hankel_blocks
from src.core.stages.svd import make_fixed_rank_svd_step
from src.facade.purifier import AudioPurifier


def test_process_frame_matches_manual_chain() -> None:
    rng = np.random.default_rng(501)
    wl = 6
    stereo = rng.standard_normal((20, 2))
    k_h = stereo.shape[0] - wl + 1
    k_req = min(wl, 2 * k_h)
    step = make_fixed_rank_svd_step(k_req)

    h_l, h_r = hankel_embed(wl, stereo)
    joint = combine_hankel_blocks(h_l, h_r)
    manual = diagonal_reconstruct(step(joint))

    fn = partial(process_frame, window_length=wl, svd_step=step)
    assert np.array_equal(fn(stereo), manual)


def test_audio_purifier_partial_process_frame_runs() -> None:
    p = AudioPurifier(
        8,
        truncation_rank=4,
        frame_size=24,
    )
    fn = p._make_denoise_frame_fn()
    x = np.random.default_rng(0).standard_normal((20, 2))
    y = fn(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
