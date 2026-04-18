"""Single-frame stereo MSSA: Hankel → joint block → SVD step → diagonal reconstruct."""

from __future__ import annotations

from collections.abc import Callable

from .array_types import FloatArray
from .stages.a_hankel import hankel_embed
from .stages.b_multichannel import combine_hankel_blocks
from .stages.d_diagonal import diagonal_reconstruct


def process_frame(
    frame: FloatArray,
    *,
    window_length: int,
    svd_step: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    """Run one OLA frame through A→B→C→D (no Stage/Pipeline objects).

    Parameters
    ----------
    frame
        Stereo samples, shape ``(F, 2)`` with ``F >= window_length``.
    window_length
        Hankel window length ``L``.
    svd_step
        Truncation + optional W-correlation (see ``make_svd_step`` in ``c_svd``).

    Returns
    -------
    Denoised stereo frame, same shape as ``frame``.
    """
    h_l, h_r = hankel_embed(window_length, frame)
    joint = combine_hankel_blocks(h_l, h_r)
    truncated = svd_step(joint)
    return diagonal_reconstruct(truncated)
