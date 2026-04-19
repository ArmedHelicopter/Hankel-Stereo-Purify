"""Exercise Phase0 placeholder APIs (expect NotImplementedError) for coverage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.core.process_frame import process_frame
from src.core.stages import (
    combine_hankel_blocks,
    diagonal_reconstruct,
    hankel_embed,
    make_svd_step,
)
from src.core.strategies import FixedRankStrategy, apply_analysis_window
from src.core.strategies.grouping import group_components_by_w_correlation
from src.facade import (
    AudioPurifier,
    PcmProducer,
    SoundfileOlaEngine,
    overlap_add_merge,
)
from src.io import read_stereo_pcm_head
from src.utils import project_root_marker


def test_fixed_rank_strategy_dataclass() -> None:
    s = FixedRankStrategy(rank=3)
    assert s.rank == 3


def test_array_types_alias() -> None:
    from src.core.array_types import Float64Array

    x: Float64Array = np.zeros(2, dtype=np.float64)
    assert x.shape == (2,)


def test_exception_hierarchy() -> None:
    from src.core.exceptions import HankelStereoPurifyError
    from src.core.linalg_errors import MssaLinearAlgebraError

    assert issubclass(MssaLinearAlgebraError, HankelStereoPurifyError)
    err = MssaLinearAlgebraError("svd")
    assert str(err) == "svd"


@pytest.mark.parametrize(
    "func,args,kwargs",
    [
        (
            hankel_embed,
            (np.zeros(4, dtype=np.float64), 2),
            {},
        ),
        (
            combine_hankel_blocks,
            (
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            {},
        ),
        (
            make_svd_step,
            (),
            {"strategy": FixedRankStrategy(rank=1), "window_length": 8},
        ),
        (diagonal_reconstruct, (np.zeros((2, 3), dtype=np.float64),), {}),
        (
            apply_analysis_window,
            (np.zeros(8, dtype=np.float64),),
            {"window_length": 8},
        ),
        (
            group_components_by_w_correlation,
            (np.zeros((2, 2), dtype=np.float64),),
            {"window_length": 4},
        ),
        (
            process_frame,
            (
                8,
                np.zeros(8, dtype=np.float64),
                np.zeros(8, dtype=np.float64),
            ),
            {"svd_step": lambda x: x},
        ),
        (
            overlap_add_merge,
            (np.zeros((2, 4), dtype=np.float64),),
            {"hop": 2, "frame_length": 4},
        ),
        (
            read_stereo_pcm_head,
            (Path("x.wav"),),
            {"max_frames": 1},
        ),
    ],
)
def test_not_implemented_raises(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(NotImplementedError, match="tutorial"):
        func(*args, **kwargs)


def test_facade_constructors_and_methods() -> None:
    with pytest.raises(NotImplementedError, match="tutorial"):
        AudioPurifier()
    ap = object.__new__(AudioPurifier)
    with pytest.raises(NotImplementedError, match="tutorial"):
        AudioPurifier.process_file(ap, Path("a.wav"), Path("b.wav"))

    with pytest.raises(NotImplementedError, match="tutorial"):
        SoundfileOlaEngine()
    eng = object.__new__(SoundfileOlaEngine)
    with pytest.raises(NotImplementedError, match="tutorial"):
        SoundfileOlaEngine.run(eng)

    with pytest.raises(NotImplementedError, match="tutorial"):
        PcmProducer()
    prod = object.__new__(PcmProducer)
    with pytest.raises(NotImplementedError, match="tutorial"):
        PcmProducer.start(prod)


def test_utils_placeholder() -> None:
    with pytest.raises(NotImplementedError, match="tutorial"):
        project_root_marker()
