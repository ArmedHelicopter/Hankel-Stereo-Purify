import numpy as np

from src.core.pipeline import Pipeline
from src.core.stages.a_hankel import AHankelStage
from src.core.stages.b_multichannel import BMultichannelStage
from src.core.stages.c_svd import CSVDStage
from src.core.stages.d_diagonal import DDiagonalStage
from src.core.strategies.truncation import FixedRankStrategy


def test_mssa_pipeline_e2e_shape_and_finite() -> None:
    rng = np.random.default_rng(7)
    n_rows = 30
    stereo = rng.standard_normal((n_rows, 2))
    window_length = 10
    k_dim = n_rows - window_length + 1
    k_trunc = min(window_length, 2 * k_dim)

    pipeline = Pipeline(
        [
            AHankelStage(window_length=window_length),
            BMultichannelStage(),
            CSVDStage(FixedRankStrategy(k_trunc)),
            DDiagonalStage(),
        ]
    )
    out = pipeline.execute(stereo)
    assert out.shape == stereo.shape
    assert np.all(np.isfinite(out))


def test_mssa_pipeline_full_rank_recover_input() -> None:
    rng = np.random.default_rng(42)
    n_rows = 20
    stereo = rng.standard_normal((n_rows, 2))
    window_length = 8
    k_dim = n_rows - window_length + 1
    k_trunc = min(window_length, 2 * k_dim)

    pipeline = Pipeline(
        [
            AHankelStage(window_length=window_length),
            BMultichannelStage(),
            CSVDStage(FixedRankStrategy(k_trunc)),
            DDiagonalStage(),
        ]
    )
    out = pipeline.execute(stereo)
    np.testing.assert_allclose(out, stereo, rtol=1e-9, atol=1e-9)
