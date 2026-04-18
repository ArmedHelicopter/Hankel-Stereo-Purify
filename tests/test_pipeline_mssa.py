import numpy as np
from numpy.typing import NDArray

from src.core.stages.a_hankel import hankel_embed
from src.core.stages.b_multichannel import combine_hankel_blocks
from src.core.stages.c_svd import make_svd_step
from src.core.stages.d_diagonal import diagonal_reconstruct
from src.core.strategies.truncation import EnergyThresholdStrategy, FixedRankStrategy


def _run_mssa(
    stereo: NDArray[np.float64],
    *,
    window_length: int,
    strategy: FixedRankStrategy | EnergyThresholdStrategy,
) -> NDArray[np.float64]:
    h_l, h_r = hankel_embed(window_length, stereo)
    block_mat = combine_hankel_blocks(h_l, h_r)
    svd_step = make_svd_step(strategy)
    truncated_mat = svd_step(block_mat)
    return diagonal_reconstruct(truncated_mat)


def test_mssa_pipeline_e2e_shape_and_finite() -> None:
    rng = np.random.default_rng(7)
    n_rows = 30
    stereo = rng.standard_normal((n_rows, 2))
    window_length = 10
    k_dim = n_rows - window_length + 1
    k_trunc = min(window_length, 2 * k_dim)

    out = _run_mssa(
        stereo,
        window_length=window_length,
        strategy=FixedRankStrategy(k_trunc),
    )
    assert out.shape == stereo.shape
    assert np.all(np.isfinite(out))


def test_mssa_pipeline_full_rank_recover_input() -> None:
    rng = np.random.default_rng(42)
    n_rows = 20
    stereo = rng.standard_normal((n_rows, 2))
    window_length = 8
    k_dim = n_rows - window_length + 1
    k_trunc = min(window_length, 2 * k_dim)

    out = _run_mssa(
        stereo,
        window_length=window_length,
        strategy=FixedRankStrategy(k_trunc),
    )
    np.testing.assert_allclose(out, stereo, rtol=1e-9, atol=1e-9)


def test_mssa_pipeline_energy_fraction_shape_and_finite() -> None:
    rng = np.random.default_rng(8)
    n_rows = 24
    stereo = rng.standard_normal((n_rows, 2))
    window_length = 8

    out = _run_mssa(
        stereo,
        window_length=window_length,
        strategy=EnergyThresholdStrategy(0.99),
    )
    assert out.shape == stereo.shape
    assert np.all(np.isfinite(out))
