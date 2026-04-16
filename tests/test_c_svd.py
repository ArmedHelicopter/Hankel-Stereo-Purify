from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import src.core.stages.c_svd as c_svd_module
from src.core.stages.c_svd import CSVDStage, TruncatedSVDStage, _reconstruct_usvh
from src.core.strategies.truncation import EnergyThresholdStrategy, FixedRankStrategy


def test_reconstruct_usvh_matches_explicit_diag() -> None:
    rng = np.random.default_rng(42)
    m, r, n = 7, 4, 9
    u = rng.standard_normal((m, r))
    s = np.abs(rng.standard_normal(r))
    vh = rng.standard_normal((r, n))
    explicit = u @ np.diag(s) @ vh
    np.testing.assert_allclose(_reconstruct_usvh(u, s, vh), explicit, rtol=1e-14)


def test_svd_full_rank_recover_matrix() -> None:
    rng = np.random.default_rng(0)
    l_row, cols = 5, 8
    x = rng.standard_normal((l_row, cols))
    k = min(l_row, cols)
    y = CSVDStage(FixedRankStrategy(k)).execute(x)
    np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-10)


def test_truncated_svd_stage_matches_csvd_fixed_rank() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, 10))
    k = 3
    y1 = CSVDStage(FixedRankStrategy(k)).execute(x)
    y2 = TruncatedSVDStage(k).execute(x)
    np.testing.assert_allclose(y1, y2, rtol=1e-14, atol=1e-14)


def test_stages_package_exports_truncated_svd_stage() -> None:
    """``TruncatedSVDStage`` is part of the public ``src.core.stages`` API."""
    import src.core.stages as stages

    assert "TruncatedSVDStage" in stages.__all__
    assert stages.TruncatedSVDStage is TruncatedSVDStage


def test_svd_truncated_rank_reduces_information() -> None:
    rng = np.random.default_rng(1)
    l_row, cols = 6, 10
    x = rng.standard_normal((l_row, cols))
    y = CSVDStage(FixedRankStrategy(2)).execute(x)
    assert y.shape == x.shape
    err = np.linalg.norm(y - x, ord="fro")
    assert err > 1e-6


def test_c_svd_w_corr_requires_window_length() -> None:
    x = np.ones((4, 6), dtype=np.float64)
    stage = CSVDStage(FixedRankStrategy(2), w_corr_threshold=0.5)
    with pytest.raises(ValueError, match="window_length"):
        stage.execute(x)


def test_c_svd_w_corr_rejects_nonpositive_window_length() -> None:
    x = np.ones((4, 6), dtype=np.float64)
    stage = CSVDStage(
        FixedRankStrategy(2),
        w_corr_threshold=0.5,
        window_length=0,
    )
    with pytest.raises(ValueError, match="window_length"):
        stage.execute(x)


def test_c_svd_w_corr_threshold_runs_and_matches_shape() -> None:
    rng = np.random.default_rng(99)
    x = rng.standard_normal((5, 8))
    base = CSVDStage(FixedRankStrategy(3)).execute(x)
    y = CSVDStage(
        FixedRankStrategy(3),
        w_corr_threshold=0.0,
        window_length=5,
    ).execute(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    # threshold 0: W[i,0] in [0,1] so W[i,0] >= 0 keeps components; matches no-op filter
    np.testing.assert_allclose(y, base, rtol=1e-14, atol=1e-14)


def test_c_svd_energy_strategy_with_w_corr_runs() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 8))
    y = CSVDStage(
        EnergyThresholdStrategy(0.99),
        w_corr_threshold=0.0,
        window_length=5,
    ).execute(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_w_corr_calls_w_corr_keep_indices_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second frame must not re-enter full W-matrix path when energy rank varies."""
    calls: list[int] = []
    real = c_svd_module._w_corr_keep_indices

    def counting(
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        window_length: int,
        w_corr_threshold: float,
    ) -> NDArray[Any]:
        calls.append(1)
        return real(u, s, vh, window_length, w_corr_threshold)

    monkeypatch.setattr(c_svd_module, "_w_corr_keep_indices", counting)

    rng = np.random.default_rng(202)
    stage = CSVDStage(
        EnergyThresholdStrategy(0.85),
        w_corr_threshold=0.0,
        window_length=4,
    )
    stage.execute(rng.standard_normal((6, 10)))
    assert len(calls) == 1
    # Different matrix -> different per-frame energy rank k; still no recompute.
    stage.execute(rng.standard_normal((6, 10)))
    assert len(calls) == 1


def test_fixed_rank_w_corr_still_one_call_when_k_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = c_svd_module._w_corr_keep_indices

    def counting(
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        window_length: int,
        w_corr_threshold: float,
    ) -> NDArray[Any]:
        calls.append(1)
        return real(u, s, vh, window_length, w_corr_threshold)

    monkeypatch.setattr(c_svd_module, "_w_corr_keep_indices", counting)

    rng = np.random.default_rng(203)
    stage = CSVDStage(
        FixedRankStrategy(3),
        w_corr_threshold=0.0,
        window_length=4,
    )
    stage.execute(rng.standard_normal((5, 8)))
    stage.execute(rng.standard_normal((5, 8)))
    assert len(calls) == 1
