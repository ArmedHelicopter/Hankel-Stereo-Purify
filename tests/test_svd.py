from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import src.core.stages.svd as svd_module
from src.core.stages.svd import (
    _reconstruct_usvh,
    _smallest_k_for_threshold_energy,
    _sort_usvh_descending,
    make_fixed_rank_svd_step,
    make_svd_step,
)
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
    step = make_svd_step(FixedRankStrategy(k))
    y = step(x)
    np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-10)


def test_make_svd_step_rejects_unknown_strategy_type() -> None:
    """Factory dispatches on concrete types; other objects fail at construction."""

    class _NotAStrategy:
        pass

    with pytest.raises(TypeError, match="FixedRankStrategy or EnergyThresholdStrategy"):
        make_svd_step(_NotAStrategy())  # type: ignore[arg-type]


def test_make_fixed_rank_svd_step_rejects_nonpositive_rank_at_factory() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_fixed_rank_svd_step(0)
    with pytest.raises(ValueError, match="positive"):
        make_fixed_rank_svd_step(-1)


def test_svd_step_rejects_empty_matrix() -> None:
    step = make_svd_step(FixedRankStrategy(1))
    with pytest.raises(ValueError, match="non-empty"):
        step(np.zeros((0, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="non-empty"):
        step(np.zeros((3, 0), dtype=np.float64))


def test_sort_usvh_descending_noop_when_empty_s() -> None:
    u = np.zeros((3, 0), dtype=np.float64)
    s = np.zeros((0,), dtype=np.float64)
    vh = np.zeros((0, 4), dtype=np.float64)
    u2, s2, vh2 = _sort_usvh_descending(u, s, vh)
    assert s2.size == 0
    assert u2.shape == u.shape and vh2.shape == vh.shape


def test_smallest_k_for_threshold_energy_edge_cases() -> None:
    assert (
        _smallest_k_for_threshold_energy(
            np.array([], dtype=np.float64),
            0.9,
            1.0,
        )
        is None
    )
    # Partial spectrum energy far below target fraction of full Frobenius norm.
    s_partial = np.array([0.1, 0.1], dtype=np.float64)
    assert _smallest_k_for_threshold_energy(s_partial, 0.99, fro_sq=100.0) is None


def test_smallest_k_for_threshold_energy_finds_rank_via_searchsorted() -> None:
    """Covers successful ``searchsorted`` path (lines 91–92 in ``svd``)."""
    s = np.array([0.8, 0.5, 0.3], dtype=np.float64)
    k = _smallest_k_for_threshold_energy(s, 0.5, fro_sq=1.0)
    assert k is not None
    assert 1 <= k <= int(s.size)


def test_fixed_rank_svd_step_matches_make_svd_fixed_strategy() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, 10))
    k = 3
    y1 = make_svd_step(FixedRankStrategy(k))(x)
    y2 = make_fixed_rank_svd_step(k)(x)
    np.testing.assert_allclose(y1, y2, rtol=1e-14, atol=1e-14)


def test_stages_package_exports_svd_factories() -> None:
    import src.core.stages as stages

    assert "make_svd_step" in stages.__all__
    assert "make_fixed_rank_svd_step" in stages.__all__
    assert stages.make_fixed_rank_svd_step is make_fixed_rank_svd_step


def test_svd_truncated_rank_reduces_information() -> None:
    rng = np.random.default_rng(1)
    l_row, cols = 6, 10
    x = rng.standard_normal((l_row, cols))
    y = make_svd_step(FixedRankStrategy(2))(x)
    assert y.shape == x.shape
    err = np.linalg.norm(y - x, ord="fro")
    assert err > 1e-6


def _fake_energy_factors_rank2(
    a: NDArray[np.float64],
    _strat: Any,
    _state: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    m, n = int(a.shape[0]), int(a.shape[1])
    u = np.zeros((m, 2), dtype=np.float64)
    u[0, 0] = 1.0
    u[1, 1] = 1.0
    s = np.array([5.0, 3.0], dtype=np.float64)
    vh = np.zeros((2, n), dtype=np.float64)
    vh[0, 0] = 1.0
    vh[1, 1] = 1.0
    return u, s, vh


def _fake_energy_factors_rank3(
    a: NDArray[np.float64],
    _strat: Any,
    _state: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    m, n = int(a.shape[0]), int(a.shape[1])
    u = np.zeros((m, 3), dtype=np.float64)
    for i in range(min(3, m)):
        u[i, i] = 1.0
    s = np.array([4.0, 3.0, 2.0], dtype=np.float64)
    vh = np.zeros((3, n), dtype=np.float64)
    for i in range(min(3, n)):
        vh[i, i] = 1.0
    return u, s, vh


def test_energy_truncated_skips_probe_loop_when_cap_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full ``svd`` fallback (tail of ``_energy_truncated_factors``) with no probes."""
    monkeypatch.setattr(svd_module, "_SVDS_ENERGY_PROBE_CAP", 0)
    rng = np.random.default_rng(901)
    x = rng.standard_normal((18, 22))
    y = make_svd_step(EnergyThresholdStrategy(0.92))(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_returns_early_from_partial_svds_when_k_probe_lt_mn() -> None:
    """Energy step: partial ``svds`` when initial ``k_probe`` is below ``min(m,n)``."""
    rng = np.random.default_rng(602)
    x = rng.standard_normal((14, 18))
    y = make_svd_step(EnergyThresholdStrategy(0.88))(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_probe_k_need_none_then_full_svd_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single probe, ``k_need is None`` branch (148–151), then full ``svd``."""
    monkeypatch.setattr(svd_module, "_SVDS_ENERGY_PROBE_CAP", 1)
    monkeypatch.setattr(
        svd_module,
        "_smallest_k_for_threshold_energy",
        lambda *_a, **_kw: None,
    )
    rng = np.random.default_rng(902)
    x = rng.standard_normal((24, 30))
    # Low threshold so the partial top block almost always clears the ``top_e`` gate.
    y = make_svd_step(EnergyThresholdStrategy(1e-12))(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_probe_insufficient_top_e_increases_k_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low partial energy: increase ``k_probe`` before any full ``svd``."""

    def tiny_svds(
        a: NDArray[np.float64],
        *,
        k: int,
        which: str = "LM",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        del which
        m, n = int(a.shape[0]), int(a.shape[1])
        kk = min(k, m, n)
        u = np.eye(m, dtype=np.float64)[:, :kk]
        sv = np.full(kk, 1e-6, dtype=np.float64)
        vh = np.eye(kk, n, dtype=np.float64)
        return u, sv, vh

    monkeypatch.setattr(svd_module, "svds", tiny_svds)
    monkeypatch.setattr(svd_module, "_SVDS_ENERGY_PROBE_CAP", 3)
    x = np.ones((18, 20), dtype=np.float64)
    y = make_svd_step(EnergyThresholdStrategy(0.95))(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
