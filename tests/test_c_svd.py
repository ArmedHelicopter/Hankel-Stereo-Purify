from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import src.core.stages.c_svd as c_svd_module
from src.core.stages.c_svd import (
    _reconstruct_usvh,
    _smallest_k_for_threshold_energy,
    _sort_usvh_descending,
    _w_corr_keep_indices,
    make_fixed_rank_svd_step,
    make_svd_step,
)
from src.core.stages.d_diagonal import fast_diagonal_average
from src.core.strategies.grouping import compute_w_correlation_matrix
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
    """Covers successful ``searchsorted`` path (lines 91–92 in ``c_svd``)."""
    s = np.array([0.8, 0.5, 0.3], dtype=np.float64)
    k = _smallest_k_for_threshold_energy(s, 0.5, fro_sq=1.0)
    assert k is not None
    assert 1 <= k <= int(s.size)


def test_w_corr_keep_indices_rank_one_returns_zero_only() -> None:
    rng = np.random.default_rng(88)
    m, n, k = 4, 5, 1
    u = rng.standard_normal((m, k))
    s = np.array([3.0], dtype=np.float64)
    vh = rng.standard_normal((k, n))
    got = _w_corr_keep_indices(u, s, vh, window_length=3, w_corr_threshold=0.0)
    np.testing.assert_array_equal(got, np.array([0], dtype=np.intp))


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


def test_c_svd_w_corr_requires_window_length() -> None:
    x = np.ones((4, 6), dtype=np.float64)
    step = make_svd_step(FixedRankStrategy(2), w_corr_threshold=0.5)
    with pytest.raises(ValueError, match="window_length"):
        step(x)


def test_c_svd_w_corr_rejects_nonpositive_window_length() -> None:
    x = np.ones((4, 6), dtype=np.float64)
    step = make_svd_step(
        FixedRankStrategy(2),
        w_corr_threshold=0.5,
        window_length=0,
    )
    with pytest.raises(ValueError, match="window_length"):
        step(x)


def test_c_svd_w_corr_threshold_runs_and_matches_shape() -> None:
    rng = np.random.default_rng(99)
    x = rng.standard_normal((5, 8))
    base = make_svd_step(FixedRankStrategy(3))(x)
    y = make_svd_step(
        FixedRankStrategy(3),
        w_corr_threshold=0.0,
        window_length=5,
    )(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, base, rtol=1e-14, atol=1e-14)


def test_c_svd_energy_strategy_with_w_corr_runs() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 8))
    y = make_svd_step(
        EnergyThresholdStrategy(0.99),
        w_corr_threshold=0.0,
        window_length=5,
    )(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_w_corr_calls_w_corr_keep_indices_only_once(
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

    rng = np.random.default_rng(202)
    step = make_svd_step(
        EnergyThresholdStrategy(0.85),
        w_corr_threshold=0.0,
        window_length=4,
    )
    step(rng.standard_normal((6, 10)))
    assert len(calls) == 1
    step(rng.standard_normal((6, 10)))
    assert len(calls) == 1


def _reference_w_corr_keep_indices_loop(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
    window_length: int,
    w_corr_threshold: float,
) -> NDArray[np.intp]:
    """Explicit loop + per-matrix diagonal average (parity reference)."""
    k = int(s.shape[0])
    if k <= 1:
        return np.arange(k, dtype=np.intp)
    rows: list[NDArray[np.float64]] = []
    for i in range(k):
        rank1 = (u[:, i : i + 1] * s[i]) @ vh[i : i + 1, :]
        rows.append(fast_diagonal_average(rank1))
    components_1d = np.row_stack(rows)
    w_mat = compute_w_correlation_matrix(components_1d, window_length)
    valid: list[int] = [0]
    for i in range(1, k):
        if float(w_mat[i, 0]) >= w_corr_threshold:
            valid.append(i)
    return np.asarray(valid, dtype=np.intp)


def test_w_corr_keep_indices_matches_reference_loop() -> None:
    rng = np.random.default_rng(501)
    for k in (2, 5, 8):
        m, n = 6, 9
        u = rng.standard_normal((m, k))
        s = np.abs(rng.standard_normal(k))
        vh = rng.standard_normal((k, n))
        wl = 4
        thr = 0.35
        got = _w_corr_keep_indices(u, s, vh, wl, thr)
        ref = _reference_w_corr_keep_indices_loop(u, s, vh, wl, thr)
        np.testing.assert_array_equal(got, ref)


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
    step = make_svd_step(
        FixedRankStrategy(3),
        w_corr_threshold=0.0,
        window_length=4,
    )
    step(rng.standard_normal((5, 8)))
    step(rng.standard_normal((5, 8)))
    assert len(calls) == 1


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
    monkeypatch.setattr(c_svd_module, "_SVDS_ENERGY_PROBE_CAP", 0)
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
    monkeypatch.setattr(c_svd_module, "_SVDS_ENERGY_PROBE_CAP", 1)
    monkeypatch.setattr(
        c_svd_module,
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

    monkeypatch.setattr(c_svd_module, "svds", tiny_svds)
    monkeypatch.setattr(c_svd_module, "_SVDS_ENERGY_PROBE_CAP", 3)
    x = np.ones((18, 20), dtype=np.float64)
    y = make_svd_step(EnergyThresholdStrategy(0.95))(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_w_corr_prepends_zero_when_frozen_indices_exceed_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen keep indices all ``>= k`` → empty selection → repair with ``0``."""

    def w_idx(
        *_a: Any,
        **_kw: Any,
    ) -> NDArray[np.intp]:
        return np.array([2, 3], dtype=np.intp)

    monkeypatch.setattr(
        c_svd_module,
        "_energy_truncated_factors",
        _fake_energy_factors_rank2,
    )
    monkeypatch.setattr(c_svd_module, "_w_corr_keep_indices", w_idx)

    rng = np.random.default_rng(904)
    x = rng.standard_normal((6, 9))
    step = make_svd_step(
        EnergyThresholdStrategy(0.88),
        w_corr_threshold=0.0,
        window_length=4,
    )
    y = step(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_energy_w_corr_inserts_zero_when_frozen_omits_leading_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen indices miss ``0`` but include other ranks → prepend ``0``."""

    def w_idx(
        *_a: Any,
        **_kw: Any,
    ) -> NDArray[np.intp]:
        return np.array([1, 2], dtype=np.intp)

    monkeypatch.setattr(
        c_svd_module,
        "_energy_truncated_factors",
        _fake_energy_factors_rank3,
    )
    monkeypatch.setattr(c_svd_module, "_w_corr_keep_indices", w_idx)

    rng = np.random.default_rng(903)
    x = rng.standard_normal((8, 10))
    step = make_svd_step(
        EnergyThresholdStrategy(0.9),
        w_corr_threshold=0.0,
        window_length=4,
    )
    y = step(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
