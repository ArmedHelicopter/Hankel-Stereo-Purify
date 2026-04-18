"""SVD step: Lanczos ``svds`` with full-SVD fallback when ``k >= min(shape)``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse.linalg import svds  # type: ignore[import-untyped]

from ..array_types import FloatArray
from ..exceptions import validate_w_corr_threshold
from ..strategies.grouping import compute_w_correlation_matrix
from ..strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)
from .d_diagonal import batched_diagonal_average

# Partial ``svds`` probe iterations before a single full ``scipy.linalg.svd`` fallback.
# Tests may monkeypatch this (see ``tests/test_c_svd.py``); keep default at 8.
_SVDS_ENERGY_PROBE_CAP = 8


def _energy_frobenius_tol(fro_sq: float) -> float:
    """Tolerance for Frobenius-energy comparisons (partial vs full spectrum)."""
    return 1e-12 * max(float(fro_sq), 1.0)


@dataclass
class _SvdStepState:
    """Per-OLA-run state for :func:`make_svd_step` (W-cache + energy warm-start)."""

    cached_valid_indices: NDArray[np.intp] | None = None
    cached_k: int | None = None
    energy_w_corr_frozen: bool = False
    frozen_w_corr_keep_indices: NDArray[np.intp] | None = None
    energy_k_prev: int | None = None


def _reconstruct_usvh(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
) -> FloatArray:
    """Low-rank ``U @ diag(S) @ Vh`` via column scaling (no ``diag(S)`` matrix)."""
    return (u * s) @ vh


def _sort_usvh_descending(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if s.size == 0:
        return u, s, vh
    order = np.argsort(s)[::-1]
    return u[:, order], s[order], vh[order, :]


def _fixed_rank_truncated_factors(
    a: NDArray[np.float64],
    k_req: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    m, n = int(a.shape[0]), int(a.shape[1])
    mn = min(m, n)
    if k_req >= mn:
        u, s, vh = scipy.linalg.svd(a, full_matrices=False)
        k_eff = min(k_req, int(s.size))
    else:
        u, s, vh = svds(a, k=k_req, which="LM")
        u, s, vh = _sort_usvh_descending(u, s, vh)
        k_eff = k_req
    return u[:, :k_eff], s[:k_eff].copy(), vh[:k_eff, :]


def _frobenius_squared(a: NDArray[np.float64]) -> float:
    return float(np.sum(np.square(a)))


def _smallest_k_for_threshold_energy(
    s_desc: NDArray[np.float64],
    energy_fraction: float,
    fro_sq: float,
) -> int | None:
    """Rank k from partial singular values vs Frobenius energy; None if insufficient."""
    s_desc = np.asarray(s_desc, dtype=np.float64).ravel()
    n = int(s_desc.size)
    if n == 0:
        return None
    e = s_desc * s_desc
    cum = np.cumsum(e)
    target = energy_fraction * fro_sq
    tol = _energy_frobenius_tol(fro_sq)
    if cum[-1] + tol < target:
        return None
    idx = int(np.searchsorted(cum, target, side="left"))
    return max(1, min(idx + 1, n))


def _energy_truncated_factors(
    a: NDArray[np.float64],
    strat: EnergyThresholdStrategy,
    state: _SvdStepState,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Energy threshold via partial ``svds`` warm-start; full SVD only when needed.

    The ``for _ in range(_SVDS_ENERGY_PROBE_CAP)`` loop is a **backoff cap**:
    each iteration probes a larger ``k_probe`` (until ``k_probe >= min(m, n)``,
    when we take a single full ``scipy.linalg.svd`` and return). If after that
    many attempts we still
    cannot certify the energy threshold from partial spectra, we **fall through**
    to a full ``svd`` (same as the loop body when ``k_probe >= mn``). This
    bounds worst-case repeated ``svds`` work without changing the mathematical
    energy rule (which still uses ``strat.get_k`` on the final singular values).
    """
    m, n = int(a.shape[0]), int(a.shape[1])
    mn = min(m, n)
    fro_sq = _frobenius_squared(a)
    thr = strat.threshold
    margin = max(2, min(16, max(mn // 8, 1)))

    k_prev = state.energy_k_prev

    if k_prev is None:
        k_probe = min(mn, max(8, (mn + 3) // 4))
    else:
        k_probe = min(mn, k_prev + margin)

    for _ in range(_SVDS_ENERGY_PROBE_CAP):
        if k_probe >= mn:
            u, singular_values, vh = scipy.linalg.svd(a, full_matrices=False)
            k = int(strat.get_k(singular_values))
            max_rank = min(m, n, len(singular_values))
            rank = min(k, max_rank)
            state.energy_k_prev = int(k)
            return (
                u[:, :rank],
                singular_values[:rank].copy(),
                vh[:rank, :],
            )

        u, sv, vh = svds(a, k=k_probe, which="LM")
        u, sv, vh = _sort_usvh_descending(u, sv, vh)
        top_e = float(np.sum(sv * sv))
        if top_e + _energy_frobenius_tol(fro_sq) < thr * fro_sq:
            k_probe = min(mn, max(k_probe + margin, int(k_probe * 1.5) + 1))
            continue

        k_need = _smallest_k_for_threshold_energy(sv, thr, fro_sq)
        if k_need is None:
            k_probe = min(mn, k_probe + margin)
            continue
        rank = min(k_need, len(sv), mn)
        state.energy_k_prev = int(k_need)
        return u[:, :rank], sv[:rank].copy(), vh[:rank, :]

    u, singular_values, vh = scipy.linalg.svd(a, full_matrices=False)
    k = int(strat.get_k(singular_values))
    max_rank = min(m, n, len(singular_values))
    rank = min(k, max_rank)
    state.energy_k_prev = int(k)
    return u[:, :rank], singular_values[:rank].copy(), vh[:rank, :]


def _w_corr_keep_indices(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
    window_length: int,
    w_corr_threshold: float,
) -> NDArray[np.intp]:
    k = int(s.shape[0])
    if k <= 1:
        return np.arange(k, dtype=np.intp)
    # (k, m, n): rank-1 tensors via broadcasting (equivalent to einsum mi,i,in->imn).
    rank1_batch = u.T[:, :, np.newaxis] * (s[:, np.newaxis] * vh)[:, np.newaxis, :]
    components_1d = batched_diagonal_average(rank1_batch)
    w_mat = compute_w_correlation_matrix(components_1d, window_length)
    rest = w_mat[1:, 0] >= w_corr_threshold
    idx_gt0 = np.flatnonzero(rest) + 1
    valid = np.empty(1 + idx_gt0.size, dtype=np.intp)
    valid[0] = 0
    valid[1:] = idx_gt0.astype(np.intp, copy=False)
    return valid


def _zero_s_except_indices(
    s: NDArray[np.float64],
    keep_indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    k = int(s.shape[0])
    out = s.copy()
    keep = np.zeros(k, dtype=bool)
    keep[keep_indices] = True
    out[~keep] = 0.0
    return out


def _prepare_svd_frame(
    data: FloatArray,
    w_corr_threshold: float | None,
    window_length: int | None,
) -> tuple[NDArray[np.float64], int | None]:
    """Validate frame shape and optional W-correlation window.

    Returns ``(a, corr_wl)`` where ``corr_wl`` is set iff W-correlation is on.
    """
    a = np.asarray(data, dtype=np.float64, order="C")
    m, n = int(a.shape[0]), int(a.shape[1])
    mn = min(m, n)
    if mn == 0:
        raise ValueError("SVD requires a non-empty matrix.")
    corr_wl: int | None = None
    if w_corr_threshold is not None:
        if window_length is None:
            raise ValueError(
                "window_length must be a positive int when w_corr_threshold is set.",
            )
        corr_wl = int(window_length)
        if corr_wl <= 0:
            raise ValueError(
                "window_length must be a positive int when w_corr_threshold is set.",
            )
    return a, corr_wl


def _make_svd_step_fixed_rank(
    strat: FixedRankStrategy,
    *,
    w_corr_threshold: float | None,
    window_length: int | None,
) -> Callable[[FloatArray], FloatArray]:
    state = _SvdStepState()

    def _filter_w_corr(
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        corr_wl: int,
        thr: float,
    ) -> NDArray[np.float64]:
        k = int(s.shape[0])
        cached_vi = state.cached_valid_indices
        cached_k_rank = state.cached_k
        if cached_vi is None or cached_k_rank is None or cached_k_rank != k:
            state.cached_valid_indices = _w_corr_keep_indices(u, s, vh, corr_wl, thr)
            state.cached_k = k
        keep = state.cached_valid_indices
        if keep is None:
            raise RuntimeError("W-correlation cache not populated after refresh.")
        return _zero_s_except_indices(s, keep)

    def svd_step(data: FloatArray) -> FloatArray:
        a, corr_wl = _prepare_svd_frame(data, w_corr_threshold, window_length)
        k_req = int(strat.k)
        if k_req <= 0:
            raise ValueError("Truncation rank must be positive.")
        u, s, vh = _fixed_rank_truncated_factors(a, k_req)
        if w_corr_threshold is not None:
            if corr_wl is None:
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )
            s = _filter_w_corr(
                u,
                s,
                vh,
                corr_wl,
                float(w_corr_threshold),
            )
        reconstructed: FloatArray = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)

    return svd_step


def _make_svd_step_energy(
    strat: EnergyThresholdStrategy,
    *,
    w_corr_threshold: float | None,
    window_length: int | None,
) -> Callable[[FloatArray], FloatArray]:
    state = _SvdStepState()

    def _filter_w_corr(
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        corr_wl: int,
        thr: float,
    ) -> NDArray[np.float64]:
        k = int(s.shape[0])
        if not state.energy_w_corr_frozen:
            state.frozen_w_corr_keep_indices = _w_corr_keep_indices(
                u, s, vh, corr_wl, thr
            )
            state.energy_w_corr_frozen = True
        frozen = state.frozen_w_corr_keep_indices
        if not isinstance(frozen, np.ndarray):
            raise TypeError(
                "W-correlation frozen indices must be a numpy.ndarray; "
                f"got {type(frozen).__name__}.",
            )
        keep = frozen[frozen < k].astype(np.intp, copy=False)
        if k >= 1:
            if keep.size == 0:
                keep = np.array([0], dtype=np.intp)
            elif not np.any(keep == 0):
                keep = np.sort(
                    np.unique(
                        np.concatenate(
                            (np.array([0], dtype=np.intp), keep),
                        ),
                    ),
                )
                keep = keep[keep < k]
        return _zero_s_except_indices(s, keep)

    def svd_step(data: FloatArray) -> FloatArray:
        a, corr_wl = _prepare_svd_frame(data, w_corr_threshold, window_length)
        u, s, vh = _energy_truncated_factors(a, strat, state)
        if w_corr_threshold is not None:
            if corr_wl is None:
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )
            s = _filter_w_corr(
                u,
                s,
                vh,
                corr_wl,
                float(w_corr_threshold),
            )
        reconstructed = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)

    return svd_step


def make_svd_step(
    truncation_strategy: TruncationStrategy,
    *,
    w_corr_threshold: float | None = None,
    window_length: int | None = None,
) -> Callable[[FloatArray], FloatArray]:
    """Return a stateful SVD+truncate callable (one per OLA / preview run).

    Dispatches on the concrete strategy type once at construction; the returned
    ``svd_step`` does not branch on ``isinstance`` per frame.
    """
    validate_w_corr_threshold(w_corr_threshold)
    if isinstance(truncation_strategy, FixedRankStrategy):
        return _make_svd_step_fixed_rank(
            truncation_strategy,
            w_corr_threshold=w_corr_threshold,
            window_length=window_length,
        )
    if isinstance(truncation_strategy, EnergyThresholdStrategy):
        return _make_svd_step_energy(
            truncation_strategy,
            w_corr_threshold=w_corr_threshold,
            window_length=window_length,
        )
    raise TypeError(
        "truncation_strategy must be FixedRankStrategy or EnergyThresholdStrategy, "
        f"got {type(truncation_strategy).__name__!r}",
    )


def make_fixed_rank_svd_step(
    truncation_rank: int,
    *,
    w_corr_threshold: float | None = None,
    window_length: int | None = None,
) -> Callable[[FloatArray], FloatArray]:
    """Fixed-rank SVD step (same as ``make_svd_step(FixedRankStrategy(k), ...)``)."""
    if truncation_rank <= 0:
        raise ValueError("Truncation rank must be positive.")
    return _make_svd_step_fixed_rank(
        FixedRankStrategy(truncation_rank),
        w_corr_threshold=w_corr_threshold,
        window_length=window_length,
    )
