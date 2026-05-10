"""SVD step: Lanczos ``svds`` with full-SVD fallback when ``k >= min(shape)``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse.linalg import svds  # type: ignore[import-untyped]

from ..array_types import FloatArray
from ..strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)

# Partial ``svds`` probe iterations before a single full ``scipy.linalg.svd`` fallback.
# Tests may monkeypatch this (see ``tests/test_svd.py``); keep default at 8.
_SVDS_ENERGY_PROBE_CAP = 8


def _energy_frobenius_tol(fro_sq: float) -> float:
    """Tolerance for Frobenius-energy comparisons (partial vs full spectrum)."""
    return 1e-12 * max(float(fro_sq), 1.0)


@dataclass
class _SvdStepState:
    """Per-OLA-run state for :func:`make_svd_step` (energy warm-start)."""

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


def _prepare_svd_frame(
    data: FloatArray,
) -> NDArray[np.float64]:
    """Validate frame shape and return the matrix."""
    a = np.asarray(data, dtype=np.float64, order="C")
    m, n = int(a.shape[0]), int(a.shape[1])
    mn = min(m, n)
    if mn == 0:
        raise ValueError("SVD requires a non-empty matrix.")
    return a


class _FixedRankSvdStep:
    """Fixed-rank SVD step."""

    __slots__ = ("_strat", "state")

    def __init__(
        self,
        strat: FixedRankStrategy,
    ) -> None:
        self._strat = strat
        self.state = _SvdStepState()

    def __call__(self, data: FloatArray) -> FloatArray:
        a = _prepare_svd_frame(data)
        k_req = int(self._strat.k)
        if k_req <= 0:
            raise ValueError("Truncation rank must be positive.")
        u, s, vh = _fixed_rank_truncated_factors(a, k_req)
        reconstructed: FloatArray = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)


class _EnergySvdStep:
    """Energy-threshold SVD step."""

    __slots__ = ("_strat", "state")

    def __init__(
        self,
        strat: EnergyThresholdStrategy,
    ) -> None:
        self._strat = strat
        self.state = _SvdStepState()

    def __call__(self, data: FloatArray) -> FloatArray:
        a = _prepare_svd_frame(data)
        u, s, vh = _energy_truncated_factors(a, self._strat, self.state)
        reconstructed = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)


def _make_svd_step_fixed_rank(
    strat: FixedRankStrategy,
) -> Callable[[FloatArray], FloatArray]:
    return _FixedRankSvdStep(strat)


def _make_svd_step_energy(
    strat: EnergyThresholdStrategy,
) -> Callable[[FloatArray], FloatArray]:
    return _EnergySvdStep(strat)


def make_fixed_rank_svd_step(
    truncation_rank: int,
) -> Callable[[FloatArray], FloatArray]:
    """Fixed-rank SVD step (same as ``make_svd_step(FixedRankStrategy(k))``)."""
    if truncation_rank <= 0:
        raise ValueError("Truncation rank must be positive.")
    return _make_svd_step_fixed_rank(FixedRankStrategy(truncation_rank))


# ── CUDA-accelerated SVD step ──────────────────────────────────────────────


class _CudaFixedRankSvdStep:
    """Fixed-rank SVD via GPU randomized SVD.  Per-frame upload/download."""

    __slots__ = ("_k", "_ctx")

    def __init__(self, k: int) -> None:
        self._k = k
        self._ctx: Any = None  # lazy init

    def __call__(self, data: FloatArray) -> FloatArray:
        a = _prepare_svd_frame(data)
        m, n = a.shape  # m = L, n = 2K
        if self._ctx is None:
            from src.cuda.mssa_rand_cuda import CudaRandSvdContext

            # C-contiguous (m, n) → column-major (n, m)
            self._ctx = CudaRandSvdContext(m=n, n=m, rank=self._k, oversample=16)

        results = self._ctx.run([a])
        return cast(FloatArray, results[0])


def make_svd_step(
    truncation_strategy: TruncationStrategy,
    *,
    use_cuda: bool = False,
) -> Callable[[FloatArray], FloatArray]:
    """Return a stateful SVD+truncate callable (one per OLA / preview run).

    Dispatches on the concrete strategy type once at construction; the returned
    callable does not branch on ``isinstance`` per frame.

    Parameters
    ----------
    truncation_strategy
        Fixed-rank or energy-threshold strategy.
    use_cuda
        If True and CUDA library is available, use GPU-accelerated SVD
        (only supported for FixedRankStrategy).
    """
    if isinstance(truncation_strategy, FixedRankStrategy):
        if use_cuda:
            try:
                from src.cuda.mssa_rand_cuda import is_available

                if is_available():
                    return _CudaFixedRankSvdStep(truncation_strategy.k)
            except Exception:
                pass  # Fall through to CPU
        return _make_svd_step_fixed_rank(truncation_strategy)
    if isinstance(truncation_strategy, EnergyThresholdStrategy):
        return _make_svd_step_energy(truncation_strategy)
    raise TypeError(
        "truncation_strategy must be FixedRankStrategy or EnergyThresholdStrategy, "
        f"got {type(truncation_strategy).__name__!r}",
    )
