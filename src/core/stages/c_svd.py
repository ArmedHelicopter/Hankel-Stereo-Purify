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
    WienerStrategy,
    TruncationStrategy,
)
from .d_diagonal import fast_diagonal_average

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
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    k = int(s.shape[0])
    if k <= 1:
        return np.arange(k, dtype=np.intp)
    m, n = int(u.shape[0]), int(vh.shape[1])
    l_seq = m + n - 1
    components_1d = np.empty((k, l_seq), dtype=np.float64)
    for r in range(k):
        rank1 = np.outer(u[:, r], vh[r, :]) * s[r]
        components_1d[r] = fast_diagonal_average(rank1)
    w_mat = compute_w_correlation_matrix(components_1d, window_length)

    # Convert similarity to distance, force diagonal to 0
    dist_mat = 1.0 - np.abs(w_mat)
    np.fill_diagonal(dist_mat, 0.0)

    # Hierarchical clustering (average linkage)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - w_corr_threshold, criterion="distance")

    # Energy per cluster
    total_energy = float(np.sum(s[:k] ** 2))
    cluster_energy: dict[int, float] = {}
    for i, lab in enumerate(labels):
        cluster_energy[lab] = cluster_energy.get(lab, 0.0) + float(s[i]) ** 2

    # Keep all clusters whose energy >= 5% of total (conservative filter)
    # Minimum 1 component always kept
    min_energy_frac = 0.05
    keep_clusters = {
        lab
        for lab, e in cluster_energy.items()
        if e >= min_energy_frac * total_energy
    }
    if not keep_clusters:
        # Fallback: keep the largest cluster
        keep_clusters = {max(cluster_energy, key=cluster_energy.get)}

    keep = np.array(
        [i for i, lab in enumerate(labels) if lab in keep_clusters],
        dtype=np.intp,
    )
    return np.sort(keep)


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


class _FixedRankSvdStep:
    """Fixed-rank SVD step; optional W-correlation uses :class:`_SvdStepState` cache."""

    __slots__ = ("_strat", "_w_corr_threshold", "_window_length", "state")

    def __init__(
        self,
        strat: FixedRankStrategy,
        *,
        w_corr_threshold: float | None,
        window_length: int | None,
    ) -> None:
        self._strat = strat
        self._w_corr_threshold = w_corr_threshold
        self._window_length = window_length
        self.state = _SvdStepState()

    def _filter_w_corr(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        corr_wl: int,
        thr: float,
    ) -> NDArray[np.float64]:
        k = int(s.shape[0])
        cached_vi = self.state.cached_valid_indices
        cached_k_rank = self.state.cached_k
        if cached_vi is None or cached_k_rank is None or cached_k_rank != k:
            self.state.cached_valid_indices = _w_corr_keep_indices(
                u, s, vh, corr_wl, thr
            )
            self.state.cached_k = k
        keep = self.state.cached_valid_indices
        if keep is None:
            raise RuntimeError("W-correlation cache not populated after refresh.")
        return _zero_s_except_indices(s, keep)

    def __call__(self, data: FloatArray) -> FloatArray:
        a, corr_wl = _prepare_svd_frame(
            data, self._w_corr_threshold, self._window_length
        )
        k_req = int(self._strat.k)
        if k_req <= 0:
            raise ValueError("Truncation rank must be positive.")
        u, s, vh = _fixed_rank_truncated_factors(a, k_req)
        if self._w_corr_threshold is not None:
            if corr_wl is None:
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )
            s = self._filter_w_corr(
                u,
                s,
                vh,
                corr_wl,
                float(self._w_corr_threshold),
            )
        reconstructed: FloatArray = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)


class _EnergySvdStep:
    """Energy-threshold SVD step; W-correlation indices frozen after first frame."""

    __slots__ = ("_strat", "_w_corr_threshold", "_window_length", "state")

    def __init__(
        self,
        strat: EnergyThresholdStrategy,
        *,
        w_corr_threshold: float | None,
        window_length: int | None,
    ) -> None:
        self._strat = strat
        self._w_corr_threshold = w_corr_threshold
        self._window_length = window_length
        self.state = _SvdStepState()

    def _filter_w_corr(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        corr_wl: int,
        thr: float,
    ) -> NDArray[np.float64]:
        k = int(s.shape[0])
        keep = _w_corr_keep_indices(u, s, vh, corr_wl, thr)
        if keep.size == 0:
            keep = np.array([0], dtype=np.intp)
        keep = keep[keep < k]
        return _zero_s_except_indices(s, keep)

    def __call__(self, data: FloatArray) -> FloatArray:
        a, corr_wl = _prepare_svd_frame(
            data, self._w_corr_threshold, self._window_length
        )
        u, s, vh = _energy_truncated_factors(a, self._strat, self.state)
        if self._w_corr_threshold is not None:
            if corr_wl is None:
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )
            s = self._filter_w_corr(
                u,
                s,
                vh,
                corr_wl,
                float(self._w_corr_threshold),
            )
        reconstructed = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)


class _WienerSvdStep:
    """Wiener soft weighting: continuous per-component gain instead of hard truncation.

    Computes full SVD, estimates noise variance from the bottom fraction of
    singular values, then applies Wiener gain:
        w_i = max(0, 1 - σ_noise² / σ_i²)
    to each component before reconstruction.
    """

    __slots__ = ("_noise_fraction",)

    def __init__(self, strat: WienerStrategy) -> None:
        self._noise_fraction = strat.noise_fraction

    def __call__(self, data: FloatArray) -> FloatArray:
        a = np.asarray(data, dtype=np.float64, order="C")
        u, s, vh = scipy.linalg.svd(a, full_matrices=False)
        k = int(s.size)
        if k == 0:
            return a

        # Noise estimate from tail singular values (no extra svds needed)
        n_noise = max(1, int(k * self._noise_fraction))
        noise_var = float(np.mean(s[-n_noise:] ** 2))

        # Wiener gain
        s_sq = s * s
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(s_sq > noise_var, 1.0 - noise_var / s_sq, 0.0)

        weighted_s = s * weights
        reconstructed = _reconstruct_usvh(u, weighted_s, vh)
        return reconstructed.astype(np.float64, copy=False)


class _WienerSvdStepCuda:
    """Wiener SVD via CUDA GPU acceleration. Falls back to CPU if CUDA unavailable."""

    __slots__ = ("_noise_fraction", "_cuda_fn")

    def __init__(self, strat: WienerStrategy) -> None:
        self._noise_fraction = strat.noise_fraction
        from src.cuda.wiener_cuda import wiener_svd_cuda
        self._cuda_fn = wiener_svd_cuda

    def __call__(self, data: FloatArray) -> FloatArray:
        a = np.asarray(data, dtype=np.float64, order="C")
        # CUDA operates on Hankel-embedded matrix (rows x cols)
        # Input shape is (F, 2), need Hankel embed first — but process_frame
        # calls svd_step AFTER hankel_embed + combine, so `data` is already
        # the joint matrix (rows x cols).
        return self._cuda_fn(a, self._noise_fraction)


def _make_svd_step_fixed_rank(
    strat: FixedRankStrategy,
    *,
    w_corr_threshold: float | None,
    window_length: int | None,
) -> Callable[[FloatArray], FloatArray]:
    return _FixedRankSvdStep(
        strat,
        w_corr_threshold=w_corr_threshold,
        window_length=window_length,
    )


def _make_svd_step_energy(
    strat: EnergyThresholdStrategy,
    *,
    w_corr_threshold: float | None,
    window_length: int | None,
) -> Callable[[FloatArray], FloatArray]:
    return _EnergySvdStep(
        strat,
        w_corr_threshold=w_corr_threshold,
        window_length=window_length,
    )


def _make_svd_step_wiener(
    strat: WienerStrategy,
) -> Callable[[FloatArray], FloatArray]:
    # Try CUDA first, fall back to CPU
    try:
        from src.cuda.wiener_cuda import is_available
        if is_available():
            return _WienerSvdStepCuda(strat)
    except ImportError:
        pass
    return _WienerSvdStep(strat)


def make_svd_step(
    truncation_strategy: TruncationStrategy,
    *,
    w_corr_threshold: float | None = None,
    window_length: int | None = None,
) -> Callable[[FloatArray], FloatArray]:
    """Return a stateful SVD+truncate callable (one per OLA / preview run).

    Dispatches on the concrete strategy type once at construction; the returned
    callable does not branch on ``isinstance`` per frame.
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
    if isinstance(truncation_strategy, WienerStrategy):
        return _make_svd_step_wiener(truncation_strategy)
    raise TypeError(
        "truncation_strategy must be FixedRankStrategy, EnergyThresholdStrategy, "
        f"or WienerStrategy, got {type(truncation_strategy).__name__!r}",
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
