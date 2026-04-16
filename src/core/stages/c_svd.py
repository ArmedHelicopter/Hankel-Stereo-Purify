"""SVD stage: Lanczos ``svds`` with full-SVD fallback when ``k >= min(shape)``."""

from __future__ import annotations

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse.linalg import svds  # type: ignore[import-untyped]

from ..exceptions import validate_w_corr_threshold
from ..pipeline import FloatArray, MSSAStage
from ..strategies.grouping import compute_w_correlation_matrix
from ..strategies.truncation import (
    EnergyThresholdStrategy,
    FixedRankStrategy,
    TruncationStrategy,
)
from .d_diagonal import DDiagonalStage


def _reconstruct_usvh(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
) -> FloatArray:
    """Low-rank ``U @ diag(S) @ Vh`` via column scaling (avoids allocating ``diag(S)``).

    Shapes: ``U`` (m, r), ``S`` (r,), ``Vh`` (r, n); must satisfy
    ``U.shape[1] == S.shape[0] == Vh.shape[0]``.
    """
    return (u * s) @ vh


def _sort_usvh_descending(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Ensure singular values are descending (``svds`` may return ascending)."""
    if s.size == 0:
        return u, s, vh
    order = np.argsort(s)[::-1]
    return u[:, order], s[order], vh[order, :]


def _fixed_rank_truncated_factors(
    a: NDArray[np.float64],
    k_req: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Top ``k_req`` triplets via ``svds``, or full SVD if ``k_req >= min(shape)``."""
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


def _w_corr_keep_indices(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
    window_length: int,
    w_corr_threshold: float,
) -> NDArray[np.intp]:
    """Indices of components to keep (signal PC0 + those with W[i,0] >= threshold)."""
    k = int(s.shape[0])
    if k <= 1:
        return np.arange(k, dtype=np.intp)
    rows: list[NDArray[np.float64]] = []
    for i in range(k):
        rank1: FloatArray = (u[:, i : i + 1] * s[i]) @ vh[i : i + 1, :]
        one_d = DDiagonalStage.fast_diagonal_average(rank1)
        rows.append(one_d)
    components_1d = np.row_stack(rows)
    w_mat = compute_w_correlation_matrix(components_1d, window_length)
    valid: list[int] = [0]
    for i in range(1, k):
        if float(w_mat[i, 0]) >= w_corr_threshold:
            valid.append(i)
    return np.asarray(valid, dtype=np.intp)


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


class CSVDStage(MSSAStage[FloatArray, FloatArray]):
    """SVD decomposition and truncation stage for MSSA.

    Fixed rank: ``svds`` + full-SVD fallback, optional one-shot W-correlation filtering.
    Energy threshold: full ``scipy.linalg.svd`` once, then truncate by energy.

    **W-correlation (optional):** ``w_corr_threshold`` must lie in ``[0.0, 1.0]`` (CLI /
    ``MSSAPurifierBuilder`` enforce this). After building the per-component 1D series,
    :func:`compute_w_correlation_matrix` yields ``W`` with entries in ``[0, 1]``. A
    component ``i>0`` is kept iff ``W[i, 0] >= threshold`` (component 0 is always
    retained as the reference). Higher thresholds drop more components.

    With W-correlation enabled, **energy** mode runs full ``_w_corr_keep_indices`` only
    on the **first** ``execute``; later frames reuse frozen ordinal indices intersected
    with ``range(k_curr)`` (PC0 enforced when ``k_curr>=1``). Fixed-rank mode still
    recomputes when ``k`` changes (typically never).
    """

    def __init__(
        self,
        truncation_strategy: TruncationStrategy,
        *,
        w_corr_threshold: float | None = None,
        window_length: int | None = None,
    ) -> None:
        self.truncation_strategy = truncation_strategy
        validate_w_corr_threshold(w_corr_threshold)
        self.w_corr_threshold = w_corr_threshold
        self.window_length = window_length
        self._cached_valid_indices: NDArray[np.intp] | None = None
        self._cached_k: int | None = None
        # Energy + W-corr: calibrate once on first execute; reuse ordinal indices.
        self._energy_w_corr_frozen: bool = False
        self._frozen_w_corr_keep_indices: NDArray[np.intp] | None = None

    def _filter_s_w_corr_oneshot(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        corr_wl: int,
        thr: float,
    ) -> NDArray[np.float64]:
        k = int(s.shape[0])
        # Energy mode: never re-run _w_corr_keep_indices after the first frame.
        if isinstance(self.truncation_strategy, EnergyThresholdStrategy):
            if not self._energy_w_corr_frozen:
                self._frozen_w_corr_keep_indices = _w_corr_keep_indices(
                    u, s, vh, corr_wl, thr
                )
                self._energy_w_corr_frozen = True
            assert self._frozen_w_corr_keep_indices is not None
            frozen = self._frozen_w_corr_keep_indices
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

        # Fixed rank: recompute when k changes (rare); same k uses cache.
        if (
            self._cached_valid_indices is None
            or self._cached_k is None
            or self._cached_k != k
        ):
            self._cached_valid_indices = _w_corr_keep_indices(u, s, vh, corr_wl, thr)
            self._cached_k = k
        assert self._cached_valid_indices is not None
        return _zero_s_except_indices(s, self._cached_valid_indices)

    def execute(self, data: FloatArray) -> FloatArray:
        """Perform SVD on the block matrix and truncate noise components."""
        a = np.asarray(data, dtype=np.float64, order="C")
        m, n = int(a.shape[0]), int(a.shape[1])
        mn = min(m, n)
        if mn == 0:
            raise ValueError("SVD requires a non-empty matrix.")

        corr_wl: int | None = None
        if self.w_corr_threshold is not None:
            if self.window_length is None:
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )
            corr_wl = int(self.window_length)
            if corr_wl <= 0:
                raise ValueError(
                    "window_length must be a positive int when "
                    "w_corr_threshold is set.",
                )

        if isinstance(self.truncation_strategy, FixedRankStrategy):
            k_req = int(self.truncation_strategy.k)
            if k_req <= 0:
                raise ValueError("Truncation rank must be positive.")
            u, s, vh = _fixed_rank_truncated_factors(a, k_req)
            if self.w_corr_threshold is not None:
                assert corr_wl is not None
                s = self._filter_s_w_corr_oneshot(
                    u,
                    s,
                    vh,
                    corr_wl,
                    float(self.w_corr_threshold),
                )
            reconstructed: FloatArray = _reconstruct_usvh(u, s, vh)
            return reconstructed.astype(np.float64, copy=False)

        u, singular_values, vh = scipy.linalg.svd(a, full_matrices=False)
        k = int(self.truncation_strategy.get_k(singular_values))
        if k <= 0:
            raise ValueError("Truncation rank must be positive.")
        max_rank = min(int(a.shape[0]), int(a.shape[1]), len(singular_values))
        rank = min(k, max_rank)
        u = u[:, :rank]
        s = singular_values[:rank].copy()
        vh = vh[:rank, :]
        if self.w_corr_threshold is not None:
            assert corr_wl is not None
            s = self._filter_s_w_corr_oneshot(
                u,
                s,
                vh,
                corr_wl,
                float(self.w_corr_threshold),
            )
        reconstructed = _reconstruct_usvh(u, s, vh)
        return reconstructed.astype(np.float64, copy=False)


class TruncatedSVDStage(CSVDStage):
    """Fixed-rank convenience: same as ``CSVDStage(FixedRankStrategy(k))``.

    Pass a single ``k`` instead of ``FixedRankStrategy``. Optional
    ``w_corr_threshold`` / ``window_length`` match ``CSVDStage`` (one-shot W-correlation
    cache on repeated same-rank frames).
    """

    def __init__(
        self,
        truncation_rank: int,
        *,
        w_corr_threshold: float | None = None,
        window_length: int | None = None,
    ) -> None:
        if truncation_rank <= 0:
            raise ValueError("Truncation rank must be positive.")
        super().__init__(
            FixedRankStrategy(truncation_rank),
            w_corr_threshold=w_corr_threshold,
            window_length=window_length,
        )
