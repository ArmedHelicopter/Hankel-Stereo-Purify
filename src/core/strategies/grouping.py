"""W-correlation matrix for MSSA component grouping evaluation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_w_correlation_matrix(
    components: npt.NDArray[np.float64],
    window_length: int,
) -> npt.NDArray[np.float64]:
    """Compute the weighted (W-) correlation matrix between MSSA components.

    Each row of ``components`` is one reconstructed component time series of
    length ``sequence_length``. Weights follow the standard SSA diagonal
    weighting with Hankel window length ``window_length``.

    Parameters
    ----------
    components
        Shape ``(num_components, sequence_length)``. Must be 2-dimensional.
    window_length
        Hankel window length ``L`` (positive integer).

    Returns
    -------
    w_corr
        Symmetric matrix of shape ``(num_components, num_components)`` with
        entries in ``[0, 1]`` (invalid entries from zero weighted norms are 0).
    """
    arr = np.asarray(components, dtype=np.float64)
    if arr.ndim == 1:
        raise ValueError(
            "components must be 2D with shape (num_components, sequence_length); "
            "got a 1D array.",
        )
    if arr.ndim != 2:
        raise ValueError(
            "components must be 2D with shape (num_components, sequence_length); "
            f"got {arr.ndim} dimensions.",
        )

    if not isinstance(window_length, int):
        raise TypeError(
            f"window_length must be int, got {type(window_length).__name__}.",
        )
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}.")

    sequence_length = int(arr.shape[1])
    if sequence_length == 0:
        raise ValueError("sequence_length (components.shape[1]) must be positive.")

    idx = np.arange(1, sequence_length + 1)
    weights = np.minimum(idx, np.minimum(window_length, sequence_length - idx + 1))

    weighted_components = arr * weights
    inner_products = weighted_components @ arr.T
    norms = np.sqrt(np.diag(inner_products))

    with np.errstate(divide="ignore", invalid="ignore"):
        w_corr = np.abs(inner_products / np.outer(norms, norms))

    w_corr = np.nan_to_num(w_corr, nan=0.0)
    return np.asarray(w_corr, dtype=np.float64)
