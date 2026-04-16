"""Tests for W-correlation grouping utilities."""

import numpy as np
import pytest

from src.core.strategies.grouping import compute_w_correlation_matrix


def test_w_corr_shape_and_diagonal_one() -> None:
    rng = np.random.default_rng(0)
    c, n = 4, 64
    comp = rng.standard_normal((c, n))
    w = compute_w_correlation_matrix(comp, window_length=16)
    assert w.shape == (c, c)
    assert np.allclose(np.diag(w), 1.0)


def test_w_corr_rejects_1d() -> None:
    with pytest.raises(ValueError, match="1D"):
        compute_w_correlation_matrix(np.zeros(5), window_length=3)


def test_w_corr_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_w_correlation_matrix(np.zeros((2, 10)), window_length=0)


def test_w_corr_rejects_non_int_window() -> None:
    with pytest.raises(TypeError, match="int"):
        compute_w_correlation_matrix(np.zeros((2, 10)), window_length=3.0)  # type: ignore[arg-type]


def test_w_corr_rejects_ndim_not_two() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_w_correlation_matrix(np.zeros((2, 3, 4)), window_length=3)


def test_w_corr_rejects_zero_sequence_length() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        compute_w_correlation_matrix(np.zeros((2, 0)), window_length=3)
