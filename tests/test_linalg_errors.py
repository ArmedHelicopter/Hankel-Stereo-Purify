"""``src.core.linalg_errors`` contract for MSSA numerical exception mapping."""

import numpy as np

from src.core.linalg_errors import MSSA_LINALG_ERRORS


def test_mssa_linalg_tuple_includes_numpy_and_arpack() -> None:
    assert np.linalg.LinAlgError in MSSA_LINALG_ERRORS
    from scipy.sparse.linalg import ArpackError  # type: ignore[import-untyped]

    assert ArpackError in MSSA_LINALG_ERRORS
