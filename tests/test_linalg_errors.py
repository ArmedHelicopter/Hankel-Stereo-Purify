"""``src.core.linalg_errors`` contract for MSSA numerical exception mapping."""

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    ArpackError,
    ArpackNoConvergence,
)

from src.core.linalg_errors import (
    MSSA_ARPACK_ERRORS,
    MSSA_LINALG_ERRORS,
    MSSA_NUMERIC_STEP_ERRORS,
)


def test_mssa_linalg_tuple_includes_numpy_and_scipy_linalg() -> None:
    assert np.linalg.LinAlgError in MSSA_LINALG_ERRORS
    assert scipy.linalg.LinAlgError in MSSA_LINALG_ERRORS


def test_mssa_numeric_step_includes_arpack_when_available() -> None:
    assert ArpackError in MSSA_ARPACK_ERRORS
    assert ArpackNoConvergence in MSSA_ARPACK_ERRORS
    assert MSSA_NUMERIC_STEP_ERRORS == MSSA_LINALG_ERRORS + MSSA_ARPACK_ERRORS
