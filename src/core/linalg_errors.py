"""Explicit LinAlg / ARPACK exception tuples for MSSA (no string-based dispatch)."""

from __future__ import annotations

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    ArpackError,
    ArpackNoConvergence,
)

_lin = (np.linalg.LinAlgError, scipy.linalg.LinAlgError)
MSSA_LINALG_ERRORS: tuple[type[BaseException], ...] = tuple(dict.fromkeys(_lin))

MSSA_ARPACK_ERRORS: tuple[type[BaseException], ...] = (
    ArpackError,
    ArpackNoConvergence,
)

# LinAlg + sparse ARPACK (svds); used by facade numeric mapping
MSSA_NUMERIC_STEP_ERRORS: tuple[type[BaseException], ...] = (
    MSSA_LINALG_ERRORS + MSSA_ARPACK_ERRORS
)
