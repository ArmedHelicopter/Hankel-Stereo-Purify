"""Exception types mapped to MSSA ``ProcessingError`` for linear-algebra failures."""

from __future__ import annotations

import numpy as np

# NumPy / SciPy share ``LinAlgError`` in typical builds; ARPACK adds its own types.
_types: list[type[BaseException]] = [np.linalg.LinAlgError]
try:
    from scipy.sparse.linalg import (  # type: ignore[import-untyped]
        ArpackError,
        ArpackNoConvergence,
    )

    for _t in (ArpackError, ArpackNoConvergence):
        if _t not in _types:
            _types.append(_t)
except ImportError:
    pass

MSSA_LINALG_ERRORS: tuple[type[BaseException], ...] = tuple(_types)
