"""Shared array type aliases for the MSSA stack (Phase0 placeholder types)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Float64Array: TypeAlias = NDArray[np.float64]
