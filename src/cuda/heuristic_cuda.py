"""Python wrapper for CUDA heuristic SVD.

Tries to load the compiled CUDA library. If unavailable, falls back to CPU.

Build:
  cd src/cuda && /usr/local/cuda-12.8/bin/nvcc -shared -Xcompiler -fPIC \
    -o libheuristic_svd.so heuristic_svd.cu -lcudart -lcufft
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_lib = None
_available = False


def _load_lib() -> bool:
    global _lib, _available
    if _lib is not None:
        return _available
    
    lib_path = Path(__file__).parent / "libheuristic_svd.so"
    if not lib_path.exists():
        return False
    
    try:
        _lib = ctypes.CDLL(str(lib_path))
        _lib.heuristic_svd_step.restype = ctypes.c_int
        _lib.heuristic_svd_step.argtypes = [
            ctypes.c_void_p,  # A_host (double*)
            ctypes.c_int,     # rows
            ctypes.c_int,     # cols
            ctypes.c_double,  # sfm_weight
            ctypes.c_double,  # energy_weight
            ctypes.c_double,  # temporal_weight
            ctypes.c_double,  # sfm_threshold_low
            ctypes.c_double,  # sfm_threshold_high
            ctypes.c_double,  # energy_threshold
            ctypes.c_double,  # temporal_threshold
            ctypes.c_void_p,  # weights_host (double*)
        ]
        _available = True
        return True
    except OSError:
        return False


def is_available() -> bool:
    return _load_lib()


def heuristic_svd_cuda(
    A: NDArray[np.float64],
    sfm_weight: float,
    energy_weight: float,
    temporal_weight: float,
    sfm_threshold_low: float,
    sfm_threshold_high: float,
    energy_threshold: float,
    temporal_threshold: float,
) -> NDArray[np.float64]:
    """Run heuristic SVD on GPU. Input: rows x cols float64 matrix.
    
    Returns weights for each SVD component.
    """
    if not _load_lib():
        raise RuntimeError("CUDA heuristic SVD library not available. Build with nvcc.")
    
    A = np.ascontiguousarray(A, dtype=np.float64)
    rows, cols = A.shape
    
    # Determine number of components (min of rows and cols)
    k = min(rows, cols)
    weights = np.empty(k, dtype=np.float64)
    
    ret = _lib.heuristic_svd_step(
        A.ctypes.data,
        rows, cols,
        sfm_weight,
        energy_weight,
        temporal_weight,
        sfm_threshold_low,
        sfm_threshold_high,
        energy_threshold,
        temporal_threshold,
        weights.ctypes.data
    )
    
    if ret != 0:
        raise RuntimeError(f"CUDA heuristic_svd_step failed with code {ret}")
    
    return weights
