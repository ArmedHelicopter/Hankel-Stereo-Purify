"""Python wrapper for CUDA Wiener SVD.

Tries to load the compiled CUDA library. If unavailable, falls back to CPU.

Build:
  cd src/cuda && /usr/local/cuda-12.8/bin/nvcc -shared -Xcompiler -fPIC \
    -o libwiener_svd.so wiener_svd.cu -lcublas -lcusolver -lcudart
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
    
    lib_path = Path(__file__).parent / "libwiener_svd.so"
    if not lib_path.exists():
        return False
    
    try:
        _lib = ctypes.CDLL(str(lib_path))
        _lib.wiener_svd_step.restype = ctypes.c_int
        _lib.wiener_svd_step.argtypes = [
            ctypes.c_void_p,  # A_host (double*)
            ctypes.c_int,     # rows
            ctypes.c_int,     # cols
            ctypes.c_double,  # noise_fraction
            ctypes.c_void_p,  # out_host (double*)
        ]
        _available = True
        return True
    except OSError:
        return False


def is_available() -> bool:
    return _load_lib()


def wiener_svd_cuda(A: NDArray[np.float64], noise_fraction: float) -> NDArray[np.float64]:
    """Run Wiener SVD on GPU. Input: rows x cols float64 matrix."""
    if not _load_lib():
        raise RuntimeError("CUDA Wiener SVD library not available. Build with nvcc.")
    
    A = np.ascontiguousarray(A, dtype=np.float64)
    rows, cols = A.shape
    out = np.empty((rows, cols), dtype=np.float64)
    
    ret = _lib.wiener_svd_step(
        A.ctypes.data, rows, cols, noise_fraction, out.ctypes.data
    )
    if ret != 0:
        raise RuntimeError(f"CUDA wiener_svd_step failed with code {ret}")
    
    return out
