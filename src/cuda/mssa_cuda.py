"""Python wrapper for CUDA batched SVD truncation + reconstruction.

Supports two usage patterns:
  1. One-shot:  mssa_svd_batch(matrices, rank) — simple, allocates per call
  2. Persistent: CudaSvdContext(m, n) → .run() × N → .cleanup()

Build:
  cd src/cuda && /usr/local/cuda-12.8/bin/nvcc -shared -Xcompiler -fPIC \
    -o libmssa_svd.so mssa_svd.cu -lcudart -lcusolver
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_lib: ctypes.CDLL | None = None
_available: bool | None = None


def _load_lib() -> ctypes.CDLL | None:
    global _lib, _available
    if _available is not None:
        return _lib

    lib_path = Path(__file__).parent / "libmssa_svd.so"
    if not lib_path.exists():
        _available = False
        return None

    try:
        _lib = ctypes.CDLL(str(lib_path))

        _lib.mssa_svd_init.restype = ctypes.c_int
        _lib.mssa_svd_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        _lib.mssa_svd_cleanup.restype = None
        _lib.mssa_svd_cleanup.argtypes = []

        _lib.mssa_svd_upload.restype = ctypes.c_int
        _lib.mssa_svd_upload.argtypes = [ctypes.c_void_p, ctypes.c_int]

        _lib.mssa_svd_run.restype = ctypes.c_int
        _lib.mssa_svd_run.argtypes = [ctypes.c_int, ctypes.c_int]

        _lib.mssa_svd_download.restype = ctypes.c_int
        _lib.mssa_svd_download.argtypes = [ctypes.c_void_p, ctypes.c_int]

        _available = True
        return _lib
    except OSError:
        _available = False
        return None


def is_available() -> bool:
    return _load_lib() is not None


class CudaSvdContext:
    """Persistent GPU context for batched SVD — avoids per-frame alloc/free.

    Parameters
    ----------
    m, n : int
        Column-major matrix dimensions. For a C-contiguous (orig_m, orig_n)
        matrix, pass m=orig_n, n=orig_m (swap because row-major (m,n) in
        memory = column-major (n,m) for cuSOLVER).
    max_batch : int
        Maximum batch size for pre-allocation.
    """

    def __init__(self, m: int, n: int, max_batch: int = 512) -> None:
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("CUDA SVD library not available.")
        self._m = m  # cu_m (column-major rows)
        self._n = n  # cu_n (column-major cols)
        self._mn = min(m, n)
        rc = lib.mssa_svd_init(m, n, max_batch)
        if rc != 0:
            raise RuntimeError(f"mssa_svd_init({m}, {n}, {max_batch}) failed: {rc}")

    def run(
        self,
        matrices: list[NDArray[np.float64]],
        rank: int,
    ) -> list[NDArray[np.float64]]:
        """SVD truncation + reconstruction on GPU.

        Parameters
        ----------
        matrices : list of NDArray, each C-contiguous (orig_m, orig_n)
        rank : int

        Returns
        -------
        list of NDArray, each C-contiguous (orig_m, orig_n)
        """
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("CUDA SVD library not available.")

        N = len(matrices)
        if N == 0:
            return []

        k = min(rank, self._mn)

        stacked = np.stack(
            [np.ascontiguousarray(A, dtype=np.float64) for A in matrices]
        )
        output_buf = np.empty(N * self._m * self._n, dtype=np.float64)

        rc = lib.mssa_svd_upload(stacked.ctypes.data, N)
        if rc != 0:
            raise RuntimeError(f"mssa_svd_upload failed: {rc}")

        rc = lib.mssa_svd_run(N, k)
        if rc != 0:
            raise RuntimeError(f"mssa_svd_run failed: {rc}")

        rc = lib.mssa_svd_download(output_buf.ctypes.data, N)
        if rc != 0:
            raise RuntimeError(f"mssa_svd_download failed: {rc}")

        # output_buf is column-major cu_m×cu_n per frame = C-contiguous cu_n×cu_m
        # We want C-contiguous orig_m×orig_n = cu_n×cu_m ✓ (same layout!)
        # cu_n = orig_m, cu_m = orig_n
        results = output_buf.reshape(N, self._n, self._m)  # (N, orig_m, orig_n)
        return [results[i].copy() for i in range(N)]

    def cleanup(self) -> None:
        lib = _load_lib()
        if lib is not None:
            lib.mssa_svd_cleanup()
