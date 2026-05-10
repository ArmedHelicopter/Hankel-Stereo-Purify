"""Python wrapper for CUDA randomized SVD truncation + reconstruction.

Build:
  cd src/cuda && /usr/local/cuda-12.8/bin/nvcc -shared -Xcompiler -fPIC \
    -o libmssa_rand_svd.so mssa_rand_svd.cu -lcudart -lcusolver -lcublas -lcurand
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_lib: ctypes.CDLL | None = None
_available: bool | None = None


def _load_lib() -> ctypes.CDLL | None:
    global _lib, _available
    if _available is not None:
        return _lib

    lib_path = Path(__file__).parent / "libmssa_rand_svd.so"
    if not lib_path.exists():
        _available = False
        return None

    try:
        _lib = ctypes.CDLL(str(lib_path))

        _lib.mssa_rand_svd_init.restype = ctypes.c_int
        _lib.mssa_rand_svd_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        _lib.mssa_rand_svd_cleanup.restype = None
        _lib.mssa_rand_svd_cleanup.argtypes = []

        _lib.mssa_rand_svd_upload.restype = ctypes.c_int
        _lib.mssa_rand_svd_upload.argtypes = [ctypes.c_void_p, ctypes.c_int]

        _lib.mssa_rand_svd_run.restype = ctypes.c_int
        _lib.mssa_rand_svd_run.argtypes = [ctypes.c_int]

        _lib.mssa_rand_svd_download.restype = ctypes.c_int
        _lib.mssa_rand_svd_download.argtypes = [ctypes.c_void_p, ctypes.c_int]

        _available = True
        return _lib
    except OSError:
        _available = False
        return None


def is_available() -> bool:
    return _load_lib() is not None


class CudaRandSvdContext:
    """Persistent GPU context for randomized SVD.

    Parameters
    ----------
    m, n : int
        Column-major matrix dimensions. For a C-contiguous (orig_m, orig_n)
        matrix, pass m=orig_n, n=orig_m.
    rank : int
        Target truncation rank k.
    oversample : int
        Oversampling parameter p (default 16). More = more accurate but slower.
    """

    def __init__(
        self, m: int, n: int, rank: int, oversample: int = 16,
    ) -> None:
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("CUDA randomized SVD library not available.")
        self._m = m
        self._n = n
        self._k = rank
        self._p = oversample
        self._kp = rank + oversample
        rc = lib.mssa_rand_svd_init(m, n, rank, oversample)
        if rc != 0:
            raise RuntimeError(f"mssa_rand_svd_init failed: {rc}")

    def run(
        self,
        matrices: list[NDArray[np.float64]],
    ) -> list[NDArray[np.float64]]:
        """Randomized SVD truncation + reconstruction on GPU.

        Parameters
        ----------
        matrices : list of NDArray, each C-contiguous (orig_m, orig_n)

        Returns
        -------
        list of NDArray, each C-contiguous (orig_m, orig_n)
        """
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("CUDA randomized SVD library not available.")

        N = len(matrices)
        if N == 0:
            return []

        stacked = np.stack(
            [np.ascontiguousarray(A, dtype=np.float64) for A in matrices]
        )

        rc = lib.mssa_rand_svd_upload(stacked.ctypes.data, N)
        if rc != 0:
            raise RuntimeError(f"mssa_rand_svd_upload failed: {rc}")

        rc = lib.mssa_rand_svd_run(N)
        if rc != 0:
            raise RuntimeError(f"mssa_rand_svd_run failed: {rc}")

        output_buf = np.empty(N * self._m * self._n, dtype=np.float64)
        rc = lib.mssa_rand_svd_download(output_buf.ctypes.data, N)
        if rc != 0:
            raise RuntimeError(f"mssa_rand_svd_download failed: {rc}")

        # output is column-major m×n per frame = C-contiguous cu_n×cu_m
        results = output_buf.reshape(N, self._n, self._m)
        return [results[i].copy() for i in range(N)]

    def cleanup(self) -> None:
        lib = _load_lib()
        if lib is not None:
            lib.mssa_rand_svd_cleanup()
