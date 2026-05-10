"""Compatibility alias for legacy ``src.core.stages.c_svd`` imports.

The implementation was renamed to :mod:`src.core.stages.svd`.  This module is
an alias, not a copy, so monkeypatching attributes such as ``svds`` affects the
real SVD implementation.
"""

from __future__ import annotations

import sys
from importlib import import_module

_svd = import_module("src.core.stages.svd")
sys.modules[__name__] = _svd
setattr(sys.modules[__package__], "c_svd", _svd)
