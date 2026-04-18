from enum import Enum
from pathlib import Path


class HankelPurifyError(Exception):
    """Base exception for Hankel-Stereo-Purify failures."""


class AudioIOError(HankelPurifyError):
    """Raised when audio I/O or format validation fails.

    Opening, reading, or writing via soundfile/libsndfile should surface here
    (mapped from ``OSError`` / ``soundfile.LibsndfileError``) for a stable API.
    """


class ConfigurationError(HankelPurifyError):
    """Raised when the purifier is configured with invalid parameters."""


class ProcessingFailureCode(str, Enum):
    """Optional machine-readable tag for :class:`ProcessingError` (logs / metrics)."""

    MSSA_NUMERIC = "mssa_numeric"
    CONSTRAINT_VALUE = "constraint_value"
    UNEXPECTED = "unexpected"


class ProcessingError(HankelPurifyError):
    """Raised when MSSA numerical steps fail or ``process_file`` maps an internal error.

    Typical causes: linear algebra / SVD failures mapped from NumPy/SciPy, constraint
    violations, or unexpected exceptions wrapped at the facade boundary.

    ``code`` is optional and does not change ``str(exc)`` for end users.
    ``origin_exception_type`` is set when this error wraps a concrete exception at
    the facade (``module.QualName``), so callers and logs can distinguish buckets
    (e.g. ``builtins.MemoryError`` vs ``builtins.KeyError``) without parsing messages.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ProcessingFailureCode | None = None,
        origin_exception_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.origin_exception_type = origin_exception_type


def exception_fully_qualified_name(exc: BaseException) -> str:
    """Stable machine-readable ``module.QualName`` for an exception instance."""
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def format_exception_origin(exc: BaseException) -> str:
    """Innermost frame as ``file:lineno:func`` (no full traceback string)."""
    tb = exc.__traceback__
    if tb is None:
        return "unknown"
    while tb.tb_next is not None:
        tb = tb.tb_next
    co = tb.tb_frame.f_code
    return f"{Path(co.co_filename).name}:{tb.tb_lineno}:{co.co_name}"


def validate_w_corr_threshold(value: float | None) -> None:
    """Raise ``ConfigurationError`` if ``value`` is set and outside ``[0.0, 1.0]``.

    W-correlation entries lie in ``[0, 1]``; the CLI threshold is compared
    with ``W[i, 0]`` in the SVD step from :func:`src.core.stages.c_svd.make_svd_step`.
    """
    if value is None:
        return
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ConfigurationError(
            "w_corr_threshold must be in [0.0, 1.0] inclusive "
            "(same range as W-correlation matrix entries)."
        )
