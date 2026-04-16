class HankelPurifyError(Exception):
    """Base exception for Hankel-Stereo-Purify failures."""


class AudioIOError(HankelPurifyError):
    """Raised when audio I/O or format validation fails.

    Opening, reading, or writing via soundfile/libsndfile should surface here
    (mapped from ``OSError`` / ``soundfile.LibsndfileError``) for a stable API.
    """


class ConfigurationError(HankelPurifyError):
    """Raised when the purifier is configured with invalid parameters."""


class ProcessingError(HankelPurifyError):
    """Raised when MSSA numerical steps fail or ``process_file`` maps an internal error.

    Typical causes: linear algebra / SVD failures mapped from NumPy/SciPy, constraint
    violations, or unexpected exceptions wrapped at the facade boundary.
    """


def validate_w_corr_threshold(value: float | None) -> None:
    """Raise ``ConfigurationError`` if ``value`` is set and outside ``[0.0, 1.0]``.

    W-correlation entries lie in ``[0, 1]``; the CLI / builder threshold is compared
    with ``W[i, 0]`` in :class:`src.core.stages.c_svd.CSVDStage`.
    """
    if value is None:
        return
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ConfigurationError(
            "w_corr_threshold must be in [0.0, 1.0] inclusive "
            "(same range as W-correlation matrix entries)."
        )
