class HankelPurifyError(Exception):
    """Base exception for Hankel-Stereo-Purify failures."""


class AudioIOError(HankelPurifyError):
    """Raised when audio I/O or format validation fails."""


class ConfigurationError(HankelPurifyError):
    """Raised when the purifier is configured with invalid parameters."""
