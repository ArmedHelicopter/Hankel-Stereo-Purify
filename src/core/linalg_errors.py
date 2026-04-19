"""Linear-algebra related errors (Phase0 skeleton)."""

from src.core.exceptions import HankelStereoPurifyError


class MssaLinearAlgebraError(HankelStereoPurifyError):
    """Raised when SVD or related LA steps fail in a non-recoverable way."""
