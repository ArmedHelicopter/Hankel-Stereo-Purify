"""Optional introspection of the linked libsndfile build (local, read-only).

Whitelists in ``audio_formats`` are defensive; actual decode/encode still depends
on the libsndfile binary. Use :func:`libsndfile_build_summary` for diagnostics.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def libsndfile_build_summary() -> str | None:
    """Return a short human-readable summary, or ``None`` if nothing is exposed."""
    parts: list[str] = []
    ver = getattr(sf, "libsndfile_version", None)
    if isinstance(ver, str) and ver:
        parts.append(f"libsndfile {ver}")

    af = getattr(sf, "available_formats", None)
    if callable(af):
        try:
            fmts: Any = af()
            if isinstance(fmts, dict):
                names = sorted(fmts.keys())
            elif isinstance(fmts, (list, tuple, set, frozenset)):
                names = sorted(str(x) for x in fmts)
            else:
                names = [str(fmts)]
            tail = ", ".join(names)
            if len(tail) > 400:
                tail = tail[:400] + "…"
            parts.append(f"available_formats: {tail}")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"libsndfile format introspection failed: {exc}",
                UserWarning,
                stacklevel=1,
            )

    return " | ".join(parts) if parts else None


def can_write_ogg_vorbis() -> bool:
    """Return True if writing a minimal stereo OGG/Vorbis file succeeds.

    Depends on the local libsndfile build (Vorbis encoder). Used by tests to
    ``pytest.skip`` when the codec is unavailable.
    """
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tf:
        path = Path(tf.name)
    try:
        x = np.zeros((8, 2), dtype=np.float64)
        sf.write(
            str(path),
            x,
            8_000,
            format="OGG",
            subtype="VORBIS",
        )
    except (OSError, sf.LibsndfileError, ValueError, RuntimeError):
        return False
    else:
        return True
    finally:
        path.unlink(missing_ok=True)
