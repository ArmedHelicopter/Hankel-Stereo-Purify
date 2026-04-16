"""Whitelist and soundfile write parameters for supported containers (libsndfile).

Extensions are defensive only; actual decode/encode still depends on the local
libsndfile build. Prefer lossless PCM_24 for file-based hi-fi parity with the
previous FLAC-only path; OGG uses Vorbis (lossy) — document in README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.exceptions import ConfigurationError

# Containers we allow for read (soundfile open).
ALLOWED_INPUT_SUFFIXES: frozenset[str] = frozenset(
    {".flac", ".wav", ".aiff", ".aif", ".ogg"}
)

# Containers we allow for write.
ALLOWED_OUTPUT_SUFFIXES: frozenset[str] = frozenset(
    {".flac", ".wav", ".aiff", ".aif", ".ogg"}
)

# Output path suffix -> (format, subtype) for soundfile.write().
_OUTPUT_SF_KWDS: dict[str, tuple[str, str]] = {
    ".flac": ("FLAC", "PCM_24"),
    ".wav": ("WAV", "PCM_24"),
    ".aiff": ("AIFF", "PCM_24"),
    ".aif": ("AIFF", "PCM_24"),
    ".ogg": ("OGG", "VORBIS"),
}


def _norm_suffix(path: Path) -> str:
    return path.suffix.lower()


def validate_io_paths(input_path: Path | str, output_path: Path | str) -> None:
    """Raise ConfigurationError if input/output extensions are not whitelisted."""
    si = _norm_suffix(Path(input_path))
    so = _norm_suffix(Path(output_path))
    if si not in ALLOWED_INPUT_SUFFIXES:
        raise ConfigurationError(
            f"Unsupported input extension {si!r}; "
            f"allowed: {sorted(ALLOWED_INPUT_SUFFIXES)}"
        )
    if so not in ALLOWED_OUTPUT_SUFFIXES:
        raise ConfigurationError(
            f"Unsupported output extension {so!r}; "
            f"allowed: {sorted(ALLOWED_OUTPUT_SUFFIXES)}"
        )


def soundfile_write_kwargs(output_path: Path | str) -> dict[str, Any]:
    """Keyword arguments for soundfile.write for the given output path."""
    so = _norm_suffix(Path(output_path))
    if so not in _OUTPUT_SF_KWDS:
        raise ConfigurationError(
            f"Unsupported output extension {so!r}; "
            f"allowed: {sorted(ALLOWED_OUTPUT_SUFFIXES)}"
        )
    fmt, subtype = _OUTPUT_SF_KWDS[so]
    return {"format": fmt, "subtype": subtype}
