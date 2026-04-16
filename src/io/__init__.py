"""I/O package for audio streaming and file boundary handling."""

from .audio_formats import (
    ALLOWED_INPUT_SUFFIXES,
    ALLOWED_OUTPUT_SUFFIXES,
    validate_io_paths,
)
from .audio_stream import AudioStream, read_audio_metadata
from .sndfile_capabilities import can_write_ogg_vorbis, libsndfile_build_summary
from .stereo_soundfile import open_stereo_soundfile

__all__ = [
    "ALLOWED_INPUT_SUFFIXES",
    "ALLOWED_OUTPUT_SUFFIXES",
    "AudioStream",
    "can_write_ogg_vorbis",
    "libsndfile_build_summary",
    "open_stereo_soundfile",
    "read_audio_metadata",
    "validate_io_paths",
]
