"""Optional libsndfile introspection (may be partial on some builds)."""

from src.io.sndfile_capabilities import libsndfile_build_summary


def test_libsndfile_build_summary_returns_str_or_none() -> None:
    s = libsndfile_build_summary()
    assert s is None or isinstance(s, str)
