"""Exception helpers and :class:`ProcessingError` metadata."""

from src.core.exceptions import ProcessingError, exception_fully_qualified_name


def test_exception_fully_qualified_name_builtin() -> None:
    assert exception_fully_qualified_name(ValueError("x")) == "builtins.ValueError"


def test_processing_error_optional_origin_type() -> None:
    err = ProcessingError("msg", origin_exception_type="builtins.RuntimeError")
    assert err.origin_exception_type == "builtins.RuntimeError"
