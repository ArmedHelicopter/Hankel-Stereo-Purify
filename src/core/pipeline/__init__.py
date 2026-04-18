"""Compatibility: re-export :func:`src.core.process_frame.process_frame`.

The former ``MssaFramePipeline`` / Stage dataclasses were removed; callers should
import ``process_frame`` from ``src.core.process_frame`` directly.
"""

from ..process_frame import process_frame

__all__ = ["process_frame"]
