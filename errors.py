"""Typed error hierarchy for the clip pipeline.

Distinguishes configuration problems (wrong setup, missing tools — fail fast and
loudly) from transient runtime problems (network hiccups, stream stalls — log,
skip the iteration, keep monitoring).
"""

from __future__ import annotations


class ClipPipelineError(Exception):
    """Base class for all pipeline-specific failures."""


class PipelineConfigError(ClipPipelineError):
    """Invalid or missing configuration (bad config JSON, absent credentials)."""


class ToolNotFoundError(PipelineConfigError):
    """A required external binary (ffmpeg, streamlink) is not on PATH."""

    def __init__(self, tool: str, hint: str = ""):
        self.tool = tool
        msg = f"required tool not found on PATH: {tool}"
        if hint:
            msg = f"{msg} ({hint})"
        super().__init__(msg)


class StreamResolutionError(ClipPipelineError):
    """A stream page URL could not be resolved to a playable media URL."""


class RecordingError(ClipPipelineError):
    """Screenshot capture or round recording failed at runtime."""
