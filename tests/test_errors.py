"""Unit tests for the typed error hierarchy in errors.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from errors import (  # noqa: E402
    ClipPipelineError,
    PipelineConfigError,
    RecordingError,
    StreamResolutionError,
    ToolNotFoundError,
)


class HierarchyTests(unittest.TestCase):
    def test_all_derive_from_base(self):
        for exc_type in (PipelineConfigError, ToolNotFoundError, StreamResolutionError, RecordingError):
            self.assertTrue(issubclass(exc_type, ClipPipelineError))

    def test_tool_not_found_is_config_error(self):
        # Missing binaries are a setup problem, so callers catching config
        # errors (fail-fast paths) must also see missing tools.
        self.assertTrue(issubclass(ToolNotFoundError, PipelineConfigError))

    def test_runtime_errors_are_not_config_errors(self):
        self.assertFalse(issubclass(StreamResolutionError, PipelineConfigError))
        self.assertFalse(issubclass(RecordingError, PipelineConfigError))


class ToolNotFoundMessageTests(unittest.TestCase):
    def test_message_names_the_tool(self):
        exc = ToolNotFoundError("ffmpeg")
        self.assertIn("ffmpeg", str(exc))
        self.assertEqual(exc.tool, "ffmpeg")

    def test_hint_is_appended(self):
        exc = ToolNotFoundError("streamlink", hint="pip install streamlink")
        self.assertIn("pip install streamlink", str(exc))


if __name__ == "__main__":
    unittest.main()
