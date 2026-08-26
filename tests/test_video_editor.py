"""Unit tests for video_editor.apply_portrait_blur — ffmpeg fully mocked.

The function's contract is the ffmpeg invocation it builds; these tests pin
the filter graph, encoder settings, and failure handling without running ffmpeg.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_editor  # noqa: E402
from video_editor import apply_portrait_blur  # noqa: E402


def _run_and_capture(**kwargs):
    """Call apply_portrait_blur with mocked subprocess; return the ffmpeg argv."""
    captured = {}

    def fake_run(cmd, **_):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(video_editor.subprocess, "run", side_effect=fake_run):
            apply_portrait_blur("in.mp4", str(Path(tmp) / "dir" / "out.mp4"), **kwargs)
    return captured["cmd"]


class PortraitBlurCommandTests(unittest.TestCase):
    def test_default_geometry_and_codecs(self):
        cmd = _run_and_capture()
        joined = " ".join(cmd)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase", joined)
        self.assertIn("crop=1080:1920", joined)
        self.assertIn("boxblur=20:8", joined)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", joined)
        self.assertIn("libx264", cmd)
        self.assertIn("aac", cmd)
        # default fps is applied
        self.assertIn("-r", cmd)
        self.assertEqual(cmd[cmd.index("-r") + 1], "30.0")

    def test_custom_dimensions_and_blur(self):
        cmd = _run_and_capture(width=720, height=1280, blur_luma_radius=5, blur_chroma_radius=3)
        joined = " ".join(cmd)
        self.assertIn("scale=720:1280", joined)
        self.assertIn("boxblur=5:3", joined)

    def test_fps_none_omits_rate_flag(self):
        cmd = _run_and_capture(fps=None)
        self.assertNotIn("-r", cmd)

    def test_quality_settings_forwarded(self):
        cmd = _run_and_capture(crf=23, preset="veryfast", audio_bitrate="96k")
        self.assertEqual(cmd[cmd.index("-crf") + 1], "23")
        self.assertEqual(cmd[cmd.index("-preset") + 1], "veryfast")
        self.assertEqual(cmd[cmd.index("-b:a") + 1], "96k")

    def test_audio_stream_is_optional_in_mapping(self):
        cmd = _run_and_capture()
        # '0:a?' keeps clips without audio tracks working
        self.assertIn("0:a?", cmd)

    def test_output_is_last_argument(self):
        cmd = _run_and_capture()
        self.assertTrue(cmd[-1].endswith("out.mp4"))

    def test_nonzero_exit_raises_with_stderr(self):
        fake = SimpleNamespace(returncode=1, stderr="boom")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(video_editor.subprocess, "run", return_value=fake):
                with self.assertRaises(RuntimeError) as ctx:
                    apply_portrait_blur("in.mp4", str(Path(tmp) / "out.mp4"))
        self.assertIn("boom", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
