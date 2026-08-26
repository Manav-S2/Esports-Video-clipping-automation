"""Unit tests for the vertical (9:16) edit in media_tools — ffmpeg mocked."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import media_tools  # noqa: E402
from media_tools import (  # noqa: E402
    build_vertical_caption_command,
    build_vertical_layout_filter,
    escape_subtitles_path,
    main,
    run_vertical_caption_edit,
)


class EscapeSubtitlesPathTests(unittest.TestCase):
    def test_backslashes_become_forward_slashes(self):
        out = escape_subtitles_path(Path("captions.srt"))
        self.assertNotIn("\\\\", out)

    def test_drive_colon_is_escaped(self):
        # Any absolute path resolved on Windows carries a drive colon that the
        # ffmpeg filter parser would otherwise treat as an option separator.
        out = escape_subtitles_path(Path("captions.srt"))
        for i, ch in enumerate(out):
            if ch == ":":
                self.assertEqual(out[i - 1], "\\", "every colon must be backslash-escaped")


class LayoutFilterTests(unittest.TestCase):
    def test_default_canvas_and_graph_shape(self):
        f = build_vertical_layout_filter()
        self.assertIn("scale=1080:1920", f)
        self.assertIn("boxblur=26:26", f)
        self.assertIn("overlay=(W-w)/2:(H-h)/2[vbase]", f)
        self.assertIn("split=2", f)

    def test_custom_canvas_and_blur(self):
        f = build_vertical_layout_filter(720, 1280, blur_strength=10, background_darken=0.3)
        self.assertIn("scale=720:1280", f)
        self.assertIn("boxblur=10:10", f)
        self.assertIn("eq=brightness=-0.3", f)

    def test_out_of_range_values_rejected(self):
        with self.assertRaises(ValueError):
            build_vertical_layout_filter(100, 1920)
        with self.assertRaises(ValueError):
            build_vertical_layout_filter(1080, 100)
        with self.assertRaises(ValueError):
            build_vertical_layout_filter(blur_strength=200)
        with self.assertRaises(ValueError):
            build_vertical_layout_filter(background_darken=5.0)


class VerticalCommandTests(unittest.TestCase):
    def _cmd(self, **kwargs):
        return build_vertical_caption_command(Path("in.mp4"), Path("out.mp4"), **kwargs)

    def test_without_captions_uses_null_passthrough(self):
        cmd = self._cmd()
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[vbase]null[vout]", fc)
        self.assertNotIn("subtitles=", fc)

    def test_with_captions_adds_subtitles_filter_and_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt = Path(tmp) / "c.srt"
            srt.write_text("1\n", encoding="utf-8")
            cmd = self._cmd(captions_file=srt)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("subtitles=", fc)
        self.assertIn("Alignment=8", fc)  # top-centered
        self.assertIn("Bold=1", fc)

    def test_caption_font_and_size_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt = Path(tmp) / "c.srt"
            srt.write_text("1\n", encoding="utf-8")
            cmd = self._cmd(captions_file=srt, caption_font="Impact", caption_font_size=90)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("FontName=Impact", fc)
        self.assertIn("FontSize=90", fc)

    def test_audio_copied_and_optional(self):
        cmd = self._cmd()
        self.assertIn("0:a?", cmd)
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")

    def test_invalid_font_size_rejected(self):
        with self.assertRaises(ValueError):
            self._cmd(caption_font_size=500)

    def test_output_is_last_argument(self):
        self.assertEqual(self._cmd()[-1], "out.mp4")


class RunVerticalEditTests(unittest.TestCase):
    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            run_vertical_caption_edit(Path("nope.mp4"))

    def test_missing_captions_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "clip.mp4"
            src.write_bytes(b"fake")
            with self.assertRaises(FileNotFoundError):
                run_vertical_caption_edit(src, captions_file=Path(tmp) / "absent.srt")

    def test_default_output_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "clip.mp4"
            src.write_bytes(b"fake")
            with mock.patch.object(media_tools.shutil, "which", return_value="ffmpeg"), \
                 mock.patch.object(
                     media_tools.subprocess, "run",
                     return_value=SimpleNamespace(returncode=0, stderr=""),
                 ):
                out = run_vertical_caption_edit(src)
        self.assertEqual(out.name, "clip.vertical.mp4")

    def test_failure_propagates_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "clip.mp4"
            src.write_bytes(b"fake")
            with mock.patch.object(media_tools.shutil, "which", return_value="ffmpeg"), \
                 mock.patch.object(
                     media_tools.subprocess, "run",
                     return_value=SimpleNamespace(returncode=1, stderr="nope"),
                 ):
                with self.assertRaises(RuntimeError) as ctx:
                    run_vertical_caption_edit(src)
        self.assertIn("nope", str(ctx.exception))


class VerticalCliTests(unittest.TestCase):
    def test_cli_forwards_flags(self):
        captured = {}

        def fake_run(input_video, output_video, **kwargs):
            captured.update(kwargs)
            return Path("out.mp4")

        with mock.patch.object(media_tools, "run_vertical_caption_edit", side_effect=fake_run):
            rc = main(["vertical-edit", "--input", "in.mp4", "--canvas-width", "720", "--blur-strength", "40"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["canvas_width"], 720)
        self.assertEqual(captured["blur_strength"], 40)


if __name__ == "__main__":
    unittest.main()
