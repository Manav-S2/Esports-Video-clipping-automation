"""Unit tests for media_tools — ffmpeg argv construction, no ffmpeg needed."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import media_tools  # noqa: E402
from errors import ToolNotFoundError  # noqa: E402
from media_tools import (  # noqa: E402
    build_ocr_filter_chain,
    build_ocr_optimize_command,
    default_ocr_output_path,
    main,
    resolve_ffmpeg,
    run_ocr_optimize,
)


class OutputPathTests(unittest.TestCase):
    def test_compressed_default_is_mp4(self):
        out = default_ocr_output_path(Path("/clips/round_03.mp4"))
        self.assertEqual(out.name, "round_03.ocr-max.mp4")

    def test_lossless_default_is_mkv(self):
        out = default_ocr_output_path(Path("/clips/round_03.mp4"), lossless=True)
        self.assertEqual(out.name, "round_03.ocr-max.mkv")

    def test_output_stays_beside_input(self):
        out = default_ocr_output_path(Path("/clips/sub/x.mkv"))
        self.assertEqual(out.parent, Path("/clips/sub"))


class FilterChainTests(unittest.TestCase):
    def test_scale_factor_is_applied(self):
        chain = build_ocr_filter_chain(3)
        self.assertIn("scale=iw*3:ih*3", chain)
        self.assertIn("lanczos", chain)

    def test_denoise_and_sharpen_present(self):
        chain = build_ocr_filter_chain()
        self.assertIn("hqdn3d=", chain)
        self.assertIn("unsharp=", chain)

    def test_binarize_appends_threshold_filter(self):
        chain = build_ocr_filter_chain(2, binarize=True, threshold=170)
        self.assertIn("lutyuv=y='if(gte(val,170),255,0)'", chain)

    def test_no_threshold_filter_without_binarize(self):
        self.assertNotIn("lutyuv", build_ocr_filter_chain(2, threshold=170))

    def test_invalid_scale_factor_rejected(self):
        for bad in (0, 5, -1):
            with self.assertRaises(ValueError):
                build_ocr_filter_chain(bad)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            build_ocr_filter_chain(2, binarize=True, threshold=999)


class CommandBuildTests(unittest.TestCase):
    def _cmd(self, **kwargs):
        return build_ocr_optimize_command(Path("in.mp4"), Path("out.mp4"), **kwargs)

    def test_h264_defaults(self):
        cmd = self._cmd()
        self.assertIn("libx264", cmd)
        self.assertEqual(cmd[cmd.index("-crf") + 1], "18")
        self.assertEqual(cmd[cmd.index("-preset") + 1], "slow")
        self.assertIn("+faststart", cmd)

    def test_h265_uses_hvc1_tag(self):
        cmd = self._cmd(codec="h265")
        self.assertIn("libx265", cmd)
        self.assertEqual(cmd[cmd.index("-tag:v") + 1], "hvc1")

    def test_lossless_uses_ffv1_and_drops_crf(self):
        cmd = self._cmd(lossless=True)
        self.assertIn("ffv1", cmd)
        self.assertNotIn("-crf", cmd)
        self.assertNotIn("libx264", cmd)

    def test_audio_and_subtitles_are_dropped(self):
        cmd = self._cmd()
        self.assertIn("-an", cmd)
        self.assertIn("-sn", cmd)

    def test_output_is_last_argument(self):
        self.assertEqual(self._cmd()[-1], "out.mp4")

    def test_invalid_codec_rejected(self):
        with self.assertRaises(ValueError):
            self._cmd(codec="av1")

    def test_invalid_preset_rejected(self):
        with self.assertRaises(ValueError):
            self._cmd(preset="turbo")

    def test_invalid_crf_rejected(self):
        with self.assertRaises(ValueError):
            self._cmd(crf=99)


class ResolveFfmpegTests(unittest.TestCase):
    def test_missing_binary_raises_typed_error(self):
        with mock.patch.object(media_tools.shutil, "which", return_value=None):
            with self.assertRaises(ToolNotFoundError):
                resolve_ffmpeg()

    def test_found_binary_is_returned(self):
        with mock.patch.object(media_tools.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(resolve_ffmpeg(), "/usr/bin/ffmpeg")


class RunOcrOptimizeTests(unittest.TestCase):
    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            run_ocr_optimize(Path("nope-does-not-exist.mp4"))

    def test_success_returns_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "clip.mp4"
            src.write_bytes(b"fake")
            with mock.patch.object(media_tools.shutil, "which", return_value="ffmpeg"), \
                 mock.patch.object(
                     media_tools.subprocess, "run",
                     return_value=SimpleNamespace(returncode=0, stderr=""),
                 ):
                out = run_ocr_optimize(src)
        self.assertEqual(out.name, "clip.ocr-max.mp4")

    def test_ffmpeg_failure_raises_with_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "clip.mp4"
            src.write_bytes(b"fake")
            with mock.patch.object(media_tools.shutil, "which", return_value="ffmpeg"), \
                 mock.patch.object(
                     media_tools.subprocess, "run",
                     return_value=SimpleNamespace(returncode=1, stderr="explode"),
                 ):
                with self.assertRaises(RuntimeError) as ctx:
                    run_ocr_optimize(src)
        self.assertIn("explode", str(ctx.exception))


class CliTests(unittest.TestCase):
    def test_cli_forwards_flags(self):
        captured = {}

        def fake_run(input_video, output_video, **kwargs):
            captured.update(kwargs)
            return Path("out.mp4")

        with mock.patch.object(media_tools, "run_ocr_optimize", side_effect=fake_run):
            rc = main(["ocr-optimize", "--input", "in.mp4", "--binarize", "--threshold", "160", "--codec", "h265"])
        self.assertEqual(rc, 0)
        self.assertTrue(captured["binarize"])
        self.assertEqual(captured["threshold"], 160)
        self.assertEqual(captured["codec"], "h265")

    def test_cli_requires_subcommand(self):
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
