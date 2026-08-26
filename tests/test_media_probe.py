"""Unit tests for pipeline.media_probe — subprocess fully mocked."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import media_probe  # noqa: E402
from pipeline.media_probe import (  # noqa: E402
    clip_duration_for_analysis,
    ffmpeg_demuxer_duration_sec,
    ffprobe_duration_sec,
    resolve_ffprobe_bin,
    run_ffmpeg,
)

CLIP = Path("clip.mp4")


class RunFfmpegTests(unittest.TestCase):
    def test_success_is_silent(self):
        with mock.patch.object(
            media_probe.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            self.assertIsNone(run_ffmpeg(["ffmpeg", "-i", "a.mp4"]))

    def test_failure_includes_command_and_stderr(self):
        with mock.patch.object(
            media_probe.subprocess, "run",
            return_value=SimpleNamespace(returncode=1, stdout="out", stderr="bad codec"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                run_ffmpeg(["ffmpeg", "-i", "a.mp4"])
        message = str(ctx.exception)
        self.assertIn("bad codec", message)
        self.assertIn("ffmpeg -i a.mp4", message)


class ResolveFfprobeTests(unittest.TestCase):
    def test_path_lookup_wins(self):
        with mock.patch.object(media_probe.shutil, "which", return_value="/usr/bin/ffprobe"):
            self.assertEqual(resolve_ffprobe_bin(None), "/usr/bin/ffprobe")

    def test_falls_back_to_sibling_of_ffmpeg(self):
        with mock.patch.object(media_probe.shutil, "which", return_value=None), \
             mock.patch.object(Path, "is_file", return_value=True):
            result = resolve_ffprobe_bin("/opt/tools/ffmpeg")
        self.assertIsNotNone(result)
        self.assertIn("ffprobe", result)

    def test_none_when_nothing_found(self):
        with mock.patch.object(media_probe.shutil, "which", return_value=None), \
             mock.patch.object(Path, "is_file", return_value=False):
            self.assertIsNone(resolve_ffprobe_bin("/opt/tools/ffmpeg"))

    def test_none_when_no_hint_and_not_on_path(self):
        with mock.patch.object(media_probe.shutil, "which", return_value=None):
            self.assertIsNone(resolve_ffprobe_bin(None))


class FfprobeDurationTests(unittest.TestCase):
    def _probe(self, **run_kwargs):
        with mock.patch.object(media_probe, "resolve_ffprobe_bin", return_value="ffprobe"), \
             mock.patch.object(media_probe.subprocess, "run", **run_kwargs):
            return ffprobe_duration_sec(CLIP, "ffmpeg")

    def test_parses_duration(self):
        result = self._probe(return_value=SimpleNamespace(returncode=0, stdout="123.45\n"))
        self.assertAlmostEqual(result, 123.45)

    def test_missing_ffprobe_returns_none(self):
        with mock.patch.object(media_probe, "resolve_ffprobe_bin", return_value=None):
            self.assertIsNone(ffprobe_duration_sec(CLIP, "ffmpeg"))

    def test_nonzero_exit_returns_none(self):
        self.assertIsNone(self._probe(return_value=SimpleNamespace(returncode=1, stdout="")))

    def test_empty_output_returns_none(self):
        self.assertIsNone(self._probe(return_value=SimpleNamespace(returncode=0, stdout="  \n")))

    def test_unparsable_output_returns_none(self):
        self.assertIsNone(self._probe(return_value=SimpleNamespace(returncode=0, stdout="N/A\n")))

    def test_zero_and_negative_durations_rejected(self):
        self.assertIsNone(self._probe(return_value=SimpleNamespace(returncode=0, stdout="0\n")))
        self.assertIsNone(self._probe(return_value=SimpleNamespace(returncode=0, stdout="-5\n")))

    def test_nan_rejected(self):
        self.assertIsNone(self._probe(return_value=SimpleNamespace(returncode=0, stdout="nan\n")))

    def test_timeout_returns_none(self):
        self.assertIsNone(self._probe(side_effect=subprocess.TimeoutExpired("ffprobe", 90)))

    def test_oserror_returns_none(self):
        self.assertIsNone(self._probe(side_effect=OSError("boom")))


class FfmpegDemuxerDurationTests(unittest.TestCase):
    def _parse(self, stderr: str):
        with mock.patch.object(
            media_probe.subprocess, "run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr=stderr),
        ):
            return ffmpeg_demuxer_duration_sec(CLIP, "ffmpeg")

    def test_parses_header_duration(self):
        # ffmpeg writes its header banner to stderr and exits non-zero with no output file.
        self.assertAlmostEqual(self._parse("  Duration: 00:02:03.50, start: 0.0"), 123.5)

    def test_hours_are_included(self):
        self.assertAlmostEqual(self._parse("Duration: 01:00:00.00,"), 3600.0)

    def test_fractional_precision_varies(self):
        self.assertAlmostEqual(self._parse("Duration: 00:00:01.5,"), 1.5)
        self.assertAlmostEqual(self._parse("Duration: 00:00:01.250,"), 1.25)

    def test_no_duration_line_returns_none(self):
        self.assertIsNone(self._parse("Invalid data found when processing input"))

    def test_zero_duration_returns_none(self):
        self.assertIsNone(self._parse("Duration: 00:00:00.00,"))

    def test_missing_ffmpeg_bin_returns_none(self):
        self.assertIsNone(ffmpeg_demuxer_duration_sec(CLIP, None))

    def test_timeout_returns_none(self):
        with mock.patch.object(
            media_probe.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 120),
        ):
            self.assertIsNone(ffmpeg_demuxer_duration_sec(CLIP, "ffmpeg"))


class ClipDurationFallbackTests(unittest.TestCase):
    def test_prefers_ffprobe(self):
        with mock.patch.object(media_probe, "ffprobe_duration_sec", return_value=42.0), \
             mock.patch.object(media_probe, "ffmpeg_demuxer_duration_sec") as demuxer:
            self.assertEqual(clip_duration_for_analysis(CLIP, "ffmpeg"), 42.0)
        demuxer.assert_not_called()

    def test_falls_back_to_demuxer(self):
        with mock.patch.object(media_probe, "ffprobe_duration_sec", return_value=None), \
             mock.patch.object(media_probe, "ffmpeg_demuxer_duration_sec", return_value=7.5):
            self.assertEqual(clip_duration_for_analysis(CLIP, "ffmpeg"), 7.5)

    def test_none_when_both_fail(self):
        with mock.patch.object(media_probe, "ffprobe_duration_sec", return_value=None), \
             mock.patch.object(media_probe, "ffmpeg_demuxer_duration_sec", return_value=None):
            self.assertIsNone(clip_duration_for_analysis(CLIP, "ffmpeg"))


if __name__ == "__main__":
    unittest.main()
