"""Unit tests for the pure caption/subtitle helpers in speech_google_captions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_google_captions import (  # noqa: E402
    _dur_to_seconds,
    _srt_ts,
    _wrap_lines,
    sanitize_srt_text,
)


class SrtTimestampTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(_srt_ts(0.0), "00:00:00,000")

    def test_ms_precision_and_units(self):
        self.assertEqual(_srt_ts(3723.5), "01:02:03,500")

    def test_negative_clamps_to_zero(self):
        self.assertEqual(_srt_ts(-4.2), "00:00:00,000")

    def test_nan_clamps_to_zero(self):
        self.assertEqual(_srt_ts(float("nan")), "00:00:00,000")


class SanitizeSrtTextTests(unittest.TestCase):
    def test_strips_control_chars_keeps_newlines(self):
        self.assertEqual(sanitize_srt_text("A\x00B\x07C\nD"), "ABC\nD")

    def test_plain_text_unchanged(self):
        self.assertEqual(sanitize_srt_text("NICE SHOT!"), "NICE SHOT!")


class WrapLinesTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(_wrap_lines("", 20, 2), [])

    def test_respects_max_chars(self):
        lines = _wrap_lines("one two three four five six", 10, 10)
        self.assertTrue(all(len(line) <= 10 for line in lines))
        self.assertEqual(" ".join(lines), "one two three four five six")

    def test_respects_max_lines(self):
        lines = _wrap_lines("a b c d e f g h i j", 3, 2)
        self.assertLessEqual(len(lines), 2)

    def test_overlong_word_is_truncated(self):
        lines = _wrap_lines("supercalifragilistic", 5, 2)
        self.assertEqual(lines[0], "super")


class DurToSecondsTests(unittest.TestCase):
    def test_none_is_zero(self):
        self.assertEqual(_dur_to_seconds(None), 0.0)

    def test_protobuf_like_duration(self):
        class D:
            seconds = 2
            nanos = 500_000_000

        self.assertAlmostEqual(_dur_to_seconds(D()), 2.5)

    def test_object_without_fields(self):
        self.assertEqual(_dur_to_seconds(object()), 0.0)


if __name__ == "__main__":
    unittest.main()
