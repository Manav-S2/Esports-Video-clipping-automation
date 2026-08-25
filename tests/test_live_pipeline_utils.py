"""Unit tests for pure utilities in live_stream_highlight_pipeline.

The module imports numpy/PIL at top level; skip cleanly when the environment
lacks them (the unit suite must run without media dependencies installed).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from live_stream_highlight_pipeline import (
        _highlight_analysis_equipart_times,
        _parse_seek_seconds,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # numpy / PIL missing
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"optional deps missing: {_IMPORT_ERROR}")
class ParseSeekSecondsTests(unittest.TestCase):
    def test_none_and_bool(self):
        self.assertEqual(_parse_seek_seconds(None), 0.0)
        self.assertEqual(_parse_seek_seconds(True), 0.0)

    def test_numbers(self):
        self.assertEqual(_parse_seek_seconds(90), 90.0)
        self.assertEqual(_parse_seek_seconds(12.5), 12.5)
        self.assertEqual(_parse_seek_seconds(-3), 0.0)

    def test_numeric_string(self):
        self.assertEqual(_parse_seek_seconds("42"), 42.0)

    def test_clock_strings(self):
        self.assertEqual(_parse_seek_seconds("1:02:03"), 3723.0)
        self.assertEqual(_parse_seek_seconds("12:30"), 750.0)

    def test_garbage_string(self):
        self.assertEqual(_parse_seek_seconds("not a time"), 0.0)


@unittest.skipIf(_IMPORT_ERROR is not None, f"optional deps missing: {_IMPORT_ERROR}")
class EquipartTimesTests(unittest.TestCase):
    def test_zero_duration(self):
        self.assertEqual(_highlight_analysis_equipart_times(0.0, 4), [0.0] * 4)

    def test_count_and_ordering(self):
        times = _highlight_analysis_equipart_times(120.0, 6)
        self.assertEqual(len(times), 6)
        self.assertEqual(times, sorted(times))

    def test_times_within_duration(self):
        times = _highlight_analysis_equipart_times(90.0, 9)
        self.assertTrue(all(0.0 <= t <= 90.0 for t in times))

    def test_min_one_division(self):
        self.assertEqual(len(_highlight_analysis_equipart_times(30.0, 0)), 1)


if __name__ == "__main__":
    unittest.main()
