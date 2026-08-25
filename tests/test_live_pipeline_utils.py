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
        _close_unbalanced_curly,
        _extract_json,
        _highlight_analysis_equipart_times,
        _parse_seek_seconds,
        _sanitize_llm_json_blob,
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
class LlmJsonRecoveryTests(unittest.TestCase):
    def test_extract_plain_json(self):
        self.assertEqual(_extract_json('{"a": 1}'), {"a": 1})

    def test_strips_markdown_fences(self):
        text = "```json\n{\"is_highlight\": true}\n```"
        self.assertEqual(_extract_json(text), {"is_highlight": True})

    def test_quotes_bare_keys(self):
        self.assertEqual(_extract_json("{round: 7, valid: true}"), {"round": 7, "valid": True})

    def test_removes_trailing_commas(self):
        self.assertEqual(_extract_json('{"a": [1, 2,], "b": 3,}'), {"a": [1, 2], "b": 3})

    def test_prose_prefix_before_json(self):
        self.assertEqual(_extract_json('The answer is: {"x": 1}'), {"x": 1})

    def test_salvages_truncated_response(self):
        truncated = '{"rounds": [1, 2, 3], "score": 8.1, "reason": "the play was incr'
        obj = _extract_json(truncated)
        self.assertEqual(obj.get("rounds"), [1, 2, 3])

    def test_no_json_raises(self):
        with self.assertRaises(RuntimeError):
            _extract_json("no braces at all")

    def test_close_unbalanced_curly(self):
        self.assertTrue(_close_unbalanced_curly('{"a": {"b": 1').count("}") >= 2)

    def test_sanitize_is_idempotent_on_valid_json(self):
        blob = '{"a": 1, "b": [2, 3]}'
        self.assertEqual(_sanitize_llm_json_blob(blob), blob)


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
