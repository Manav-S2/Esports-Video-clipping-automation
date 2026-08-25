"""Unit tests for the demo-based highlight scoring in detect_cs2_highlight."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect_cs2_highlight import (  # noqa: E402
    _extract_json_block,
    _map_score_from_rounds,
    _pick_column,
    _score_demo_highlight,
)


def _features(**overrides):
    base = {
        "kills_total": 0.0,
        "kills_per_minute": 0.0,
        "headshot_ratio": 0.0,
        "multi_kill_burst_score": 0.0,
        "unique_killers": 0.0,
    }
    base.update(overrides)
    return base


class ScoreDemoHighlightTests(unittest.TestCase):
    def test_empty_round_is_not_highlight(self):
        score, is_highlight, confidence = _score_demo_highlight(_features())
        self.assertEqual(score, 0.0)
        self.assertFalse(is_highlight)
        self.assertLess(confidence, 0.5)

    def test_stacked_round_is_highlight(self):
        score, is_highlight, confidence = _score_demo_highlight(
            _features(
                kills_total=10.0,
                kills_per_minute=8.0,
                headshot_ratio=0.6,
                multi_kill_burst_score=6.0,
                unique_killers=5.0,
            )
        )
        self.assertGreaterEqual(score, 7.5)
        self.assertTrue(is_highlight)
        self.assertGreater(confidence, 0.5)

    def test_kills_are_capped(self):
        capped = _score_demo_highlight(_features(kills_total=12.0))[0]
        over = _score_demo_highlight(_features(kills_total=50.0))[0]
        self.assertEqual(capped, over)

    def test_confidence_is_monotonic_in_score(self):
        low = _score_demo_highlight(_features(kills_total=2.0))[2]
        high = _score_demo_highlight(_features(kills_total=12.0, kills_per_minute=10.0))[2]
        self.assertLess(low, high)


class MapScoreTests(unittest.TestCase):
    def test_empty_rounds_return_none(self):
        self.assertIsNone(_map_score_from_rounds([]))

    def test_mean_of_top_three(self):
        rows = [{"round_score": s} for s in (1.0, 9.0, 5.0, 7.0)]
        # top-3 = 9, 7, 5 -> mean 7.0
        self.assertAlmostEqual(_map_score_from_rounds(rows), 7.0)

    def test_fewer_than_three_rounds(self):
        rows = [{"round_score": 4.0}, {"round_score": 6.0}]
        self.assertAlmostEqual(_map_score_from_rounds(rows), 5.0)


class PickColumnTests(unittest.TestCase):
    def test_case_insensitive_match_preserves_original_name(self):
        cols = ["Attacker_Name", "victim_name", "tick"]
        self.assertEqual(_pick_column(cols, ["attacker_name"]), "Attacker_Name")

    def test_candidate_priority_order(self):
        cols = ["user_name", "attacker_name"]
        self.assertEqual(_pick_column(cols, ["attacker_name", "user_name"]), "attacker_name")

    def test_no_match_returns_none(self):
        self.assertIsNone(_pick_column(["a", "b"], ["c"]))


class ExtractJsonBlockTests(unittest.TestCase):
    def test_extracts_embedded_object(self):
        text = "Sure! Here is the JSON:\n{\"is_highlight\": true, \"score\": 8}\nHope that helps."
        obj = _extract_json_block(text)
        self.assertEqual(obj, {"is_highlight": True, "score": 8})

    def test_raises_without_json(self):
        with self.assertRaises(RuntimeError):
            _extract_json_block("no json here")


if __name__ == "__main__":
    unittest.main()
