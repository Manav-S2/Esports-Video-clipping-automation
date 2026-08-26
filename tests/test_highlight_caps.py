"""Unit tests for the per-map highlight cap in detect_cs2_highlight."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect_cs2_highlight import _apply_per_map_highlight_cap  # noqa: E402


def _row(demo: str, rnd: int, score: float) -> dict[str, object]:
    return {"demo": demo, "round": rnd, "round_score": score}


class PerMapHighlightCapTests(unittest.TestCase):
    def test_zero_cap_marks_nothing(self):
        rows = [_row("de_mirage", 1, 9.0), _row("de_mirage", 2, 8.0)]
        updated, cutoffs, notes = _apply_per_map_highlight_cap(rows, 0)
        self.assertTrue(all(not r["is_round_highlight"] for r in updated))
        self.assertEqual(cutoffs, {})
        self.assertTrue(notes)

    def test_top_n_per_map_are_marked(self):
        rows = [
            _row("de_mirage", 1, 3.0),
            _row("de_mirage", 2, 9.0),
            _row("de_mirage", 3, 7.0),
            _row("de_mirage", 4, 5.0),
        ]
        updated, cutoffs, _ = _apply_per_map_highlight_cap(rows, 2)
        marked = {r["round"] for r in updated if r["is_round_highlight"]}
        self.assertEqual(marked, {2, 3})
        self.assertEqual(cutoffs["de_mirage"], 7.0)

    def test_cap_is_per_demo_not_global(self):
        rows = [
            _row("map_a", 1, 9.0),
            _row("map_a", 2, 8.0),
            _row("map_b", 1, 2.0),
            _row("map_b", 2, 1.0),
        ]
        updated, cutoffs, _ = _apply_per_map_highlight_cap(rows, 1)
        marked = {(r["demo"], r["round"]) for r in updated if r["is_round_highlight"]}
        # map_b's best round qualifies even though it scores below map_a's worst
        self.assertEqual(marked, {("map_a", 1), ("map_b", 1)})
        self.assertEqual(set(cutoffs), {"map_a", "map_b"})

    def test_cap_larger_than_round_count_marks_all(self):
        rows = [_row("de_nuke", 1, 4.0), _row("de_nuke", 2, 6.0)]
        updated, cutoffs, _ = _apply_per_map_highlight_cap(rows, 10)
        self.assertTrue(all(r["is_round_highlight"] for r in updated))
        self.assertEqual(cutoffs["de_nuke"], 4.0)  # cutoff falls to the weakest round

    def test_output_order_is_demo_then_round(self):
        rows = [
            _row("map_b", 2, 1.0),
            _row("map_a", 2, 9.0),
            _row("map_b", 1, 2.0),
            _row("map_a", 1, 3.0),
        ]
        updated, _, _ = _apply_per_map_highlight_cap(rows, 1)
        order = [(r["demo"], r["round"]) for r in updated]
        self.assertEqual(order, [("map_a", 1), ("map_a", 2), ("map_b", 1), ("map_b", 2)])

    def test_input_rows_are_not_mutated(self):
        rows = [_row("de_inferno", 1, 5.0)]
        _apply_per_map_highlight_cap(rows, 1)
        self.assertNotIn("is_round_highlight", rows[0])


if __name__ == "__main__":
    unittest.main()
