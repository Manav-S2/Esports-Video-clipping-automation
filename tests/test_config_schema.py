"""Unit tests for config_schema — validation of live_pipeline_config.json."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_schema import load_config, validate_config  # noqa: E402
from errors import PipelineConfigError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _valid(**overrides):
    cfg = {
        "stream_url": "https://twitch.tv/example",
        "screenshot_interval_sec": 5,
        "min_round_record_sec": 20,
        "max_round_record_sec": 900,
        "round_detection_min_confidence": 0.45,
        "max_round_jump": 3,
        "api_provider": "rekognition",
        "highlight_api_provider": "vertex",
        "instagram_enabled": False,
        "round_roi_x": 0.3,
        "round_roi_y": 0.0,
        "round_roi_w": 0.36,
        "round_roi_h": 0.24,
    }
    cfg.update(overrides)
    return cfg


class ShippedExampleTests(unittest.TestCase):
    def test_example_config_validates(self):
        # The committed example must always pass, or onboarding is broken.
        example = json.loads(
            (REPO_ROOT / "live_pipeline_config.example.json").read_text(encoding="utf-8")
        )
        report = validate_config(example)
        self.assertTrue(report.ok, f"example config failed validation: {report.errors}")

    def test_example_config_has_no_unknown_keys(self):
        example = json.loads(
            (REPO_ROOT / "live_pipeline_config.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validate_config(example).warnings, [])


class RequiredFieldTests(unittest.TestCase):
    def test_missing_stream_url_is_error(self):
        cfg = _valid()
        del cfg["stream_url"]
        self.assertFalse(validate_config(cfg).ok)

    def test_blank_stream_url_is_error(self):
        self.assertFalse(validate_config(_valid(stream_url="   ")).ok)

    def test_non_object_root_rejected(self):
        self.assertFalse(validate_config([1, 2, 3]).ok)


class TypeCheckTests(unittest.TestCase):
    def test_string_where_number_expected(self):
        report = validate_config(_valid(screenshot_interval_sec="five"))
        self.assertFalse(report.ok)
        self.assertIn("screenshot_interval_sec", report.errors[0])

    def test_bool_is_not_accepted_as_number(self):
        self.assertFalse(validate_config(_valid(max_round_jump=True)).ok)

    def test_float_where_int_expected(self):
        self.assertFalse(validate_config(_valid(max_round_jump=2.5)).ok)

    def test_number_where_bool_expected(self):
        self.assertFalse(validate_config(_valid(instagram_enabled=1)).ok)

    def test_list_entries_must_be_strings(self):
        self.assertFalse(validate_config(_valid(streamlink_extra_args=["ok", 5])).ok)

    def test_list_field_must_be_a_list(self):
        self.assertFalse(validate_config(_valid(vertex_api_keys="key")).ok)


class RangeCheckTests(unittest.TestCase):
    def test_confidence_above_one_rejected(self):
        self.assertFalse(validate_config(_valid(round_detection_min_confidence=1.5)).ok)

    def test_negative_confidence_rejected(self):
        self.assertFalse(validate_config(_valid(round_detection_min_confidence=-0.1)).ok)

    def test_crf_out_of_range_rejected(self):
        self.assertFalse(validate_config(_valid(portrait_blur_crf=99)).ok)

    def test_boundary_values_accepted(self):
        self.assertTrue(validate_config(_valid(round_detection_min_confidence=0.0)).ok)
        self.assertTrue(validate_config(_valid(round_detection_min_confidence=1.0)).ok)


class RoiTests(unittest.TestCase):
    def test_roi_outside_frame_rejected(self):
        self.assertFalse(validate_config(_valid(round_roi_x=1.4)).ok)

    def test_roi_width_overflow_rejected(self):
        report = validate_config(_valid(round_roi_x=0.8, round_roi_w=0.5))
        self.assertFalse(report.ok)
        self.assertTrue(any("frame width" in e for e in report.errors))

    def test_roi_height_overflow_rejected(self):
        report = validate_config(_valid(round_roi_y=0.9, round_roi_h=0.3))
        self.assertTrue(any("frame height" in e for e in report.errors))

    def test_roi_exactly_filling_frame_is_allowed(self):
        self.assertTrue(validate_config(_valid(round_roi_x=0.5, round_roi_w=0.5)).ok)


class EnumTests(unittest.TestCase):
    def test_unknown_provider_rejected(self):
        report = validate_config(_valid(api_provider="magic"))
        self.assertFalse(report.ok)
        self.assertIn("magic", report.errors[0])

    def test_unknown_preset_rejected(self):
        self.assertFalse(validate_config(_valid(portrait_blur_preset="turbo")).ok)

    def test_empty_enum_value_is_allowed(self):
        # Empty means "use the built-in default" throughout the config.
        self.assertTrue(validate_config(_valid(caption_provider="")).ok)


class ConsistencyTests(unittest.TestCase):
    def test_min_greater_than_max_rejected(self):
        report = validate_config(_valid(min_round_record_sec=900, max_round_record_sec=20))
        self.assertFalse(report.ok)
        self.assertTrue(any("greater than" in e for e in report.errors))

    def test_instagram_enabled_without_credentials_rejected(self):
        report = validate_config(_valid(instagram_enabled=True))
        self.assertFalse(report.ok)
        self.assertTrue(any("instagram" in e.lower() for e in report.errors))

    def test_instagram_enabled_with_credentials_accepted(self):
        cfg = _valid(instagram_enabled=True, instagram_username="u", instagram_password="p")
        self.assertTrue(validate_config(cfg).ok)


class WarningTests(unittest.TestCase):
    def test_unknown_key_warns_but_does_not_fail(self):
        report = validate_config(_valid(some_future_option=1))
        self.assertTrue(report.ok)
        self.assertTrue(any("some_future_option" in w for w in report.warnings))


class ErrorAggregationTests(unittest.TestCase):
    def test_all_problems_reported_together(self):
        cfg = _valid(
            screenshot_interval_sec="x",
            round_detection_min_confidence=9,
            api_provider="nope",
        )
        self.assertGreaterEqual(len(validate_config(cfg).errors), 3)


class LoadConfigTests(unittest.TestCase):
    def test_missing_file_raises_typed_error(self):
        with self.assertRaises(PipelineConfigError):
            load_config(Path("no-such-config.json"))

    def test_malformed_json_raises_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(PipelineConfigError) as ctx:
                load_config(p)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_invalid_config_lists_every_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            p.write_text(json.dumps(_valid(stream_url="", max_round_jump=99)), encoding="utf-8")
            with self.assertRaises(PipelineConfigError) as ctx:
                load_config(p)
        message = str(ctx.exception)
        self.assertIn("stream_url", message)
        self.assertIn("max_round_jump", message)

    def test_valid_config_is_returned_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            payload = _valid()
            p.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_config(p), payload)


if __name__ == "__main__":
    unittest.main()
