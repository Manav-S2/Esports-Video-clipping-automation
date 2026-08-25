"""Unit tests for credential resolution in burn_karaoke_captions.

Uses temp directories and env patching only — no real credentials touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from burn_karaoke_captions import (  # noqa: E402
    _default_output,
    _resolve_speech_api_key,
    _speech_api_key_local_paths,
)


class ResolveSpeechApiKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg_path = Path(self._tmp.name) / "config.json"
        self.cfg_path.write_text("{}", encoding="utf-8")

    def _write_local_key(self, key: str) -> None:
        (Path(self._tmp.name) / "speech_api_key.local.json").write_text(
            json.dumps({"speech_api_key": key}), encoding="utf-8"
        )

    def test_env_var_wins_over_everything(self):
        self._write_local_key("from-file")
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": "from-env"}):
            key = _resolve_speech_api_key({"speech_api_key": "from-cfg"}, self.cfg_path)
        self.assertEqual(key, "from-env")

    def test_local_file_beats_config(self):
        self._write_local_key("from-file")
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            key = _resolve_speech_api_key({"speech_api_key": "from-cfg"}, self.cfg_path)
        self.assertEqual(key, "from-file")

    def test_config_direct_key(self):
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            key = _resolve_speech_api_key({"speech_api_key": "direct"}, self.cfg_path)
        self.assertEqual(key, "direct")

    def test_vertex_key_fallback_order(self):
        cfg = {"vertex_api_key": "vk", "gemini_api_key": "gk"}
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            self.assertEqual(_resolve_speech_api_key(cfg, self.cfg_path), "vk")

    def test_key_list_fallbacks_skip_blanks(self):
        cfg = {"vertex_api_keys": ["", "  ", "vk2"]}
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            self.assertEqual(_resolve_speech_api_key(cfg, self.cfg_path), "vk2")

    def test_gemini_key_is_last_resort(self):
        cfg = {"gemini_api_keys": ["gk-list"]}
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            self.assertEqual(_resolve_speech_api_key(cfg, self.cfg_path), "gk-list")

    def test_nothing_configured_returns_empty(self):
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            self.assertEqual(_resolve_speech_api_key({}, self.cfg_path), "")

    def test_corrupt_local_file_is_skipped(self):
        (Path(self._tmp.name) / "speech_api_key.local.json").write_text(
            "not json{", encoding="utf-8"
        )
        with mock.patch.dict("os.environ", {"GOOGLE_SPEECH_API_KEY": ""}):
            key = _resolve_speech_api_key({"speech_api_key": "cfg"}, self.cfg_path)
        self.assertEqual(key, "cfg")


class LocalKeyPathTests(unittest.TestCase):
    def test_paths_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            paths = _speech_api_key_local_paths(cfg)
            resolved = [str(p.resolve()) for p in paths]
            self.assertEqual(len(resolved), len(set(resolved)))
            self.assertTrue(all(p.name == "speech_api_key.local.json" for p in paths))


class DefaultOutputTests(unittest.TestCase):
    def test_appends_karaoke_suffix(self):
        out = _default_output(Path("C:/clips/round_03_portrait.mp4"))
        self.assertNotEqual(out, Path("C:/clips/round_03_portrait.mp4"))
        self.assertEqual(out.suffix, ".mp4")


if __name__ == "__main__":
    unittest.main()
