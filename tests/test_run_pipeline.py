"""Unit tests for the run_pipeline launcher — no subprocess actually spawned."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline  # noqa: E402
from run_pipeline import (  # noqa: E402
    build_command,
    build_environment,
    main,
    resolve_ca_bundle,
)


class ResolveCaBundleTests(unittest.TestCase):
    def test_existing_valid_ssl_cert_file_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            pem = Path(tmp) / "ca.pem"
            pem.write_text("cert", encoding="utf-8")
            with mock.patch.dict("os.environ", {"SSL_CERT_FILE": str(pem)}):
                self.assertEqual(resolve_ca_bundle(), str(pem))

    def test_nonexistent_ssl_cert_file_falls_back_to_certifi(self):
        with mock.patch.dict("os.environ", {"SSL_CERT_FILE": "/definitely/not/here.pem"}):
            result = resolve_ca_bundle()
        # certifi is a hard dependency, so a real bundle path is expected
        self.assertTrue(result is None or Path(result).is_file())

    def test_missing_certifi_returns_none(self):
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "certifi":
                raise ImportError("no certifi")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict("os.environ", {"SSL_CERT_FILE": ""}), \
             mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertIsNone(resolve_ca_bundle())


class BuildEnvironmentTests(unittest.TestCase):
    def test_sets_unbuffered_output(self):
        env = build_environment({"PATH": "/usr/bin"})
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_preserves_existing_variables(self):
        env = build_environment({"CUSTOM": "value"})
        self.assertEqual(env["CUSTOM"], "value")

    def test_sets_ca_bundle_when_available(self):
        with mock.patch.object(run_pipeline, "resolve_ca_bundle", return_value="/tmp/ca.pem"):
            env = build_environment({})
        self.assertEqual(env["SSL_CERT_FILE"], "/tmp/ca.pem")

    def test_omits_ca_bundle_when_unavailable(self):
        with mock.patch.object(run_pipeline, "resolve_ca_bundle", return_value=None):
            env = build_environment({})
        self.assertNotIn("SSL_CERT_FILE", env)


class BuildCommandTests(unittest.TestCase):
    def test_default_config_is_injected(self):
        cmd = build_command([], python_bin="python")
        self.assertIn("--config", cmd)
        self.assertTrue(cmd[cmd.index("--config") + 1].endswith("live_pipeline_config.json"))

    def test_explicit_config_is_not_duplicated(self):
        cmd = build_command(["--config", "other.json"], python_bin="python")
        self.assertEqual(cmd.count("--config"), 1)
        self.assertEqual(cmd[cmd.index("--config") + 1], "other.json")

    def test_extra_args_are_forwarded(self):
        cmd = build_command(["--dry-run", "--verbose"], python_bin="python")
        self.assertIn("--dry-run", cmd)
        self.assertIn("--verbose", cmd)

    def test_targets_the_pipeline_entrypoint(self):
        cmd = build_command([], python_bin="python")
        self.assertTrue(cmd[1].endswith("live_stream_highlight_pipeline.py"))


class MainTests(unittest.TestCase):
    def test_returncode_is_propagated(self):
        with mock.patch.object(
            run_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=3)
        ):
            self.assertEqual(main([]), 3)

    def test_child_receives_prepared_environment(self):
        captured = {}

        def fake_run(cmd, env=None, **kwargs):
            captured["env"] = env
            return SimpleNamespace(returncode=0)

        with mock.patch.object(run_pipeline.subprocess, "run", side_effect=fake_run):
            main([])
        self.assertEqual(captured["env"]["PYTHONUNBUFFERED"], "1")


if __name__ == "__main__":
    unittest.main()
