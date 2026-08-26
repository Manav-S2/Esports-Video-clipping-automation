"""Unit tests for StreamRoundRecorder — subprocess and Gemini fully mocked.

No network, no ffmpeg, no streamlink: subprocess/shutil.which and the genai
client are patched so the URL resolution, screenshot flow, and round-number
parsing logic are exercised in isolation.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stream_recorder  # noqa: E402
from errors import StreamResolutionError, ToolNotFoundError  # noqa: E402
from stream_recorder import StreamRoundRecorder  # noqa: E402


def _recorder(tmp: str, url: str = "https://cdn.example/stream.m3u8") -> StreamRoundRecorder:
    return StreamRoundRecorder(
        stream_url=url,
        gemini_api_key="",  # no client — tests inject fakes explicitly
        gemini_model="gemini-test",
        output_root=Path(tmp),
    )


class ResolveStreamInputTests(unittest.TestCase):
    def test_direct_url_passes_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)
            self.assertEqual(rec._resolve_stream_input(), "https://cdn.example/stream.m3u8")

    def test_twitch_url_resolved_via_streamlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp, url="https://twitch.tv/somechannel")
            with mock.patch.object(stream_recorder.shutil, "which", return_value="streamlink"), \
                 mock.patch.object(
                     stream_recorder.subprocess, "run",
                     return_value=SimpleNamespace(stdout="https://real.m3u8\n"),
                 ) as run:
                self.assertEqual(rec._resolve_stream_input(), "https://real.m3u8")
            self.assertIn("--stream-url", run.call_args[0][0])

    def test_twitch_url_without_streamlink_raises_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp, url="https://twitch.tv/somechannel")
            with mock.patch.object(stream_recorder.shutil, "which", return_value=None):
                with self.assertRaises(ToolNotFoundError):
                    rec._resolve_stream_input()

    def test_streamlink_failure_raises_stream_resolution_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp, url="https://twitch.tv/somechannel")
            err = stream_recorder.subprocess.CalledProcessError(1, ["streamlink"], stderr="boom")
            with mock.patch.object(stream_recorder.shutil, "which", return_value="streamlink"), \
                 mock.patch.object(stream_recorder.subprocess, "run", side_effect=err):
                with self.assertRaises(StreamResolutionError):
                    rec._resolve_stream_input()

    def test_resolution_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp, url="https://twitch.tv/c")
            with mock.patch.object(stream_recorder.shutil, "which", return_value="streamlink"), \
                 mock.patch.object(
                     stream_recorder.subprocess, "run",
                     return_value=SimpleNamespace(stdout="https://real.m3u8\n"),
                 ) as run:
                rec._resolve_stream_input()
                rec._resolve_stream_input()
            self.assertEqual(run.call_count, 1)


class CaptureScreenshotTests(unittest.TestCase):
    def test_returns_path_when_ffmpeg_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)

            def fake_run(cmd, **kwargs):
                Path(cmd[-1]).write_bytes(b"jpeg")
                return SimpleNamespace(returncode=0)

            with mock.patch.object(stream_recorder.shutil, "which", return_value="ffmpeg"), \
                 mock.patch.object(stream_recorder.subprocess, "run", side_effect=fake_run):
                out = rec.capture_screenshot()
            self.assertIsNotNone(out)
            self.assertTrue(out.exists())

    def test_missing_ffmpeg_raises_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)
            with mock.patch.object(stream_recorder.shutil, "which", return_value=None):
                with self.assertRaises(ToolNotFoundError):
                    rec.capture_screenshot()

    def test_returns_none_when_no_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)
            with mock.patch.object(stream_recorder.shutil, "which", return_value="ffmpeg"), \
                 mock.patch.object(
                     stream_recorder.subprocess, "run",
                     return_value=SimpleNamespace(returncode=0),
                 ):
                self.assertIsNone(rec.capture_screenshot())


class _FakePart:
    @staticmethod
    def from_text(_):
        return "text-part"

    @staticmethod
    def from_bytes(data=None, mime_type=None):
        return "image-part"


def _fake_client(reply_text: str):
    response = SimpleNamespace(text=reply_text)
    models = SimpleNamespace(generate_content=mock.Mock(return_value=response))
    return SimpleNamespace(models=models)


class DetectRoundTests(unittest.TestCase):
    def _detect(self, reply_text: str):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)
            rec.client = _fake_client(reply_text)
            shot = Path(tmp) / "shot.jpg"
            shot.write_bytes(b"jpeg")
            fake_types = SimpleNamespace(Part=_FakePart)
            with mock.patch.object(stream_recorder, "genai_types", fake_types):
                result = rec.detect_round_from_screenshot(shot)
            self.assertFalse(shot.exists(), "screenshot should be cleaned up")
            return result

    def test_plain_integer_reply(self):
        self.assertEqual(self._detect("14"), 14)

    def test_json_reply_with_round_field(self):
        self.assertEqual(self._detect("{'round': 7}"), 7)

    def test_null_reply_returns_none(self):
        self.assertIsNone(self._detect("null"))

    def test_garbage_reply_returns_none(self):
        self.assertIsNone(self._detect("I cannot tell"))

    def test_no_client_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)
            shot = Path(tmp) / "shot.jpg"
            shot.write_bytes(b"jpeg")
            self.assertIsNone(rec.detect_round_from_screenshot(shot))


if __name__ == "__main__":
    unittest.main()
