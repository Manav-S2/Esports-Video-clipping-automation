"""Unit tests for llm_client: JSON recovery, chat parsing, retry policy."""

from __future__ import annotations

import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import (  # noqa: E402
    _chat_text,
    _close_unbalanced_curly,
    _extract_json,
    _google_gemini_retry_delay,
    _is_retryable_urllib_failure,
    _sanitize_llm_json_blob,
    _urllib_retry_delay_after_network_error,
)


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

    def test_non_object_root_raises(self):
        with self.assertRaises(RuntimeError):
            _extract_json("[1, 2, 3]")

    def test_close_unbalanced_curly(self):
        self.assertGreaterEqual(_close_unbalanced_curly('{"a": {"b": 1').count("}"), 2)

    def test_sanitize_is_idempotent_on_valid_json(self):
        blob = '{"a": 1, "b": [2, 3]}'
        self.assertEqual(_sanitize_llm_json_blob(blob), blob)


class ChatTextTests(unittest.TestCase):
    def test_empty_response(self):
        self.assertEqual(_chat_text({}), "")
        self.assertEqual(_chat_text({"choices": []}), "")

    def test_string_content(self):
        resp = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(_chat_text(resp), "hello")

    def test_multipart_content(self):
        resp = {
            "choices": [
                {"message": {"content": [{"text": "a"}, {"type": "image"}, {"text": "b"}]}}
            ]
        }
        self.assertEqual(_chat_text(resp), "a\nb")


def _http_error(code: int, headers=None, body: bytes = b"") -> urllib.error.HTTPError:
    import email.message
    import io

    hdrs = email.message.Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return urllib.error.HTTPError("http://x", code, "err", hdrs, io.BytesIO(body))


class RetryPolicyTests(unittest.TestCase):
    def test_retry_after_header_wins(self):
        exc = _http_error(429, {"Retry-After": "7"})
        self.assertEqual(_google_gemini_retry_delay(exc, "", 0, exponential_cap=60.0), 7.0)

    def test_please_retry_in_message(self):
        exc = _http_error(429)
        body = '{"error": {"message": "Please retry in 3.5s"}}'
        self.assertAlmostEqual(
            _google_gemini_retry_delay(exc, body, 0, exponential_cap=60.0), 3.85
        )

    def test_resource_exhausted_backoff_grows_with_attempt(self):
        exc = _http_error(429)
        body = '{"error": {"status": "RESOURCE_EXHAUSTED", "message": ""}}'
        d0 = _google_gemini_retry_delay(exc, body, 0, exponential_cap=60.0)
        d2 = _google_gemini_retry_delay(exc, body, 2, exponential_cap=60.0)
        self.assertLess(d0, d2)

    def test_fallback_respects_cap_for_non_429(self):
        exc = _http_error(503)
        d = _google_gemini_retry_delay(exc, "not json", 10, exponential_cap=20.0)
        self.assertLessEqual(d, 20.0)

    def test_network_backoff_is_capped(self):
        self.assertEqual(_urllib_retry_delay_after_network_error(0), 3.0)
        self.assertEqual(_urllib_retry_delay_after_network_error(10), 60.0)


class RetryableFailureTests(unittest.TestCase):
    def test_timeout_is_retryable(self):
        self.assertTrue(_is_retryable_urllib_failure(TimeoutError()))

    def test_http_error_is_not_retryable(self):
        self.assertFalse(_is_retryable_urllib_failure(_http_error(500)))

    def test_urlerror_with_connection_reset_is_retryable(self):
        self.assertTrue(_is_retryable_urllib_failure(urllib.error.URLError(ConnectionResetError())))

    def test_urlerror_with_timeout_text_is_retryable(self):
        self.assertTrue(_is_retryable_urllib_failure(urllib.error.URLError(OSError("read timed out"))))

    def test_plain_oserror_without_timeout_not_retryable(self):
        self.assertFalse(_is_retryable_urllib_failure(OSError("permission denied")))


if __name__ == "__main__":
    unittest.main()
