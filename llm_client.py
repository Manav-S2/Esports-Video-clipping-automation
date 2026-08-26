"""Shared LLM REST client helpers (Gemini / Vertex / OpenAI-compatible endpoints).

Extracted from ``live_stream_highlight_pipeline`` so the transport, retry policy, and
defensive JSON recovery for model responses live in one place and stay unit-testable
without the media pipeline's dependencies.

Model responses are treated as hostile input: never ``json.loads`` a raw reply — route
it through :func:`_extract_json` so markdown fences, bare keys, trailing commas, and
truncated blobs are handled.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _make_ssl_context() -> ssl.SSLContext:
    """Strict TLS verification using ``SSL_CERT_FILE`` or certifi (no unverified fallback)."""
    ca = (os.environ.get("SSL_CERT_FILE") or "").strip()
    if ca and Path(ca).is_file():
        return ssl.create_default_context(cafile=ca)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError as exc:
        raise RuntimeError(
            "TLS: install certifi (`pip install certifi`) or set SSL_CERT_FILE to a PEM CA bundle."
        ) from exc


def _sanitize_llm_json_blob(blob: str) -> str:
    """Fix common vision-model JSON mistakes: unquoted keys, trailing commas."""
    b = blob
    # Quote bare identifiers used as keys: { foo: → { "foo":
    for _ in range(16):
        nb = re.sub(r'([\{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', b)
        if nb == b:
            break
        b = nb
    prev = ""
    while prev != b:
        prev = b
        b = re.sub(r",\s*}", "}", b)
        b = re.sub(r",\s*]", "]", b)
    return b


def _close_unbalanced_curly(s: str) -> str:
    """Append ``}`` so truncated ``{ ... `` fragments may parse (best-effort)."""
    diff = s.count("{") - s.count("}")
    if diff > 0:
        return s + ("}" * diff)
    return s


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    # Gemini / Vertex often wrap JSON in markdown fences.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())
    idx = cleaned.find("{")
    if idx < 0:
        raise RuntimeError(f"Model did not return JSON object: {text[:400]}")
    sub = cleaned[idx:]
    sub = _sanitize_llm_json_blob(sub)
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(sub)
    except json.JSONDecodeError as exc:
        err_pos = getattr(exc, "pos", len(sub))
        head = sub[: err_pos].rstrip().rstrip(",")
        salvaged = None
        if head:
            last_comma = head.rfind(",")
            if last_comma > 0:
                shorter = head[:last_comma].rstrip().rstrip(",")
                shorter = _close_unbalanced_curly(shorter)
                shorter = _sanitize_llm_json_blob(shorter)
                try:
                    salvaged, _ = decoder.raw_decode(shorter)
                except json.JSONDecodeError:
                    salvaged = None
            if salvaged is None:
                shorter = _close_unbalanced_curly(head.rstrip(","))
                shorter = _sanitize_llm_json_blob(shorter)
                try:
                    salvaged, _ = decoder.raw_decode(shorter)
                except json.JSONDecodeError:
                    salvaged = None
        if isinstance(salvaged, dict):
            return salvaged
        raise RuntimeError(
            "Model JSON was truncated or invalid: "
            f"{exc}. First chars after '{{': {cleaned[idx : idx + 500]!r}"
        ) from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"Model JSON root was not an object: {type(obj).__name__}")
    return obj


def _file_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _json_post(url: str, api_key: str, payload: dict[str, Any], timeout_sec: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ssl_ctx = _make_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ssl_ctx) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {err_text[:1000]}") from exc
    return json.loads(raw)


def _chat_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts)
    return ""


def _google_gemini_retry_delay(
    exc: urllib.error.HTTPError,
    err_body: str,
    attempt: int,
    *,
    exponential_cap: float,
) -> float:
    """Seconds to sleep before retrying Gemini / Vertex REST calls (quota, overload).

    Honors Retry-After, embedded ``Please retry in Ns``, google.rpc.RetryInfo, and uses
    stronger backoff for RESOURCE_EXHAUSTED when no hint is present (free-tier RPM).
    """
    if exc.headers:
        ra = exc.headers.get("Retry-After")
        if ra:
            try:
                return min(120.0, max(0.5, float(str(ra).strip())))
            except ValueError:
                pass
    try:
        obj = json.loads(err_body)
        err_obj = obj.get("error") or {}
        msg = str(err_obj.get("message", ""))
        m = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)\s*s", msg, re.I)
        if m:
            return min(120.0, max(0.5, float(m.group(1)) + 0.35))
        status = str(err_obj.get("status", ""))
        if status == "RESOURCE_EXHAUSTED" or "Quota exceeded" in msg:
            return min(120.0, max(15.0, 12.0 * (attempt + 1)))
        for det in err_obj.get("details") or []:
            if not isinstance(det, dict):
                continue
            rd = det.get("retryDelay")
            if rd is None:
                continue
            if isinstance(rd, str) and rd.endswith("s"):
                return min(120.0, max(0.5, float(rd[:-1]) + 0.35))
            if isinstance(rd, dict):
                sec = rd.get("seconds")
                if sec is not None:
                    nanos = int(rd.get("nanos") or 0)
                    return min(120.0, max(0.5, float(sec) + nanos / 1e9 + 0.35))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    fallback = min(exponential_cap, 0.75 * (2**attempt))
    if exc.code == 429:
        fallback = max(fallback, min(90.0, 12.0 * (attempt + 1)))
    return fallback


def _urllib_retry_delay_after_network_error(attempt: int) -> float:
    """Backoff for read/connect timeouts and transient TLS/TCP failures."""
    return min(60.0, 3.0 * (2**attempt))


def _is_retryable_urllib_failure(exc: BaseException) -> bool:
    """True when ``urlopen`` failed due to timeout or likely-transient connection issues."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        r = exc.reason
        if isinstance(r, TimeoutError | socket.timeout | BrokenPipeError | ConnectionResetError):
            return True
        if isinstance(r, ConnectionError):
            return True
        if isinstance(r, OSError):
            msg = str(r).lower()
            if "timed out" in msg or "time out" in msg:
                return True
        msg = str(exc).lower()
        if "timed out" in msg or "time out" in msg:
            return True
    if isinstance(exc, BrokenPipeError | ConnectionResetError):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if "timed out" in msg or "time out" in msg:
            return True
    return False
