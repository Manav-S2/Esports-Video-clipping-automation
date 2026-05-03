#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import multiprocessing
import mimetypes
import os
import re
import shutil
import signal
import sys
import socket
import ssl
import subprocess
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import warnings

# MSYS2 UCRT64 / some Windows builds: NumPy warns about longdouble signature probe; harmless here (we use float32).
warnings.filterwarnings("ignore", category=UserWarning, module="numpy._core.getlimits")

import numpy as np
from PIL import Image, ImageFilter

from detect_cs2_highlight import (
    _assign_round_numbers,
    _load_kill_events,
    _load_round_start_events,
    _normalize_kill_time_column,
    _score_rounds,
)
from speech_google_captions import transcribe_and_burn
from video_editor import apply_portrait_blur

# Strict live HUD cadence: wall-clock period between successive crop → AI round checks.
# ``PipelineConfig.screenshot_interval_sec`` is not used for this loop (kept for compatibility / timeouts).
LIVE_HUD_ROUND_CHECK_INTERVAL_SEC = 5.0
# Avoid calling streamlink on every HUD grab; tokens usually last long enough; retry with refresh on failure.
STREAMLINK_RESOLVE_CACHE_SEC = 75.0
# Fail fast if the HLS read stalls (microseconds for ffmpeg ``-rw_timeout``).
HUD_FFMPEG_RW_TIMEOUT_US = 12_000_000


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


def _parse_seek_seconds(val: Any) -> float:
    """Seconds from JSON config: number, numeric string, or clock \"H:MM:SS\" / \"MM:SS\"."""
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 0.0
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return 0.0
        try:
            return max(0.0, float(s))
        except ValueError:
            pass
        parts = s.replace(",", ".").split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return 0.0
        if len(nums) == 3:
            return max(0.0, nums[0] * 3600 + nums[1] * 60 + nums[2])
        if len(nums) == 2:
            return max(0.0, nums[0] * 60 + nums[1])
        if len(nums) == 1:
            return max(0.0, nums[0])
        return 0.0
    return 0.0


def _extract_json(text: str) -> Dict[str, Any]:
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


def _extract_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path, "r") as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    xml = re.sub(r"\n{3,}", "\n\n", xml)
    return xml.strip()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _run_ffmpeg(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clear_dir_contents(path: Path) -> None:
    """Remove all files and subdirectories inside ``path``; keep the directory itself."""
    if not path.is_dir():
        return
    for child in list(path.iterdir()):
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        except OSError as exc:
            print(f"[live] warning: could not remove {child}: {exc}", flush=True)


def _numpy_enhance_rgb_float(
    arr: np.ndarray,
    contrast_clip_percent: float,
    saturation_boost: float,
    unsharp_radius: float,
    unsharp_amount: float,
) -> np.ndarray:
    """Color-preserving HUD enhancement on a small RGB float32 array (HxWx3)."""
    clip_pct = max(0.0, min(10.0, float(contrast_clip_percent)))
    low = np.percentile(arr, clip_pct, axis=(0, 1), keepdims=True)
    high = np.percentile(arr, 100.0 - clip_pct, axis=(0, 1), keepdims=True)
    arr = (arr - low) * (255.0 / np.maximum(high - low, 1.0))
    arr = np.clip(arr, 0.0, 255.0)

    sat = max(0.5, min(2.5, float(saturation_boost)))
    luma = arr.mean(axis=2, keepdims=True)
    arr = np.clip(luma + (arr - luma) * sat, 0.0, 255.0)

    radius = max(0.1, float(unsharp_radius))
    amount = max(0.0, min(3.0, float(unsharp_amount)))
    base = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    blur = base.filter(ImageFilter.GaussianBlur(radius=radius))
    arr_blur = np.asarray(blur, dtype=np.float32)
    return np.clip(arr + amount * (arr - arr_blur), 0.0, 255.0)


def _file_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _json_post(url: str, api_key: str, payload: Dict[str, Any], timeout_sec: int = 120) -> Dict[str, Any]:
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
    ssl_ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ssl_ctx) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    return json.loads(raw)


def _chat_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts)
    return ""


def _deprioritize_background_thread() -> None:
    """Lower calling thread CPU priority so screenshot/ffmpeg/HUD paths stay responsive (Windows)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        THREAD_PRIORITY_BELOW_NORMAL = -1
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass


def _os_suspend_pid(pid: int, *, tag: str = "") -> bool:
    """Suspend every thread in ``pid`` (Windows ``NtSuspendProcess``; POSIX ``SIGSTOP``). Best-effort."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            ntdll = ctypes.windll.ntdll
            # PROCESS_SUSPEND_RESUME alone often fails OpenProcess on child ffmpeg; add query rights.
            PROCESS_SUSPEND_RESUME = 0x0800
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            access = PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION
            h = kernel32.OpenProcess(access, False, ctypes.c_uint(pid))
            if not h:
                err = int(kernel32.GetLastError())
                suffix = f" {tag}" if tag else ""
                print(
                    f"[live] WARN: OpenProcess(suspend) failed pid={pid} winerr={err}{suffix}",
                    flush=True,
                )
                return False
            try:
                status = int(ntdll.NtSuspendProcess(h))
                if status != 0:
                    suffix = f" {tag}" if tag else ""
                    print(
                        f"[live] WARN: NtSuspendProcess failed pid={pid} status={status:#x}{suffix}",
                        flush=True,
                    )
                return status == 0
            finally:
                kernel32.CloseHandle(h)
        except Exception as exc:
            suffix = f" {tag}" if tag else ""
            print(f"[live] WARN: suspend pid={pid} raised {exc!r}{suffix}", flush=True)
            return False
    try:
        os.kill(pid, signal.SIGSTOP)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _os_resume_pid(pid: int, *, tag: str = "") -> bool:
    """Resume ``pid`` after :func:`_os_suspend_pid` (Windows ``NtResumeProcess``; POSIX ``SIGCONT``)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            ntdll = ctypes.windll.ntdll
            PROCESS_SUSPEND_RESUME = 0x0800
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            access = PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION
            h = kernel32.OpenProcess(access, False, ctypes.c_uint(pid))
            if not h:
                err = int(kernel32.GetLastError())
                suffix = f" {tag}" if tag else ""
                print(
                    f"[live] WARN: OpenProcess(resume) failed pid={pid} winerr={err}{suffix}",
                    flush=True,
                )
                return False
            try:
                status = int(ntdll.NtResumeProcess(h))
                if status != 0:
                    suffix = f" {tag}" if tag else ""
                    print(
                        f"[live] WARN: NtResumeProcess failed pid={pid} status={status:#x}{suffix}",
                        flush=True,
                    )
                return status == 0
            finally:
                kernel32.CloseHandle(h)
        except Exception as exc:
            suffix = f" {tag}" if tag else ""
            print(f"[live] WARN: resume pid={pid} raised {exc!r}{suffix}", flush=True)
            return False
    try:
        os.kill(pid, signal.SIGCONT)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


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
        if isinstance(r, (TimeoutError, socket.timeout, BrokenPipeError, ConnectionResetError)):
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
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if "timed out" in msg or "time out" in msg:
            return True
    return False


def _vertex_generate_content(
    vertex_api_key: str,
    vertex_project_id: str,
    vertex_location: str,
    model: str,
    prompt: str,
    image_path: Optional[Path] = None,
    *,
    max_output_tokens: int = 1024,
) -> str:
    """Gemini on Vertex AI via REST + API key (see Vertex AI Express Mode / API key auth)."""
    model_resource = model.strip()
    if model_resource.startswith("projects/"):
        endpoint = (
            f"https://{vertex_location}-aiplatform.googleapis.com/v1/"
            f"{model_resource}:generateContent"
        )
    else:
        endpoint = (
            f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project_id}/"
            f"locations/{vertex_location}/publishers/google/models/{model_resource}:generateContent"
        )
    endpoint = f"{endpoint}?key={urllib.parse.quote(vertex_api_key)}"

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if image_path is not None:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                }
            }
        )

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
        },
    }

    ssl_ctx = ssl._create_unverified_context()
    transient_http = frozenset({429, 500, 502, 503, 504})
    max_attempts = 8
    # Multimodal requests (large inline image) often exceed default ~120s server-side latency.
    read_timeout_sec = 120.0 if image_path is not None else 180.0
    raw = ""
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=read_timeout_sec, context=ssl_ctx) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            break
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            if exc.code in transient_http and attempt < max_attempts - 1:
                delay = _google_gemini_retry_delay(
                    exc, err_body, attempt, exponential_cap=60.0
                )
                tag = (
                    "quota/rate limit"
                    if exc.code == 429 or "RESOURCE_EXHAUSTED" in err_body
                    else "transient"
                )
                print(
                    f"[live] Vertex HTTP {exc.code} ({tag}), retry in {delay:.1f}s "
                    f"({attempt + 1}/{max_attempts})",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Vertex AI HTTP {exc.code}: {err_body[:1500]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_attempts - 1 and _is_retryable_urllib_failure(exc):
                delay = _urllib_retry_delay_after_network_error(attempt)
                print(
                    f"[live] Vertex network error ({type(exc).__name__}), retry in {delay:.1f}s "
                    f"({attempt + 1}/{max_attempts}): {exc}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Vertex AI request failed: {exc}") from exc
    data = json.loads(raw)
    texts: List[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    combined = "\n".join(texts).strip()
    if not combined:
        raise RuntimeError(f"Vertex AI response had no text: {raw[:1200]}")
    return combined


def _gemini_generate_text(
    api_key: str,
    model: str,
    prompt: str,
    image_path: Optional[Path] = None,
    *,
    max_output_tokens: int = 1024,
) -> str:
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if image_path is not None:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = json.dumps(payload).encode("utf-8")
    ssl_ctx = ssl._create_unverified_context()
    transient_http = frozenset({429, 500, 502, 503, 504})
    max_attempts = 8
    read_timeout_sec = 120.0 if image_path is not None else 180.0
    raw = ""
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=read_timeout_sec, context=ssl_ctx) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            break
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            if exc.code in transient_http and attempt < max_attempts - 1:
                delay = _google_gemini_retry_delay(
                    exc, err_body, attempt, exponential_cap=60.0
                )
                if exc.code == 429 or "RESOURCE_EXHAUSTED" in err_body:
                    tag = "quota/rate limit"
                elif exc.code == 503:
                    tag = "high demand"
                else:
                    tag = "transient"
                print(
                    f"[live] Gemini HTTP {exc.code} ({tag}), retry in {delay:.1f}s "
                    f"({attempt + 1}/{max_attempts})",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini HTTP {exc.code}: {err_body[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_attempts - 1 and _is_retryable_urllib_failure(exc):
                delay = _urllib_retry_delay_after_network_error(attempt)
                print(
                    f"[live] Gemini network error ({type(exc).__name__}), retry in {delay:.1f}s "
                    f"({attempt + 1}/{max_attempts}): {exc}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
    data = json.loads(raw)
    texts: List[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _live_pipeline_repo_root() -> Path:
    """Repo root containing ``CAPTIONS`` and ``Esports-Video-clipping-automation``."""
    return Path(__file__).resolve().parent.parent


def _captions_vertex_burn_script_path() -> Path:
    return _live_pipeline_repo_root() / "CAPTIONS" / "burn_karaoke_captions.py"


def _input_nonempty(prompt: str) -> str:
    while True:
        try:
            s = input(prompt).strip()
        except EOFError:
            return ""
        if s:
            return s
        print("[live] Please enter a non-empty value.", flush=True)


def _prompt_live_or_recorded() -> str:
    print("", flush=True)
    print("[live] Stream source:", flush=True)
    print("  1 = LIVE stream", flush=True)
    print("  2 = RECORDED (VOD / past broadcast)", flush=True)
    while True:
        try:
            choice = input("[live] Choose 1 or 2: ").strip()
        except EOFError:
            return "live"
        if choice == "1":
            return "live"
        if choice == "2":
            return "vod"
        low = choice.lower()
        if low in ("live", "l"):
            return "live"
        if low in ("recorded", "vod", "r"):
            return "vod"
        print("[live] Type 1 for live or 2 for recorded/VOD.", flush=True)


def _read_multiline_match_context(end_sentinel: str = "END") -> str:
    print("", flush=True)
    print(
        "[live] Paste match context: teams, players, casters, maps - anything useful for captions.",
        flush=True,
    )
    print(
        f"[live] When finished, type {end_sentinel} on its own line and press Enter.",
        flush=True,
    )
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == end_sentinel:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_match_context_for_captions(
    cfg: PipelineConfig,
    raw_notes: str,
) -> Tuple[str, Dict[str, Any]]:
    """Expand informal notes into roster text (+ structured extract) via Gemini or Vertex."""
    if not raw_notes.strip():
        return "", {}

    prompt = (
        "You extract structured esports match context from informal notes for speech-to-text captioning.\n\n"
        "USER NOTES:\n"
        f"{raw_notes}\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "caption_roster_text": string,\n'
        '  "teams": [{"team_name": string, "players": [string]}],\n'
        '  "casters_or_talent": [string],\n'
        '  "aliases_or_handles": [string],\n'
        '  "summary_one_line": string\n'
        "}\n\n"
        "caption_roster_text must be plaintext (multiple lines allowed) listing authoritative spellings: "
        "team names, player nicknames and handles, caster names - concise and ASCII-friendly.\n"
        "Infer teams and players from messy prose.\n"
    )

    raw_response = ""
    last_api_err: Optional[BaseException] = None

    if cfg.vertex_project_id and cfg.vertex_api_keys:
        try:
            raw_response = _vertex_generate_content(
                cfg.vertex_api_keys[0],
                cfg.vertex_project_id,
                cfg.vertex_location,
                cfg.gemini_model,
                prompt,
                None,
                max_output_tokens=4096,
            )
        except RuntimeError as exc:
            last_api_err = exc
            print(
                "[live] Vertex roster call failed (often DNS/offline: errno 11001 getaddrinfo). "
                "Trying Gemini API next if configured.",
                flush=True,
            )

    if not (raw_response or "").strip() and cfg.gemini_api_keys:
        try:
            raw_response = _gemini_generate_text(
                cfg.gemini_api_keys[0],
                cfg.gemini_model,
                prompt,
                None,
                max_output_tokens=4096,
            )
        except Exception as exc:
            last_api_err = exc
            print(f"[live] Gemini roster call failed ({exc}); using your pasted text as-is.", flush=True)

    if not (raw_response or "").strip():
        if not cfg.vertex_api_keys and not cfg.gemini_api_keys:
            print(
                "[live] No Vertex/Gemini keys; saving your pasted paragraph as the caption roster verbatim.",
                flush=True,
            )
        else:
            print(
                "[live] AI unreachable — saving your pasted paragraph as the caption roster "
                "(fix DNS/network or run when online).",
                flush=True,
            )
        hint = repr(last_api_err) if last_api_err else ""
        return raw_notes, {"fallback": "raw_notes_only", "last_error": hint[:500]}

    try:
        obj = _extract_json(raw_response)
    except RuntimeError:
        return raw_notes, {"parse_failed": True, "model_raw_excerpt": (raw_response or "")[:1200]}

    roster = str(obj.get("caption_roster_text", "") or "").strip()
    if not roster:
        roster = raw_notes

    structured = {
        "teams": obj.get("teams"),
        "casters_or_talent": obj.get("casters_or_talent"),
        "aliases_or_handles": obj.get("aliases_or_handles"),
        "summary_one_line": obj.get("summary_one_line"),
    }
    combined = roster.rstrip() + "\n\n--- structured_extract ---\n"
    combined += json.dumps(structured, ensure_ascii=False, indent=2)
    return combined, structured


def _interactive_session_configure(cfg: PipelineConfig) -> PipelineConfig:
    """Opt-in via ``--interactive``: prompts for URL, VOD seek, roster paste; writes karaoke_vertex roster."""
    mode = _prompt_live_or_recorded()
    url = _input_nonempty("[live] Paste stream / VOD link: ")
    if not url:
        raise RuntimeError("Empty stream URL.")

    seek_sec = 0.0
    if mode == "vod":
        ts_raw = input(
            "[live] VOD start offset from beginning (seconds, MM:SS, H:MM:SS; blank = 0): ",
        ).strip()
        seek_sec = _parse_seek_seconds(ts_raw if ts_raw else "0")

    raw_ctx = _read_multiline_match_context()
    if not raw_ctx.strip():
        print("[live] No match context pasted; roster file will be minimal.", flush=True)

    root = Path(cfg.output_root).resolve()
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    roster_path = meta / "session_vertex_caption_roster.txt"
    session_log = meta / "interactive_session.json"

    structured_summary: Dict[str, Any] = {}
    roster_blob = ""
    if raw_ctx.strip():
        print("[live] Calling Gemini/Vertex to extract teams & players from your notes...", flush=True)
        roster_blob, structured_summary = _normalize_match_context_for_captions(cfg, raw_ctx)
    if not roster_blob.strip():
        roster_blob = "(no roster context provided)\n"

    roster_path.write_text(roster_blob, encoding="utf-8")
    session_log.write_text(
        json.dumps(
            {
                "timestamp": _now_stamp(),
                "stream_mode": mode,
                "stream_url": url,
                "stream_input_seek_sec": seek_sec,
                "roster_path": str(roster_path.resolve()),
                "structured_extract_keys": list(structured_summary.keys()),
                "user_notes_chars": len(raw_ctx),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[live] Session roster written: {roster_path}", flush=True)
    print("[live] Interactive setup complete; starting round detection.", flush=True)

    return replace(
        cfg,
        stream_url=url,
        stream_input_seek_sec=float(max(0.0, seek_sec)),
        caption_provider="karaoke_vertex",
        karaoke_vertex_roster_path=str(roster_path.resolve()),
    )


def _derived_round_from_scores(detection: Dict[str, Any]) -> Optional[int]:
    """Infer round index from team scores when visible: ``round_sum = score_left + score_right``, round = sum + 1."""
    score_left = detection.get("score_left")
    score_right = detection.get("score_right")
    if score_left is not None and score_right is not None:
        try:
            left = int(score_left)
            right = int(score_right)
            round_sum = left + right
            detection["round_sum"] = round_sum
            rn = round_sum + 1
            return rn if rn > 0 else None
        except (TypeError, ValueError):
            pass

    raw_round_sum = detection.get("round_sum")
    if raw_round_sum is not None:
        try:
            round_num = int(raw_round_sum) + 1
            return round_num if round_num > 0 else None
        except (TypeError, ValueError):
            pass
    return None


def _rekognition_scores_from_crop(
    crop_path: Path,
    min_confidence_0_1: float,
    region_name: str,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
) -> Dict[str, Any]:
    """Call Rekognition ``detect_text`` on JPEG bytes; map left/right team totals from digit WORDs.

    Credentials: pass explicit keys (from env merged into config, or JSON), else default boto3
    chain (``~/.aws/credentials``, IAM role, etc.).
    """
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "HUD uses AWS Rekognition but boto3 is not installed for this Python:\n"
            f"  {sys.executable}\n"
            "Install: py -3 -m pip install boto3   (use Windows CPython, not MSYS python if pip is missing)\n"
            "Or run the pipeline with: py -3 live_stream_highlight_pipeline.py --config <config.json>"
        ) from exc

    img = crop_path.read_bytes()
    if len(img) > 15 * 1024 * 1024:
        raise ValueError("HUD crop exceeds Rekognition 15MB limit")

    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

    client_kwargs: Dict[str, Any] = {"region_name": region_name}
    ak = (aws_access_key_id or "").strip()
    sk = (aws_secret_access_key or "").strip()
    if ak and sk:
        client_kwargs["aws_access_key_id"] = ak
        client_kwargs["aws_secret_access_key"] = sk
        st = (aws_session_token or "").strip()
        if st:
            client_kwargs["aws_session_token"] = st

    try:
        client = boto3.client("rekognition", **client_kwargs)
        resp = client.detect_text(Image={"Bytes": img})
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise RuntimeError(
            "AWS Rekognition: no usable credentials.\n"
            f"  Interpreter: {sys.executable}\n"
            f"  Region (rekognition): {region_name}\n"
            "Configure one of:\n"
            "  • Environment: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY "
            "(optional AWS_SESSION_TOKEN for temp keys)\n"
            "  • Windows: run `aws configure` (install AWS CLI) → %USERPROFILE%\\.aws\\credentials\n"
            "  • Optional: aws_access_key_id / aws_secret_access_key in main pipeline JSON, or gitignored "
            "aws_credentials.local.json next to it (see aws_credentials.local.example.json)\n"
            "  • Env vars override values from JSON / local file\n"
            "IAM policy must allow rekognition:DetectText on resource * (image bytes)."
        ) from exc
    except ClientError as exc:
        err = str(exc)
        if "Unable to locate credentials" in err or "could not be found" in err.lower():
            raise RuntimeError(
                "AWS Rekognition: Unable to locate credentials.\n"
                f"  Interpreter: {sys.executable}\n"
                f"  Region: {region_name}\n"
                "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or run `aws configure`."
            ) from exc
        raise

    detections = resp.get("TextDetections") or []
    id_map: Dict[int, Dict[str, Any]] = {}
    for d in detections:
        raw_id = d.get("Id")
        if raw_id is None:
            continue
        try:
            id_map[int(raw_id)] = d
        except (TypeError, ValueError):
            continue

    def _line_text_for_detection(d: Dict[str, Any]) -> str:
        """Walk ParentId chain to the enclosing LINE's text (if any)."""
        pid = d.get("ParentId")
        hops = 0
        while pid is not None and hops < 16:
            try:
                i = int(pid)
            except (TypeError, ValueError):
                break
            p = id_map.get(i)
            if not p:
                break
            if p.get("Type") == "LINE":
                return str(p.get("DetectedText") or "")
            pid = p.get("ParentId")
            hops += 1
        return ""

    def _line_is_timer_line(line_text: str) -> bool:
        return bool(re.search(r"\d\s*:\s*\d", line_text or ""))

    def _normalize_hud_score_text(text: str) -> str:
        """Rekognition often reads team totals as letter O instead of zero."""
        t = (text or "").strip()
        if re.fullmatch(r"[Oo]", t):
            return "0"
        return t

    def _should_skip_word_for_parent_line(ctx: str, cx_norm: float) -> bool:
        """Parent LINE often bundles ROUND/timer *and* edge scores — skip only *center* digits."""
        if not (ctx or "").strip():
            return False
        c = ctx.upper()
        # Team scores sit on the wings; round counter + timer sit near center.
        mid = 0.18 < cx_norm < 0.82
        if "ROUND" in c and mid:
            return True
        if _line_is_timer_line(ctx) and mid:
            return True
        return False

    def _line_is_score_noise_line(text: str) -> bool:
        """Standalone LINE nodes: drop whole lines that are clearly round/timer."""
        t = (text or "").strip()
        if not t:
            return False
        u = t.upper()
        if "ROUND" in u:
            return True
        if _line_is_timer_line(t):
            return True
        return False

    # Rekognition confidence is 0–100; HUD digits are often a bit weaker — don't match LLM thresholds 1:1.
    eff_min_conf = max(18.0, min(100.0, float(min_confidence_0_1) * 65.0))

    candidates: List[Tuple[float, float, int, float, str]] = []  # cy, cx, val, conf, kind

    for d in detections:
        dtype = d.get("Type")
        if dtype == "WORD":
            geom0 = d.get("Geometry") or {}
            bb0 = geom0.get("BoundingBox") or {}
            left0 = float(bb0.get("Left", 0.0))
            w0 = float(bb0.get("Width", 0.0))
            cx_word = left0 + w0 / 2.0
            ctx = _line_text_for_detection(d)
            if _should_skip_word_for_parent_line(ctx, cx_word):
                continue
            text = _normalize_hud_score_text(d.get("DetectedText") or "")
            if not re.fullmatch(r"\d{1,2}", text):
                continue
            val = int(text)
            if not (0 <= val <= 24):
                continue
            conf = float(d.get("Confidence") or 0.0)
            if conf < eff_min_conf:
                continue
            geom = d.get("Geometry") or {}
            bb = geom.get("BoundingBox") or {}
            left = float(bb.get("Left", 0.0))
            top = float(bb.get("Top", 0.0))
            w = float(bb.get("Width", 0.0))
            h = float(bb.get("Height", 0.0))
            cx = left + w / 2.0
            cy = top + h / 2.0
            if cy < 0.03 or cy > 0.97:
                continue
            candidates.append((cy, cx, val, conf, "WORD"))
        elif dtype == "LINE":
            text = _normalize_hud_score_text(d.get("DetectedText") or "")
            if _line_is_score_noise_line(d.get("DetectedText") or ""):
                continue
            if not re.fullmatch(r"\d{1,2}", text):
                continue
            val = int(text)
            if not (0 <= val <= 24):
                continue
            conf = float(d.get("Confidence") or 0.0)
            if conf < eff_min_conf:
                continue
            geom = d.get("Geometry") or {}
            bb = geom.get("BoundingBox") or {}
            left = float(bb.get("Left", 0.0))
            top = float(bb.get("Top", 0.0))
            w = float(bb.get("Width", 0.0))
            h = float(bb.get("Height", 0.0))
            cx = left + w / 2.0
            cy = top + h / 2.0
            if cy < 0.03 or cy > 0.97:
                continue
            candidates.append((cy, cx, val, conf, "LINE"))

    # Prefer candidates in the vertical band where team scores sit (exclude top strip with ROUND text).
    score_band = [c for c in candidates if c[0] >= 0.20]
    use = score_band if len(score_band) >= 2 else candidates

    use.sort(key=lambda t: t[1])
    if len(use) < 2:
        snippet = []
        for d in detections[:24]:
            snippet.append(f"{d.get('Type')}:{(d.get('DetectedText') or '')[:40]!r}")
        return {
            "score_left": None,
            "score_right": None,
            "confidence": 0.0,
            "source": "aws_rekognition",
            "rekognition_note": (
                f"candidates={len(use)} (words+lines); sample={'; '.join(snippet[:8])}"
            ),
        }

    left_tok = use[0]
    right_tok = use[-1]
    sl, sr = int(left_tok[2]), int(right_tok[2])
    conf01 = min(float(left_tok[3]), float(right_tok[3])) / 100.0
    return {
        "score_left": sl,
        "score_right": sr,
        "confidence": conf01,
        "source": "aws_rekognition",
        "rekognition_numeric_words": len(use),
        "rekognition_pick": f"left={left_tok[4]} right={right_tok[4]}",
    }


@dataclass
class PipelineConfig:
    stream_url: str
    # HUD round scores: always AWS Rekognition DetectText (loader forces ``rekognition``).
    api_provider: str
    # Post-round highlights / titles: nvidia | gemini | vertex (not Rekognition).
    highlight_api_provider: str
    # Background threads draining ``_highlight_queue`` (contact-sheet Vertex/Gemini/NVIDIA work).
    highlight_parallel_workers: int
    # When True, highlight multimodal waits whenever HUD Rekognition/HTTP holds ``_hud_remote_calls_active``.
    highlight_yield_to_hud_vision: bool
    gemini_api_key: str
    gemini_api_keys: List[str]
    gemini_model: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_model: str
    aws_rekognition_region: str
    # Optional IAM access keys for Rekognition (prefer env AWS_*; JSON for local dev only — do not commit).
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    demo_file: str
    rules_docx: str
    output_root: str
    # Legacy / compat only: the live HUD loop uses :data:`LIVE_HUD_ROUND_CHECK_INTERVAL_SEC` (5s), not this field.
    screenshot_interval_sec: int
    min_round_record_sec: int
    max_round_record_sec: int  # 0 = no limit (split only on HUD round change); >0 safety cap in seconds
    clip_start_offset_sec: int
    # For Twitch/YouTube VOD replay: ffmpeg -ss offset from stream start (seconds); advances with wall clock so each screenshot/recording stays in sync.
    stream_input_seek_sec: float
    round_detection_min_confidence: float
    round_started_required: bool
    max_round_jump: int
    # Require N consecutive vision reads showing the same round before arming the first record (1 = first good read starts).
    stable_round_reads_to_start: int
    # Require N consecutive reads showing the *next* round number before we cut the file (reduces 1 file spanning multiple rounds).
    round_transition_confirmations: int
    # If True, only allow round_num == current_round + 1 (reject HUD skips like 10 -> 12).
    require_consecutive_round_increments: bool
    # If True, run highlight pipeline on segments split only because of max duration (usually incomplete rounds).
    process_partial_on_max_duration: bool
    # Ignored (always off): round recorder suspend during HUD idle was removed from the live loop.
    record_suspend_while_hud_idle: bool
    # Target max width after HUD crop (height auto); capped at 3840 for UHD intake to vision APIs.
    screenshot_4k_width: int
    screenshot_4k_height: int
    numpy_contrast_clip_percent: float
    numpy_saturation_boost: float
    numpy_unsharp_radius: float
    numpy_unsharp_amount: float
    # Portrait export (apply_portrait_blur): faster presets shorten CPU time vs ``slow`` + low CRF.
    portrait_blur_preset: str
    portrait_blur_crf: int
    # Normalized ROI (0..1) for HUD score strip — ffmpeg crops this region before JPEG for vision.
    round_roi_x: float
    round_roi_y: float
    round_roi_w: float
    round_roi_h: float
    caption_cmd_template: str
    caption_hook_timeout_sec: int
    caption_provider: str  # auto | none | google_speech | shell | karaoke_whisper | karaoke_google | karaoke_vertex
    speech_language_code: str
    speech_recognition_timeout_sec: int
    speech_api_key: str  # optional; prefer env GOOGLE_SPEECH_API_KEY
    instagram_enabled: bool
    instagram_username: str
    instagram_password: str
    # Gemini on Vertex AI (REST + API key): requires project id and Vertex-enabled API key.
    vertex_project_id: str
    vertex_location: str
    vertex_api_keys: List[str]
    # Absolute path to the pipeline JSON (Google Speech key resolution for karaoke_google).
    pipeline_config_path: Path
    # Karaoke ASS + optional bottom PNG overlay via isolated child process (``multiprocessing`` spawn).
    karaoke_whisper_model: str
    karaoke_margin_top_ratio: float
    karaoke_overlay_width_frac: float
    karaoke_overlay_margin_bottom_px: int
    karaoke_use_adc: bool
    karaoke_no_overlay: bool
    # Optional PNG/JPEG; relative paths are resolved against ``pipeline_config_path``'s parent directory.
    karaoke_overlay_image: str
    # When True, karaoke burns in a background thread so titles/Instagram run on portrait immediately.
    karaoke_async: bool
    # libx264 for karaoke subtitle burn + branding overlay (two passes when overlay on).
    karaoke_ffmpeg_preset: str
    karaoke_ffmpeg_crf: int
    # Roster text path for CAPTIONS Vertex full-video karaoke (set by interactive session or JSON).
    karaoke_vertex_roster_path: str


def _karaoke_vertex_burn_child_main(payload: Dict[str, Any]) -> None:
    """Spawn CAPTIONS ``burn_karaoke_captions.py`` (Vertex Gemini video karaoke)."""
    import subprocess
    import sys

    script = Path(payload["captions_script"])
    roster = str(payload.get("karaoke_vertex_roster_path") or "").strip()
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--video",
        str(payload["video_path"]),
        "--output",
        str(payload["video_out"]),
        "--config",
        str(payload["pipeline_config"]),
        "--work-dir",
        str(payload["work_dir"]),
    ]
    if roster:
        cmd += ["--match-stats-file", roster]
    proc = subprocess.run(cmd, cwd=str(script.parent))
    code = proc.returncode if proc.returncode is not None else 1
    raise SystemExit(code)


def _karaoke_burn_child_main(payload: Dict[str, Any]) -> None:
    """Run ``transcribe_and_burn_karaoke`` in a spawned interpreter (isolates Whisper/GPU load).

    Module-level entry point required for ``multiprocessing`` spawn on Windows.
    """
    from pathlib import Path

    from speech_google_captions import transcribe_and_burn_karaoke

    overlay_raw = payload.get("overlay_image")
    overlay_image = Path(overlay_raw).resolve() if overlay_raw else None

    api_raw = payload.get("api_key")
    api_key = str(api_raw).strip() if api_raw else None

    transcribe_and_burn_karaoke(
        Path(payload["video_path"]),
        Path(payload["work_dir"]),
        Path(payload["video_out"]),
        str(payload["ffmpeg_bin"]),
        backend=str(payload["backend"]),
        whisper_model=str(payload["whisper_model"]),
        language_code=str(payload["language_code"]),
        timeout_sec=float(payload["timeout_sec"]),
        api_key=api_key,
        margin_v_from_top_ratio=float(payload["margin_v_from_top_ratio"]),
        overlay_image=overlay_image,
        overlay_width_frac=float(payload["overlay_width_frac"]),
        overlay_margin_bottom_px=int(payload["overlay_margin_bottom_px"]),
        encode_preset=str(payload.get("encode_preset") or "medium"),
        encode_crf=int(payload.get("encode_crf", 20)),
    )


class LiveRoundPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

        # Gemini is called via REST so the local Python environment does not need google-genai installed.
        self.client = None
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH.")

        self.root = Path(config.output_root).resolve()
        self.screen_dir = self.root / "screens"
        self.raw_dir = self.root / "round_raw"
        self.edit_dir = self.root / "round_edited"
        self.final_dir = self.root / "round_final"
        self.meta_dir = self.root / "meta"
        for p in [self.root, self.screen_dir, self.raw_dir, self.edit_dir, self.final_dir, self.meta_dir]:
            _ensure_dir(p)

        _clear_dir_contents(self.screen_dir)
        _clear_dir_contents(self.raw_dir)
        print("[live] cleared screens/ and round_raw/ for this run", flush=True)

        base_seek = float(self.cfg.stream_input_seek_sec or 0)
        if base_seek > 0:
            ih = int(base_seek // 3600)
            im = int((base_seek % 3600) // 60)
            s_rem = base_seek - ih * 3600 - im * 60
            print(
                f"[live] VOD start: ffmpeg -ss base {base_seek:.3f}s "
                f"({ih:d}:{im:02d}:{s_rem:06.3f} h:m:s from stream start)",
                flush=True,
            )

        self.demo_path = Path(config.demo_file).resolve()
        self.rules_docx_path = Path(config.rules_docx).resolve()
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {self.demo_path}")
        if not self.rules_docx_path.exists():
            raise FileNotFoundError(f"Rules DOCX not found: {self.rules_docx_path}")

        self.rules_text = _extract_docx_text(self.rules_docx_path)
        cp = (self.cfg.caption_provider or "auto").strip().lower()
        if cp == "auto":
            if (self.cfg.speech_api_key or "").strip():
                print("[live] caption_provider=auto -> Google Speech captions (API key configured)", flush=True)
            elif (self.cfg.caption_cmd_template or "").strip():
                print("[live] caption_provider=auto -> shell caption_cmd_template", flush=True)
            else:
                print(
                    "[live] caption_provider=auto -> no Speech key / no caption template (captions off)",
                    flush=True,
                )
        elif cp in ("karaoke_whisper", "karaoke_google"):
            async_note = "async (titles/post don't wait)" if self.cfg.karaoke_async else "blocking"
            print(
                f"[live] caption_provider={cp} -> karaoke via multiprocessing child "
                f"({async_note}; transcribe_and_burn_karaoke)",
                flush=True,
            )
        elif cp == "karaoke_vertex":
            async_note = "async (titles/post don't wait)" if self.cfg.karaoke_async else "blocking"
            print(
                f"[live] caption_provider=karaoke_vertex -> CAPTIONS Vertex Gemini karaoke "
                f"({async_note}; burn_karaoke_captions.py)",
                flush=True,
            )
        print("[live] loading demo context for round hints (may take a minute)...", flush=True)
        self.round_context_text = self._build_demo_context_text(self.demo_path)
        self.state_file = self.meta_dir / "round_state.json"
        self.detection_log_file = self.meta_dir / "round_detections.jsonl"

        self.current_round: Optional[int] = None
        self.record_proc: Optional[subprocess.Popen[str]] = None
        self.record_round: Optional[int] = None
        self.record_started_at: float = 0.0
        self.record_path: Optional[Path] = None
        self.record_log_path: Optional[Path] = None
        self.screenshot_proc: Optional[subprocess.Popen[str]] = None
        self.screenshot_log_path: Optional[Path] = None
        self._resolved_input_url: str = ""
        self._resolved_at: float = 0.0
        # Wall-clock anchor when stream_input_seek_sec > 0 (VOD simulated playback rate ~1x).
        self._vod_playback_anchor_monotonic: float = 0.0
        self._logged_vod_recording_re: bool = False
        # Arm first recording only after the same round is seen this many times in a row.
        self._arm_round: Optional[int] = None
        self._arm_same_count: int = 0
        # Debounce round boundary: must see the next round this many times before cutting.
        self._pending_transition_to: Optional[int] = None
        self._pending_transition_count: int = 0
        self._gemini_key_index: int = 0
        self._vertex_key_index: int = 0
        self._highlight_queue: queue.Queue[tuple[Path, int]] = queue.Queue()
        self._highlight_workers_started = 0
        self._highlight_worker_lock = threading.Lock()
        self._vision_coord_cv = threading.Condition(threading.Lock())
        self._hud_remote_calls_active = 0
        self._record_suspended_for_hud_idle = False

    @contextlib.contextmanager
    def _hud_remote_activity_scope(self):
        """Mark HUD multimodal HTTP in-flight so highlight worker yields before heavy vision calls."""
        with self._vision_coord_cv:
            self._hud_remote_calls_active += 1
        try:
            yield
        finally:
            with self._vision_coord_cv:
                self._hud_remote_calls_active -= 1
                self._vision_coord_cv.notify_all()

    def _yield_while_hud_remote_busy(self) -> None:
        """Wait until no HUD remote request is active (optional; default off for parallel highlights)."""
        if not self.cfg.highlight_yield_to_hud_vision:
            return
        while True:
            with self._vision_coord_cv:
                if self._hud_remote_calls_active <= 0:
                    return
                self._vision_coord_cv.wait(timeout=120.0)

    def _highlight_worker_loop(self) -> None:
        _deprioritize_background_thread()
        while True:
            clip_path, round_number = self._highlight_queue.get()
            try:
                self._process_completed_round(clip_path, round_number)
            except Exception as exc:
                print(f"[live] round post-process failed for round={round_number}: {exc}", flush=True)
            finally:
                self._highlight_queue.task_done()

    def _ensure_highlight_workers(self) -> None:
        target = max(1, min(8, int(self.cfg.highlight_parallel_workers)))
        with self._highlight_worker_lock:
            while self._highlight_workers_started < target:
                idx = self._highlight_workers_started
                threading.Thread(
                    target=self._highlight_worker_loop,
                    name=f"highlight-queue-worker-{idx}",
                    daemon=True,
                ).start()
                self._highlight_workers_started += 1
                print(
                    f"[live] highlight worker {idx + 1}/{target} started "
                    f"(HUD Rekognition runs on main loop; highlights run here in parallel; "
                    f"highlight_yield_to_hud_vision={self.cfg.highlight_yield_to_hud_vision})",
                    flush=True,
                )

    def _write_meta_json(self, filename: str, payload: Dict[str, Any]) -> Path:
        out = self.meta_dir / filename
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _gemini_generate_text_with_fallback(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        *,
        max_output_tokens: int = 1024,
    ) -> str:
        keys = self.cfg.gemini_api_keys or ([self.cfg.gemini_api_key] if self.cfg.gemini_api_key else [])
        if not keys:
            raise RuntimeError("No Gemini API keys configured.")

        errors: List[str] = []
        for offset in range(len(keys)):
            idx = (self._gemini_key_index + offset) % len(keys)
            key = keys[idx]
            try:
                text = _gemini_generate_text(
                    key,
                    self.cfg.gemini_model,
                    prompt,
                    image_path,
                    max_output_tokens=max_output_tokens,
                )
                if idx != self._gemini_key_index:
                    print(f"[live] Gemini API key switched to key#{idx + 1} ({_mask_secret(key)})")
                self._gemini_key_index = idx
                return text
            except Exception as exc:
                errors.append(f"key#{idx + 1}: {exc}")
                next_idx = (idx + 1) % len(keys)
                print(
                    f"[live] Gemini API key failed key#{idx + 1} ({_mask_secret(key)}); "
                    f"trying key#{next_idx + 1}"
                )
                self._gemini_key_index = next_idx

        raise RuntimeError("All Gemini API keys failed: " + " | ".join(errors))

    def _vertex_generate_text_with_fallback(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        *,
        max_output_tokens: int = 1024,
    ) -> str:
        keys = self.cfg.vertex_api_keys
        if not keys:
            raise RuntimeError("No Vertex API keys configured (vertex_api_keys / VERTEX_API_KEY).")
        pid = (self.cfg.vertex_project_id or "").strip()
        if not pid:
            raise RuntimeError("vertex_project_id is required for Vertex AI.")

        errors: List[str] = []
        model = self.cfg.gemini_model.strip()
        loc = self.cfg.vertex_location.strip() or "us-central1"
        for offset in range(len(keys)):
            idx = (self._vertex_key_index + offset) % len(keys)
            key = keys[idx]
            try:
                text = _vertex_generate_content(
                    key,
                    pid,
                    loc,
                    model,
                    prompt,
                    image_path,
                    max_output_tokens=max_output_tokens,
                )
                if idx != self._vertex_key_index:
                    print(f"[live] Vertex API key switched to key#{idx + 1} ({_mask_secret(key)})")
                self._vertex_key_index = idx
                return text
            except Exception as exc:
                errors.append(f"key#{idx + 1}: {exc}")
                next_idx = (idx + 1) % len(keys)
                print(
                    f"[live] Vertex API key failed key#{idx + 1} ({_mask_secret(key)}); "
                    f"trying key#{next_idx + 1}"
                )
                self._vertex_key_index = next_idx

        raise RuntimeError("All Vertex API keys failed: " + " | ".join(errors))

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _log_detection(self, detection: Dict[str, Any], screenshot: Path, status: str) -> None:
        self._append_jsonl(
            self.detection_log_file,
            {
                "timestamp": _now_stamp(),
                "status": status,
                "screenshot": str(screenshot),
                "current_round": self.current_round,
                "record_round": self.record_round,
                "detection": detection,
            },
        )

    def _print_ai_hud_readout(self, screenshot_full: Path, detection: Dict[str, Any]) -> None:
        """Echo score reads + derived round (matches scoreboard-sum contract)."""
        sl = detection.get("score_left")
        sr = detection.get("score_right")
        rs = detection.get("round_sum")
        derived = detection.get("scores_derived_round")
        src = detection.get("source")
        src_bit = f" {src}" if src else ""
        print(
            f"[live] AI read [{screenshot_full.name}] scores={sl}/{sr} sum={rs} round_from_scores={derived}{src_bit}",
            flush=True,
        )

    def _resolve_stream_input(self, *, force_refresh: bool = False) -> str:
        """Resolve page URLs (e.g., Twitch) to a direct playable stream URL.

        For plain media URLs (m3u8/mp4/etc.), returns the original URL.

        ``force_refresh``: bypass URL reuse cache and call streamlink again (slower; use after read errors).

        Resolved URLs are reused for :data:`STREAMLINK_RESOLVE_CACHE_SEC` wall seconds so each HUD frame
        does not pay for a new streamlink subprocess unless the cache expired or a grab failed.
        """
        src = self.cfg.stream_url.strip()
        if not src:
            raise ValueError("stream_url is empty")

        now = time.time()
        if (
            not force_refresh
            and self._resolved_input_url
            and (now - self._resolved_at) < STREAMLINK_RESOLVE_CACHE_SEC
        ):
            return self._resolved_input_url

        lower = src.lower()
        if "twitch.tv/" in lower or "youtube.com/" in lower or "youtu.be/" in lower:
            # Prioritize local portable streamlink executable (supports nested extracted layout).
            streamlink_exec: Optional[str] = None
            portable_candidates = [
                Path("./streamlink_portable/streamlink.exe"),
                Path("./streamlink_portable/streamlink-8.3.0-1-py314-x86_64/bin/streamlink.exe"),
            ]
            for candidate in portable_candidates:
                if candidate.exists():
                    streamlink_exec = str(candidate.resolve())
                    break
            if not streamlink_exec:
                streamlink_exec = shutil.which("streamlink")

            if not streamlink_exec:
                raise RuntimeError(
                    "streamlink is required for page URLs like Twitch/YouTube. "
                    "Please ensure streamlink.exe is in ./streamlink_portable/ or streamlink is installed and in PATH."
                )
            try:
                proc = subprocess.run(
                    [streamlink_exec, "--stream-url", src, "best"],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "streamlink --stream-url timed out after 45s (network, Twitch, or streamlink stuck)."
                ) from exc
            if proc.returncode != 0:
                raise RuntimeError(
                    "Failed to resolve stream URL via streamlink.\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
            resolved = (proc.stdout or "").strip().splitlines()[-1].strip()
            if not resolved:
                raise RuntimeError("streamlink returned empty stream URL.")
            self._resolved_input_url = resolved
            self._resolved_at = now
            return resolved

        self._resolved_input_url = src
        self._resolved_at = now
        return src

    def _vod_seek_anchor_elapsed_sec(self) -> float:
        """Elapsed wall time since first seek after anchor (advances virtual playhead through VOD)."""
        base = float(self.cfg.stream_input_seek_sec or 0)
        if base <= 0:
            return 0.0
        if self._vod_playback_anchor_monotonic <= 0:
            self._vod_playback_anchor_monotonic = time.monotonic()
            print(f"[live] VOD playback anchor set base_seek_sec={base:.3f}", flush=True)
        return max(0.0, time.monotonic() - self._vod_playback_anchor_monotonic)

    def _ffmpeg_seek_args_before_input(self) -> List[str]:
        """Input-side seek for ffmpeg (VOD / archive); omitted when stream_input_seek_sec is 0."""
        base = float(self.cfg.stream_input_seek_sec or 0)
        if base <= 0:
            return []
        pos = base + self._vod_seek_anchor_elapsed_sec()
        return ["-ss", f"{pos:.3f}"]

    def _ffmpeg_realtime_read_args_before_input(self) -> List[str]:
        """Throttle input demux to ~1× for **long-running** encoders so media time tracks wall clock.

        Used by the round **recorder** on VOD. Do **not** add this for one-shot ``-frames:v 1`` HUD grabs:
        with ``-ss`` into HLS/VOD, ``-re`` forces realtime demux and can take ~one GOP interval of wall
        time (often tens of seconds) before the first output frame after each new process.
        """
        if float(self.cfg.stream_input_seek_sec or 0) > 0:
            return ["-re"]
        return []

    def _build_demo_context_text(self, demo_path: Path) -> str:
        if not demo_path.exists():
            return f"Demo file not found at {demo_path}. Relying on video analysis only."
        try:
            kills_df = _load_kill_events(demo_path)
            rs_df = _load_round_start_events(demo_path)
            kills_df, _ = _normalize_kill_time_column(kills_df)
            kills_df, _ = _assign_round_numbers(kills_df, rs_df)
            rows, _ = _score_rounds(kills_df, demo_path.name)

            lines = [f"demo_file={demo_path.name}", "round_summary:"]
            for r in sorted(rows, key=lambda x: int(x.get("round", 0))):
                lines.append(
                    "round={round} score={score:.3f} kills={kills:.0f} hs={hs:.3f} max_multi={mm:.0f}".format(
                        round=int(r.get("round", 0)),
                        score=float(r.get("round_score", 0.0)),
                        kills=float(r.get("kills_total", 0.0)),
                        hs=float(r.get("headshot_ratio", 0.0)),
                        mm=float(r.get("max_multikill_by_player", 0.0)),
                    )
                )
            return "\n".join(lines)
        except ModuleNotFoundError:
            return "demoparser2 is not installed. Demo context unavailable. Relying on video analysis only."
        except Exception as e:
            return f"Error parsing demo file {demo_path}: {e}. Relying on video analysis only."

    def _hud_vision_scale_width(self) -> int:
        """Clamp ``screenshot_4k_width`` to HUD crop width for ffmpeg scale (up to 4K-wide)."""
        return max(320, min(3840, int(self.cfg.screenshot_4k_width)))

    def _build_hud_capture_vf(self, fps_interval: Optional[int]) -> str:
        """fps (optional) → crop ``round_roi_*`` → upscale to vision width → yuv420p for MJPEG."""
        rx = float(max(0.0, min(1.0, self.cfg.round_roi_x)))
        ry = float(max(0.0, min(1.0, self.cfg.round_roi_y)))
        rw = float(max(0.01, min(1.0, self.cfg.round_roi_w)))
        rh = float(max(0.01, min(1.0, self.cfg.round_roi_h)))
        tw = self._hud_vision_scale_width()
        parts: List[str] = []
        if fps_interval is not None:
            parts.append(f"fps=1/{max(1, int(fps_interval))}")
        parts.append(
            f"crop=floor(iw*{rw:.6f}):floor(ih*{rh:.6f}):floor(iw*{rx:.6f}):floor(ih*{ry:.6f})"
        )
        parts.append(f"scale={tw}:-2:flags=lanczos")
        parts.append("format=yuvj420p")
        return ",".join(parts)

    def _crop_round_hud_roi(self, image_path: Path) -> Path:
        """Optional local crop+enhance (debug). Live pipeline crops in ffmpeg instead."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        x = int(max(0.0, min(1.0, self.cfg.round_roi_x)) * w)
        y = int(max(0.0, min(1.0, self.cfg.round_roi_y)) * h)
        rw = int(max(0.01, min(1.0, self.cfg.round_roi_w)) * w)
        rh = int(max(0.01, min(1.0, self.cfg.round_roi_h)) * h)

        x2 = min(w, x + rw)
        y2 = min(h, y + rh)
        x = min(x, x2 - 1)
        y = min(y, y2 - 1)

        roi = img.crop((x, y, x2, y2))
        arr = np.asarray(roi, dtype=np.float32)
        arr = _numpy_enhance_rgb_float(
            arr,
            self.cfg.numpy_contrast_clip_percent,
            self.cfg.numpy_saturation_boost,
            self.cfg.numpy_unsharp_radius,
            self.cfg.numpy_unsharp_amount,
        )
        roi_enhanced = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        # Bilinear is much faster than LANCZOS on small HUD crops; detail is preserved after 2x upscale.
        roi_enhanced = roi_enhanced.resize(
            (roi.width * 2, roi.height * 2),
            Image.Resampling.BILINEAR,
        )

        out = image_path.with_name(f"{image_path.stem}_roi{image_path.suffix}")
        roi_enhanced.save(out, format="JPEG", quality=87, optimize=False)
        return out

    def _capture_screenshot(self) -> Path:
        out = self.screen_dir / f"screen_{_now_stamp()}.jpg"
        stream_input = self._resolve_stream_input()
        vf = self._build_hud_capture_vf(None)
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            *self._ffmpeg_seek_args_before_input(),
            "-i",
            stream_input,
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
        _run_ffmpeg(cmd)
        return out

    def _start_screenshot_capture(self) -> None:
        """Start a long-lived ffmpeg that writes JPEGs to a strftime path (legacy).

        Prefer :meth:`_grab_one_hud_jpeg_blocking` in the main loop: without ``-frames:v 1`` this mode can
        emit many frames per second until ``q`` is sent.
        """
        if self.screenshot_proc is not None and self.screenshot_proc.poll() is None:
            return

        stream_input = self._resolve_stream_input()
        base_seek = float(self.cfg.stream_input_seek_sec or 0)
        # Same anchor as recording / one-shot grabs: first start uses elapsed=0; daemon restarts seek
        # to base + wall_elapsed so we do not rewind the VOD when only the screenshot ffmpeg restarts.
        seek_prefix: List[str] = []
        if base_seek > 0:
            elapsed_wall = self._vod_seek_anchor_elapsed_sec()
            pos = base_seek + elapsed_wall
            seek_prefix = ["-ss", f"{pos:.3f}"]

        log_path = self.meta_dir / f"screenshot_capture_{_now_stamp()}.ffmpeg.log"
        out_pattern = self.screen_dir / "screen_%Y-%m-%d_%H-%M-%S.jpg"
        interval = max(1, int(self.cfg.screenshot_interval_sec))
        # Single-frame grabs are spaced in run_forever's producer (wall clock). fps= in vf would not
        # delay the first output when ffmpeg is stopped after one frame.
        vf = self._build_hud_capture_vf(None)
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            *seek_prefix,
            *self._ffmpeg_realtime_read_args_before_input(),
            "-i",
            stream_input,
            "-vf",
            vf,
            "-q:v",
            "2",
            "-strftime",
            "1",
            str(out_pattern),
        ]

        stderr_file = log_path.open("w", encoding="utf-8")
        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": stderr_file,
            "text": True,
            "stdin": subprocess.PIPE,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        try:
            self.screenshot_proc = subprocess.Popen(cmd, **popen_kwargs)
            self.screenshot_log_path = log_path
            print(
                f"[live] screenshot capture started target_spacing={interval}s_wall_clock "
                f"(HUD crop JPEG q=2 max_width={self._hud_vision_scale_width()}px)",
                flush=True,
            )
        finally:
            stderr_file.close()

    def _stop_screenshot_capture(self) -> None:
        if self.screenshot_proc is None:
            return
        try:
            if self.screenshot_proc.stdin:
                self.screenshot_proc.stdin.write("q\n")
                self.screenshot_proc.stdin.flush()
                self.screenshot_proc.stdin.close()
        except Exception:
            pass
        try:
            self.screenshot_proc.wait(timeout=10)
        except Exception:
            try:
                self.screenshot_proc.kill()
            except Exception:
                pass
            try:
                self.screenshot_proc.wait(timeout=5)
            except Exception:
                pass
        self.screenshot_proc = None

    def _grab_one_hud_jpeg_blocking(self) -> Path:
        """Run ffmpeg once and write exactly one HUD crop JPEG (no frame burst on disk).

        Tries reusing a cached streamlink URL (fast); on failure invalidates cache and retries once with a
        fresh resolve. Uses ``-rw_timeout`` so a bad/stalled read does not block for minutes.
        """
        out = self.screen_dir / f"screen_{_now_stamp()}.jpg"
        log_path = self.meta_dir / f"screenshot_one_{_now_stamp()}.ffmpeg.log"
        vf = self._build_hud_capture_vf(None)
        timeout_sec = min(120.0, max(50.0, float(LIVE_HUD_ROUND_CHECK_INTERVAL_SEC) * 12.0))

        for attempt in range(2):
            force_refresh = attempt > 0
            try:
                if attempt and out.exists():
                    try:
                        out.unlink()
                    except OSError:
                        pass
                stream_input = self._resolve_stream_input(force_refresh=force_refresh)
                if attempt > 0:
                    print("[live] HUD grab: retry with fresh streamlink URL", flush=True)

                cmd = [
                    self.ffmpeg,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-rw_timeout",
                    str(int(HUD_FFMPEG_RW_TIMEOUT_US)),
                    *self._ffmpeg_seek_args_before_input(),
                    "-i",
                    stream_input,
                    "-vf",
                    vf,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(out),
                ]
                stderr_file = log_path.open("w", encoding="utf-8")
                try:
                    proc = subprocess.run(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_file,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    raise TimeoutError(
                        f"HUD screenshot ffmpeg timed out after {timeout_sec:.0f}s; log={log_path}"
                    ) from None
                finally:
                    stderr_file.close()

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"HUD screenshot ffmpeg failed code={proc.returncode} log={log_path}"
                    )
                if not out.exists() or out.stat().st_size == 0:
                    raise RuntimeError(f"HUD screenshot missing/empty: {out} log={log_path}")
                return out
            except (TimeoutError, RuntimeError, OSError) as exc:
                self._resolved_input_url = ""
                self._resolved_at = 0.0
                if attempt >= 1:
                    raise

    _SCREEN_CAPTURE_GLOBS = ("screen_*.jpg", "screen_*.png")

    def _iter_screen_capture_paths(self) -> List[Path]:
        """Paths to HUD crop frames (JPEG from ffmpeg); legacy PNG full frames included. Excludes *_roi*."""
        found: List[Path] = []
        for pattern in LiveRoundPipeline._SCREEN_CAPTURE_GLOBS:
            found.extend(self.screen_dir.glob(pattern))
        return found

    def _latest_screen_capture_mtime(self) -> float:
        """Newest mtime among HUD crop screenshots (excludes *_roi debug crops)."""
        best = 0.0
        for p in self._iter_screen_capture_paths():
            if "_roi" in p.stem:
                continue
            try:
                best = max(best, p.stat().st_mtime)
            except OSError:
                continue
        return best

    def _wait_next_screen_capture_path(
        self,
        *,
        after_mtime: float,
        after_name: str,
        deadline: float,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Path:
        """Block until a new screenshot image appears from the ffmpeg daemon (stable size), or deadline.

        Ordering uses ``(mtime, filename)`` so a new file is not missed when mtimes collide on FAT/
        coarse clocks. ``stop_flag`` returns True to abort (producer shutdown).
        """
        stable_ok = 0
        last_key: Optional[tuple[str, int]] = None
        after_key = (after_mtime, after_name)

        def is_newer(st: Any, name: str) -> bool:
            return (st.st_mtime, name) > after_key

        while time.time() < deadline:
            if stop_flag is not None and stop_flag():
                raise InterruptedError("screenshot producer stopped")

            if self.screenshot_proc is not None:
                code = self.screenshot_proc.poll()
                if code is not None:
                    log_hint = ""
                    if getattr(self, "screenshot_log_path", None):
                        log_hint = f" log={self.screenshot_log_path}"
                    raise RuntimeError(f"screenshot ffmpeg exited early code={code}{log_hint}")

            best_p: Optional[Path] = None
            best_tuple: Tuple[float, str] = (-1.0, "")

            for p in self._iter_screen_capture_paths():
                if "_roi" in p.stem:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                if not is_newer(st, p.name):
                    continue
                cand = (st.st_mtime, p.name)
                if cand > best_tuple:
                    best_tuple = cand
                    best_p = p

            if best_p is not None:
                try:
                    sz = best_p.stat().st_size
                except OSError:
                    time.sleep(0.05)
                    continue
                key = (str(best_p.resolve()), sz)
                if key == last_key:
                    stable_ok += 1
                    if stable_ok >= 2:
                        return best_p
                else:
                    stable_ok = 0
                    last_key = key
            else:
                stable_ok = 0
                last_key = None

            time.sleep(0.05)

        log_hint = self.screenshot_log_path if self.screenshot_log_path else "(no screenshot log)"
        raise TimeoutError(
            f"No new screenshot before deadline (after={after_key}); "
            f"check stream URL and {log_hint}"
        )

    def _detect_round_from_hud_crop(self, crop_path: Path) -> Dict[str, Any]:
        """HUD scores via AWS Rekognition ``DetectText`` on the crop (``api_provider`` is always rekognition)."""
        region = (self.cfg.aws_rekognition_region or "").strip()
        if not region:
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        with self._hud_remote_activity_scope():
            return _rekognition_scores_from_crop(
                crop_path,
                self.cfg.round_detection_min_confidence,
                region,
                aws_access_key_id=self.cfg.aws_access_key_id,
                aws_secret_access_key=self.cfg.aws_secret_access_key,
                aws_session_token=self.cfg.aws_session_token,
            )

    def _detect_round_from_screenshot(self, image_path: Path) -> Dict[str, Any]:
        """Alias: ``image_path`` is the HUD crop file from ffmpeg."""
        return self._detect_round_from_hud_crop(image_path)

    def _round_from_detection(self, detection: Dict[str, Any]) -> Optional[int]:
        """Round index from broadcast scores only: ``score_left + score_right + 1``."""
        scores_round = _derived_round_from_scores(detection)
        detection["scores_derived_round"] = scores_round
        if scores_round is None:
            return None
        detection["round_number"] = scores_round
        return scores_round

    def _detection_confidence(self, detection: Dict[str, Any]) -> float:
        """Round HUD JSON omits confidence — treat missing as confident so clears pass ``round_detection_min_confidence``."""
        raw = detection.get("confidence")
        if raw is None:
            return 1.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _scores_derived_agrees_with_round(self, detection: Dict[str, Any], round_num: int) -> bool:
        """True when ``scores_derived_round`` matches ``round_num`` (same-frame consistency check)."""
        sd = detection.get("scores_derived_round")
        if sd is None:
            return False
        try:
            return int(sd) == int(round_num)
        except (TypeError, ValueError):
            return False

    def _accept_detected_round(self, round_num: int, detection: Dict[str, Any]) -> Tuple[bool, str]:
        confidence = self._detection_confidence(detection)
        if confidence < self.cfg.round_detection_min_confidence:
            return False, f"low_confidence_{confidence:.2f}"

        if self.cfg.round_started_required and not bool(detection.get("round_started", False)):
            return False, "round_not_started"

        if self.current_round is None:
            return True, "initial_round"

        if round_num == self.current_round:
            return True, "same_round"

        if round_num < self.current_round:
            return False, f"stale_or_replay_round_{round_num}"

        jump = round_num - self.current_round
        scores_agree = self._scores_derived_agrees_with_round(detection, round_num)

        if self.cfg.require_consecutive_round_increments and jump != 1:
            # OCR may skip an increment briefly; trust score-derived round when it matches ``round_num``.
            if not scores_agree:
                return False, f"non_consecutive_increase_{self.current_round}_to_{round_num}"
        if jump > self.cfg.max_round_jump:
            return False, f"round_jump_too_large_{self.current_round}_to_{round_num}"

        return True, f"new_round_{self.current_round}_to_{round_num}"

    def _start_round_recording(self, round_number: int) -> None:
        if self.record_proc is not None:
            raise RuntimeError(f"Cannot start round {round_number}; round {self.record_round} is still recording")

        out = self.raw_dir / f"round_{round_number:02d}_{_now_stamp()}.mp4"
        log_path = out.with_suffix(".ffmpeg.log")
        stream_input = self._resolve_stream_input()
        vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        rt_args = self._ffmpeg_realtime_read_args_before_input()
        if rt_args and not self._logged_vod_recording_re:
            print(
                "[live] VOD recording: enabling ffmpeg -re so recording clock matches wall time / OCR seeks "
                "(fixes multi-minute drift and wrong clip lengths).",
                flush=True,
            )
            self._logged_vod_recording_re = True
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            # Do not use -nostdin: graceful stop sends "q" on stdin; -nostdin prevents ffmpeg from reading it,
            # so shutdown waited for the full poll timeout (~45s) before SIGBREAK/kill.
            "-y",
            *self._ffmpeg_seek_args_before_input(),
            *rt_args,
            "-i",
            stream_input,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(out),
        ]

        stderr_file = log_path.open("w", encoding="utf-8")
        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": stderr_file,
            "text": True,
            "stdin": subprocess.PIPE,  # send 'q' for graceful shutdown on Windows
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        try:
            self.record_proc = subprocess.Popen(cmd, **popen_kwargs)
        finally:
            stderr_file.close()
        self.record_round = round_number
        self.record_started_at = time.time()
        self.record_path = out
        self.record_log_path = log_path
        self._record_suspended_for_hud_idle = False
        print(f"[live] recording started round={round_number} path={out}")

    def _wake_round_recorder_if_suspended(self) -> None:
        """Ensure recorder runs before signalling quit — suspended ffmpeg often ignores stdin ``q``."""
        if self.record_proc is not None and self.record_proc.poll() is None:
            _os_resume_pid(self.record_proc.pid, tag="(wake_for_stop)")
        self._record_suspended_for_hud_idle = False

    def _suspend_round_recorder_while_hud_capture_idle(self) -> None:
        """Pause round ffmpeg while HUD screenshot subprocess is stopped (Vision gap — no extra pull)."""
        if not self.cfg.record_suspend_while_hud_idle:
            return
        if self.record_proc is None or self.record_proc.poll() is not None:
            self._record_suspended_for_hud_idle = False
            return
        if self._record_suspended_for_hud_idle:
            return
        if _os_suspend_pid(self.record_proc.pid, tag="(hud_capture_idle)"):
            self._record_suspended_for_hud_idle = True
            print("[live] round recorder suspended (HUD capture paused)", flush=True)

    def _resume_round_recorder_when_hud_capture_runs(self) -> None:
        """Resume round ffmpeg when HUD screenshot subprocess starts decoding again."""
        if not self.cfg.record_suspend_while_hud_idle:
            return
        if not self._record_suspended_for_hud_idle:
            return
        if self.record_proc is None or self.record_proc.poll() is not None:
            self._record_suspended_for_hud_idle = False
            return
        if _os_resume_pid(self.record_proc.pid, tag="(before_hud_grab)"):
            self._record_suspended_for_hud_idle = False
            print("[live] round recorder resumed (HUD capture running)", flush=True)

    def _stop_round_recording(self) -> Optional[Path]:
        if not self.record_proc or not self.record_path:
            return None

        self._wake_round_recorder_if_suspended()

        # Prefer graceful ffmpeg quit so the MP4 moov atom is written (CTRL_BREAK often truncates).
        try:
            if self.record_proc.stdin:
                self.record_proc.stdin.write("q\n")
                try:
                    self.record_proc.stdin.flush()
                except Exception:
                    pass
                self.record_proc.stdin.close()
        except Exception:
            pass
        try:
            self.record_proc.wait(timeout=25)
        except KeyboardInterrupt:
            print("[live] KeyboardInterrupt during recorder shutdown — terminating ffmpeg", flush=True)
            try:
                self.record_proc.kill()
            except Exception:
                pass
            try:
                self.record_proc.wait(timeout=5)
            except Exception:
                pass
            raise
        except Exception:
            try:
                if os.name == "nt":
                    self.record_proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[name-defined]
                else:
                    self.record_proc.terminate()
            except Exception:
                pass
        try:
            if self.record_proc.poll() is None:
                self.record_proc.wait(timeout=5)
        except KeyboardInterrupt:
            print("[live] KeyboardInterrupt during recorder shutdown — terminating ffmpeg", flush=True)
            try:
                self.record_proc.kill()
            except Exception:
                pass
            try:
                self.record_proc.wait(timeout=5)
            except Exception:
                pass
            raise
        except Exception:
            pass
        if self.record_proc.poll() is None:
            try:
                self.record_proc.kill()
            except Exception:
                pass
            try:
                self.record_proc.wait(timeout=5)
            except Exception:
                pass

        clip = self.record_path
        round_num = self.record_round
        log_path = self.record_log_path
        if not clip.exists() or clip.stat().st_size == 0:
            print(f"[live] recording stopped but output is missing/empty round={round_num} path={clip}")
            clip = None
        else:
            print(f"[live] recording stopped round={round_num} path={clip}")

        self.record_proc = None
        self.record_round = None
        self.record_started_at = 0.0
        self.record_path = None
        self.record_log_path = None
        self._record_suspended_for_hud_idle = False
        if log_path:
            self._append_jsonl(
                self.meta_dir / "recording_events.jsonl",
                {
                    "timestamp": _now_stamp(),
                    "event": "recording_stopped",
                    "round": round_num,
                    "clip": str(clip) if clip else "",
                    "ffmpeg_log": str(log_path),
                },
            )
        return clip

    def _extract_clip_analysis_frames(self, clip_path: Path, round_number: int) -> List[Path]:
        frame_dir = self.meta_dir / f"clip_frames_round_{round_number:02d}_{_now_stamp()}"
        _ensure_dir(frame_dir)
        pattern = frame_dir / "frame_%03d.jpg"
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(clip_path),
            "-vf",
            "fps=1/8,scale=960:-2",
            "-frames:v",
            "10",
            "-q:v",
            "3",
            str(pattern),
        ]
        _run_ffmpeg(cmd)
        return sorted(frame_dir.glob("frame_*.jpg"))

    def _build_clip_contact_sheet(self, frames: List[Path], round_number: int) -> Path:
        selected = frames[:6]
        if not selected:
            raise RuntimeError("No frames available for contact sheet")

        thumbs: List[Image.Image] = []
        for frame in selected:
            with Image.open(frame) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((480, 270), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (480, 270), (0, 0, 0))
                canvas.paste(thumb, ((480 - thumb.width) // 2, (270 - thumb.height) // 2))
                thumbs.append(canvas)

        cols = 3
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 480, rows * 270), (0, 0, 0))
        for idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((idx % cols) * 480, (idx // cols) * 270))

        out = frames[0].parent / f"round_{round_number:02d}_contact_sheet.jpg"
        sheet.save(out, "JPEG", quality=85, optimize=True)
        return out

    def _analyze_clip_with_nvidia(self, clip_path: Path, round_number: int) -> Dict[str, Any]:
        frames = self._extract_clip_analysis_frames(clip_path, round_number)
        if not frames:
            return {
                "is_highlight": False,
                "confidence": 0.0,
                "why_highlight": [],
                "why_not_highlight": ["no_analysis_frames_extracted"],
                "final_reason": "No frames could be extracted for highlight analysis.",
            }

        contact_sheet = self._build_clip_contact_sheet(frames, round_number)
        prompt = (
            "You are judging a completed CS2 round clip from a contact sheet of sampled video frames. "
            "Use ONLY the highlight criteria in rules_context as the decision rules. "
            "Return strict JSON only: "
            '{"is_highlight": boolean, "confidence": number 0-1, "why_highlight": [string], '
            '"why_not_highlight": [string], "final_reason": string}. '
            "Mark is_highlight=true only if the clip clearly matches the rules_context. "
            "Reject normal rounds, low-action rounds, unclear footage, or clips that do not match the rules. "
            f"Round context: round {round_number}.\n\n"
            "rules_context:\n"
            f"{self.rules_text[:5000]}"
        )

        payload = {
            "model": self.cfg.nvidia_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _file_data_url(contact_sheet)}},
                    ],
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.0,
        }
        try:
            if self.cfg.highlight_yield_to_hud_vision:
                self._yield_while_hud_remote_busy()
            resp = _json_post(
                f"{self.cfg.nvidia_base_url}/chat/completions",
                self.cfg.nvidia_api_key,
                payload,
                timeout_sec=180,
            )
            return _extract_json(_chat_text(resp))
        finally:
            contact_sheet.unlink(missing_ok=True)
            for frame in frames:
                frame.unlink(missing_ok=True)
            if frames:
                frames[0].parent.rmdir()

    def _analyze_clip(self, clip_path: Path, round_number: int) -> Dict[str, Any]:
        if self.cfg.highlight_api_provider == "nvidia":
            return self._analyze_clip_with_nvidia(clip_path, round_number)

        frames = self._extract_clip_analysis_frames(clip_path, round_number)
        if not frames:
            return {
                "is_highlight": False,
                "confidence": 0.0,
                "why_highlight": [],
                "why_not_highlight": ["no_analysis_frames_extracted"],
                "final_reason": "No frames could be extracted for highlight analysis.",
            }

        contact_sheet = self._build_clip_contact_sheet(frames, round_number)
        prompt = (
            "You are judging a completed CS2 round clip from a contact sheet of sampled video frames. "
            "Use ONLY the highlight criteria in rules_context as the decision rules. "
            "Return strict JSON only: "
            '{"is_highlight": boolean, "confidence": number 0-1, "why_highlight": [string], '
            '"why_not_highlight": [string], "final_reason": string}. '
            "Mark is_highlight=true only if the clip clearly matches the rules_context. "
            "Reject normal rounds, low-action rounds, unclear footage, or clips that do not match the rules. "
            f"Round context: round {round_number}.\n\n"
            "rules_context:\n"
            f"{self.rules_text[:5000]}"
        )
        try:
            if self.cfg.highlight_yield_to_hud_vision:
                self._yield_while_hud_remote_busy()
            if self.cfg.highlight_api_provider == "vertex":
                txt = self._vertex_generate_text_with_fallback(
                    prompt, contact_sheet, max_output_tokens=8192
                )
            elif self.cfg.highlight_api_provider == "gemini":
                txt = self._gemini_generate_text_with_fallback(
                    prompt, contact_sheet, max_output_tokens=8192
                )
            else:
                raise RuntimeError(f"Unexpected highlight_api_provider: {self.cfg.highlight_api_provider}")
            return _extract_json(txt)
        finally:
            contact_sheet.unlink(missing_ok=True)
            for frame in frames:
                frame.unlink(missing_ok=True)
            if frames:
                frames[0].parent.rmdir()

    def _edit_portrait_blur(self, clip_path: Path, round_number: int) -> Path:
        out = self.edit_dir / f"round_{round_number:02d}_{_now_stamp()}_portrait.mp4"
        apply_portrait_blur(
            clip_path,
            out,
            ffmpeg_bin=self.ffmpeg,
            crf=self.cfg.portrait_blur_crf,
            preset=self.cfg.portrait_blur_preset,
            fps=30.0,
        )
        print(f"[live] portrait edit saved: {out.name}")
        return out

    def _run_caption_hook(self, edited_path: Path) -> Path:
        provider = (self.cfg.caption_provider or "auto").strip().lower()
        tpl = (self.cfg.caption_cmd_template or "").strip()
        key = (self.cfg.speech_api_key or "").strip()

        if provider == "none":
            return edited_path

        if provider in ("karaoke_whisper", "karaoke_google", "karaoke_vertex"):
            if self.cfg.karaoke_async:
                self._karaoke_start_background(edited_path)
                return edited_path
            return self._caption_with_karaoke_subprocess(edited_path)

        use_google = provider == "google_speech" or (provider == "auto" and bool(key))
        if use_google:
            return self._caption_with_google_speech(edited_path)

        use_shell = provider == "shell" or (provider == "auto" and tpl)
        if not use_shell or not tpl:
            return edited_path

        # Custom shell caption command (caption_cmd_template).
        out = self.final_dir / f"{edited_path.stem}_captioned.mp4"
        cmd = tpl.format(input=str(edited_path), output=str(out))
        hook_log = self.meta_dir / f"{edited_path.stem}_caption_hook.json"
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=max(1, self.cfg.caption_hook_timeout_sec),
            )
        except subprocess.TimeoutExpired as exc:
            hook_log.write_text(
                json.dumps(
                    {
                        "timestamp": _now_stamp(),
                        "command": cmd,
                        "input": str(edited_path),
                        "expected_output": str(out),
                        "timeout_sec": self.cfg.caption_hook_timeout_sec,
                        "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                        "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("[live] caption hook timed out, using uncaptioned video")
            return edited_path
        hook_log.write_text(
            json.dumps(
                {
                    "timestamp": _now_stamp(),
                    "command": cmd,
                    "input": str(edited_path),
                    "expected_output": str(out),
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            print(f"[live] caption hook failed, using uncaptioned video: {proc.stderr[:400]}")
            return edited_path
        if not out.exists() or out.stat().st_size == 0:
            print("[live] caption hook did not create a usable output; using uncaptioned video")
            return edited_path
        return out

    def _resolve_karaoke_overlay_image(self, edited_path: Path) -> Optional[Path]:
        """PNG/JPEG path for branding overlay, or None."""
        if self.cfg.karaoke_no_overlay:
            return None
        raw_ov = (self.cfg.karaoke_overlay_image or "").strip()
        if raw_ov:
            ov_path = Path(raw_ov)
            if not ov_path.is_absolute():
                ov_path = (self.cfg.pipeline_config_path.parent / ov_path).resolve()
            if ov_path.is_file():
                return ov_path
            print(
                f"[live] karaoke_overlay_image not found ({ov_path}); "
                "falling back to auto PNG beside clip if present",
                flush=True,
            )
        cand = edited_path.parent / "Screenshot 2026-05-01 164644.png"
        return cand if cand.is_file() else None

    def _karaoke_validate_can_run(self, edited_path: Path) -> bool:
        """Return False if karaoke_google / karaoke_vertex prerequisites are missing (Whisper path always OK)."""
        provider = (self.cfg.caption_provider or "").strip().lower()
        if provider == "karaoke_vertex":
            if not self.cfg.vertex_project_id or not self.cfg.vertex_api_keys:
                print("[live] karaoke_vertex: Vertex credentials missing; skipping karaoke task", flush=True)
                return False
            script = _captions_vertex_burn_script_path()
            if not script.is_file():
                print(f"[live] karaoke_vertex: CAPTIONS script not found ({script}); skipping karaoke task", flush=True)
                return False
            rp = (self.cfg.karaoke_vertex_roster_path or "").strip()
            if rp and not Path(rp).is_file():
                print(f"[live] karaoke_vertex: roster file missing ({rp}); skipping karaoke task", flush=True)
                return False
            return True
        backend = "whisper" if provider == "karaoke_whisper" else "google"
        if backend != "google":
            return True
        cfg_path = self.cfg.pipeline_config_path
        if not cfg_path.is_file():
            print(f"[live] karaoke_google: pipeline config missing ({cfg_path}); skipping karaoke task")
            return False
        if not self.cfg.karaoke_use_adc and not (self.cfg.speech_api_key or "").strip():
            print(
                "[live] karaoke_google: need speech_api_key (or karaoke_use_adc=true); skipping karaoke task",
                flush=True,
            )
            return False
        return True

    def _karaoke_start_background(self, edited_path: Path) -> None:
        """Run ``_caption_with_karaoke_subprocess`` on a daemon thread (does not block clip pipeline)."""
        if not self._karaoke_validate_can_run(edited_path):
            return

        stem = edited_path.stem
        pipeline = self
        expected = self.final_dir / f"{stem}_karaoke.mp4"
        pending = self.meta_dir / f"{stem}_karaoke_pending.json"
        try:
            pending.write_text(
                json.dumps(
                    {
                        "timestamp": _now_stamp(),
                        "edited_input": str(edited_path.resolve()),
                        "expected_karaoke_mp4": str(expected.resolve()),
                        "note": "Deleted when karaoke finishes; completion details in *_karaoke_subprocess.json",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

        def runner_wrap() -> None:
            try:
                pipeline._caption_with_karaoke_subprocess(edited_path)
            except Exception as exc:
                print(f"[live] karaoke background failed ({stem}): {exc}", flush=True)
            finally:
                try:
                    pending.unlink(missing_ok=True)
                except OSError:
                    pass

        threading.Thread(target=runner_wrap, name=f"karaoke-bg-{stem}", daemon=True).start()
        print(
            f"[live] karaoke burning in background → {expected.name}; "
            "titles/post use portrait copy immediately",
            flush=True,
        )

    def _caption_with_karaoke_subprocess(self, edited_path: Path) -> Path:
        """Isolate karaoke transcribe+burn in a spawned child (``multiprocessing`` spawn).

        Dispatches Whisper/Google karaoke or CAPTIONS Vertex full-video karaoke.
        """
        provider = (self.cfg.caption_provider or "").strip().lower()
        if provider == "karaoke_vertex":
            return self._caption_with_karaoke_vertex_subprocess(edited_path)

        backend = "whisper" if provider == "karaoke_whisper" else "google"

        if backend == "google":
            cfg_path = self.cfg.pipeline_config_path
            if not cfg_path.is_file():
                print(f"[live] karaoke_google: pipeline config missing ({cfg_path}); using portrait-only video")
                return edited_path
            if not self.cfg.karaoke_use_adc and not (self.cfg.speech_api_key or "").strip():
                print(
                    "[live] karaoke_google: need speech_api_key (or karaoke_use_adc=true); "
                    "using portrait-only video",
                    flush=True,
                )
                return edited_path

        out_pref = self.final_dir / f"{edited_path.stem}_karaoke.mp4"
        work = self.meta_dir / "karaoke_work"
        work.mkdir(parents=True, exist_ok=True)
        hook_log = self.meta_dir / f"{edited_path.stem}_karaoke_subprocess.json"

        overlay_resolved = self._resolve_karaoke_overlay_image(edited_path)
        speech_timeout = float(max(60, self.cfg.speech_recognition_timeout_sec))
        api_key_val: Optional[str]
        if backend == "google" and not self.cfg.karaoke_use_adc:
            api_key_val = (self.cfg.speech_api_key or "").strip() or None
        else:
            api_key_val = None

        payload: Dict[str, Any] = {
            "video_path": str(edited_path.resolve()),
            "work_dir": str(work.resolve()),
            "video_out": str(out_pref.resolve()),
            "ffmpeg_bin": self.ffmpeg or "",
            "backend": backend,
            "whisper_model": self.cfg.karaoke_whisper_model,
            "language_code": self.cfg.speech_language_code,
            "timeout_sec": speech_timeout,
            "api_key": api_key_val,
            "margin_v_from_top_ratio": float(self.cfg.karaoke_margin_top_ratio),
            "overlay_width_frac": float(self.cfg.karaoke_overlay_width_frac),
            "overlay_margin_bottom_px": int(self.cfg.karaoke_overlay_margin_bottom_px),
            "overlay_image": str(overlay_resolved.resolve()) if overlay_resolved else None,
            "encode_preset": str(self.cfg.karaoke_ffmpeg_preset),
            "encode_crf": int(self.cfg.karaoke_ffmpeg_crf),
        }

        timeout_sec = max(1, self.cfg.caption_hook_timeout_sec)
        started = time.time()

        def _pick_karaoke_output() -> Optional[Path]:
            stem = edited_path.stem
            matches = sorted(
                self.final_dir.glob(f"{stem}_karaoke*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            slack = 30.0
            for path in matches:
                try:
                    if path.stat().st_size <= 0:
                        continue
                except OSError:
                    continue
                try:
                    if path.stat().st_mtime >= started - slack:
                        return path
                except OSError:
                    continue
            return None

        def _safe_hook_payload() -> Dict[str, Any]:
            logged = dict(payload)
            ak = logged.get("api_key")
            if ak:
                logged["api_key"] = _mask_secret(str(ak))
            return logged

        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=_karaoke_burn_child_main, args=(payload,), name="karaoke-burn")
        child.start()
        child.join(timeout=float(timeout_sec))

        timed_out = child.is_alive()
        if timed_out:
            child.terminate()
            child.join(15)
            if child.is_alive():
                child.kill()
                child.join(5)

        hook_obj: Dict[str, Any] = {
            "timestamp": _now_stamp(),
            "transport": "multiprocessing_spawn",
            "backend": backend,
            "timeout_sec": timeout_sec,
            "child_alive_after_join": timed_out,
            "exitcode": child.exitcode,
            "payload": _safe_hook_payload(),
            "input": str(edited_path),
            "expected_output": str(out_pref),
        }
        hook_log.write_text(json.dumps(hook_obj, indent=2), encoding="utf-8")

        if timed_out:
            print("[live] karaoke child process timed out; using portrait-only video", flush=True)
            return edited_path

        if child.exitcode != 0:
            print(
                f"[live] karaoke child exited with code {child.exitcode}; "
                "using portrait-only video (see hook log for payload)",
                flush=True,
            )
            return edited_path

        picked = _pick_karaoke_output()
        if picked is None:
            print("[live] karaoke child finished but no output MP4 found; using portrait-only video")
            return edited_path

        print(f"[live] karaoke captions saved: {picked.name}", flush=True)
        return picked

    def _caption_with_karaoke_vertex_subprocess(self, edited_path: Path) -> Path:
        """Vertex Gemini video captions via repo ``CAPTIONS/burn_karaoke_captions.py``."""
        script = _captions_vertex_burn_script_path()
        if not script.is_file():
            print(f"[live] karaoke_vertex: missing {script}; using portrait-only video", flush=True)
            return edited_path

        roster = (self.cfg.karaoke_vertex_roster_path or "").strip()
        out_pref = self.final_dir / f"{edited_path.stem}_karaoke.mp4"
        work = self.meta_dir / "karaoke_vertex_work"
        work.mkdir(parents=True, exist_ok=True)
        hook_log = self.meta_dir / f"{edited_path.stem}_karaoke_vertex_subprocess.json"

        payload: Dict[str, Any] = {
            "captions_script": str(script.resolve()),
            "pipeline_config": str(self.cfg.pipeline_config_path.resolve()),
            "video_path": str(edited_path.resolve()),
            "work_dir": str(work.resolve()),
            "video_out": str(out_pref.resolve()),
            "karaoke_vertex_roster_path": roster,
        }

        timeout_sec = max(1, self.cfg.caption_hook_timeout_sec)
        started = time.time()

        def _pick_vertex_output() -> Optional[Path]:
            stem = edited_path.stem
            matches = sorted(
                self.final_dir.glob(f"{stem}_karaoke*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            slack = 45.0
            for path in matches:
                try:
                    if path.stat().st_size <= 0:
                        continue
                except OSError:
                    continue
                try:
                    if path.stat().st_mtime >= started - slack:
                        return path
                except OSError:
                    continue
            return None

        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=_karaoke_vertex_burn_child_main, args=(payload,), name="karaoke-vertex")
        child.start()
        child.join(timeout=float(timeout_sec))

        timed_out = child.is_alive()
        if timed_out:
            child.terminate()
            child.join(15)
            if child.is_alive():
                child.kill()
                child.join(5)

        hook_obj: Dict[str, Any] = {
            "timestamp": _now_stamp(),
            "transport": "multiprocessing_spawn",
            "backend": "karaoke_vertex",
            "captions_script": str(script),
            "timeout_sec": timeout_sec,
            "child_alive_after_join": timed_out,
            "exitcode": child.exitcode,
            "payload": payload,
            "input": str(edited_path),
            "expected_output": str(out_pref),
        }
        hook_log.write_text(json.dumps(hook_obj, indent=2), encoding="utf-8")

        if timed_out:
            print("[live] karaoke_vertex child timed out; using portrait-only video", flush=True)
            return edited_path
        if child.exitcode != 0:
            print(
                f"[live] karaoke_vertex exited with code {child.exitcode}; "
                "using portrait-only video (see hook log)",
                flush=True,
            )
            return edited_path

        picked = _pick_vertex_output()
        if picked is None:
            print("[live] karaoke_vertex finished but no karaoke MP4 found; using portrait-only video", flush=True)
            return edited_path

        print(f"[live] karaoke_vertex captions saved: {picked.name}", flush=True)
        return picked

    def _caption_with_google_speech(self, edited_path: Path) -> Path:
        out = self.final_dir / f"{edited_path.stem}_captioned.mp4"
        work = self.meta_dir / "speech_work"
        hook_log = self.meta_dir / f"{edited_path.stem}_google_speech.json"
        try:
            payload = transcribe_and_burn(
                edited_path,
                work,
                out,
                self.ffmpeg,
                language_code=self.cfg.speech_language_code,
                timeout_sec=float(max(60, self.cfg.speech_recognition_timeout_sec)),
                api_key=(self.cfg.speech_api_key or None),
            )
            hook_log.write_text(
                json.dumps(
                    {"timestamp": _now_stamp(), "input": str(edited_path), "output": str(out), **payload},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[live] Google Speech captions saved: {out.name}")
            return out
        except Exception as exc:
            hook_log.write_text(
                json.dumps(
                    {
                        "timestamp": _now_stamp(),
                        "input": str(edited_path),
                        "error": repr(exc),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[live] Google Speech captions failed, using uncaptioned video: {exc}")
            return edited_path

    def _final_video_path(self, edited_path: Path, captioned_path: Path) -> Path:
        if captioned_path != edited_path:
            return captioned_path

        out = self.final_dir / f"{edited_path.stem}_final.mp4"
        shutil.copy2(edited_path, out)
        return out

    def _generate_title_and_seo(self, analysis: Dict[str, Any], round_number: int) -> Dict[str, Any]:
        prompt = (
            "Generate an Instagram Reel title and SEO tags for this CS2 clip. Return strict JSON only: "
            '{"title": string, "caption": string, "seo_keywords": [string]}. '
            "Keep the title punchy, keep the caption natural, and include searchable CS2/esports keywords. "
            f"Round number context: {round_number}. Analysis JSON: {json.dumps(analysis)}"
        )
        fallback = {
            "title": f"CS2 Round {round_number} Highlight",
            "caption": f"CS2 round {round_number} highlight.",
            "seo_keywords": ["CS2", "CounterStrike2", "esports", "gaming", "highlight"],
        }
        # Match highlight clip analysis provider — do not force NVIDIA just because nvidia_api_key exists.
        provider = (self.cfg.highlight_api_provider or "nvidia").strip().lower()
        parsed: Dict[str, Any]
        try:
            if provider == "vertex":
                txt = self._vertex_generate_text_with_fallback(
                    prompt, None, max_output_tokens=1024
                )
                parsed = _extract_json(txt)
            elif provider == "gemini":
                txt = self._gemini_generate_text_with_fallback(
                    prompt, None, max_output_tokens=1024
                )
                parsed = _extract_json(txt)
            elif provider == "nvidia":
                nv_key = (self.cfg.nvidia_api_key or "").strip()
                if not nv_key:
                    raise RuntimeError("nvidia_api_key required when highlight_api_provider is nvidia")
                payload = {
                    "model": self.cfg.nvidia_model,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                }
                resp = _json_post(
                    f"{self.cfg.nvidia_base_url}/chat/completions",
                    nv_key,
                    payload,
                    timeout_sec=120,
                )
                parsed = _extract_json(_chat_text(resp))
            else:
                raise RuntimeError(f"Unsupported highlight_api_provider for titles: {provider}")
        except Exception as exc:
            print(f"[live] title/SEO generation failed, using fallback caption: {exc}")
            parsed = dict(fallback)

        keywords = parsed.get("seo_keywords")
        if not isinstance(keywords, list):
            parsed["seo_keywords"] = []
        return parsed

    def _post_to_instagram(self, video_path: Path, text_pack: Dict[str, Any]) -> bool:
        if not self.cfg.instagram_enabled:
            return False

        try:
            from instagrapi import Client  # type: ignore
        except Exception:
            print("[live] instagrapi not installed; skipping Instagram upload.")
            return False

        user = (self.cfg.instagram_username or "").strip()
        pwd = (self.cfg.instagram_password or "").strip()
        if not user or not pwd:
            print("[live] Instagram credentials missing; skipping upload.")
            return False

        caption = str(text_pack.get("caption", "") or "")
        keywords = text_pack.get("seo_keywords", []) if isinstance(text_pack.get("seo_keywords"), list) else []
        hashtags = []
        for keyword in keywords[:20]:
            tag = re.sub(r"[^A-Za-z0-9_]", "", str(keyword).replace(" ", ""))
            if tag:
                hashtags.append(f"#{tag}")
        suffix = "\n\n" + " ".join(hashtags) if hashtags else ""
        full_caption = (caption or str(text_pack.get("title", "") or video_path.stem)) + suffix
        full_caption = full_caption[:2200]

        try:
            ig = Client()
            ig.login(user, pwd)
            ig.clip_upload(str(video_path), caption=full_caption)
            print(f"[live] posted to Instagram: {video_path.name}")
            return True
        except Exception as exc:
            print(f"[live] Instagram upload failed: {exc}")
            return False

    def _process_completed_round(self, clip_path: Path, round_number: int) -> None:
        print(f"[live] analyzing round clip round={round_number} path={clip_path}")
        analysis = self._analyze_clip(clip_path, round_number)

        meta = {
            "round": round_number,
            "clip": str(clip_path),
            "analysis": analysis,
            "timestamp": _now_stamp(),
        }

        if not bool(analysis.get("is_highlight", False)):
            meta["status"] = "rejected_non_highlight"
            (self.meta_dir / f"round_{round_number:02d}_{_now_stamp()}_rejected.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"[live] round={round_number} rejected by highlight rules")
            return

        edited = self._edit_portrait_blur(clip_path, round_number)
        cp_live = (self.cfg.caption_provider or "auto").strip().lower()
        captioned = self._run_caption_hook(edited)
        final_video = self._final_video_path(edited, captioned)
        text_pack = self._generate_title_and_seo(analysis, round_number)
        posted = self._post_to_instagram(final_video, text_pack)

        meta_extra: Dict[str, Any] = {}
        if cp_live in ("karaoke_whisper", "karaoke_google", "karaoke_vertex") and self.cfg.karaoke_async:
            meta_extra["karaoke_async"] = True
            meta_extra["karaoke_expected_mp4"] = str((self.final_dir / f"{edited.stem}_karaoke.mp4").resolve())

        meta.update(
            {
                "status": "highlight_ready" if not posted else "posted",
                "edited_video": str(edited),
                "captioned_video": str(captioned) if captioned != edited else "",
                "final_video": str(final_video),
                "title_pack": text_pack,
                "posted": posted,
                **meta_extra,
            }
        )
        self._write_meta_json(f"round_{round_number:02d}_{_now_stamp()}_result.json", meta)
        print(f"[live] round={round_number} processed status={meta['status']}")

    def _process_completed_round_async(self, clip_path: Path, round_number: int) -> None:
        self._ensure_highlight_workers()
        self._highlight_queue.put((clip_path, round_number))
        print(
            f"[live] highlight queued round={round_number} clip={clip_path.name} "
            f"(pending={self._highlight_queue.qsize()})",
            flush=True,
        )

    def _save_state(self) -> None:
        self.state_file.write_text(
            json.dumps(
                {
                    "current_round": self.current_round,
                    "record_round": self.record_round,
                    "record_path": str(self.record_path) if self.record_path else "",
                    "arm_round": self._arm_round,
                    "arm_same_count": self._arm_same_count,
                    "pending_transition_to": self._pending_transition_to,
                    "pending_transition_count": self._pending_transition_count,
                    "updated_at": _now_stamp(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _reset_round_tracking(self) -> None:
        self.current_round = None
        self._arm_round = None
        self._arm_same_count = 0
        self._pending_transition_to = None
        self._pending_transition_count = 0

    def _process_or_save_partial_clip(self, clip: Optional[Path], round_number: Optional[int]) -> None:
        if clip is None or round_number is None:
            return
        if self.cfg.process_partial_on_max_duration:
            self._process_completed_round_async(clip, round_number)
            return

        part_path = self.raw_dir / f"{clip.stem}_exceeded_max{clip.suffix}"
        try:
            clip.rename(part_path)
        except OSError:
            part_path = clip
        self._write_meta_json(
            f"round_{round_number:02d}_{_now_stamp()}_exceeded_max.json",
            {
                "round": round_number,
                "partial_clip": str(part_path),
                "reason": "max_round_record_sec (not a full round boundary; highlight skipped)",
            },
        )
        print(f"[live] max duration: saved partial, skipped highlight: {part_path.name}")

    def _check_recording_health(self) -> None:
        if self.record_proc is not None and self.record_proc.poll() is not None:
            finished_round = self.record_round or self.current_round
            self._wake_round_recorder_if_suspended()
            finished_clip = self._stop_round_recording()
            self._reset_round_tracking()
            self._process_or_save_partial_clip(finished_clip, finished_round)

        if (
            self.cfg.max_round_record_sec > 0
            and self.record_proc
            and self.record_started_at
            and (time.time() - self.record_started_at) > self.cfg.max_round_record_sec
        ):
            finished_round = self.current_round
            print(
                f"[live] recording stopped: max_round_record_sec exceeded "
                f"({self.cfg.max_round_record_sec}s), not necessarily a round boundary "
                f"(round_label={finished_round})",
                flush=True,
            )
            finished_clip = self._stop_round_recording()
            self._process_or_save_partial_clip(finished_clip, finished_round)
            if self.current_round is not None and self.record_proc is None:
                self._start_round_recording(self.current_round)

    def _apply_detection_result(self, screenshot_full: Path, detection: Dict[str, Any]) -> None:
        self._check_recording_health()
        round_num = self._round_from_detection(detection)
        self._print_ai_hud_readout(screenshot_full, detection)
        if round_num is None:
            self._log_detection(detection, screenshot_full, "no_round")
            self._save_state()
            return

        accepted, status = self._accept_detected_round(round_num, detection)
        self._log_detection(detection, screenshot_full, status)
        if not accepted:
            self._save_state()
            return

        if self.current_round is not None and round_num == self.current_round:
            self._pending_transition_to = None
            self._pending_transition_count = 0

        if self.current_round is None:
            if self._arm_round != round_num:
                self._arm_round = round_num
                self._arm_same_count = 1
            else:
                self._arm_same_count += 1
            if self._arm_same_count >= self.cfg.stable_round_reads_to_start:
                self.current_round = round_num
                if self.record_proc is None:
                    self._start_round_recording(round_num)
        elif round_num > self.current_round:
            if self._pending_transition_to != round_num:
                self._pending_transition_to = round_num
                self._pending_transition_count = 1
            else:
                self._pending_transition_count += 1
            if self._pending_transition_count >= self.cfg.round_transition_confirmations:
                # Ignore a bogus "next round" right after arming: brief vision flicker before scores stabilize.
                boundary_floor_sec = max(12.0, float(self.cfg.min_round_record_sec))
                elapsed_record = (
                    max(0.0, time.time() - self.record_started_at)
                    if self.record_proc is not None and self.record_started_at
                    else boundary_floor_sec
                )
                if self.record_proc is not None and elapsed_record < boundary_floor_sec:
                    print(
                        f"[live] ignoring premature round boundary "
                        f"(elapsed={elapsed_record:.1f}s < floor={boundary_floor_sec:.0f}s); "
                        f"would_be_round={round_num} — likely score flicker; reset confirmations",
                        flush=True,
                    )
                    self._pending_transition_to = None
                    self._pending_transition_count = 0
                else:
                    # Score sum increased → new regulation round — cut regardless of bomb/defuse overlays.
                    finished_clip = self._stop_round_recording()
                    finished_round = self.current_round
                    self._pending_transition_to = None
                    self._pending_transition_count = 0
                    self._arm_round = None
                    self._arm_same_count = 0
                    self.current_round = round_num
                    if self.current_round is not None and self.record_proc is None:
                        self._start_round_recording(self.current_round)
                    if finished_clip is not None and finished_round is not None:
                        sl = detection.get("score_left")
                        sr = detection.get("score_right")
                        rs = detection.get("round_sum")
                        print(
                            f"[live] round boundary confirmed "
                            f"(scores={sl}/{sr} sum={rs} -> round_index={round_num}) "
                            f"finished_round={finished_round} clip={finished_clip.name}",
                            flush=True,
                        )
                        self._process_completed_round_async(finished_clip, finished_round)

        self._check_recording_health()
        self._save_state()

    def _process_detection_screenshot(self, hud_crop_path: Path) -> None:
        """Run HUD vision (Rekognition / NVIDIA / Gemini) and advance recording state (blocking)."""
        self._check_recording_health()
        detection = self._detect_round_from_hud_crop(hud_crop_path)
        self._apply_detection_result(hud_crop_path, detection)

    def run_forever(self) -> None:
        period = float(LIVE_HUD_ROUND_CHECK_INTERVAL_SEC)
        cfg_hint = int(self.cfg.screenshot_interval_sec)
        if cfg_hint != int(period):
            print(
                f"[live] note: screenshot_interval_sec={cfg_hint} in config is ignored — "
                f"HUD round checks use a fixed {period:g}s period (crop → AI → pad to {period:g}s)",
                flush=True,
            )
        print(
            f"[live] pipeline started (strict HUD: one crop every {period:g}s wall clock → Rekognition → "
            "round state; stop/split on confirmed next-round scores)",
            flush=True,
        )

        stop_requested = False
        while not stop_requested:
            try:
                while True:
                    t_iter = time.monotonic()
                    loop_start = time.time()
                    t_cap = time.time()
                    hud_path = self._grab_one_hud_jpeg_blocking()
                    t_after_cap = time.time()

                    t_vis = time.time()
                    self._process_detection_screenshot(hud_path)
                    t_after_det = time.time()

                    print(
                        f"[live] hud_crop+detect path={hud_path.name} "
                        f"capture_sec={t_after_cap - t_cap:.1f} vision_sec={t_after_det - t_vis:.1f} "
                        f"total_sec={time.time() - loop_start:.1f} "
                        f"(target_period_sec={period:g})",
                        flush=True,
                    )
                    self._check_recording_health()
                    delay = period - (time.monotonic() - t_iter)
                    if delay > 0:
                        time.sleep(delay)
            except KeyboardInterrupt:
                print("[live] stopping pipeline")
                stop_requested = True
            except Exception as exc:
                self._resolved_input_url = ""
                self._resolved_at = 0.0
                if self.record_proc is not None and self.record_started_at:
                    finished_round = self.record_round or self.current_round
                    finished_clip = self._stop_round_recording()
                    self._reset_round_tracking()
                    self._process_or_save_partial_clip(finished_clip, finished_round)
                print(f"[live] capture/Vision loop error: {exc}")
            finally:
                self._stop_screenshot_capture()

            if stop_requested:
                break
            # On Exception we reset inputs and sleep above, then continue this outer loop — inner `while True`
            # runs again. The inner loop only stops on KeyboardInterrupt (stop_requested) or a nested
            # Exception (handled here), never "normally", so we must not break after a successful run.

        try:
            self._stop_round_recording()
        except KeyboardInterrupt:
            print("[live] aborted during final recorder shutdown", flush=True)


def _resolve_speech_api_key(cfg: Dict[str, Any], config_path: Path) -> str:
    """Resolve key for Cloud Speech-to-Text captions.

    Precedence: ``GOOGLE_SPEECH_API_KEY`` env → ``speech_api_key.local.json`` beside main config
    → ``speech_api_key`` in main JSON → same key as ``vertex_api_key`` / ``vertex_api_keys[0]``
    / ``gemini_api_key`` / ``gemini_api_keys[0]`` (one GCP key for vision + screenshots + captions).
    """
    env_k = os.getenv("GOOGLE_SPEECH_API_KEY", "").strip()
    if env_k:
        return env_k
    local_file = config_path.resolve().parent / "speech_api_key.local.json"
    if local_file.exists():
        try:
            blob = json.loads(local_file.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                k = str(blob.get("speech_api_key", "") or "").strip()
                if k:
                    return k
        except (json.JSONDecodeError, OSError):
            pass
    direct = str(cfg.get("speech_api_key", "") or "").strip()
    if direct:
        return direct
    vk = str(cfg.get("vertex_api_key", "") or "").strip()
    if vk:
        return vk
    raw_v = cfg.get("vertex_api_keys", [])
    if isinstance(raw_v, list):
        for item in raw_v:
            sk = str(item).strip()
            if sk:
                return sk
    gk = str(cfg.get("gemini_api_key", "") or "").strip()
    if gk:
        return gk
    raw_g = cfg.get("gemini_api_keys", [])
    if isinstance(raw_g, list):
        for item in raw_g:
            sk = str(item).strip()
            if sk:
                return sk
    return ""


def _pipeline_builtin_json_defaults() -> Dict[str, Any]:
    """Defaults merged **before** JSON values from disk — keys present in JSON override these.

    Session: NAVI vs GamerLegion (maps Mirage / Ancient); roster body:
    ``default_match_roster_navigl.txt`` next to ``live_pipeline_config.json``.
    Change these constants when switching default VOD / seek / roster until JSON overrides them.
    """
    return {
        "stream_url": "https://www.twitch.tv/videos/2760697668",
        "stream_input_seek_hms": "10:08:25",
        "karaoke_vertex_roster_path": "default_match_roster_navigl.txt",
        "screenshot_interval_sec": 5,
        "screenshot_4k_width": 3840,
        "api_provider": "rekognition",
        "highlight_api_provider": "nvidia",
        "highlight_parallel_workers": 2,
        "highlight_yield_to_hud_vision": False,
    }


def _merge_aws_credentials_local_file(config_path: Path, cfg: Dict[str, Any]) -> None:
    """Merge ``aws_credentials.local.json`` (same directory as the pipeline JSON) into ``cfg``.

    Use this for Rekognition keys so they are not committed. File is gitignored.
    Env vars ``AWS_*`` still win when building :class:`PipelineConfig`.
    """
    p = config_path.resolve().parent / "aws_credentials.local.json"
    if not p.is_file():
        return
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[live] warning: could not read {p.name}: {exc}", flush=True)
        return
    if not isinstance(blob, dict):
        return
    merged = 0
    for k in (
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_rekognition_region",
    ):
        v = blob.get(k)
        if v is not None and str(v).strip():
            cfg[k] = str(v).strip()
            merged += 1
    if merged:
        print(f"[live] merged {merged} AWS field(s) from {p.name} (keep out of git)", flush=True)


def _load_config(path: Path) -> PipelineConfig:
    raw_cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise ValueError("pipeline config JSON must be an object")
    cfg: Dict[str, Any] = {**_pipeline_builtin_json_defaults(), **raw_cfg}
    _merge_aws_credentials_local_file(path, cfg)

    local_creds = path.resolve().parent / "aws_credentials.local.json"
    ak_merged = str(cfg.get("aws_access_key_id", "") or "").strip()
    sk_merged = str(cfg.get("aws_secret_access_key", "") or "").strip()
    ak_env = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    sk_env = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    if (not ak_env or not sk_env) and (not ak_merged or not sk_merged) and not local_creds.is_file():
        print(
            f"[live] Rekognition: no credentials and file missing: {local_creds}\n"
            f"[live] Copy aws_credentials.local.example.json → aws_credentials.local.json here and fill keys, "
            f"or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY.",
            flush=True,
        )
    raw_api = str(cfg.get("api_provider", "") or "").strip().lower()
    if raw_api and raw_api != "rekognition":
        print(
            f"[live] ignoring api_provider={raw_api!r} — HUD round detection uses AWS Rekognition only",
            flush=True,
        )
    api_provider = "rekognition"

    hl_raw = cfg.get("highlight_api_provider")
    if hl_raw is None or str(hl_raw).strip() == "":
        highlight_api_provider = "nvidia"
    else:
        highlight_api_provider = str(hl_raw).strip().lower()

    gemini_api_key_val = cfg.get("gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")
    gemini_api_keys_val: List[str] = []
    raw_gemini_keys = cfg.get("gemini_api_keys", [])
    if isinstance(raw_gemini_keys, list):
        gemini_api_keys_val.extend(str(key).strip() for key in raw_gemini_keys if str(key).strip())
    if gemini_api_key_val:
        gemini_api_keys_val.insert(0, gemini_api_key_val)
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        gemini_api_keys_val.append(env_key)
    gemini_api_keys_val = list(dict.fromkeys(gemini_api_keys_val))

    if highlight_api_provider == "gemini" and not gemini_api_keys_val:
        raise ValueError("highlight_api_provider=gemini requires gemini_api_key / gemini_api_keys in config or env")

    vertex_project_id = (
        str(cfg.get("vertex_project_id", "") or "").strip()
        or os.getenv("VERTEX_PROJECT_ID", "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    vertex_location = str(cfg.get("vertex_location", "us-central1") or "us-central1").strip()

    vertex_api_keys_val: List[str] = []
    vertex_single = str(cfg.get("vertex_api_key", "") or "").strip() or os.getenv("VERTEX_API_KEY", "").strip()
    raw_vertex_keys = cfg.get("vertex_api_keys", [])
    if isinstance(raw_vertex_keys, list):
        vertex_api_keys_val.extend(str(k).strip() for k in raw_vertex_keys if str(k).strip())
    if vertex_single:
        vertex_api_keys_val.insert(0, vertex_single)
    env_vertex = os.getenv("VERTEX_API_KEY", "").strip()
    if env_vertex:
        vertex_api_keys_val.append(env_vertex)
    vertex_api_keys_val = list(dict.fromkeys(vertex_api_keys_val))

    uses_vertex = highlight_api_provider == "vertex"
    if uses_vertex:
        if not vertex_project_id:
            raise ValueError(
                "vertex_project_id missing (set in JSON or VERTEX_PROJECT_ID / GOOGLE_CLOUD_PROJECT) "
                "when using api_provider/highlight_api_provider vertex"
            )
        if not vertex_api_keys_val:
            raise ValueError(
                "vertex_api_keys missing (set vertex_api_key / vertex_api_keys or VERTEX_API_KEY) "
                "when using Vertex AI"
            )

    seek_sec = _parse_seek_seconds(cfg.get("stream_input_seek_sec"))
    if seek_sec <= 0:
        seek_sec = _parse_seek_seconds(cfg.get("stream_input_seek_hms"))

    karaoke_vertex_roster_raw = str(cfg.get("karaoke_vertex_roster_path", "") or "").strip()
    karaoke_vertex_roster_resolved = ""
    if karaoke_vertex_roster_raw:
        rp = Path(karaoke_vertex_roster_raw)
        if not rp.is_absolute():
            rp = (path.resolve().parent / rp).resolve()
        karaoke_vertex_roster_resolved = str(rp)

    hl_workers_raw = cfg.get("highlight_parallel_workers", 2)
    try:
        highlight_parallel_workers = int(hl_workers_raw)
    except (TypeError, ValueError):
        highlight_parallel_workers = 2
    highlight_parallel_workers = max(1, min(8, highlight_parallel_workers))

    hl_yield_raw = cfg.get("highlight_yield_to_hud_vision")
    highlight_yield_to_hud_vision = False if hl_yield_raw is None else bool(hl_yield_raw)

    config = PipelineConfig(
        stream_url=cfg["stream_url"],
        api_provider=api_provider,
        highlight_api_provider=highlight_api_provider,
        highlight_parallel_workers=highlight_parallel_workers,
        highlight_yield_to_hud_vision=highlight_yield_to_hud_vision,
        gemini_api_key=gemini_api_keys_val[0] if gemini_api_keys_val else "",
        gemini_api_keys=gemini_api_keys_val,
        gemini_model=cfg.get("gemini_model", "gemini-2.5-flash"),
        nvidia_api_key=cfg.get("nvidia_api_key", "") or os.getenv("NVIDIA_API_KEY", ""),
        nvidia_base_url=cfg.get("nvidia_base_url", "https://integrate.api.nvidia.com/v1"),
        nvidia_model=cfg.get("nvidia_model", "meta/llama-3.2-90b-vision-instruct"),
        aws_rekognition_region=(
            str(cfg.get("aws_rekognition_region", "") or os.getenv("AWS_DEFAULT_REGION", "") or "").strip()
            or "us-east-1"
        ),
        aws_access_key_id=(
            (os.getenv("AWS_ACCESS_KEY_ID") or "").strip() or str(cfg.get("aws_access_key_id", "") or "").strip()
        ),
        aws_secret_access_key=(
            (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
            or str(cfg.get("aws_secret_access_key", "") or "").strip()
        ),
        aws_session_token=(
            (os.getenv("AWS_SESSION_TOKEN") or "").strip() or str(cfg.get("aws_session_token", "") or "").strip()
        ),
        demo_file=cfg["demo_file"],
        rules_docx=cfg["rules_docx"],
        output_root=cfg.get("output_root", "live_pipeline_output"),
        screenshot_interval_sec=int(cfg.get("screenshot_interval_sec", 5)),
        min_round_record_sec=int(cfg.get("min_round_record_sec", 20)),
        max_round_record_sec=int(cfg.get("max_round_record_sec", 900)),
        clip_start_offset_sec=int(cfg.get("clip_start_offset_sec", 0)),
        stream_input_seek_sec=float(seek_sec),
        round_detection_min_confidence=float(cfg.get("round_detection_min_confidence", 0.45)),
        round_started_required=bool(cfg.get("round_started_required", False)),
        max_round_jump=int(cfg.get("max_round_jump", 3)),
        stable_round_reads_to_start=int(cfg.get("stable_round_reads_to_start", 1)),
        round_transition_confirmations=int(cfg.get("round_transition_confirmations", 2)),
        require_consecutive_round_increments=bool(cfg.get("require_consecutive_round_increments", True)),
        process_partial_on_max_duration=bool(cfg.get("process_partial_on_max_duration", False)),
        record_suspend_while_hud_idle=False,
        screenshot_4k_width=int(cfg.get("screenshot_4k_width", 3840)),
        screenshot_4k_height=int(cfg.get("screenshot_4k_height", 1080)),
        numpy_contrast_clip_percent=float(cfg.get("numpy_contrast_clip_percent", 1.0)),
        numpy_saturation_boost=float(cfg.get("numpy_saturation_boost", 1.15)),
        numpy_unsharp_radius=float(cfg.get("numpy_unsharp_radius", 1.2)),
        numpy_unsharp_amount=float(cfg.get("numpy_unsharp_amount", 1.35)),
        portrait_blur_preset=str(cfg.get("portrait_blur_preset", "medium") or "medium"),
        portrait_blur_crf=int(cfg.get("portrait_blur_crf", 20)),
        round_roi_x=float(cfg.get("round_roi_x", 0.22)),
        round_roi_y=float(cfg.get("round_roi_y", 0.00)),
        round_roi_w=float(cfg.get("round_roi_w", 0.56)),
        round_roi_h=float(cfg.get("round_roi_h", 0.24)),
        caption_cmd_template=cfg.get("caption_cmd_template", ""),
        caption_hook_timeout_sec=int(cfg.get("caption_hook_timeout_sec", 900)),
        caption_provider=str(cfg.get("caption_provider", "auto")).strip().lower(),
        speech_language_code=str(cfg.get("speech_language_code", "en-US")),
        speech_recognition_timeout_sec=int(cfg.get("speech_recognition_timeout_sec", 600)),
        speech_api_key=_resolve_speech_api_key(cfg, path),
        instagram_enabled=bool(cfg.get("instagram_enabled", False)),
        instagram_username=cfg.get("instagram_username", "") or os.getenv("INSTA_USER", ""),
        instagram_password=cfg.get("instagram_password", "") or os.getenv("INSTA_PASS", ""),
        vertex_project_id=vertex_project_id,
        vertex_location=vertex_location,
        vertex_api_keys=vertex_api_keys_val,
        pipeline_config_path=path.resolve(),
        karaoke_whisper_model=str(cfg.get("karaoke_whisper_model", "small")),
        karaoke_margin_top_ratio=float(cfg.get("karaoke_margin_top_ratio", 0.22)),
        karaoke_overlay_width_frac=float(cfg.get("karaoke_overlay_width_frac", 0.52)),
        karaoke_overlay_margin_bottom_px=int(cfg.get("karaoke_overlay_margin_bottom_px", 140)),
        karaoke_use_adc=bool(cfg.get("karaoke_use_adc", False)),
        karaoke_no_overlay=bool(cfg.get("karaoke_no_overlay", False)),
        karaoke_overlay_image=str(cfg.get("karaoke_overlay_image", "") or ""),
        karaoke_async=bool(cfg.get("karaoke_async", True)),
        karaoke_ffmpeg_preset=str(cfg.get("karaoke_ffmpeg_preset", "medium") or "medium"),
        karaoke_ffmpeg_crf=int(cfg.get("karaoke_ffmpeg_crf", 20)),
        karaoke_vertex_roster_path=karaoke_vertex_roster_resolved,
    )
    if config.screenshot_interval_sec < 1:
        raise ValueError("screenshot_interval_sec must be >= 1")
    if config.min_round_record_sec < 0:
        raise ValueError("min_round_record_sec must be >= 0")
    if config.max_round_record_sec != 0 and config.max_round_record_sec <= config.min_round_record_sec:
        raise ValueError(
            "max_round_record_sec must be greater than min_round_record_sec (use 0 for no max-duration split)"
        )
    if config.stream_input_seek_sec < 0:
        raise ValueError("stream_input_seek_sec must be >= 0")
    if not (0.0 <= config.round_detection_min_confidence <= 1.0):
        raise ValueError("round_detection_min_confidence must be between 0 and 1")
    if config.max_round_jump < 1:
        raise ValueError("max_round_jump must be >= 1")
    if config.stable_round_reads_to_start < 1:
        raise ValueError("stable_round_reads_to_start must be >= 1")
    if config.round_transition_confirmations < 1:
        raise ValueError("round_transition_confirmations must be >= 1")
    if config.highlight_api_provider == "nvidia" and not (config.nvidia_api_key or "").strip():
        raise ValueError("nvidia_api_key missing in config/env when highlight_api_provider is nvidia")
    if config.api_provider != "rekognition":
        raise ValueError("api_provider must be rekognition (HUD uses AWS DetectText only)")
    if config.screenshot_4k_width < 320 or config.screenshot_4k_width > 3840:
        raise ValueError("screenshot_4k_width must be between 320 and 3840 (HUD crop scale for vision)")
    if config.screenshot_4k_height < 240:
        raise ValueError("screenshot_4k_height must be >= 240")
    for name, val in [
        ("round_roi_x", config.round_roi_x),
        ("round_roi_y", config.round_roi_y),
        ("round_roi_w", config.round_roi_w),
        ("round_roi_h", config.round_roi_h),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1")
    if config.round_roi_w <= 0.0 or config.round_roi_h <= 0.0:
        raise ValueError("round_roi_w and round_roi_h must be > 0")
    if not (10 <= config.portrait_blur_crf <= 51):
        raise ValueError("portrait_blur_crf must be between 10 and 51")
    if not (10 <= config.karaoke_ffmpeg_crf <= 51):
        raise ValueError("karaoke_ffmpeg_crf must be between 10 and 51")
    if not str(config.portrait_blur_preset or "").strip():
        raise ValueError("portrait_blur_preset must be non-empty")
    if not str(config.karaoke_ffmpeg_preset or "").strip():
        raise ValueError("karaoke_ffmpeg_preset must be non-empty")
    allowed_cp = {
        "none",
        "google_speech",
        "shell",
        "auto",
        "karaoke_whisper",
        "karaoke_google",
        "karaoke_vertex",
    }
    if config.caption_provider not in allowed_cp:
        raise ValueError(f"caption_provider must be one of {sorted(allowed_cp)}")
    return config


def _bootstrap_ssl_cert_file_from_certifi() -> None:
    """If ``SSL_CERT_FILE`` is unset, set it to ``certifi``'s PEM so HTTPS (incl. Speech) trusts public CAs.

    Does nothing when the user already exported ``SSL_CERT_FILE``. Requires ``pip install certifi``
    (listed in ``requirements.txt``).
    """
    if (os.environ.get("SSL_CERT_FILE") or "").strip():
        return
    try:
        import certifi
    except ImportError:
        print(
            "[live] certifi not installed — Speech HTTPS may fail on this Python. "
            "Run: pip install certifi",
            flush=True,
        )
        return
    ca = certifi.where()
    if ca and Path(ca).is_file():
        os.environ["SSL_CERT_FILE"] = ca
        print(f"[live] SSL_CERT_FILE set from certifi -> {ca}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live stream CS2 highlight automation pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Prompt for stream URL, VOD timestamp, match roster paste (karaoke_vertex session). "
            "Default is non-interactive: use JSON + builtin defaults only."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--caption-provider",
        default=None,
        metavar="MODE",
        help=(
            "Override caption_provider JSON field: "
            "none | google_speech | shell | auto | karaoke_whisper | karaoke_google | karaoke_vertex"
        ),
    )
    parser.add_argument(
        "--karaoke-whisper-model",
        default=None,
        metavar="MODEL",
        help="Override karaoke_whisper_model for Esports Whisper karaoke only",
    )
    parser.add_argument(
        "--karaoke-sync",
        action="store_true",
        help="Wait for karaoke burn before titles/Instagram (overrides karaoke_async in JSON)",
    )
    args = parser.parse_args()

    _bootstrap_ssl_cert_file_from_certifi()

    cfg_path = Path(args.config).resolve()
    config = _load_config(cfg_path)

    if args.interactive:
        print("", flush=True)
        print("[live] --- Interactive session ---", flush=True)
        config = _interactive_session_configure(config)

    overrides: Dict[str, Any] = {}
    if args.caption_provider is not None:
        overrides["caption_provider"] = str(args.caption_provider).strip().lower()
    if args.karaoke_whisper_model is not None:
        overrides["karaoke_whisper_model"] = str(args.karaoke_whisper_model).strip()
    if args.karaoke_sync:
        overrides["karaoke_async"] = False
    if overrides:
        config = replace(config, **overrides)
        allowed_cp = {
            "none",
            "google_speech",
            "shell",
            "auto",
            "karaoke_whisper",
            "karaoke_google",
            "karaoke_vertex",
        }
        if config.caption_provider not in allowed_cp:
            raise ValueError(f"caption_provider must be one of {sorted(allowed_cp)}")
    if config.highlight_api_provider == "gemini" and not config.gemini_api_keys:
        raise ValueError("gemini_api_key missing in config/env")

    pipeline = LiveRoundPipeline(config)
    pipeline.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
