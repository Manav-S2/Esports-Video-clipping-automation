#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import mimetypes
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
import wave
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

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
from llm_client import (
    _extract_json,
    _google_gemini_retry_delay,
    _is_retryable_urllib_failure,
    _make_ssl_context,
    _urllib_retry_delay_after_network_error,
)
from speech_google_captions import transcribe_and_burn, transcribe_google_long_wav
from video_editor import apply_portrait_blur

# Strict live HUD cadence: wall-clock period between successive crop → AI round checks.
# ``PipelineConfig.screenshot_interval_sec`` is not used for this loop (kept for compatibility / timeouts).
LIVE_HUD_ROUND_CHECK_INTERVAL_SEC = 5.0

# Vertex highlight analysis: by default **9** equipart snapshots + clip audio; optional ``highlight_vertex_audio_only``
# sends audio + ``rules_docx`` text only (no frame snapshots / contact sheet).
HIGHLIGHT_ANALYSIS_FRAME_COUNT = 9
# Inline Vertex `generateContent` payload must stay ~≤10 MiB; 180 s @ 128 kbps mono is safe for typical rounds.
HIGHLIGHT_VERTEX_INLINE_AUDIO_MAX_SEC = 180
HIGHLIGHT_ANALYSIS_AUDIO_MAX_SEC = HIGHLIGHT_VERTEX_INLINE_AUDIO_MAX_SEC

# Pre-analysis STT: Google REST syncRecognize is limited to ~1 min; we send at most this many seconds of LINEAR16.
HIGHLIGHT_PREANALYSIS_STT_MAX_SEC = 58

# Hype cues for transcript keyword pass (case-insensitive).
HIGHLIGHT_HYPE_KEYWORD_PATTERNS = [
    r"\bclutch\b",
    r"\bace\b",
    r"\binsane\b",
    r"\bcrazy\b",
    r"\bholy\b",
    r"\bwtf\b",
    r"\bclip\b",
    r"\blet'?s go\b",
    r"\bone v \d\b",
    r"\b1v\d\b",
    r"\bquad\b",
    r"\bunbelievable\b",
    r"\bwhat a play\b",
    r"\bno way\b",
    r"\brofl\b",
]

HIGHLIGHT_RULES_CONTEXT_MAX_CHARS = 8000

# Default highlight-rules Word document: ``…\Esports-Video-clipping-automation\CS2_Highlights.docx``
# (same folder as this module). Used when JSON ``rules_docx`` is missing or blank.
DEFAULT_HIGHLIGHT_RULES_DOCX = Path(__file__).resolve().parent / "CS2_Highlights.docx"

# Injected into the Vertex highlight prompt: how to weigh speech, prosody, and weak phrases vs. multi-signal fusion.
HIGHLIGHT_VERTEX_AUDIO_ANALYSIS_GUIDE = """
Audio + multimodal highlight scoring (listen to the attached clip audio; infer meaning and intensity — do not raw keyword-match):

Signal families (examples, not exhaustive):
1) Direct hype / excitement (very high weight): e.g. OH MY GOD, NO WAY, WHAT WAS THAT, ARE YOU SERIOUS, INSANE, CRAZY,
   WHAT THE HELL, BROOOOO, CLIP THAT, THAT'S A CLIP, SEND THAT, HIGHLIGHT RIGHT THERE.
2) Kill streak / clutch (high weight; combine with visuals): 1v2/1v3/1v4/1v5, CLUTCH, ACE, QUAD KILL,
   HE GOT ALL OF THEM, NO WAY HE WINS THIS, HE WINS THESE, HE'S HIM, LAST GUY, ONE MORE.
3) Skill / mechanics: ONE TAP, HEADSHOT, FLICK, INSANE FLICK, WHAT A SHOT, PIXEL, PRE-FIRE, SPRAYDOWN, TRACKING, CLEAN.
4) Big brain / surprise: OUTPLAYED, HE READ HIM, WHAT A PLAY, 200 IQ, BIG BRAIN, FAKE, BAITED, MIND GAMES.
5) Commentary-style: UNBELIEVABLE, ABSOLUTELY RIDICULOUS, YOU CAN'T WRITE THIS, THIS IS NOT REAL, HE'S DONE IT, WHAT A MOMENT.
6) Non-verbal audio (strong ML-style cues): sudden volume spike, screaming, laughter bursts, sharp pitch rise — especially if
   clustered with exciting visuals.
7) Chat / overlay if heard: CLIP, CLIP IT, WTF, HOLY, OMG, INSANE spam — useful mainly when combined with other signals.
8) Weak alone (avoid false positives): generic "nice", "good shot", "okay", "lol" — not highlight-worthy without stronger evidence.

Fusion rule: The best highlights are multi-signal in a short window. Example: hype phrase + visible multi-kill/clutch + loud reaction
→ high confidence. Example: only "nice shot" + routine single kill → is_highlight=false.

Visuals: You also get one JPEG contact sheet with exactly 9 thumbnails, evenly spaced in time across the full clip duration
(left-to-right, top-to-bottom, chronological). Use audio and visuals together; require multi-signal or clearly explosive moments —
do not mark highlights on generic praise or noise alone.
""".strip()

# Audio-only Vertex path (no images): rules_context + clip soundtrack; semantic alignment with the Word rules, not raw substring matching.
HIGHLIGHT_VERTEX_AUDIO_ONLY_GUIDE = """
Audio-only highlight scoring (listen to the attached clip audio; infer meaning and emotional intensity — align semantically
with rules_context when provided; do not treat rules_context as a naive literal keyword grep unless it explicitly demands verbatim phrases):

Signal families (examples, not exhaustive — same intent as the full multimodal guide):
1) Direct hype / excitement (very high weight): e.g. OH MY GOD, NO WAY, WHAT WAS THAT, ARE YOU SERIOUS, INSANE, CRAZY,
   WHAT THE HELL, BROOOOO, CLIP THAT, THAT'S A CLIP, SEND THAT, HIGHLIGHT RIGHT THERE.
2) Kill streak / clutch (high weight): 1v2/1v3/1v4/1v5, CLUTCH, ACE, QUAD KILL, HE GOT ALL OF THEM, NO WAY HE WINS THIS,
   HE WINS THESE, HE'S HIM, LAST GUY, ONE MORE.
3) Skill / mechanics: ONE TAP, HEADSHOT, FLICK, INSANE FLICK, WHAT A SHOT, PIXEL, PRE-FIRE, SPRAYDOWN, TRACKING, CLEAN.
4) Big brain / surprise: OUTPLAYED, HE READ HIM, WHAT A PLAY, 200 IQ, BIG BRAIN, FAKE, BAITED, MIND GAMES.
5) Commentary-style: UNBELIEVABLE, ABSOLUTELY RIDICULOUS, YOU CAN'T WRITE THIS, THIS IS NOT REAL, HE'S DONE IT, WHAT A MOMENT.
6) Non-verbal audio: sudden volume spikes, screaming, laughter bursts, sharp pitch rise — weight higher when clustered with strong speech.
7) Chat / overlay if heard: CLIP, CLIP IT, WTF, HOLY, OMG, INSANE spam — useful mainly when combined with other signals.
8) Weak alone: generic "nice", "good shot", "okay", "lol" — not highlight-worthy without stronger alignment to rules_context or explosive audio.

Fusion: Prefer clips where multiple strong audio cues arrive in a short window. Reject routine calm casts, muzak-only segments,
 unintelligible noise, or nothing that satisfies rules_context when rules_context is non-empty.

No video frames are supplied — never invent HUD, killfeed, economy, map, or specific round outcomes from visuals; ground claims in speech and prosody only. If stakes are ambiguous, reflect that in round_description and lower confidence accordingly.
""".strip()

# Avoid calling streamlink on every HUD grab; tokens usually last long enough; retry with refresh on failure.
STREAMLINK_RESOLVE_CACHE_SEC = 75.0
# Fail fast if the HLS read stalls (microseconds for ffmpeg ``-rw_timeout``).
HUD_FFMPEG_RW_TIMEOUT_US = 12_000_000


def _parse_seek_seconds(val: Any) -> float:
    """Seconds from JSON config: number, numeric string, or clock \"H:MM:SS\" / \"MM:SS\"."""
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 0.0
    if isinstance(val, int | float):
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


def _extract_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path, "r") as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    xml = re.sub(r"\n{3,}", "\n\n", xml)
    return xml.strip()


def _resolve_rules_docx_path(rules_raw: str, pipeline_config_path: Path) -> Path:
    """Resolve ``rules_docx`` for config + host OS.

    Windows-style absolute paths (``C:\\...``) are not POSIX-absolute, so Docker/Linux used to join
    them with ``/app`` and break. On non-Windows we map those to ``<config_dir>/<basename>`` (bind-mounted repo).
    """
    s = (rules_raw or "").strip()
    if not s:
        raise ValueError("rules_docx value is empty")

    cfg_parent = pipeline_config_path.resolve().parent
    candidate = Path(s)

    if PureWindowsPath(s).is_absolute():
        if sys.platform == "win32":
            return candidate.resolve()
        fallback = (cfg_parent / candidate.name).resolve()
        print(
            f"[live] rules_docx is a Windows absolute path; in this OS it is not usable as-is — "
            f"trying bind-mounted path: {fallback}",
            flush=True,
        )
        return fallback

    if candidate.is_absolute():
        return candidate.resolve()

    return (cfg_parent / candidate).resolve()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _resolve_ffprobe_bin(ffmpeg_bin: str | None) -> str | None:
    """Locate ``ffprobe`` on PATH or next to ``ffmpeg`` (Windows-friendly)."""
    w = shutil.which("ffprobe")
    if w:
        return w
    if ffmpeg_bin:
        parent = Path(ffmpeg_bin).resolve().parent
        for name in ("ffprobe.exe", "ffprobe"):
            cand = parent / name
            if cand.is_file():
                return str(cand)
    return None


def _ffprobe_duration_sec(media_path: Path, ffmpeg_bin: str | None) -> float | None:
    """Return container duration in seconds, or None if unknown."""
    exe = _resolve_ffprobe_bin(ffmpeg_bin)
    if not exe:
        return None
    cmd = [
        exe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            return None
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return None
        val = float(line[0])
        if val <= 0 or val != val:  # NaN check
            return None
        return val
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None


def _ffmpeg_demuxer_duration_sec(media_path: Path, ffmpeg_bin: str | None) -> float | None:
    """Parse ``Duration:`` from ``ffmpeg -i`` stderr (header read only; no full decode)."""
    if not ffmpeg_bin:
        return None
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-i", str(media_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    blob = (proc.stderr or "") + (proc.stdout or "")
    m = re.search(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d+)", blob)
    if not m:
        return None
    h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = m.group(4)
    try:
        sub = int(frac) / (10 ** len(frac))
    except ValueError:
        return None
    val = h * 3600 + mn * 60 + s + sub
    return val if val > 0 else None


def _clip_duration_for_analysis(media_path: Path, ffmpeg_bin: str | None) -> float | None:
    d = _ffprobe_duration_sec(media_path, ffmpeg_bin)
    if d is not None:
        return d
    return _ffmpeg_demuxer_duration_sec(media_path, ffmpeg_bin)


def _highlight_analysis_equipart_times(duration_sec: float, divisions: int) -> list[float]:
    """Timestamps at the center of ``divisions`` equal slices (full timeline coverage)."""
    dur = float(duration_sec)
    k = max(1, int(divisions))
    if dur <= 0:
        return [0.0] * k
    margin = max(0.25, min(dur * 0.02, 8.0))
    lo = margin
    hi = dur - margin
    if hi <= lo:
        lo, hi = 0.0, dur
    usable = hi - lo
    return [lo + usable * (i + 0.5) / k for i in range(k)]


def _mono16_wav_rms_timeline(wav_path: Path, window_sec: float = 0.5) -> tuple[list[float], list[float], float]:
    """Sliding-window RMS for mono int16 WAV. Returns (window_center_times_sec, rms_0_1, sample_rate_hz)."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            nch = int(wf.getnchannels())
            sw = int(wf.getsampwidth())
            fr = int(wf.getframerate())
            if sw != 2 or fr <= 0 or nch < 1:
                return [], [], float(fr)
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, OSError):
        return [], [], 0.0
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    w = max(int(float(fr) * float(window_sec)), 256)
    step = max(w // 2, 1)
    centers: list[float] = []
    rms_vals: list[float] = []
    i = 0
    while i + w <= len(audio):
        chunk = audio[i : i + w]
        rms = float(np.sqrt(np.mean(chunk * chunk)) / 32768.0)
        centers.append((i + w / 2) / float(fr))
        rms_vals.append(rms)
        i += step
    return centers, rms_vals, float(fr)


def _summarize_rms_spikes(centers: list[float], rms_vals: list[float]) -> str:
    if not rms_vals or not centers or len(rms_vals) != len(centers):
        return "rms_windows=unavailable"
    arr = np.asarray(rms_vals, dtype=np.float64)
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    std = float(np.std(arr))
    thresh = max(p95 * 0.9, med + max(2.5 * std, 1e-6))
    hits: list[str] = []
    for t, r in zip(centers, rms_vals, strict=False):
        if r >= thresh:
            hits.append(f"{t:.2f}s~{r:.3f}")
    top_i = int(np.argmax(arr))
    peak_note = f"global_peak {centers[top_i]:.2f}s~{rms_vals[top_i]:.3f}"
    if hits:
        return (
            f"median_rms={med:.4f} p95={p95:.4f} thresh~{thresh:.4f}; {peak_note}; "
            f"loud_windows({len(hits)}): {', '.join(hits[:10])}"
        )
    return f"median_rms={med:.4f} p95={p95:.4f}; {peak_note} (no windows above adaptive threshold)"


def _hype_hits_in_text(text: str) -> list[str]:
    """Return which hype regexes matched (substring snippets)."""
    if not (text or "").strip():
        return []
    low = text.lower()
    out: list[str] = []
    seen: set[str] = set()
    for pat in HIGHLIGHT_HYPE_KEYWORD_PATTERNS:
        if re.search(pat, low, re.IGNORECASE):
            key = pat.strip("\\b")
            if key not in seen:
                seen.add(key)
                out.append(pat.strip("^$")[:48])
    return out[:24]


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


def _subprocess_creationflags_low_priority() -> int:
    """Windows: start child processes (ffmpeg / CAPTIONS burn) below normal CPU priority.

    Keeps the main HUD capture thread and interactive loop more responsive under heavy encode load.
    On non-Windows, returns 0 (no extra flags).
    """
    if os.name != "nt":
        return 0
    try:
        import subprocess as sp

        return int(sp.CREATE_BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError, ValueError, TypeError):
        return 0


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


def _vertex_generate_content(
    vertex_api_key: str,
    vertex_project_id: str,
    vertex_location: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    *,
    audio_path: Path | None = None,
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

    parts: list[dict[str, Any]] = [{"text": prompt}]
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
    if audio_path is not None:
        audio_bytes = audio_path.read_bytes()
        audio_mime = mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg"
        print(
            f"[live] Vertex generateContent (highlight): attaching audio inline "
            f"{audio_path.name} {len(audio_bytes) / 1024:.1f} KiB mime={audio_mime}",
            flush=True,
        )
        parts.append(
            {
                "inline_data": {
                    "mime_type": audio_mime,
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                }
            }
        )

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
        },
    }

    ssl_ctx = _make_ssl_context()
    transient_http = frozenset({429, 500, 502, 503, 504})
    max_attempts = 8
    # Multimodal requests (large inline image and/or audio) often exceed ~120s server-side latency.
    read_timeout_sec = 180.0
    if image_path is not None:
        read_timeout_sec = max(read_timeout_sec, 240.0)
    if audio_path is not None:
        read_timeout_sec = max(read_timeout_sec, 360.0)
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
    texts: list[str] = []
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
    image_path: Path | None = None,
    *,
    max_output_tokens: int = 1024,
) -> str:
    parts: list[dict[str, Any]] = [{"text": prompt}]
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
    ssl_ctx = _make_ssl_context()
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
    texts: list[str] = []
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


def _captions_workspace_root() -> Path:
    """Parent directory that contains ``CAPTIONS/burn_karaoke_captions.py``.

    Mono-repo: ``ca/CAPTIONS`` next to ``ca/Esports-Video-clipping-automation`` (this file).
    Docker (default compose): mount that folder at ``/app/CAPTIONS`` so ``/app`` is the returned root.

    Override: ``CAPTIONS_BURN_SCRIPT`` = absolute path to ``burn_karaoke_captions.py``, or
    ``CAPTIONS_KARAOKE_ROOT`` = parent of the ``CAPTIONS`` directory.
    """
    env_script = os.environ.get("CAPTIONS_BURN_SCRIPT", "").strip()
    if env_script:
        p = Path(env_script).expanduser().resolve()
        if p.is_file():
            return p.parent.parent.resolve()

    env_root = os.environ.get("CAPTIONS_KARAOKE_ROOT", "").strip()
    if env_root:
        r = Path(env_root).expanduser().resolve()
        if (r / "CAPTIONS" / "burn_karaoke_captions.py").is_file():
            return r.resolve()

    live_dir = Path(__file__).resolve().parent
    for root in (live_dir.parent, live_dir):
        if (root / "CAPTIONS" / "burn_karaoke_captions.py").is_file():
            return root.resolve()
    # Expected layout when using docker-compose ``../CAPTIONS:/app/CAPTIONS`` (script not mounted yet).
    return live_dir.resolve()


def _captions_vertex_burn_script_path() -> Path:
    return _captions_workspace_root() / "CAPTIONS" / "burn_karaoke_captions.py"


def _esports_karaoke_burn_script_path() -> Path:
    """``burn_karaoke_captions.py`` next to this module (runs without CAPTIONS router subprocess)."""

    return Path(__file__).resolve().parent / "burn_karaoke_captions.py"


CAPTIONS_STANDALONE_OVERLAY_FALLBACK_NAME = "Screenshot 2026-05-01 164644.png"


def _captions_sidecar_live_pipeline_json(esports_pipeline_config_used_to_start: Path) -> Path:
    """Match ``CAPTIONS/burn_karaoke_captions.py`` default config choice for Vertex karaoke.

    When ``CAPTIONS/live_pipeline_config.json`` exists it wins over the Esports copy so margin, logo sizing,
    and encode settings match burns you run manually from CAPTIONS.
    """
    cap = _captions_workspace_root() / "CAPTIONS" / "live_pipeline_config.json"
    if cap.is_file():
        return cap.resolve()
    return Path(esports_pipeline_config_used_to_start).resolve()


def _safe_load_json_settings(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}

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


def _read_interactive_match_context(end_sentinel: str = "END") -> str:
    """Multiline paste (LIVE or VOD): team/player/caster notes → Gemini 2.5 Flash roster extract.

    Type ``SKIP`` alone on the first line to skip when you have no roster notes (optional).
    """
    print("", flush=True)
    print(
        "[live] Match context (optional): team names, player nicknames / in-game names, alternate spellings,",
        flush=True,
    )
    print(
        "[live]   caster names — anything useful for captions. "
        "Gemini 2.5 Flash runs first (then Vertex if needed) to build the roster file.",
        flush=True,
    )
    print(
        f"[live] No notes? Type SKIP alone on the first line. "
        f"Otherwise paste notes, then type {end_sentinel} on its own line and press Enter.",
        flush=True,
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == end_sentinel:
            break
        if not lines and line.strip().upper() == "SKIP":
            return ""
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_match_context_for_captions(
    cfg: PipelineConfig,
    raw_notes: str,
    *,
    prefer_gemini_api: bool = False,
    gemini_model_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Expand informal notes into roster text (+ structured extract) via Gemini or Vertex."""
    if not raw_notes.strip():
        return "", {}

    focus = ""
    if prefer_gemini_api:
        focus = (
            "Prioritize accurate team names and player rosters (handles, nicknames, common caster spellings).\n\n"
        )

    prompt = (
        "You extract structured esports match context from informal notes for speech-to-text captioning.\n\n"
        f"{focus}"
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
    last_api_err: BaseException | None = None
    model_g = (gemini_model_override or cfg.gemini_model or "gemini-2.5-flash").strip()

    if prefer_gemini_api:
        keys_gem = list(dict.fromkeys(k for k in (cfg.gemini_api_keys or []) if (k or "").strip()))
        for key in keys_gem:
            if (raw_response or "").strip():
                break
            try:
                raw_response = _gemini_generate_text(
                    key,
                    model_g,
                    prompt,
                    None,
                    max_output_tokens=4096,
                )
            except Exception as exc:
                last_api_err = exc
                print(f"[live] Gemini roster ({model_g}) failed: {exc}", flush=True)
        if not (raw_response or "").strip() and cfg.vertex_project_id and cfg.vertex_api_keys:
            try:
                raw_response = _vertex_generate_content(
                    cfg.vertex_api_keys[0],
                    cfg.vertex_project_id,
                    cfg.vertex_location,
                    (cfg.gemini_model or model_g).strip(),
                    prompt,
                    None,
                    max_output_tokens=4096,
                )
            except RuntimeError as exc:
                last_api_err = exc
                print(
                    "[live] Vertex roster fallback failed (often DNS/offline).",
                    flush=True,
                )

    elif cfg.vertex_project_id and cfg.vertex_api_keys:
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

    elif cfg.gemini_api_keys:
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


def _url_looks_like_twitch_vod(url: str) -> bool:
    """True for Twitch archive URLs ``.../videos/<id>`` (not a channel root page)."""
    return "twitch.tv/videos/" in (url or "").lower()


def _interactive_session_configure(cfg: PipelineConfig, *, stream_url_from_cli: bool = False) -> PipelineConfig:
    """Opt-in via ``--interactive``: LIVE vs VOD, URL, VOD timestamp if needed, optional match context.

    ``stream_url_from_cli``: ``True`` only when ``--stream-url`` was passed with a non-empty URL —
    URL prompts are skipped (same URL used for LIVE or VOD).

    Without CLI URL: **LIVE** always prompts for a URL — JSON ``stream_url`` is **not** reused (avoids
    analyzing a saved Blast VOD while choosing LIVE). **VOD** accepts Enter to keep JSON default.

    Context is optional: user may type ``SKIP`` on the first line. Roster file is still written
    (minimal placeholder when skipped). Gemini runs only when non-empty notes are pasted.
    """
    mode = _prompt_live_or_recorded()
    preset_url = (cfg.stream_url or "").strip()

    if mode == "live":
        if stream_url_from_cli and preset_url:
            url = preset_url
            print(f"[live] Stream URL (--stream-url): {url}", flush=True)
        else:
            if preset_url:
                print(
                    "[live] Config contains stream_url (often a saved VOD). "
                    "For LIVE it is ignored until you paste a **live channel** URL.",
                    flush=True,
                )
            url = _input_nonempty("[live] LIVE URL (e.g. https://www.twitch.tv/channelname): ")
    else:
        if stream_url_from_cli and preset_url:
            url = preset_url
            print(f"[live] Stream URL (--stream-url): {url}", flush=True)
        elif preset_url:
            typed = input("[live] VOD URL [Enter = use stream_url from config]: ").strip()
            url = typed if typed else preset_url
            print(f"[live] Stream URL: {url}", flush=True)
        else:
            url = _input_nonempty("[live] Paste VOD link: ")

    if not url:
        raise RuntimeError("Empty stream URL.")

    if mode == "live" and _url_looks_like_twitch_vod(url):
        print(
            "[live] WARN: URL looks like a Twitch **past broadcast** (/videos/…). "
            "Use https://www.twitch.tv/<channel> while they are live.",
            flush=True,
        )

    seek_sec = 0.0
    if mode == "vod":
        ts_raw = input(
            "[live] VOD start offset from beginning (seconds, MM:SS, H:MM:SS; blank = 0): ",
        ).strip()
        seek_sec = _parse_seek_seconds(ts_raw if ts_raw else "0")

    root = Path(cfg.output_root).resolve()
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    roster_path = meta / "session_vertex_caption_roster.txt"
    session_log = meta / "interactive_session.json"

    structured_summary: dict[str, Any] = {}
    roster_blob = ""
    raw_ctx = _read_interactive_match_context()
    if not raw_ctx.strip():
        print("[live] Match context skipped or empty; roster file will be minimal.", flush=True)
    if raw_ctx.strip():
        print(
            "[live] Calling Gemini 2.5 Flash (then Vertex fallback if needed) to extract teams & players...",
            flush=True,
        )
        roster_blob, structured_summary = _normalize_match_context_for_captions(
            cfg,
            raw_ctx,
            prefer_gemini_api=True,
            gemini_model_override="gemini-2.5-flash",
        )
    if not roster_blob.strip():
        roster_blob = "(no roster context provided)\n"

    roster_path.write_text(roster_blob, encoding="utf-8")
    roster_out = roster_path.resolve()

    session_log.write_text(
        json.dumps(
            {
                "timestamp": _now_stamp(),
                "stream_mode": mode,
                "stream_url": url,
                "stream_input_seek_sec": seek_sec,
                "roster_path": str(roster_out),
                "structured_extract_keys": list(structured_summary.keys()),
                "user_notes_chars": len(raw_ctx),
                "roster_context_skipped": not bool(raw_ctx.strip()),
                "roster_extract_pipeline": "gemini-2.5-flash-then-vertex"
                if raw_ctx.strip()
                else "skipped",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[live] Session roster: {roster_out}", flush=True)
    print(
        f"[live] Recording arms after HUD detects the same round {cfg.stable_round_reads_to_start} time(s) "
        "(JSON ``stable_round_reads_to_start``); ensure the stream shows in-game HUD.",
        flush=True,
    )
    print("[live] Interactive setup complete; starting round detection.", flush=True)

    return replace(
        cfg,
        stream_url=url,
        stream_input_seek_sec=float(max(0.0, seek_sec)),
        caption_provider="karaoke_google",
    )


def _derived_round_from_scores(detection: dict[str, Any]) -> int | None:
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
) -> dict[str, Any]:
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

    client_kwargs: dict[str, Any] = {"region_name": region_name}
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
        err_meta = exc.response.get("Error") or {}
        code = str(err_meta.get("Code") or "").strip()
        if code in ("UnrecognizedClientException", "InvalidClientTokenId"):
            raise RuntimeError(
                "AWS Rekognition refused these credentials "
                f"({code}: invalid/expired access key or session token).\n"
                "  • Long-term IAM: check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.\n"
                "  • Temporary keys: refresh AWS_SESSION_TOKEN (SSO/STS expire).\n"
                "  • Docker Compose: unset wrong host env vars or pass fresh ones — they override "
                "aws_credentials.local.json inside the merged config.\n"
                f"  • Region used: {region_name}"
            ) from exc
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
    id_map: dict[int, dict[str, Any]] = {}
    for d in detections:
        raw_id = d.get("Id")
        if raw_id is None:
            continue
        try:
            id_map[int(raw_id)] = d
        except (TypeError, ValueError):
            continue

    def _line_text_for_detection(d: dict[str, Any]) -> str:
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

    candidates: list[tuple[float, float, int, float, str]] = []  # cy, cx, val, conf, kind

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
    # Post-round highlight scoring (contact sheet → multimodal): **Vertex AI Gemini only** (loader forces ``vertex``).
    highlight_api_provider: str
    # Background threads draining ``_highlight_queue`` (Vertex highlight work).
    highlight_parallel_workers: int
    # When True, highlight multimodal waits whenever HUD Rekognition/HTTP holds ``_hud_remote_calls_active``.
    highlight_yield_to_hud_vision: bool
    # When True, Vertex post-round scoring sends clip audio + ``rules_docx`` text only — no JPEG grid / snapshots.
    highlight_vertex_audio_only: bool
    gemini_api_key: str
    gemini_api_keys: list[str]
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
    # JSON key ``rules_docx``: path (relative to this JSON file, or absolute) to highlight-rules ``.docx``.
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
    # Output frame size for portrait MP4 (smaller → faster encode; default 1080×1920).
    portrait_blur_width: int
    portrait_blur_height: int
    # Normalized ROI (0..1) on the **full broadcast frame**: HUD score strip for Rekognition.
    # Defaults match a tighter horizontal band (legacy strip was 0.22 + width 0.56): scoreboard plus two
    # square pics; trim dead space left and extra avatars / donation overlays right — adjust per broadcast layout.
    round_roi_x: float
    round_roi_y: float
    round_roi_w: float
    round_roi_h: float
    caption_cmd_template: str
    caption_hook_timeout_sec: int
    caption_provider: str  # auto | none | google_speech | shell | karaoke_google | karaoke_vertex
    speech_language_code: str
    speech_recognition_timeout_sec: int
    speech_api_key: str  # optional; prefer env GOOGLE_SPEECH_API_KEY
    instagram_enabled: bool
    instagram_username: str
    instagram_password: str
    # Gemini on Vertex AI (REST + API key): requires project id and Vertex-enabled API key.
    vertex_project_id: str
    vertex_location: str
    vertex_api_keys: list[str]
    # Absolute path to the pipeline JSON (Google Speech key resolution for karaoke_google).
    pipeline_config_path: Path
    # Karaoke ASS + optional bottom PNG overlay via isolated child process (``multiprocessing`` spawn).
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
    # Extra ``streamlink`` CLI args before ``--stream-url`` (all page URLs).
    streamlink_extra_args: list[str]
    # Prepended for twitch.tv URLs only (defaults help Docker/Twitch flakey segments).
    streamlink_twitch_extra_args: list[str]
    # Timeout seconds for ``streamlink --stream-url`` (Docker/WSL DNS can need >45s).
    streamlink_resolve_timeout_sec: int


def _vertex_karaoke_argv_from_prefs(prefs: dict[str, Any]) -> list[str]:
    """CLI flags for CAPTIONS ``burn_karaoke_captions.py`` **Vertex** mode (not Esports delegate).

    Matches margin, overlay layout, Vertex limits/GCS/roster/time-offset knobs from pipeline prefs JSON.
    """
    out: list[str] = []
    lang = str(prefs.get("speech_language_code") or "").strip()
    if lang:
        out += ["--language", lang]
    out += ["--margin-top-ratio", str(float(prefs["margin_v_from_top_ratio"]))]
    out += ["--overlay-width-frac", str(float(prefs["overlay_width_frac"]))]
    out += ["--overlay-margin-bottom", str(int(prefs["overlay_margin_bottom_px"]))]
    preset = str(prefs.get("encode_preset") or "medium").strip()
    if preset:
        out += ["--encode-preset", preset]
    out += ["--encode-crf", str(int(prefs.get("encode_crf", 20)))]
    if prefs.get("karaoke_no_overlay"):
        out.append("--no-overlay")
    else:
        ov = prefs.get("overlay_image")
        if isinstance(ov, Path) and ov.is_file():
            out += ["--overlay-image", str(ov.resolve())]
    vmx = prefs.get("karaoke_vertex_inline_video_max_mb")
    if vmx is not None:
        try:
            fv = float(vmx)
            if fv > 0:
                out += ["--vertex-inline-video-max-mb", str(fv)]
        except (TypeError, ValueError):
            pass
    if prefs.get("karaoke_vertex_send_full_video"):
        out.append("--vertex-send-full-video")
    if prefs.get("karaoke_vertex_audio_only"):
        out.append("--vertex-audio-only")
    v_au = prefs.get("vertex_audio_gcs_uri")
    if isinstance(v_au, str) and v_au.strip():
        out += ["--vertex-audio-gcs-uri", v_au.strip()]
    ko = prefs.get("karaoke_caption_time_offset_sec")
    if ko is not None:
        try:
            out += ["--karaoke-caption-time-offset-sec", str(float(ko))]
        except (TypeError, ValueError):
            pass
    if prefs.get("karaoke_vertex_invert_mux_timing_fix"):
        out.append("--vertex-invert-mux-timing")
    return out


def _karaoke_vertex_burn_child_main(payload: dict[str, Any]) -> None:
    """Spawn CAPTIONS ``burn_karaoke_captions.py`` (Vertex Gemini karaoke; video when under cap, else MP3)."""
    script = Path(payload["captions_script"])
    roster = str(payload.get("karaoke_vertex_roster_path") or "").strip()
    cmd: list[str] = [sys.executable, str(script)]
    ffmpeg_bin = str(payload.get("ffmpeg_bin") or "").strip()
    if ffmpeg_bin:
        cmd += ["--ffmpeg", ffmpeg_bin]
    cmd += [
        "--video",
        str(payload["video_path"]),
        "--output",
        str(payload["video_out"]),
        "--config",
        str(payload["pipeline_config"]),
        "--work-dir",
        str(payload["work_dir"]),
    ]
    extras = payload.get("karaoke_cli_extras")
    if isinstance(extras, list) and extras:
        cmd.extend(str(x) for x in extras)
    if roster:
        cmd += ["--match-stats-file", roster]
    low = _subprocess_creationflags_low_priority()
    proc = subprocess.run(
        cmd,
        cwd=str(script.parent),
        **({"creationflags": low} if low else {}),
    )
    code = proc.returncode if proc.returncode is not None else 1
    raise SystemExit(code)


def _karaoke_google_caps_delegate_child_main(payload: dict[str, Any]) -> None:
    """Same entry as standalone ``CAPTIONS/burn_karaoke_captions.py --delegate-esports-karaoke`` → Esports Speech burn."""

    caps_script = Path(payload["captions_script"]).resolve()
    out_mp4 = Path(payload["video_out"]).resolve()
    cmd: list[str] = [
        sys.executable,
        str(caps_script),
        "--delegate-esports-karaoke",
        "--config",
        str(Path(payload["pipeline_config"]).resolve()),
        "--video",
        str(Path(payload["video_path"]).resolve()),
        "--output",
        str(out_mp4),
        "--work-dir",
        str(out_mp4.parent),
    ]
    ffmpeg_bin = str(payload.get("ffmpeg_bin") or "").strip()
    if ffmpeg_bin:
        cmd += ["--ffmpeg", ffmpeg_bin]

    low = _subprocess_creationflags_low_priority()
    proc = subprocess.run(
        cmd,
        cwd=str(caps_script.parent),
        **({"creationflags": low} if low else {}),
    )
    code = proc.returncode if proc.returncode is not None else 1
    raise SystemExit(code)


def _karaoke_prefs_for_spawn(prefs: dict[str, Any]) -> dict[str, Any]:
    """Pickle-friendly prefs copy for multiprocessing spawn (Path → str where needed)."""
    bp = dict(prefs)
    ov = bp.get("overlay_image")
    if isinstance(ov, Path):
        bp["overlay_image"] = str(ov.resolve())
    elif ov is not None:
        bp["overlay_image"] = str(ov)
    cfg = bp.get("config_path_used")
    if isinstance(cfg, Path):
        bp["config_path_used"] = str(cfg.resolve())
    return bp


def _karaoke_google_esports_direct_child_main(payload: dict[str, Any]) -> None:
    """Call Esports ``burn_karaoke_captions.py`` when CAPTIONS router is absent (standalone clone / minimal layout)."""

    es_main = Path(payload["esports_script"]).resolve()
    prefs_raw = payload.get("burn_prefs")
    prefs: dict[str, Any] = prefs_raw if isinstance(prefs_raw, dict) else {}

    cmd: list[str] = [sys.executable, str(es_main)]

    ffmpeg_bin = str(payload.get("ffmpeg_bin") or "").strip()
    if ffmpeg_bin:
        cmd += ["--ffmpeg", ffmpeg_bin]

    cmd += [
        "--config",
        str(Path(payload["pipeline_config"]).resolve()),
        "--video",
        str(Path(payload["video_path"]).resolve()),
        "--output",
        str(Path(payload["video_out"]).resolve()),
        "--work-dir",
        str(Path(payload["work_dir"]).resolve()),
    ]

    lang = str(prefs.get("speech_language_code", "") or "").strip()
    if lang:
        cmd += ["--language", lang]

    try:
        cmd += ["--margin-top-ratio", str(float(prefs.get("margin_v_from_top_ratio", 0.22)))]
        cmd += ["--overlay-width-frac", str(float(prefs.get("overlay_width_frac", 0.52)))]
        cmd += ["--overlay-margin-bottom", str(int(prefs.get("overlay_margin_bottom_px", 140)))]
    except (TypeError, ValueError):
        cmd += ["--margin-top-ratio", "0.22", "--overlay-width-frac", "0.52", "--overlay-margin-bottom", "140"]

    preset = str(prefs.get("encode_preset", "medium") or "medium").strip()
    cmd += ["--encode-preset", preset]
    try:
        crf = int(prefs.get("encode_crf", 20))
    except (TypeError, ValueError):
        crf = 20
    cmd += ["--encode-crf", str(max(10, min(51, crf)))]

    if prefs.get("karaoke_use_adc"):
        cmd.append("--use-adc")
    if prefs.get("karaoke_no_overlay"):
        cmd.append("--no-overlay")
    else:
        raw_ov = str(prefs.get("overlay_image") or "").strip()
        if raw_ov:
            ov_path = Path(raw_ov)
            if ov_path.is_file():
                cmd += ["--overlay-image", str(ov_path.resolve())]

    if prefs.get("karaoke_vertex_invert_mux_timing_fix"):
        cmd.append("--karaoke-invert-mux-timing")

    low = _subprocess_creationflags_low_priority()
    proc = subprocess.run(
        cmd,
        cwd=str(es_main.parent),
        **({"creationflags": low} if low else {}),
    )
    code = proc.returncode if proc.returncode is not None else 1
    raise SystemExit(code)


class LiveRoundPipeline:
    def __init__(self, config: PipelineConfig, *, init_mode: str = "live"):
        if init_mode not in ("live", "captions_batch"):
            raise ValueError(f"init_mode must be 'live' or 'captions_batch', not {init_mode!r}")

        self.cfg = config

        # Gemini is called via REST so the local Python environment does not need google-genai installed.
        self.client = None
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH.")

        if init_mode == "captions_batch":
            self._init_captions_batch_mode()
            return

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
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {self.demo_path}")

        rules_raw = (config.rules_docx or "").strip()
        self.rules_text = ""
        if rules_raw:
            rp = _resolve_rules_docx_path(rules_raw, config.pipeline_config_path)
            self.rules_docx_path = rp
            if not self.rules_docx_path.is_file():
                raise FileNotFoundError(f"Rules DOCX not found: {self.rules_docx_path}")
            self.rules_text = _extract_docx_text(self.rules_docx_path)
            print(
                f"[live] highlight rules_docx: {self.rules_docx_path.name} "
                f"({len(self.rules_text)} chars → rules_context in Vertex prompt)",
                flush=True,
            )
        else:
            self.rules_docx_path = None
            print("[live] highlight rules_docx: (none) — Vertex judge uses frames + generic criteria only", flush=True)

        cp_raw = (self.cfg.caption_provider or "auto").strip().lower()
        if cp_raw == "auto":
            eff = self._resolve_caption_provider()
            if eff == "shell":
                print("[live] caption_provider=auto -> shell caption_cmd_template", flush=True)
            elif eff == "google_speech":
                print("[live] caption_provider=auto -> Google Speech captions (simple burn API)", flush=True)
            elif eff == "karaoke_google":
                async_note = "async (titles/post don't wait)" if self.cfg.karaoke_async else "blocking"
                print(
                    f"[live] caption_provider=auto -> karaoke_google ({async_note}; Cloud Speech-to-Text karaoke)",
                    flush=True,
                )
            else:
                print(
                    "[live] caption_provider=auto -> captions off "
                    "(need Speech API key or karaoke_use_adc; CAPTIONS router optional if Esports burn script present)",
                    flush=True,
                )
        elif cp_raw == "none":
            print("[live] caption_provider=none (captions disabled)", flush=True)
        elif cp_raw == "karaoke_google":
            async_note = "async (titles/post don't wait)" if self.cfg.karaoke_async else "blocking"
            print(
                f"[live] caption_provider=karaoke_google -> CAPTIONS burn_karaoke_captions.py --delegate-esports-karaoke "
                f"→ Esports Speech burn ({async_note})",
                flush=True,
            )
        elif cp_raw == "karaoke_vertex":
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

        self.current_round: int | None = None
        self.record_proc: subprocess.Popen[str] | None = None
        self.record_round: int | None = None
        self.record_started_at: float = 0.0
        self.record_path: Path | None = None
        self.record_log_path: Path | None = None
        self.screenshot_proc: subprocess.Popen[str] | None = None
        self.screenshot_log_path: Path | None = None
        self._resolved_input_url: str = ""
        self._resolved_at: float = 0.0
        # Wall-clock anchor when stream_input_seek_sec > 0 (VOD simulated playback rate ~1x).
        self._vod_playback_anchor_monotonic: float = 0.0
        self._logged_vod_recording_re: bool = False
        # Arm first recording only after the same round is seen this many times in a row.
        self._arm_round: int | None = None
        self._arm_same_count: int = 0
        # Debounce round boundary: must see the next round this many times before cutting.
        self._pending_transition_to: int | None = None
        self._pending_transition_count: int = 0
        self._gemini_key_index: int = 0
        self._vertex_key_index: int = 0
        self._highlight_queue: queue.Queue[tuple[Path, int]] = queue.Queue()
        self._highlight_workers_started = 0
        self._highlight_worker_lock = threading.Lock()
        self._vision_coord_cv = threading.Condition(threading.Lock())
        self._hud_remote_calls_active = 0
        self._record_suspended_for_hud_idle = False
        self._vision_auth_error_last_emit_monotonic: float = 0.0

    def _init_captions_batch_mode(self) -> None:
        """Minimal dirs + worker state for ``--captions-batch-only`` (no capture, HUD, demo, highlights)."""
        self.root = Path(self.cfg.output_root).resolve()
        self.screen_dir = self.root / "screens"
        self.raw_dir = self.root / "round_raw"
        self.edit_dir = self.root / "round_edited"
        self.final_dir = self.root / "round_final"
        self.meta_dir = self.root / "meta"
        for p in (self.root, self.edit_dir, self.final_dir, self.meta_dir):
            _ensure_dir(p)

        self.demo_path = Path(self.cfg.demo_file).expanduser().resolve()
        self.rules_docx_path = None
        self.rules_text = ""
        self.round_context_text = ""

        print(
            "[live] captions-batch-only: skipping live stream, Rekognition HUD, demo/rules load, highlights; "
            f"reading {self.edit_dir} (+ optional ``round_final/*_portrait_final.mp4`` staged); writing karaoke + *_final.mp4 under {self.final_dir}",
            flush=True,
        )
        prov_resolved = self._resolve_caption_provider()
        vtx_script = _captions_vertex_burn_script_path()
        print(
            f"[live] captions-batch resolved_provider={prov_resolved}; "
            f"CAPTIONS burn script exists={vtx_script.is_file()} → {vtx_script}",
            flush=True,
        )

        self.state_file = self.meta_dir / "round_state.json"
        self.detection_log_file = self.meta_dir / "round_detections.jsonl"

        self.current_round: int | None = None
        self.record_proc: subprocess.Popen[str] | None = None
        self.record_round: int | None = None
        self.record_started_at: float = 0.0
        self.record_path: Path | None = None
        self.record_log_path: Path | None = None
        self.screenshot_proc: subprocess.Popen[str] | None = None
        self.screenshot_log_path: Path | None = None
        self._resolved_input_url: str = ""
        self._resolved_at: float = 0.0
        self._vod_playback_anchor_monotonic: float = 0.0
        self._logged_vod_recording_re: bool = False
        self._arm_round: int | None = None
        self._arm_same_count: int = 0
        self._pending_transition_to: int | None = None
        self._pending_transition_count: int = 0
        self._gemini_key_index: int = 0
        self._vertex_key_index: int = 0
        self._highlight_queue: queue.Queue[tuple[Path, int]] = queue.Queue()
        self._highlight_workers_started = 0
        self._highlight_worker_lock = threading.Lock()
        self._vision_coord_cv = threading.Condition(threading.Lock())
        self._hud_remote_calls_active = 0
        self._record_suspended_for_hud_idle = False
        self._vision_auth_error_last_emit_monotonic: float = 0.0

    @staticmethod
    def _caption_batch_dedupe_key(stem: str) -> str:
        """Normalize filename stem so ``foo_portrait`` and ``foo_portrait_final`` dedupe."""
        stem_l = stem.lower()
        suf = "_final"
        if stem_l.endswith(suf):
            stem = stem[: len(stem) - len(suf)]
        return stem.lower()

    def _caption_batch_repair_mp4_inplace(self, path: Path, *, source_hint: str) -> None:
        """Try stream-copy remux when ffprobe cannot read a staged MP4 (e.g. missing moov due to truncate)."""
        if _ffprobe_duration_sec(path, self.ffmpeg) is not None:
            return
        ffmpeg_bin = (self.ffmpeg or "").strip() or shutil.which("ffmpeg") or ""
        if not ffmpeg_bin:
            print(
                f"[live] captions-batch WARN: ffprobe failed on {path.name} and ffmpeg not found; "
                f"source={source_hint}",
                flush=True,
            )
            return
        tmp = path.parent / f"{path.stem}__moovrepair{path.suffix}"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-i",
            str(path),
            "-map",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        except (OSError, subprocess.TimeoutExpired) as exc:
            tmp.unlink(missing_ok=True)
            print(
                f"[live] captions-batch WARN: remux aborted for staged {path.name}: {exc}; "
                f"source={source_hint}",
                flush=True,
            )
            return
        if proc.returncode != 0:
            tail = ((proc.stderr or "") + (proc.stdout or ""))[-800:]
            tmp.unlink(missing_ok=True)
            print(
                f"[live] captions-batch WARN: remux failed (code {proc.returncode}) for {path.name}; "
                f"source={source_hint}\n{tail}",
                flush=True,
            )
            return
        if _ffprobe_duration_sec(tmp, self.ffmpeg) is None:
            tmp.unlink(missing_ok=True)
            print(
                f"[live] captions-batch WARN: ffprobe still failed after remux for {path.name}; "
                f"source={source_hint}",
                flush=True,
            )
            return
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            print(
                f"[live] captions-batch WARN: could not unlink broken {path.name}: {exc}; "
                f"source={source_hint}",
                flush=True,
            )
            return
        try:
            tmp.rename(path)
        except OSError:
            try:
                shutil.move(str(tmp), str(path))
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                print(
                    f"[live] captions-batch WARN: could not move remux to {path.name}: {exc}; "
                    f"source={source_hint}",
                    flush=True,
                )
                return
        print(
            f"[live] captions-batch: remux repaired unreadable staged file → {path.name} "
            f"(from {source_hint})",
            flush=True,
        )

    def _caption_batch_gather_inputs(self, sources: frozenset[str]) -> tuple[list[Path], list[Path]]:
        """Build ordered caption inputs and ephemeral staged files to delete afterward.

        * ``edited`` — ``*_portrait.mp4`` plus other ``*.mp4`` excluding karaoke/captioned/final artefacts.
        * ``final`` — ``*_portrait_final.mp4`` portrait-only finals copied under ``meta/captions_batch_stage/``
          with stem ``*_portrait`` so karaoke output names match ``live`` runs.

        Duplicate logical clips dedupe ``edited`` over ``final``.
        """
        work: list[Path] = []
        staged_cleanup: list[Path] = []
        seen_keys: set[str] = set()

        def try_add(path: Path) -> None:
            key = self._caption_batch_dedupe_key(path.stem)
            if key in seen_keys:
                return
            seen_keys.add(key)
            work.append(path)

        if "edited" in sources and self.edit_dir.is_dir():
            primary = sorted(self.edit_dir.glob("*_portrait.mp4"))
            resolved_already: set[Path] = set()
            for p in primary:
                self._caption_batch_repair_mp4_inplace(p, source_hint=p.name)
                if _ffprobe_duration_sec(p, self.ffmpeg) is None:
                    print(
                        f"[live] captions-batch: skip {p.name} in round_edited — unreadable MP4 (fix or replace).",
                        flush=True,
                    )
                    continue
                try_add(p)
                resolved_already.add(p.resolve())
            for p in sorted(self.edit_dir.glob("*.mp4")):
                st_l = p.stem.lower()
                if "_karaoke" in st_l or "captioned" in st_l:
                    continue
                if st_l.endswith("_final"):
                    continue
                try:
                    if p.resolve() in resolved_already:
                        continue
                except OSError:
                    continue
                self._caption_batch_repair_mp4_inplace(p, source_hint=p.name)
                if _ffprobe_duration_sec(p, self.ffmpeg) is None:
                    print(
                        f"[live] captions-batch: skip {p.name} in round_edited — unreadable MP4 (fix or replace).",
                        flush=True,
                    )
                    continue
                try_add(p)
                resolved_already.add(p.resolve())

        if "final" in sources and self.final_dir.is_dir():
            stage_root = self.meta_dir / "captions_batch_stage"
            stage_root.mkdir(parents=True, exist_ok=True)
            for fp in sorted(self.final_dir.glob("*_portrait_final.mp4")):
                st = fp.stem
                st_l = st.lower()
                if not st_l.endswith("_portrait_final"):
                    continue
                base = st[: len(st) - len("_final")]
                key = base.lower()
                if key in seen_keys:
                    continue
                staged = stage_root / f"{base}.mp4"
                try:
                    shutil.copy2(fp, staged)
                    now = time.time()
                    os.utime(staged, (now, now))
                    self._caption_batch_repair_mp4_inplace(staged, source_hint=fp.name)
                    if _ffprobe_duration_sec(staged, self.ffmpeg) is None:
                        print(
                            f"[live] captions-batch: skip {fp.name} — MP4 unreadable (moov missing / corrupt). "
                            f"Fix or replace source file:\n  {fp.resolve()}",
                            flush=True,
                        )
                        try:
                            staged.unlink(missing_ok=True)
                        except OSError:
                            pass
                        continue
                    seen_keys.add(key)
                    staged_cleanup.append(staged)
                    work.append(staged)
                except OSError as exc:
                    print(f"[live] captions-batch: could not stage {fp.name}: {exc}", flush=True)

        return work, staged_cleanup

    def _captions_batch_should_skip_processed(self, portrait_input: Path, *, redo_all: bool) -> bool:
        """Return True only when ``round_final/{{stem}}_karaoke.mp4`` already exists and is fresh enough.

        Comparing ``*_final.mp4`` timestamps is incorrect: portrait-only finals often match/copy mtime back
        from ``*_portrait_final`` staging and would skip karaoke forever without this check.
        """
        if redo_all:
            return False
        stem = portrait_input.stem
        karaoke_out = self.final_dir / f"{stem}_karaoke.mp4"
        if not karaoke_out.is_file():
            return False
        try:
            if karaoke_out.stat().st_size < 4096:
                return False
        except OSError:
            return False
        try:
            return karaoke_out.stat().st_mtime >= portrait_input.stat().st_mtime
        except OSError:
            return False

    def run_captions_batch_on_round_edited(
        self,
        *,
        redo_all: bool = False,
        sources: frozenset[str] | None = None,
    ) -> int:
        """Burn captions onto gathered portrait-ish MP4 sources; deliver ``*_final.mp4`` under ``round_final``.

        Uses the configured ``caption_provider`` (default ``karaoke_google``: Cloud Speech-to-Text karaoke burn).
        Legacy ``karaoke_vertex`` uses Gemini on Vertex instead of Speech — opt-in only.

        Sources (see ``--captions-batch-sources``): ``edited`` (``round_edited``) and optionally ``final``
        (``round_final`` ``*_portrait_final.mp4`` staged with a ``*_portrait.mp4`` filename).
        """
        src = sources if sources is not None else frozenset({"edited", "final"})
        work_paths, staged_cleanup = self._caption_batch_gather_inputs(src)

        try:
            if not work_paths:
                print(
                    f"[live] captions-batch: no MP4 inputs (sources={sorted(src)}).\n"
                    f"       round_edited: {self.edit_dir}\n"
                    f"       round_final:  {self.final_dir}\n"
                    f"       Expect ``*_portrait.mp4`` in round_edited, or ``*_portrait_final.mp4`` in round_final; "
                    f"or pass ``--captions-batch-sources edited`` / ``final`` explicitly.",
                    flush=True,
                )
                return 1

            provider = self._resolve_caption_provider()
            if provider == "none":
                raise RuntimeError(
                    'caption_provider is "none"; set karaoke_google (recommended), google_speech, karaoke_vertex '
                    "(legacy), or shell in JSON / --caption-provider."
                )

            print(
                f"[live] captions-batch: {len(work_paths)} clip(s); sources={sorted(src)}; provider={provider}; "
                f"redo_all={redo_all}",
                flush=True,
            )

            ok = skipped = failures = 0
            outcomes: list[dict[str, Any]] = []

            for edited in work_paths:
                if self._captions_batch_should_skip_processed(edited, redo_all=redo_all):
                    print(
                        f"[live] captions-batch: skip (karaoke MP4 exists and newer than input): "
                        f"{(self.final_dir / f'{edited.stem}_karaoke.mp4').name}",
                        flush=True,
                    )
                    skipped += 1
                    outcomes.append({"input": str(edited), "status": "skipped_karaoke_up_to_date"})
                    continue

                print(f"[live] captions-batch: processing {edited.name}", flush=True)
                captioned = self._run_caption_hook(edited)
                if captioned.resolve() != edited.resolve() and captioned.is_file():
                    ok += 1
                    outcomes.append(
                        {"input": str(edited), "status": "captioned", "captioned": str(captioned)}
                    )
                elif captioned.resolve() == edited.resolve():
                    failures += 1
                    outcomes.append({"input": str(edited), "status": "failed_or_no_caption_file"})
                    if provider in ("karaoke_google", "karaoke_vertex"):
                        print(
                            "[live] captions-batch WARN: captions path equals portrait-only input — "
                            "check Speech/Vertex credentials, caption_hook_timeout_sec, logs under meta/ "
                            "(CAPTIONS router optional; Esports karaoke may run without it).",
                            flush=True,
                        )
                final_path = self._final_video_path(edited, captioned)
                print(f"[live] captions-batch: wrote {final_path.name}", flush=True)

            summary = {
                "timestamp": _now_stamp(),
                "sources": sorted(src),
                "edit_dir": str(self.edit_dir),
                "final_dir": str(self.final_dir),
                "provider": provider,
                "redo_all": redo_all,
                "ok": ok,
                "skipped": skipped,
                "failures": failures,
                "items": outcomes,
            }
            try:
                out_log = self.meta_dir / f"captions_batch_{_now_stamp()}.json"
                out_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"[live] captions-batch summary: {out_log.name}", flush=True)
            except OSError:
                print(f"[live] captions-batch summary (could not write file): {summary}", flush=True)

            if failures and not ok and not skipped:
                return 1
            return 0
        finally:
            for s in staged_cleanup:
                try:
                    s.unlink(missing_ok=True)
                except OSError:
                    pass

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

    def _write_meta_json(self, filename: str, payload: dict[str, Any]) -> Path:
        out = self.meta_dir / filename
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _gemini_generate_text_with_fallback(
        self,
        prompt: str,
        image_path: Path | None = None,
        *,
        max_output_tokens: int = 1024,
    ) -> str:
        keys = self.cfg.gemini_api_keys or ([self.cfg.gemini_api_key] if self.cfg.gemini_api_key else [])
        if not keys:
            raise RuntimeError("No Gemini API keys configured.")

        errors: list[str] = []
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
        image_path: Path | None = None,
        *,
        audio_path: Path | None = None,
        max_output_tokens: int = 1024,
    ) -> str:
        keys = self.cfg.vertex_api_keys
        if not keys:
            raise RuntimeError("No Vertex API keys configured (vertex_api_keys / VERTEX_API_KEY).")
        pid = (self.cfg.vertex_project_id or "").strip()
        if not pid:
            raise RuntimeError("vertex_project_id is required for Vertex AI.")

        errors: list[str] = []
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
                    audio_path=audio_path,
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

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _log_detection(self, detection: dict[str, Any], screenshot: Path, status: str) -> None:
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

    def _print_ai_hud_readout(self, screenshot_full: Path, detection: dict[str, Any]) -> None:
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
            # Prefer portable Windows builds only on native Windows. Under Linux/macOS (including Docker bind-mounts),
            # ./streamlink_portable/*.exe may exist from the host but must not run — use pip/system ``streamlink``.
            streamlink_exec: str | None = None
            if sys.platform == "win32":
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
                    "On Windows: put streamlink.exe in ./streamlink_portable/ or install streamlink on PATH. "
                    "On Linux/Docker: pip install streamlink (already in container image) or install streamlink on PATH."
                )

            twitch_low = "twitch.tv/" in lower
            extras: list[str] = []
            if twitch_low:
                extras.extend(str(a).strip() for a in (self.cfg.streamlink_twitch_extra_args or []) if str(a).strip())
            extras.extend(str(a).strip() for a in (self.cfg.streamlink_extra_args or []) if str(a).strip())
            cmd = [streamlink_exec] + extras + ["--stream-url", src, "best"]
            timeout_sec = max(15, int(self.cfg.streamlink_resolve_timeout_sec or 60))
            if extras:
                print(f"[live] streamlink resolve cmd: {streamlink_exec} {' '.join(extras)} --stream-url <url> best", flush=True)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_sec),
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"streamlink --stream-url timed out after {timeout_sec}s (network, Twitch, or Docker/WSL DNS)."
                ) from exc
            if proc.returncode != 0:
                hint = ""
                err_blob = f"{proc.stderr or ''}\n{proc.stdout or ''}".lower()
                if not (proc.stderr or "").strip() and not (proc.stdout or "").strip():
                    hint = (
                        "\n[live] Hint: empty streamlink output often happens under Docker Desktop + WSL2 "
                        "(vsock/socket errors). Try: restart Docker Desktop, update it, disable VPN briefly, "
                        "or run the pipeline on Windows host (not container). "
                        "Smoke test: docker compose run --rm streamlink-debug --stream-url \"TWITCH_URL\" best\n"
                    )
                elif "vsock" in err_blob or "utilbindvsock" in err_blob.replace(" ", ""):
                    hint = (
                        "\n[live] Hint: WSL/Docker networking glitch — restart Docker Desktop or run streamlink on "
                        "the Windows host.\n"
                    )
                raise RuntimeError(
                    "Failed to resolve stream URL via streamlink.\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                    f"{hint}"
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

    def _ffmpeg_seek_args_before_input(self) -> list[str]:
        """Input-side seek for ffmpeg (VOD / archive); omitted when stream_input_seek_sec is 0."""
        base = float(self.cfg.stream_input_seek_sec or 0)
        if base <= 0:
            return []
        pos = base + self._vod_seek_anchor_elapsed_sec()
        return ["-ss", f"{pos:.3f}"]

    def _ffmpeg_realtime_read_args_before_input(self) -> list[str]:
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

    def _build_hud_capture_vf(self, fps_interval: int | None) -> str:
        """fps (optional) → crop ``round_roi_*`` → upscale to vision width → yuv420p for MJPEG."""
        rx = float(max(0.0, min(1.0, self.cfg.round_roi_x)))
        ry = float(max(0.0, min(1.0, self.cfg.round_roi_y)))
        rw = float(max(0.01, min(1.0, self.cfg.round_roi_w)))
        rh = float(max(0.01, min(1.0, self.cfg.round_roi_h)))
        tw = self._hud_vision_scale_width()
        parts: list[str] = []
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
        seek_prefix: list[str] = []
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
        popen_kwargs: dict[str, Any] = {
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

    def _iter_screen_capture_paths(self) -> list[Path]:
        """Paths to HUD crop frames (JPEG from ffmpeg); legacy PNG full frames included. Excludes *_roi*."""
        found: list[Path] = []
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
        stop_flag: Callable[[], bool] | None = None,
    ) -> Path:
        """Block until a new screenshot image appears from the ffmpeg daemon (stable size), or deadline.

        Ordering uses ``(mtime, filename)`` so a new file is not missed when mtimes collide on FAT/
        coarse clocks. ``stop_flag`` returns True to abort (producer shutdown).
        """
        stable_ok = 0
        last_key: tuple[str, int] | None = None
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

            best_p: Path | None = None
            best_tuple: tuple[float, str] = (-1.0, "")

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

    def _detect_round_from_hud_crop(self, crop_path: Path) -> dict[str, Any]:
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

    def _detect_round_from_screenshot(self, image_path: Path) -> dict[str, Any]:
        """Alias: ``image_path`` is the HUD crop file from ffmpeg."""
        return self._detect_round_from_hud_crop(image_path)

    def _round_from_detection(self, detection: dict[str, Any]) -> int | None:
        """Round index from broadcast scores only: ``score_left + score_right + 1``."""
        scores_round = _derived_round_from_scores(detection)
        detection["scores_derived_round"] = scores_round
        if scores_round is None:
            return None
        detection["round_number"] = scores_round
        return scores_round

    def _detection_confidence(self, detection: dict[str, Any]) -> float:
        """Round HUD JSON omits confidence — treat missing as confident so clears pass ``round_detection_min_confidence``."""
        raw = detection.get("confidence")
        if raw is None:
            return 1.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _scores_derived_agrees_with_round(self, detection: dict[str, Any], round_num: int) -> bool:
        """True when ``scores_derived_round`` matches ``round_num`` (same-frame consistency check)."""
        sd = detection.get("scores_derived_round")
        if sd is None:
            return False
        try:
            return int(sd) == int(round_num)
        except (TypeError, ValueError):
            return False

    def _accept_detected_round(self, round_num: int, detection: dict[str, Any]) -> tuple[bool, str]:
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
        popen_kwargs: dict[str, Any] = {
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

    def _stop_round_recording(self) -> Path | None:
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

    def _analyze_audio_energy_and_transcript(self, clip_path: Path, work_dir: Path) -> str:
        """RMS loudness timeline + short STT scan; returned block is injected before Vertex as ``AUDIO_PRE_ANALYSIS``."""
        ff = self.ffmpeg
        if not ff:
            return ""
        dur = _clip_duration_for_analysis(clip_path, ff)
        cap = float(HIGHLIGHT_VERTEX_INLINE_AUDIO_MAX_SEC)
        t_audio = cap if dur is None or dur <= 0 else min(cap, float(dur))

        pre_wav = work_dir / "pre_analysis_mono16k.wav"
        try:
            _run_ffmpeg(
                [
                    ff,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(clip_path),
                    "-vn",
                    "-t",
                    f"{t_audio:.3f}",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(pre_wav),
                ]
            )
        except RuntimeError as exc:
            return (
                "AUDIO_PRE_ANALYSIS (local; failed):\n"
                f"- wav_extract_error: {exc}\n"
            )

        if not pre_wav.is_file() or pre_wav.stat().st_size < 800:
            return (
                "AUDIO_PRE_ANALYSIS (local):\n"
                "- no_usable_wav: clip may lack audio or be near-silent.\n"
            )

        centers, rms, _sr = _mono16_wav_rms_timeline(pre_wav, window_sec=0.5)
        rms_line = _summarize_rms_spikes(centers, rms)

        stt_sec = min(float(HIGHLIGHT_PREANALYSIS_STT_MAX_SEC), t_audio)
        stt_wav = work_dir / "pre_analysis_stt_mono16k.wav"
        try:
            _run_ffmpeg(
                [
                    ff,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(pre_wav),
                    "-t",
                    f"{stt_sec:.3f}",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(stt_wav),
                ]
            )
        except RuntimeError as exc:
            transcript = ""
            stt_note = f"stt_wav_trim_failed: {exc}"
        else:
            stt_note = ""
            transcript = ""
            lang = (self.cfg.speech_language_code or "en-US").strip() or "en-US"
            api_key = (self.cfg.speech_api_key or "").strip()
            try:
                ff_stt = (self.ffmpeg or "").strip() or shutil.which("ffmpeg") or "ffmpeg"
                _words, transcript = transcribe_google_long_wav(
                    stt_wav,
                    language_code=lang,
                    timeout_sec=min(180.0, float(self.cfg.speech_recognition_timeout_sec or 600)),
                    api_key=api_key if api_key else None,
                    ffmpeg_bin=ff_stt,
                )
                stt_note = "google_cloud_speech (REST key or ADC)"
            except Exception as exc:
                transcript = ""
                stt_note = f"google_speech_failed: {exc}"

        hype_hits = _hype_hits_in_text(transcript or "")
        hype_line = ", ".join(hype_hits) if hype_hits else "(no hype-keyword pattern hits)"

        return (
            "AUDIO_PRE_ANALYSIS (instrumental facts from local preprocessing; weight these heavily; "
            "do not contradict them with invented audio claims):\n"
            f"- excerpt_sec: min(clip, {HIGHLIGHT_VERTEX_INLINE_AUDIO_MAX_SEC}s) -> used {t_audio:.1f}s mono 16 kHz WAV\n"
            f"- rms_loudness: {rms_line}\n"
            f"- stt: {stt_note}\n"
            f"- transcript_excerpt: {(transcript or '(empty)')[:1400]}\n"
            f"- hype_keyword_hits: {hype_line}\n"
        )

    def _extract_clip_analysis_audio(self, clip_path: Path, frame_dir: Path) -> Path | None:
        """Mono excerpt for Vertex: prefer 128 kbps MP3 @ 22050 Hz; fallback AAC then WAV (PCM)."""
        ff = self.ffmpeg
        if not ff:
            return None
        cap = str(int(HIGHLIGHT_ANALYSIS_AUDIO_MAX_SEC))
        out_mp3 = frame_dir / "highlight_analysis_audio.mp3"
        cmd_mp3 = [
            ff,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(clip_path),
            "-vn",
            "-t",
            cap,
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(out_mp3),
        ]
        try:
            _run_ffmpeg(cmd_mp3)
            if out_mp3.exists() and out_mp3.stat().st_size > 0:
                sz = out_mp3.stat().st_size
                print(
                    f"[live] highlight analysis audio (MP3): {sz / 1024:.1f} KiB "
                    f"(mono 22050 Hz 128 kbps, ≤{HIGHLIGHT_ANALYSIS_AUDIO_MAX_SEC}s)",
                    flush=True,
                )
                return out_mp3
        except RuntimeError as exc:
            print(f"[live] highlight analysis audio MP3 failed ({exc}); trying AAC…", flush=True)
        out_m4a = frame_dir / "highlight_analysis_audio.m4a"
        cmd_aac = [
            ff,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(clip_path),
            "-vn",
            "-t",
            cap,
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out_m4a),
        ]
        try:
            _run_ffmpeg(cmd_aac)
            if out_m4a.exists() and out_m4a.stat().st_size > 0:
                sz = out_m4a.stat().st_size
                print(
                    f"[live] highlight analysis audio (AAC): {sz / 1024:.1f} KiB "
                    f"(mono 22050 Hz 128 kbps, ≤{HIGHLIGHT_ANALYSIS_AUDIO_MAX_SEC}s)",
                    flush=True,
                )
                return out_m4a
        except RuntimeError as exc:
            print(f"[live] highlight analysis audio AAC failed ({exc}); trying WAV PCM…", flush=True)
        out_wav = frame_dir / "highlight_analysis_audio.wav"
        cmd_wav = [
            ff,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(clip_path),
            "-vn",
            "-t",
            cap,
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        try:
            _run_ffmpeg(cmd_wav)
            if out_wav.exists() and out_wav.stat().st_size > 0:
                sz = out_wav.stat().st_size
                print(
                    f"[live] highlight analysis audio (WAV PCM): {sz / 1024:.1f} KiB "
                    f"(mono 22050 Hz, ≤{HIGHLIGHT_ANALYSIS_AUDIO_MAX_SEC}s)",
                    flush=True,
                )
                return out_wav
        except RuntimeError as exc:
            print(f"[live] highlight analysis WAV extract failed ({exc})", flush=True)
        if self.cfg.highlight_vertex_audio_only:
            print(
                "[live] highlight analysis audio: all extract modes failed — audio-only mode needs a readable audio track",
                flush=True,
            )
        else:
            print(
                "[live] highlight analysis audio: extraction failed — Vertex highlight may be vision-only",
                flush=True,
            )
        return None

    def _extract_clip_analysis_frames(self, clip_path: Path, round_number: int) -> tuple[list[Path], Path | None]:
        """Extract equipart JPEGs via **one** batched ffmpeg (``select``) when possible; else fps fallback."""
        frame_dir = self.meta_dir / f"clip_frames_round_{round_number:02d}_{_now_stamp()}"
        _ensure_dir(frame_dir)
        ff = self.ffmpeg
        if not ff:
            raise RuntimeError("ffmpeg not configured")

        k = HIGHLIGHT_ANALYSIS_FRAME_COUNT
        dur = _clip_duration_for_analysis(clip_path, ff)
        frames: list[Path] = []

        if dur is not None and dur > 0:
            times = _highlight_analysis_equipart_times(dur, k)
            span = max(0.03, min(0.12, float(dur) * 0.003))
            expr = "+".join(
                f"between(t\\,{max(0.0, t - span * 0.35):.6f}\\,{t + span:.6f})" for t in times
            )
            vf = f"select='{expr}',setpts=N/FRAME_RATE/TB,scale=960:-2"
            out_pat = str(frame_dir / "eq_%03d.jpg")
            batched_ok = False
            try:
                cmd_b = [
                    ff,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(clip_path),
                    "-vf",
                    vf,
                    "-frames:v",
                    str(k),
                    "-q:v",
                    "3",
                    out_pat,
                ]
                _run_ffmpeg(cmd_b)
                cand = sorted(frame_dir.glob("eq_*.jpg"))
                if len(cand) >= k:
                    frames = cand[:k]
                    batched_ok = True
                elif len(cand) > 0:
                    frames = cand
            except RuntimeError as exc:
                print(
                    f"[live] highlight frames: batched select failed ({exc}); falling back to per-timestamp decode",
                    flush=True,
                )
            if not batched_ok:
                for p in frame_dir.glob("eq_*.jpg"):
                    p.unlink(missing_ok=True)
                frames = []
                for i, t in enumerate(times):
                    out_j = frame_dir / f"frame_{i + 1:05d}.jpg"
                    cmd = [
                        ff,
                        "-hide_banner",
                        "-nostdin",
                        "-y",
                        "-ss",
                        f"{t:.3f}",
                        "-i",
                        str(clip_path),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=960:-2",
                        "-q:v",
                        "3",
                        str(out_j),
                    ]
                    try:
                        _run_ffmpeg(cmd)
                        if out_j.exists() and out_j.stat().st_size > 0:
                            frames.append(out_j)
                    except RuntimeError:
                        pass
                if len(frames) < k:
                    print(
                        f"[live] highlight frames: equipart decode got {len(frames)}/{k}; clearing for fps fallback "
                        f"(round={round_number})",
                        flush=True,
                    )
                    for p in frames:
                        p.unlink(missing_ok=True)
                    for p in frame_dir.glob("frame_*.jpg"):
                        p.unlink(missing_ok=True)
                    frames = []

        if not frames:
            pattern = frame_dir / "frame_%05d.jpg"
            cmd = [
                ff,
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(clip_path),
                "-vf",
                "fps=1,scale=960:-2",
                "-frames:v",
                str(k),
                "-q:v",
                "3",
                str(pattern),
            ]
            _run_ffmpeg(cmd)
            frames = sorted(frame_dir.glob("frame_*.jpg"))[:k]

        audio_path = self._extract_clip_analysis_audio(clip_path, frame_dir)
        if dur is not None:
            extra = f"; clip ~{dur:.1f}s"
        else:
            extra = ""
        au = " + audio for Vertex" if audio_path else " (no audio → vision-only)"
        print(
            f"[live] highlight frames: {len(frames)} /{k}{extra}{au}; round={round_number}",
            flush=True,
        )
        return frames, audio_path

    def _build_clip_contact_sheet(self, frames: list[Path], round_number: int) -> Path:
        selected = list(frames)
        if not selected:
            raise RuntimeError("No frames available for contact sheet")

        thumbs: list[Image.Image] = []
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

    def _normalize_highlight_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Ensure ``round_description`` is a string and ``rejection_reason`` exists when rejected."""
        rd_raw = analysis.get("round_description")
        rd = rd_raw.strip() if isinstance(rd_raw, str) else ""
        analysis = {**analysis, "round_description": rd}

        ih = bool(analysis.get("is_highlight", False))
        raw_rr = analysis.get("rejection_reason")
        rr = raw_rr.strip() if isinstance(raw_rr, str) else ""
        if ih:
            return {**analysis, "rejection_reason": "", "round_description": rd}
        if rr:
            return {**analysis, "rejection_reason": rr}
        parts: list[str] = []
        fr = analysis.get("final_reason")
        if isinstance(fr, str) and fr.strip():
            parts.append(fr.strip())
        wnh = analysis.get("why_not_highlight")
        if isinstance(wnh, list) and wnh:
            joined = "; ".join(str(x).strip() for x in wnh if str(x).strip())
            if joined:
                parts.append(joined)
        fallback = " ".join(parts) if parts else "Model returned is_highlight=false without a rejection_reason."
        return {**analysis, "rejection_reason": fallback, "round_description": rd}

    def _analyze_clip(self, clip_path: Path, round_number: int) -> dict[str, Any]:
        """Judge highlight-worthiness via Vertex Gemini: default 9-frame sheet + audio; or audio + ``rules_docx`` only."""
        frames: list[Path] = []
        audio_path: Path | None = None
        contact_sheet: Path | None = None
        cleanup_dir: Path | None = None

        audio_only = bool(self.cfg.highlight_vertex_audio_only)
        rules_body = (self.rules_text or "").strip()
        audio_pre_block = ""

        try:
            if audio_only:
                cleanup_dir = self.meta_dir / f"clip_vertex_audio_only_round_{round_number:02d}_{_now_stamp()}"
                _ensure_dir(cleanup_dir)
                audio_path = self._extract_clip_analysis_audio(clip_path, cleanup_dir)
                if audio_path is None:
                    return {
                        "is_highlight": False,
                        "confidence": 0.0,
                        "why_highlight": [],
                        "why_not_highlight": ["no_highlight_analysis_audio"],
                        "final_reason": "Highlight analysis audio could not be extracted from this clip.",
                        "rejection_reason": "audio-only Vertex mode requires a clip audio stream; ffmpeg extraction failed.",
                        "round_description": "No soundtrack available for inferred summary.",
                    }
                ff = self.ffmpeg
                dur = _clip_duration_for_analysis(clip_path, ff) if ff else None
                dur_bit = f"~{dur:.1f}s" if dur is not None and dur > 0 else "unknown duration"
                print(
                    f"[live] highlight Vertex audio-only: mono excerpt ({dur_bit}) + rules text — no frame snapshots",
                    flush=True,
                )

                audio_pre_block = self._analyze_audio_energy_and_transcript(clip_path, cleanup_dir)

                rules_section = ""
                rules_instruction = ""
                if rules_body:
                    excerpt = rules_body[:HIGHLIGHT_RULES_CONTEXT_MAX_CHARS]
                    rules_section = "rules_context:\n" + excerpt + "\n\n"
                    rules_instruction = (
                        "The following rules_context is plain text extracted from the project's highlight-rules Word document — "
                        "treat it as binding criteria for is_highlight and for rejection_reason when you reject. "
                        "Combine rules_context with what you infer from the clip audio ONLY (semantic fit, not substring matching "
                        "unless rules_context explicitly requires verbatim phrases).\n\n"
                    )

                audio_note = (
                    "Attached clip audio is the round's soundtrack (mono, excerpt). "
                    "Judge highlight-worthiness from speech, caster energy, hype/clutch vocabulary, laughs/screams, and dynamics — "
                    "aligned with rules_context.\n\n"
                )
                prompt = (
                    "You are judging Counter-Strike 2 broadcast audio for clip / social highlight potential. "
                    "NO video, thumbnails, or HUD images are supplied — ignore visuals entirely.\n\n"
                    + audio_note
                    + "\n"
                    + audio_pre_block
                    + "\n"
                    + HIGHLIGHT_VERTEX_AUDIO_ONLY_GUIDE
                    + "\n\n"
                    + rules_instruction
                    + rules_section
                    + (
                        "Decide if this excerpt is highlight–worthy using CS2/esports caster/listener judgment "
                        + ("and rules_context where provided; " if rules_body else "")
                        + "inferring stakes and hype from AUDIO ONLY. "
                    )
                    + "Return strict JSON only (no markdown, no preamble): "
                    '{"is_highlight": boolean, "confidence": number 0-1, "round_description": string, "why_highlight": [string], '
                    '"why_not_highlight": [string], "final_reason": string, "rejection_reason": string, '
                    '"audio_evidence_summary": string, "highlight_signal_mix": string}. '
                    "audio_evidence_summary: 1–3 sentences on what you heard (themes, spikes, weak vs strong). "
                    "highlight_signal_mix: explain how fused audio cues (and alignment with rules_context) support the verdict — "
                    "state explicitly that scoring is audio-only. "
                    "round_description MUST be 1–3 sentences summarizing plausible in-round narrative inferred from AUDIO ONLY "
                    "(e.g., apparent clutch hype, ace calls, chaotic teamfight energy); say when inference is uncertain. "
                    "Write round_description BEFORE verdict reasoning. "
                    "When is_highlight is false: rejection_reason MUST be one detailed paragraph on what audio showed "
                    "(or lacked)"
                    + (
                        " versus rules_context expectations — "
                        if rules_body
                        else " versus strong highlight audio patterns — "
                    )
                    + 'no vague wording like "not exciting". '
                    "When is_highlight is true: rejection_reason must be \"\" (empty string). "
                    + (
                        "Mark is_highlight=true only when rules_context (if provided) clearly supports highlighting AND "
                        "audio evidence backs it;"
                        if rules_body
                        else "Set is_highlight=true when audio shows unusually explosive or clip-worthy moments per the guide; "
                    )
                    + (
                        " reject when rules_context forbids highlighting even if casters sound loud."
                        if rules_body
                        else " prefer false for calm rundowns or ambiguous low-energy chatter."
                    )
                    + f" Round number for context only: {round_number}."
                )
                sheet_kb = 0.0
                audio_kb = audio_path.stat().st_size / 1024.0 if audio_path.is_file() else 0.0
                self._append_jsonl(
                    self.meta_dir / "highlight_vertex_multimodal.jsonl",
                    {
                        "timestamp": _now_stamp(),
                        "round": round_number,
                        "clip": str(clip_path.resolve()),
                        "highlight_vertex_audio_only": True,
                        "contact_sheet_kb": 0.0,
                        "vertex_audio_attached": True,
                        "vertex_audio_kb": round(audio_kb, 2),
                        "vertex_audio_suffix": audio_path.suffix if audio_path else "",
                        "note": "Temp audio excerpt deleted after this request; no JPEG contact sheet.",
                    },
                )
                print(
                    f"[live] highlight Vertex audio-only: sending audio ~{audio_kb:.1f} KiB ({audio_path.name}) — no JPEG",
                    flush=True,
                )
                if self.cfg.highlight_yield_to_hud_vision:
                    self._yield_while_hud_remote_busy()
                txt = self._vertex_generate_text_with_fallback(
                    prompt,
                    None,
                    audio_path=audio_path,
                    max_output_tokens=8192,
                )
                return self._normalize_highlight_analysis(_extract_json(txt))

            # Multimodal: equipart snapshots + optional audio + optional rules_docx
            pre_dir = self.meta_dir / f"hl_preanalysis_r{round_number:02d}_{_now_stamp()}"
            _ensure_dir(pre_dir)
            audio_pre_block = self._analyze_audio_energy_and_transcript(clip_path, pre_dir)

            frames, audio_path = self._extract_clip_analysis_frames(clip_path, round_number)
            cleanup_dir = frames[0].parent if frames else None
            if not frames:
                return {
                    "is_highlight": False,
                    "confidence": 0.0,
                    "why_highlight": [],
                    "why_not_highlight": ["no_analysis_frames_extracted"],
                    "final_reason": "No frames could be extracted for highlight analysis.",
                    "rejection_reason": "No frames could be extracted from the clip for vision analysis.",
                    "round_description": "No sampled frames available; the round could not be visually summarized.",
                }

            sheet_frames = frames[:HIGHLIGHT_ANALYSIS_FRAME_COUNT]
            contact_sheet = self._build_clip_contact_sheet(sheet_frames, round_number)
            nf = len(sheet_frames)
            rules_section = ""
            rules_instruction = ""
            if rules_body:
                excerpt = rules_body[:HIGHLIGHT_RULES_CONTEXT_MAX_CHARS]
                rules_section = "rules_context:\n" + excerpt + "\n\n"
                rules_instruction = (
                    "The following rules_context is plain text extracted from the project's highlight-rules Word document — "
                    "treat it as binding criteria for is_highlight and for rejection_reason when you reject. "
                    "Combine rules_context with the contact sheet AND the clip audio.\n\n"
                )
            audio_note = (
                "Attached clip audio is the round's soundtrack (mono, excerpt). Listen for caster/player hype, clutch language, "
                "skill callouts, laughter/screaming, and sudden loudness — cross-check against the 9 evenly spaced frames.\n\n"
                if audio_path
                else "No clip audio could be extracted; rely on the contact sheet only.\n\n"
            )
            prompt = (
                "You are judging a Counter-Strike 2 competitive round clip using multimodal evidence.\n\n"
                + audio_note
                + audio_pre_block
                + "\n"
                + HIGHLIGHT_VERTEX_AUDIO_ANALYSIS_GUIDE
                + "\n\n"
                + rules_instruction
                + rules_section
                + f"There are exactly {nf} stills on one JPEG grid (3 columns × 3 rows), chronological left-to-right, top-to-bottom, "
                "evenly spaced in time across the full clip duration (not uniform real-time fps sampling). "
                + (
                    "Decide if this clip is social-media highlight–worthy using CS2/esports judgment "
                    + ("and rules_context where provided; " if rules_body else "")
                    + "including exciting gunplay, clutches, aces, economy swings, standout individual plays, knife/zeus moments, "
                    "meme-worthy or caster-bait moments when rules allow. "
                )
                + "Return strict JSON only (no markdown, no preamble): "
                '{"is_highlight": boolean, "confidence": number 0-1, "round_description": string, "why_highlight": [string], '
                '"why_not_highlight": [string], "final_reason": string, "rejection_reason": string, '
                '"audio_evidence_summary": string, "highlight_signal_mix": string}. '
                "audio_evidence_summary: 1–3 sentences on what you heard (speech themes, intensity spikes, laughter/screams, weak vs strong cues). "
                "highlight_signal_mix: brief explanation of how audio + visuals combine (multi-signal vs single weak cue). "
                "round_description MUST be 1–3 sentences describing what happens in this round from the frames "
                "(map area if visible, trades, clutch, spike, economy reads, apparent outcome); say if footage is unclear. "
                "Write round_description BEFORE your verdict reasoning. "
                "When is_highlight is false: rejection_reason MUST be one detailed paragraph naming what you actually "
                "see on the contact sheet and what you heard (if audio present)"
                + (
                    " (specific cues: HUD state, kills/deaths visible, clutch timing, economy, reactions on audio)"
                    + (
                        " and how that lines up with rules_context — "
                        if rules_body
                        else " and the concrete reason this fails as a clip-worthy highlight — "
                    )
                )
                + 'not vague wording like "not exciting". '
                "When is_highlight is true: rejection_reason must be \"\" (empty string). "
                + (
                    "Mark is_highlight=true only when the clip clearly satisfies rules_context AND multimodal evidence supports it; "
                    if rules_body
                    else "Set is_highlight=true when audio and/or visuals show unusually strong or clip-worthy moments per the guide above; "
                )
                + (
                    "reject when rules_context says non-highlight even if action looks flashy."
                    if rules_body
                    else "prefer false for slow default rounds, unclear frames, absent reactions, or nothing remarkable even if audio is noisy. "
                )
                + f"Round number for context only: {round_number}."
            )
            sheet_kb = contact_sheet.stat().st_size / 1024.0 if contact_sheet.is_file() else 0.0
            audio_kb = audio_path.stat().st_size / 1024.0 if audio_path is not None and audio_path.is_file() else 0.0
            self._append_jsonl(
                self.meta_dir / "highlight_vertex_multimodal.jsonl",
                {
                    "timestamp": _now_stamp(),
                    "round": round_number,
                    "clip": str(clip_path.resolve()),
                    "highlight_vertex_audio_only": False,
                    "contact_sheet_kb": round(sheet_kb, 2),
                    "vertex_audio_attached": audio_path is not None,
                    "vertex_audio_kb": round(audio_kb, 2) if audio_path else 0.0,
                    "vertex_audio_suffix": audio_path.suffix if audio_path else "",
                    "note": "Temp audio/contact frames are deleted after this request; this log confirms payload.",
                },
            )
            if audio_path is not None:
                print(
                    f"[live] highlight Vertex multimodal: JPEG sheet ~{sheet_kb:.1f} KiB + "
                    f"audio ~{audio_kb:.1f} KiB ({audio_path.name}) — sending to Gemini",
                    flush=True,
                )
            else:
                print(
                    f"[live] highlight Vertex: JPEG sheet ~{sheet_kb:.1f} KiB only (no audio attachment)",
                    flush=True,
                )
            if self.cfg.highlight_yield_to_hud_vision:
                self._yield_while_hud_remote_busy()
            txt = self._vertex_generate_text_with_fallback(
                prompt,
                contact_sheet,
                audio_path=audio_path,
                max_output_tokens=8192,
            )
            return self._normalize_highlight_analysis(_extract_json(txt))
        finally:
            if contact_sheet is not None:
                contact_sheet.unlink(missing_ok=True)
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
            for frame in frames:
                frame.unlink(missing_ok=True)
            if cleanup_dir is not None and cleanup_dir.is_dir():
                try:
                    cleanup_dir.rmdir()
                except OSError:
                    pass

    def _edit_portrait_blur(self, clip_path: Path, round_number: int) -> Path:
        out = self.edit_dir / f"round_{round_number:02d}_{_now_stamp()}_portrait.mp4"
        t0 = time.monotonic()
        apply_portrait_blur(
            clip_path,
            out,
            ffmpeg_bin=self.ffmpeg,
            width=int(self.cfg.portrait_blur_width),
            height=int(self.cfg.portrait_blur_height),
            crf=self.cfg.portrait_blur_crf,
            preset=self.cfg.portrait_blur_preset,
            fps=30.0,
        )
        dt = time.monotonic() - t0
        print(
            f"[live] portrait edit saved: {out.name} "
            f"({dt:.1f}s {self.cfg.portrait_blur_width}x{self.cfg.portrait_blur_height} "
            f"preset={self.cfg.portrait_blur_preset} crf={self.cfg.portrait_blur_crf})",
            flush=True,
        )
        return out

    def _karaoke_google_auto_available(self) -> bool:
        """True when Google Speech karaoke burn can run (CAPTIONS router or Esports burn + Speech credentials)."""
        if not (_captions_vertex_burn_script_path().is_file() or _esports_karaoke_burn_script_path().is_file()):
            return False
        if not self.cfg.pipeline_config_path.is_file():
            return False
        speech_ok = bool((self.cfg.speech_api_key or "").strip()) or bool(self.cfg.karaoke_use_adc)
        return speech_ok

    def _resolve_caption_provider(self) -> str:
        """Expand ``caption_provider=auto`` into a concrete backend (shell / Speech karaoke / Speech legacy / none).

        Caption defaults prefer ``karaoke_google`` (Cloud Speech-to-Text → ASS karaoke burn). Vertex Gemini captions
        are opt-in via explicit ``caption_provider=karaoke_vertex`` only.
        """
        raw = (self.cfg.caption_provider or "auto").strip().lower()
        if raw != "auto":
            return raw
        tpl = (self.cfg.caption_cmd_template or "").strip()
        key = (self.cfg.speech_api_key or "").strip()
        if tpl:
            return "shell"
        if self._karaoke_google_auto_available():
            return "karaoke_google"
        if key:
            return "google_speech"
        return "none"

    def _run_caption_hook(self, edited_path: Path) -> Path:
        provider = self._resolve_caption_provider()
        tpl = (self.cfg.caption_cmd_template or "").strip()

        if provider == "none":
            return edited_path

        if provider in ("karaoke_google", "karaoke_vertex"):
            if self.cfg.karaoke_async:
                print(
                    "[live] karaoke_async=true: portrait-only copy goes to *_final.mp4 first; "
                    "karaoke overwrites *_final.mp4 when the burn finishes (or run with --karaoke-sync). "
                    "Watch for *_karaoke.mp4 in round_final/.",
                    flush=True,
                )
                self._karaoke_start_background(edited_path)
                return edited_path
            return self._caption_with_karaoke_subprocess(edited_path)

        if provider == "google_speech":
            return self._caption_with_google_speech(edited_path)

        use_shell = provider == "shell"
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

    def _karaoke_burn_preferences(self, video_hint: Path) -> dict[str, Any]:
        """Karaoke ASS + logo layout + encode — knobs read from CAPTIONS sidecar JSON (if any) matching inject.

        Prefer ``CAPTIONS/live_pipeline_config.json`` when present; otherwise Esports pipeline JSON.
        Overlay resolution matches CAPTIONS ``_resolve_overlay_image``. For ``karaoke_google`` the subprocess
        is ``CAPTIONS/burn_karaoke_captions.py --delegate-esports-karaoke``, which merges these fields into CLI.
        """
        cfg_disk = _captions_sidecar_live_pipeline_json(self.cfg.pipeline_config_path)
        raw = _safe_load_json_settings(cfg_disk)
        c = self.cfg

        def pick(key: str, fallback: Any) -> Any:
            return raw[key] if key in raw else fallback

        margin = float(pick("karaoke_margin_top_ratio", c.karaoke_margin_top_ratio))
        owf = float(pick("karaoke_overlay_width_frac", c.karaoke_overlay_width_frac))
        omb = int(pick("karaoke_overlay_margin_bottom_px", c.karaoke_overlay_margin_bottom_px))
        preset = str(pick("karaoke_ffmpeg_preset", c.karaoke_ffmpeg_preset) or "medium").strip()
        crf = int(pick("karaoke_ffmpeg_crf", c.karaoke_ffmpeg_crf))
        use_adc = bool(pick("karaoke_use_adc", c.karaoke_use_adc))
        no_overlay = bool(pick("karaoke_no_overlay", c.karaoke_no_overlay))
        lang = str(pick("speech_language_code", c.speech_language_code) or "en-US").strip()

        vmx: float | None = None
        for k_mx in ("karaoke_vertex_inline_video_max_mb", "vertex_inline_video_max_mb"):
            if k_mx not in raw:
                continue
            try:
                v_candidate = float(raw[k_mx])
            except (TypeError, ValueError):
                continue
            if v_candidate > 0:
                vmx = v_candidate
                break

        overlay: Path | None = None
        if not no_overlay:
            raw_ov = str(pick("karaoke_overlay_image", c.karaoke_overlay_image) or "").strip()
            base = cfg_disk.parent.resolve()
            if raw_ov:
                ov = Path(raw_ov)
                resolved_ov = ov.resolve() if ov.is_absolute() else (base / ov).resolve()
                if resolved_ov.is_file():
                    overlay = resolved_ov
            if overlay is None:
                cand = video_hint.parent / CAPTIONS_STANDALONE_OVERLAY_FALLBACK_NAME
                overlay = cand if cand.is_file() else None
        prefs_ret: dict[str, Any] = {
            "config_path_used": cfg_disk,
            "margin_v_from_top_ratio": margin,
            "overlay_width_frac": owf,
            "overlay_margin_bottom_px": omb,
            "encode_preset": preset,
            "encode_crf": crf,
            "karaoke_use_adc": use_adc,
            "speech_language_code": lang,
            "overlay_image": overlay,
            "karaoke_no_overlay": no_overlay,
        }
        if vmx is not None:
            prefs_ret["karaoke_vertex_inline_video_max_mb"] = float(vmx)

        vf = raw.get("karaoke_vertex_send_full_video")
        if vf is True or vf == 1 or (
            isinstance(vf, str) and vf.strip().lower() in {"1", "true", "yes", "on"}
        ):
            prefs_ret["karaoke_vertex_send_full_video"] = True

        ao = raw.get("karaoke_vertex_audio_only")
        if ao is True or ao == 1 or (
            isinstance(ao, str) and ao.strip().lower() in {"1", "true", "yes", "on"}
        ):
            prefs_ret["karaoke_vertex_audio_only"] = True

        gcs_au = str(raw.get("vertex_audio_gcs_uri", "") or "").strip()
        if gcs_au:
            prefs_ret["vertex_audio_gcs_uri"] = gcs_au

        if raw.get("karaoke_caption_time_offset_sec") is not None:
            try:
                prefs_ret["karaoke_caption_time_offset_sec"] = float(raw["karaoke_caption_time_offset_sec"])
            except (TypeError, ValueError):
                pass

        imx = raw.get("karaoke_vertex_invert_mux_timing_fix")
        if imx is True or imx == 1 or (
            isinstance(imx, str) and imx.strip().lower() in {"1", "true", "yes", "on"}
        ):
            prefs_ret["karaoke_vertex_invert_mux_timing_fix"] = True

        return prefs_ret

    def _karaoke_validate_can_run(self, edited_path: Path) -> bool:
        """Return False if karaoke_google / karaoke_vertex prerequisites are missing."""
        provider = self._resolve_caption_provider()
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
        prefs = self._karaoke_burn_preferences(edited_path)
        cfg_path = self.cfg.pipeline_config_path
        if not cfg_path.is_file():
            print(f"[live] karaoke_google: pipeline config missing ({cfg_path}); skipping karaoke task")
            return False
        if not prefs["karaoke_use_adc"] and not (self.cfg.speech_api_key or "").strip():
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
            picked: Path | None = None
            try:
                picked = pipeline._caption_with_karaoke_subprocess(edited_path)
            except Exception as exc:
                print(f"[live] karaoke background failed ({stem}): {exc}", flush=True)
            finally:
                try:
                    pending.unlink(missing_ok=True)
                except OSError:
                    pass
                # Async mode initially copies portrait-only into *_final.mp4; promote karaoke here when ready.
                if (
                    picked is not None
                    and picked.resolve() != edited_path.resolve()
                    and picked.is_file()
                    and picked.stat().st_size > 0
                ):
                    final_out = pipeline.final_dir / f"{stem}_final.mp4"
                    try:
                        shutil.copy2(picked, final_out)
                        print(
                            f"[live] karaoke async: captions burned → {picked.name}; "
                            f"updated deliverable {final_out.name}",
                            flush=True,
                        )
                    except OSError as exc:
                        print(f"[live] karaoke async: could not update {final_out.name}: {exc}", flush=True)

        threading.Thread(target=runner_wrap, name=f"karaoke-bg-{stem}", daemon=True).start()
        print(
            f"[live] karaoke burning in background → {expected.name}; "
            f"{stem}_final.mp4 is portrait-only until burn-in completes",
            flush=True,
        )

    def _caption_with_karaoke_subprocess(self, edited_path: Path) -> Path:
        """Isolate karaoke transcribe+burn in a spawned child (``multiprocessing`` spawn).

        ``karaoke_google`` prefers ``CAPTIONS/burn_karaoke_captions.py --delegate-esports-karaoke`` when that
        router exists; otherwise runs ``burn_karaoke_captions.py`` in this repo directly (same Speech burn).
        ``karaoke_vertex`` invokes Vertex mode on CAPTIONS (no Esports-only fallback).
        """
        provider = self._resolve_caption_provider()
        if provider == "karaoke_vertex":
            return self._caption_with_karaoke_vertex_subprocess(edited_path)

        prefs = self._karaoke_burn_preferences(edited_path)
        cfg_path = self.cfg.pipeline_config_path
        if not cfg_path.is_file():
            print(f"[live] karaoke_google: pipeline config missing ({cfg_path}); using portrait-only video")
            return edited_path
        if not prefs["karaoke_use_adc"] and not (self.cfg.speech_api_key or "").strip():
            print(
                "[live] karaoke_google: need speech_api_key (or karaoke_use_adc=true); "
                "using portrait-only video",
                flush=True,
            )
            return edited_path

        if prefs["config_path_used"].resolve() != cfg_path.resolve():
            print(
                f"[live] karaoke_google: --config {prefs['config_path_used']} "
                "(CAPTIONS sidecar — same knobs as standalone CAPTIONS Esports karaoke)",
                flush=True,
            )

        caps_script = _captions_vertex_burn_script_path()
        es_script = _esports_karaoke_burn_script_path()

        out_pref = self.final_dir / f"{edited_path.stem}_karaoke.mp4"
        hook_log = self.meta_dir / f"{edited_path.stem}_karaoke_subprocess.json"

        child_target: Callable[[dict[str, Any]], None]
        backend_name: str
        payload: dict[str, Any]

        if caps_script.is_file():
            child_target = _karaoke_google_caps_delegate_child_main
            backend_name = "caps_delegate_google_speech"
            payload = {
                "captions_script": str(caps_script.resolve()),
                "pipeline_config": str(prefs["config_path_used"].resolve()),
                "video_path": str(edited_path.resolve()),
                "video_out": str(out_pref.resolve()),
                "ffmpeg_bin": self.ffmpeg or "",
            }
        elif es_script.is_file():
            work_dir_final = self.final_dir.resolve()
            print(
                f"[live] karaoke_google: CAPTIONS router not found; "
                f"running Esports burn directly ({es_script.name}); "
                f"scratch + ASS → {work_dir_final}",
                flush=True,
            )
            child_target = _karaoke_google_esports_direct_child_main
            backend_name = "esports_direct_google_speech"
            payload = {
                "esports_script": str(es_script.resolve()),
                "pipeline_config": str(prefs["config_path_used"].resolve()),
                "video_path": str(edited_path.resolve()),
                "video_out": str(out_pref.resolve()),
                "work_dir": str(work_dir_final),
                "ffmpeg_bin": self.ffmpeg or "",
                "burn_prefs": _karaoke_prefs_for_spawn(prefs),
            }
        else:
            print(
                f"[live] karaoke_google: missing CAPTIONS router ({caps_script}) and Esports burn ({es_script}); "
                "using portrait-only video",
                flush=True,
            )
            return edited_path

        timeout_sec = max(1, self.cfg.caption_hook_timeout_sec)
        started = time.time()

        def _pick_karaoke_output() -> Path | None:
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

        def _safe_hook_payload() -> dict[str, Any]:
            return json.loads(json.dumps(payload, default=str))

        ctx = multiprocessing.get_context("spawn")
        proc_name = "karaoke-google-caps" if backend_name.startswith("caps") else "karaoke-google-esports"
        child = ctx.Process(target=child_target, args=(payload,), name=proc_name)
        child.start()
        child.join(timeout=float(timeout_sec))

        timed_out = child.is_alive()
        if timed_out:
            child.terminate()
            child.join(15)
            if child.is_alive():
                child.kill()
                child.join(5)

        hook_obj: dict[str, Any] = {
            "timestamp": _now_stamp(),
            "transport": "multiprocessing_spawn",
            "backend": backend_name,
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

        picked: Path | None = None
        try:
            if out_pref.is_file() and out_pref.stat().st_size > 0:
                picked = out_pref
        except OSError:
            picked = None
        if picked is None:
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
        work = (self.final_dir / f"{edited_path.stem}_vertex_karaoke_work").resolve()
        work.mkdir(parents=True, exist_ok=True)
        hook_log = self.meta_dir / f"{edited_path.stem}_karaoke_vertex_subprocess.json"

        prefs = self._karaoke_burn_preferences(edited_path)
        if prefs["config_path_used"].resolve() != self.cfg.pipeline_config_path.resolve():
            print(
                f"[live] karaoke_vertex: --config {prefs['config_path_used']} "
                "(CAPTIONS sidecar when present — same overlay/margin/logo as standalone burn_karaoke_captions.py)",
                flush=True,
            )

        payload: dict[str, Any] = {
            "captions_script": str(script.resolve()),
            "pipeline_config": str(prefs["config_path_used"].resolve()),
            "ffmpeg_bin": str(self.ffmpeg or "").strip(),
            "video_path": str(edited_path.resolve()),
            "work_dir": str(work.resolve()),
            "video_out": str(out_pref.resolve()),
            "karaoke_vertex_roster_path": roster,
            "karaoke_cli_extras": _vertex_karaoke_argv_from_prefs(prefs),
        }

        timeout_sec = max(1, self.cfg.caption_hook_timeout_sec)
        started = time.time()

        def _pick_vertex_output() -> Path | None:
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

        hook_obj: dict[str, Any] = {
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
        """Always materialize ``<stem>_final.mp4`` in ``round_final`` as the publishable deliverable.

        When captions run inline, ``captioned_path`` is the karaoke/captioned file; we copy it to
        ``*_final.mp4``. When ``karaoke_async`` is true, ``captioned_path`` equals ``edited_path``
        initially (portrait only); the background karaoke thread overwrites ``*_final.mp4`` after burn-in.
        """
        out = self.final_dir / f"{edited_path.stem}_final.mp4"
        src = captioned_path if captioned_path != edited_path else edited_path
        shutil.copy2(src, out)
        return out

    def _generate_title_and_seo(self, analysis: dict[str, Any], round_number: int) -> dict[str, Any]:
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
        parsed: dict[str, Any]
        try:
            txt = self._vertex_generate_text_with_fallback(prompt, None, max_output_tokens=1024)
            parsed = _extract_json(txt)
        except Exception as exc:
            print(f"[live] title/SEO generation failed, using fallback caption: {exc}")
            parsed = dict(fallback)

        keywords = parsed.get("seo_keywords")
        if not isinstance(keywords, list):
            parsed["seo_keywords"] = []
        return parsed

    def _post_to_instagram(self, video_path: Path, text_pack: dict[str, Any]) -> bool:
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
        rd = str(analysis.get("round_description", "") or "").strip()
        if rd:
            meta["round_description"] = rd
            print(f"[live] round={round_number} round_description: {rd}", flush=True)

        if not bool(analysis.get("is_highlight", False)):
            meta["status"] = "rejected_non_highlight"
            rr = str(analysis.get("rejection_reason", "") or "").strip()
            if rr:
                meta["rejection_reason"] = rr
            reject_path = self.meta_dir / f"round_{round_number:02d}_{_now_stamp()}_rejected.json"
            reject_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[live] round={round_number} rejected (non-highlight)", flush=True)
            if rr:
                print(f"[live] rejection_reason: {rr}", flush=True)
            else:
                print(
                    f"[live] rejection_reason missing — see analysis in meta/{reject_path.name}",
                    flush=True,
                )
            return

        edited = self._edit_portrait_blur(clip_path, round_number)
        cp_live = self._resolve_caption_provider()
        captioned = self._run_caption_hook(edited)
        if (
            cp_live in ("karaoke_google", "karaoke_vertex")
            and captioned.resolve() == edited.resolve()
        ):
            print(
                "[live] WARN karaoke captions were not burned — *_final.mp4 is portrait-only. "
                "For Speech karaoke: set GOOGLE_SPEECH_API_KEY / speech_api_key / karaoke_use_adc + ADC; "
                "CAPTIONS router is optional (Esports calls burn_karaoke_captions.py directly when missing). "
                f"Burn router (if present): {_captions_vertex_burn_script_path()}",
                flush=True,
            )
        final_video = self._final_video_path(edited, captioned)
        text_pack = self._generate_title_and_seo(analysis, round_number)
        posted = self._post_to_instagram(final_video, text_pack)

        meta_extra: dict[str, Any] = {}
        if cp_live in ("karaoke_google", "karaoke_vertex") and self.cfg.karaoke_async:
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

    def _process_or_save_partial_clip(self, clip: Path | None, round_number: int | None) -> None:
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

    def _apply_detection_result(self, screenshot_full: Path, detection: dict[str, Any]) -> None:
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
        try:
            cp = self._resolve_caption_provider()
            if cp in ("karaoke_google", "karaoke_vertex"):
                if self.cfg.karaoke_async:
                    print(
                        "[live] Karaoke: async — portrait *_final.mp4 first; captions land in *_karaoke.mp4 / "
                        "final updated later (HUD capture loop is never blocked).",
                        flush=True,
                    )
                else:
                    print(
                        "[live] Karaoke: synchronous — waits for burn before writing captioned *_final.mp4. "
                        "This work runs on highlight-queue worker threads only; the main HUD/screenshot loop "
                        "stays on its own thread so capture is not paused by ffmpeg.",
                        flush=True,
                    )
                sp = _captions_vertex_burn_script_path()
                ok = sp.is_file()
                print(
                    f"[live] karaoke CAPTIONS router: {sp} (exists={ok}). "
                    "If false in Docker, mount CAPTIONS or set CAPTIONS_BURN_SCRIPT / CAPTIONS_KARAOKE_ROOT.",
                    flush=True,
                )
        except Exception:
            pass

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
                err_text = str(exc)
                aws_auth_issue = (
                    "UnrecognizedClientException" in err_text
                    or "InvalidClientTokenId" in err_text
                    or "credentials rejected" in err_text.lower()
                    or "security token included in the request is invalid" in err_text.lower()
                )
                now_m = time.monotonic()
                if aws_auth_issue:
                    if now_m - self._vision_auth_error_last_emit_monotonic >= 45.0:
                        self._vision_auth_error_last_emit_monotonic = now_m
                        print(
                            "[live] AWS Rekognition auth failing — throttling repeats to every ~45s. "
                            "Fix credentials then restart (see RuntimeError hints above when raised).",
                            flush=True,
                        )
                        print(f"[live] capture/Vision loop error: {exc}", flush=True)
                else:
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


def _speech_api_key_local_paths(config_path: Path) -> list[Path]:
    """Locate ``speech_api_key.local.json``: beside pipeline JSON then mono-repo Esports/CAPTIONS siblings."""

    cfg_dir = config_path.resolve().parent
    grand = cfg_dir.parent
    seen_keys: set[str] = set()
    out: list[Path] = []
    for p in (
        cfg_dir / "speech_api_key.local.json",
        grand / "Esports-Video-clipping-automation" / "speech_api_key.local.json",
        grand / "CAPTIONS" / "speech_api_key.local.json",
    ):
        rk = str(p.resolve())
        if rk in seen_keys:
            continue
        seen_keys.add(rk)
        out.append(p)
    return out


def _resolve_speech_api_key(cfg: dict[str, Any], config_path: Path) -> str:
    """Resolve key for Cloud Speech-to-Text captions.

    Precedence: ``GOOGLE_SPEECH_API_KEY`` env → ``speech_api_key.local.json`` (beside main config **or**
    sibling ``Esports-Video-clipping-automation`` / ``CAPTIONS`` in a mono-repo) → ``speech_api_key`` in
    main JSON → same key as ``vertex_api_key`` / ``vertex_api_keys[0]`` / ``gemini_api_key`` / ``gemini_api_keys[0]``.
    """
    env_k = os.getenv("GOOGLE_SPEECH_API_KEY", "").strip()
    if env_k:
        return env_k
    for local_file in _speech_api_key_local_paths(config_path):
        if not local_file.is_file():
            continue
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


def _pipeline_builtin_json_defaults() -> dict[str, Any]:
    """Defaults merged **before** JSON values from disk — keys present in JSON override these.

    Session: NAVI vs GamerLegion (maps Mirage / Ancient); roster body:
    ``default_match_roster_navigl.txt`` next to ``live_pipeline_config.json``.
    Change these constants when switching default VOD / seek / roster until JSON overrides them.
    """
    return {
        "stream_url": "https://www.twitch.tv/videos/2760697668",
        "stream_input_seek_hms": "10:08:25",
        "karaoke_vertex_roster_path": "default_match_roster_navigl.txt",
        "caption_provider": "karaoke_google",
        "screenshot_interval_sec": 5,
        "screenshot_4k_width": 3840,
        "api_provider": "rekognition",
        "highlight_api_provider": "vertex",
        "highlight_parallel_workers": 2,
        "highlight_yield_to_hud_vision": False,
        "highlight_vertex_audio_only": False,
        "streamlink_extra_args": [],
        "streamlink_twitch_extra_args": [
            "--twitch-disable-ads",
            "--twitch-disable-reruns",
        ],
        "streamlink_resolve_timeout_sec": 90,
        # Synchronous karaoke by default so *_final.mp4 has captions (burn runs on highlight workers only).
        "karaoke_async": False,
    }


def _merge_aws_credentials_local_file(config_path: Path, cfg: dict[str, Any]) -> None:
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


def _load_config(path: Path, *, captions_batch_mode: bool = False) -> PipelineConfig:
    raw_cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise ValueError("pipeline config JSON must be an object")
    cfg: dict[str, Any] = {**_pipeline_builtin_json_defaults(), **raw_cfg}
    _merge_aws_credentials_local_file(path, cfg)

    cp_prov = str(cfg.get("caption_provider", "auto")).strip().lower()
    if cp_prov == "karaoke_whisper":
        print(
            "[live] karaoke_whisper is removed; using karaoke_google (Google Cloud Speech karaoke)",
            flush=True,
        )
        cfg["caption_provider"] = "karaoke_google"

    if not str(cfg.get("rules_docx", "") or "").strip():
        cfg["rules_docx"] = str(DEFAULT_HIGHLIGHT_RULES_DOCX)
        print(
            f"[live] rules_docx not set in JSON — using default beside this module: {cfg['rules_docx']}",
            flush=True,
        )
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
    if hl_raw is not None and str(hl_raw).strip().lower() not in ("vertex", ""):
        print(
            f"[live] ignoring highlight_api_provider={str(hl_raw).strip()!r} — "
            "highlight analysis uses Vertex AI Gemini only",
            flush=True,
        )
    highlight_api_provider = "vertex"

    gemini_api_key_val = cfg.get("gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")
    gemini_api_keys_val: list[str] = []
    raw_gemini_keys = cfg.get("gemini_api_keys", [])
    if isinstance(raw_gemini_keys, list):
        gemini_api_keys_val.extend(str(key).strip() for key in raw_gemini_keys if str(key).strip())
    if gemini_api_key_val:
        gemini_api_keys_val.insert(0, gemini_api_key_val)
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        gemini_api_keys_val.append(env_key)
    gemini_api_keys_val = list(dict.fromkeys(gemini_api_keys_val))

    vertex_project_id = (
        str(cfg.get("vertex_project_id", "") or "").strip()
        or os.getenv("VERTEX_PROJECT_ID", "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    vertex_location = str(cfg.get("vertex_location", "us-central1") or "us-central1").strip()

    vertex_api_keys_val: list[str] = []
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

    if not captions_batch_mode:
        if not vertex_project_id:
            raise ValueError(
                "vertex_project_id missing (set in JSON or VERTEX_PROJECT_ID / GOOGLE_CLOUD_PROJECT) "
                "— required for Vertex AI highlight analysis"
            )
        if not vertex_api_keys_val:
            raise ValueError(
                "vertex_api_keys missing (set vertex_api_key / vertex_api_keys or VERTEX_API_KEY) "
                "— required for Vertex AI highlight analysis"
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

    highlight_vertex_audio_only = bool(cfg.get("highlight_vertex_audio_only", False))

    raw_sl_extra = cfg.get("streamlink_extra_args")
    if isinstance(raw_sl_extra, list):
        streamlink_extra_args = [str(x).strip() for x in raw_sl_extra if str(x).strip()]
    else:
        streamlink_extra_args = []

    raw_sl_tw = cfg.get("streamlink_twitch_extra_args")
    if isinstance(raw_sl_tw, list):
        streamlink_twitch_extra_args = [str(x).strip() for x in raw_sl_tw if str(x).strip()]
    else:
        streamlink_twitch_extra_args = ["--twitch-disable-ads", "--twitch-disable-reruns"]

    try:
        streamlink_resolve_timeout_sec = int(cfg.get("streamlink_resolve_timeout_sec", 90))
    except (TypeError, ValueError):
        streamlink_resolve_timeout_sec = 90
    streamlink_resolve_timeout_sec = max(15, min(600, streamlink_resolve_timeout_sec))

    config = PipelineConfig(
        stream_url=cfg["stream_url"],
        api_provider=api_provider,
        highlight_api_provider=highlight_api_provider,
        highlight_parallel_workers=highlight_parallel_workers,
        highlight_yield_to_hud_vision=highlight_yield_to_hud_vision,
        highlight_vertex_audio_only=highlight_vertex_audio_only,
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
        rules_docx=str(cfg.get("rules_docx", "") or ""),
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
        portrait_blur_width=int(cfg.get("portrait_blur_width", 1080)),
        portrait_blur_height=int(cfg.get("portrait_blur_height", 1920)),
        round_roi_x=float(cfg.get("round_roi_x", 0.304)),
        round_roi_y=float(cfg.get("round_roi_y", 0.00)),
        round_roi_w=float(cfg.get("round_roi_w", 0.364)),
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
        karaoke_margin_top_ratio=float(cfg.get("karaoke_margin_top_ratio", 0.22)),
        karaoke_overlay_width_frac=float(cfg.get("karaoke_overlay_width_frac", 0.52)),
        karaoke_overlay_margin_bottom_px=int(cfg.get("karaoke_overlay_margin_bottom_px", 140)),
        karaoke_use_adc=bool(cfg.get("karaoke_use_adc", False)),
        karaoke_no_overlay=bool(cfg.get("karaoke_no_overlay", False)),
        karaoke_overlay_image=str(cfg.get("karaoke_overlay_image", "") or ""),
        karaoke_async=bool(cfg.get("karaoke_async", False)),
        karaoke_ffmpeg_preset=str(cfg.get("karaoke_ffmpeg_preset", "medium") or "medium"),
        karaoke_ffmpeg_crf=int(cfg.get("karaoke_ffmpeg_crf", 20)),
        karaoke_vertex_roster_path=karaoke_vertex_roster_resolved,
        streamlink_extra_args=streamlink_extra_args,
        streamlink_twitch_extra_args=streamlink_twitch_extra_args,
        streamlink_resolve_timeout_sec=streamlink_resolve_timeout_sec,
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
    if not (480 <= config.portrait_blur_width <= 2160):
        raise ValueError("portrait_blur_width must be between 480 and 2160")
    if not (854 <= config.portrait_blur_height <= 3840):
        raise ValueError("portrait_blur_height must be between 854 and 3840")
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


def _reconfigure_stdio_utf8_best_effort() -> None:
    """Windows defaults to cp1252; UTF-8 logs avoid UnicodeEncodeError on arrows/em dashes."""
    if sys.platform != "win32":
        return
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, TypeError):
            pass


def main() -> int:
    _reconfigure_stdio_utf8_best_effort()
    parser = argparse.ArgumentParser(description="Live stream CS2 highlight automation pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--stream-url",
        default=None,
        metavar="URL",
        help=(
            "Override stream_url from JSON. With --interactive, skips URL prompts "
            "(use for repeatable LIVE/VOD runs)."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Interactive setup: LIVE vs RECORDED; LIVE always asks for URL unless --stream-url is set "
            "(JSON VOD presets are not reused when you pick LIVE). VOD may Enter to keep JSON URL; "
            "optional match context (SKIP for none) → Gemini roster when notes provided."
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
            "none | google_speech | shell | auto | karaoke_google (default; Cloud Speech) | karaoke_vertex (legacy Gemini)"
        ),
    )
    parser.add_argument(
        "--captions-batch",
        dest="captions_batch_only",
        action="store_true",
        help=(
            "Same as --captions-batch-only (shorter name; avoids prefix-ambiguity without this exact flag)."
        ),
    )
    parser.add_argument(
        "--captions-batch-only",
        dest="captions_batch_only",
        action="store_true",
        help=(
            "Skip live HUD/capture/highlights; caption batch from round_edited and/or portrait-only finals in "
            "round_final (see --captions-batch-sources). Writes *_karaoke.mp4 and *_final.mp4 under round_final."
        ),
    )
    parser.add_argument(
        "--captions-batch-redo-all",
        action="store_true",
        help="With --captions-batch / --captions-batch-only: re-caption every portrait file even when karaoke is up to date.",
    )
    parser.add_argument(
        "--captions-batch-sources",
        default="edited,final",
        metavar="LIST",
        help=(
            "Comma list with --captions-batch / --captions-batch-only: edited=round_edited, "
            "final=round_final *Portrait_final clips "
            "(copied to meta/captions_batch_stage as *_portrait.mp4). Default edited,final finds finals when "
            "round_edited is empty."
        ),
    )
    parser.add_argument(
        "--karaoke-sync",
        action="store_true",
        help="Wait for karaoke burn before titles/Instagram (sets karaoke_async=false; overrides JSON)",
    )
    parser.add_argument(
        "--karaoke-async",
        action="store_true",
        help="Burn karaoke in the background (sets karaoke_async=true; portrait *_final first; overrides JSON)",
    )
    args = parser.parse_args()

    if args.captions_batch_redo_all and not args.captions_batch_only:
        parser.error("--captions-batch-redo-all requires --captions-batch / --captions-batch-only")
    if args.captions_batch_only and args.interactive:
        parser.error("--captions-batch / --captions-batch-only cannot be used with --interactive")

    _bootstrap_ssl_cert_file_from_certifi()

    cfg_path = Path(args.config).resolve()
    config = _load_config(cfg_path, captions_batch_mode=args.captions_batch_only)
    stream_url_from_cli = bool(args.stream_url is not None and str(args.stream_url).strip())
    if stream_url_from_cli:
        config = replace(config, stream_url=str(args.stream_url).strip())

    if args.interactive:
        print("", flush=True)
        print("[live] --- Interactive session ---", flush=True)
        config = _interactive_session_configure(config, stream_url_from_cli=stream_url_from_cli)

    overrides: dict[str, Any] = {}
    if args.caption_provider is not None:
        cp_cli = str(args.caption_provider).strip().lower()
        if cp_cli == "karaoke_whisper":
            print(
                "[live] karaoke_whisper is removed; using karaoke_google (Google Cloud Speech karaoke)",
                flush=True,
            )
            cp_cli = "karaoke_google"
        overrides["caption_provider"] = cp_cli
    if args.karaoke_sync and args.karaoke_async:
        parser.error("use only one of --karaoke-sync and --karaoke-async")
    if args.karaoke_sync:
        overrides["karaoke_async"] = False
    if args.karaoke_async:
        overrides["karaoke_async"] = True
    if overrides:
        config = replace(config, **overrides)
        allowed_cp = {
            "none",
            "google_speech",
            "shell",
            "auto",
            "karaoke_google",
            "karaoke_vertex",
        }
        if config.caption_provider not in allowed_cp:
            raise ValueError(f"caption_provider must be one of {sorted(allowed_cp)}")

    if args.captions_batch_only:
        config = replace(config, karaoke_async=False)
        cp_default = (config.caption_provider or "").strip().lower()
        if cp_default in ("auto", "none", ""):
            config = replace(config, caption_provider="karaoke_google")

    if not args.captions_batch_only:
        if not (config.vertex_project_id or "").strip():
            raise ValueError("vertex_project_id required for Vertex AI highlight analysis (config or env)")
        if not config.vertex_api_keys:
            raise ValueError("vertex_api_key / vertex_api_keys required for Vertex AI highlight analysis")
    else:
        cp_final = (config.caption_provider or "").strip().lower()
        if cp_final == "karaoke_google" and not config.karaoke_use_adc:
            if not (config.speech_api_key or "").strip():
                raise ValueError(
                    "captions-batch with karaoke_google requires speech credentials: speech_api_key in JSON, "
                    "GOOGLE_SPEECH_API_KEY, speech_api_key.local.json beside config, vertex_api_key fallback in JSON, "
                    "or karaoke_use_adc=true with Application Default Credentials for Cloud Speech-to-Text"
                )
        if cp_final == "karaoke_vertex" and (
            not (config.vertex_project_id or "").strip() or not config.vertex_api_keys
        ):
            raise ValueError(
                "captions-batch with karaoke_vertex requires vertex_project_id and vertex API keys (config/env)"
            )

    pipeline = LiveRoundPipeline(
        config,
        init_mode="captions_batch" if args.captions_batch_only else "live",
    )
    if args.captions_batch_only:
        caps_parts = [
            x.strip().lower()
            for x in (args.captions_batch_sources or "edited,final").split(",")
            if x.strip()
        ]
        caps_sources_fs: frozenset[str] = frozenset(caps_parts) if caps_parts else frozenset({"edited", "final"})
        unknown = caps_sources_fs - frozenset({"edited", "final"})
        if unknown:
            parser.error(f"--captions-batch-sources unknown: {sorted(unknown)} — use edited and/or final")
        return pipeline.run_captions_batch_on_round_edited(
            redo_all=args.captions_batch_redo_all,
            sources=caps_sources_fs,
        )

    if config.highlight_vertex_audio_only:
        print(
            "[live] highlight_vertex_audio_only=true — Vertex receives clip audio + rules_docx text only "
            "(no equipart JPEG / contact sheet for highlight scoring)",
            flush=True,
        )
    pipeline.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
