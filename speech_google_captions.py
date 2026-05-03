"""Google Cloud Speech-to-Text → SRT / ASS karaoke → ffmpeg burn-in (portrait/clips).

Auth (pick one):

- **Application Default Credentials** (recommended when API keys are disabled): install
  ``google-cloud-speech``, then set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account JSON or run
  ``gcloud auth application-default login``. Used by ``transcribe_google_long_wav(..., api_key=None)``
  and ``burn_karaoke_captions.py --use-adc``.

- **API key** (REST ``speech:recognize``): set ``speech_api_key`` in config or env
  ``GOOGLE_SPEECH_API_KEY``. Works only when your GCP project allows API keys on Speech-to-Text and the
  clip is under ~1 minute (synchronous recognition). Long-running Speech RPCs typically require OAuth.

Enable the Cloud Speech-to-Text API on the GCP project you authenticate against.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_speech_tls_fallback_logged = False


def _google_https_ssl_context() -> ssl.SSLContext:
    """TLS trust bundle for urllib → ``speech.googleapis.com``.

    MSYS2 / some portable Pythons ship without usable system CAs, which yields
    ``CERTIFICATE_VERIFY_FAILED``. Prefer ``SSL_CERT_FILE`` or ``certifi``.

    When neither is available, falls back to **unverified** TLS for these Speech REST calls only
    (still encrypted; weaker trust). Set ``SPEECH_TLS_STRICT=1`` to disable that fallback and fail fast.
    """
    global _speech_tls_fallback_logged
    strict = os.environ.get("SPEECH_TLS_STRICT", "").strip().lower() in ("1", "true", "yes")

    cafile = (os.environ.get("SSL_CERT_FILE") or "").strip()
    if cafile and Path(cafile).is_file():
        return ssl.create_default_context(cafile=cafile)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    if strict:
        return ssl.create_default_context()

    if not _speech_tls_fallback_logged:
        print(
            "[speech] TLS: using unverified HTTPS context for Google Speech only (install ``certifi`` "
            "or set ``SSL_CERT_FILE`` to a CA bundle for verification). "
            "Set SPEECH_TLS_STRICT=1 to forbid this fallback.",
            flush=True,
        )
        _speech_tls_fallback_logged = True
    return ssl._create_unverified_context()


def extract_linear16_wav_mono16k(video_path: Path, ffmpeg_bin: str, wav_out: Path) -> None:
    """Mono 16 kHz LINEAR16 WAV — optimal default for Speech-to-Text."""
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg WAV extract failed\n" + (proc.stderr or proc.stdout or "")[-6000:]
        )


def _dur_to_seconds(d: Any) -> float:
    if d is None:
        return 0.0
    sec = getattr(d, "seconds", 0) or 0
    nano = getattr(d, "nanos", 0) or 0
    return float(sec) + float(nano) * 1e-9


def _parse_rest_duration(val: Any) -> float:
    """REST JSON duration: ``\"1.5s\"`` string or ``{seconds, nanos}`` object."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val.endswith("s"):
            return float(val[:-1])
        return float(val)
    if isinstance(val, dict):
        return float(val.get("seconds", 0) or 0) + float(val.get("nanos", 0) or 0) * 1e-9
    return 0.0


def _json_results_to_words_and_transcript(response: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    words_out: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []
    for result in response.get("results") or []:
        alts = result.get("alternatives") or []
        if not alts:
            continue
        alt = alts[0]
        line = (alt.get("transcript") or "").strip()
        if line:
            transcript_parts.append(line)
        for w in alt.get("words") or []:
            words_out.append(
                {
                    "word": (w.get("word") or "").strip(),
                    "start": _parse_rest_duration(w.get("startTime")),
                    "end": _parse_rest_duration(w.get("endTime")),
                }
            )

    transcript = " ".join(transcript_parts).strip()
    if not words_out and transcript:
        words_out = [
            {"word": transcript, "start": 0.0, "end": max(2.0, len(transcript) * 0.08)},
        ]
    return words_out, transcript


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (wave.Error, OSError):
        data_len = max(0, len(wav_bytes) - 44)
        return data_len / 32000.0


def _speech_rest_sync_recognize(
    wav_bytes: bytes,
    api_key: str,
    language_code: str,
) -> Dict[str, Any]:
    """Synchronous ``speech:recognize`` — API keys are supported for audio up to ~1 minute.

    Returns the JSON ``RecognizeResponse`` dict (same ``results`` layout as LRO inner response).
    """
    base = "https://speech.googleapis.com/v1"
    key_q = urllib.parse.quote(api_key, safe="")
    url = f"{base}/speech:recognize?key={key_q}"
    body = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": language_code,
            "enableWordTimeOffsets": True,
            "enableAutomaticPunctuation": True,
        },
        "audio": {"content": base64.b64encode(wav_bytes).decode("ascii")},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ssl_ctx = _google_https_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=180, context=ssl_ctx) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Speech API recognize failed: HTTP {exc.code} {err_body}") from exc


def transcribe_google_long_wav(
    wav_path: Path,
    *,
    language_code: str = "en-US",
    timeout_sec: float = 600.0,
    api_key: Optional[str] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Long-running recognition on WAV bytes; returns word dicts and full transcript.

    Each word dict: ``{"word": str, "start": float, "end": float}``.

    If ``api_key`` is set, uses REST + API key: synchronous ``speech:recognize`` for clips
    up to ~58 seconds (Google limit ~1 minute). Longer audio requires Application Default
    Credentials + ``google-cloud-speech`` (``api_key=None``).

    With no API key, uses ``google.cloud.speech`` + Application Default Credentials (LRO).
    """
    wav_bytes = wav_path.read_bytes()
    limit = 10 * 1024 * 1024  # inline async limit (~10 MiB)
    if len(wav_bytes) > limit:
        raise RuntimeError(
            f"Extracted WAV is {len(wav_bytes)} bytes (> ~10 MiB inline limit). "
            "Shorten the clip or raise extract bitrate/chunking in speech_google_captions.py."
        )

    if (api_key or "").strip():
        key = api_key.strip()
        duration_sec = _wav_duration_seconds(wav_bytes)
        if duration_sec <= 58.0:
            response = _speech_rest_sync_recognize(wav_bytes, key, language_code)
            return _json_results_to_words_and_transcript(response)
        raise RuntimeError(
            f"Clip audio is ~{duration_sec:.1f}s; Google Speech REST with an API key only supports "
            "synchronous recognition for audio up to ~1 minute, and long-running recognition does not "
            "accept API keys. Install google-cloud-speech, use ``api_key=None`` / ``--use-adc``, and set "
            "``GOOGLE_APPLICATION_CREDENTIALS`` (or run ``gcloud auth application-default login``)."
        )

    try:
        from google.cloud import speech_v1 as speech  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install google-cloud-speech (``pip install google-cloud-speech``) for clips longer than "
            "~58 seconds or when not using an API key; otherwise set speech_api_key / GOOGLE_SPEECH_API_KEY "
            "for synchronous REST captions."
        ) from exc

    client = client_factory() if client_factory else speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )
    audio = speech.RecognitionAudio(content=wav_bytes)

    op = client.long_running_recognize(config=config, audio=audio)
    response = op.result(timeout=timeout_sec)

    words_out: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []

    for result in response.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        line = (alt.transcript or "").strip()
        if line:
            transcript_parts.append(line)
        if getattr(alt, "words", None):
            for w in alt.words:
                words_out.append(
                    {
                        "word": (w.word or "").strip(),
                        "start": _dur_to_seconds(w.start_time),
                        "end": _dur_to_seconds(w.end_time),
                    }
                )

    transcript = " ".join(transcript_parts).strip()
    if not words_out and transcript:
        words_out = [
            {"word": transcript, "start": 0.0, "end": max(2.0, len(transcript) * 0.08)},
        ]

    return words_out, transcript


def words_to_srt(
    words: List[Dict[str, Any]],
    *,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    max_block_seconds: float = 5.0,
    gap_seconds: float = 0.08,
) -> str:
    """Turn word-level timings into simple readable SRT cues."""
    if not words:
        return ""

    blocks: List[Tuple[float, float, str]] = []
    buf: List[str] = []
    b_start = 0.0
    b_end = 0.0
    buf_text_len = 0

    def flush() -> None:
        nonlocal buf, buf_text_len
        if not buf:
            return
        raw = " ".join(buf)
        text = "\n".join(_wrap_lines(raw, max_chars_per_line, max_lines))
        blocks.append((b_start, max(b_end, b_start + gap_seconds), text))
        buf = []
        buf_text_len = 0

    for w in words:
        word = (w.get("word") or "").strip()
        if not word:
            continue
        ws = float(w["start"])
        we = float(w["end"])

        add_len = len(word) + (1 if buf else 0)
        duration_ok = not buf or (we - b_start) <= max_block_seconds
        length_ok = not buf or (buf_text_len + add_len <= max_chars_per_line * max_lines)

        if buf and (not duration_ok or not length_ok):
            flush()

        if not buf:
            b_start = ws
            buf = [word]
            buf_text_len = len(word)
            b_end = we
        else:
            buf.append(word)
            buf_text_len += add_len
            b_end = we

    flush()

    lines: List[str] = []
    for idx, (s, e, txt) in enumerate(blocks, start=1):
        if not txt.strip():
            continue
        lines.append(str(idx))
        lines.append(f"{_srt_ts(s)} --> {_srt_ts(e)}")
        lines.append(txt)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def _wrap_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add <= max_chars:
            cur.append(w)
            cur_len += add
            continue
        if cur:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            lines.append(w[:max_chars])
            cur_len = max_chars
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    return lines[:max_lines]


def _srt_ts(seconds: float) -> str:
    if math.isnan(seconds) or seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


_SRT_BAD = re.compile(r"[\x00-\x08\x0b-\x1f]")


def sanitize_srt_text(s: str) -> str:
    return _SRT_BAD.sub("", s)


def burn_subtitles_ffmpeg(
    video_in: Path,
    srt_path: Path,
    video_out: Path,
    ffmpeg_bin: str,
    *,
    cwd_for_filter: Optional[Path] = None,
) -> None:
    """Burn subtitles with ffmpeg ``subtitles`` filter (libass)."""
    _burn_subtitles_file_ffmpeg(
        video_in, srt_path, video_out, ffmpeg_bin, cwd_for_filter=cwd_for_filter
    )


def burn_ass_subtitles_ffmpeg(
    video_in: Path,
    ass_path: Path,
    video_out: Path,
    ffmpeg_bin: str,
    *,
    cwd_for_filter: Optional[Path] = None,
    x264_preset: str = "slow",
    x264_crf: int = 18,
) -> None:
    """Burn ASS (e.g. karaoke) subtitles with ffmpeg ``subtitles`` filter (libass)."""
    _burn_subtitles_file_ffmpeg(
        video_in,
        ass_path,
        video_out,
        ffmpeg_bin,
        cwd_for_filter=cwd_for_filter,
        x264_preset=x264_preset,
        x264_crf=x264_crf,
    )


def _ffmpeg_subtitles_filter_value(sub_path: Path) -> str:
    """Build a ``subtitles=`` filter argument that works on Windows paths (drive letters, spaces)."""
    p = sub_path.expanduser().resolve().as_posix()
    if os.name == "nt" and len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    p = p.replace("'", r"\'")
    return f"subtitles='{p}'"


def _burn_subtitles_file_ffmpeg(
    video_in: Path,
    sub_path: Path,
    video_out: Path,
    ffmpeg_bin: str,
    *,
    cwd_for_filter: Optional[Path] = None,
    x264_preset: str = "slow",
    x264_crf: int = 18,
) -> None:
    video_out.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path(cwd_for_filter or sub_path.parent).resolve()
    vf = _ffmpeg_subtitles_filter_value(sub_path)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video_in.resolve()),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        x264_preset,
        "-crf",
        str(x264_crf),
        "-c:a",
        "copy",
        str(video_out.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg subtitle burn-in failed\n" + (proc.stderr or proc.stdout or "")[-8000:]
        )


def overlay_image_bottom_center_ffmpeg(
    video_in: Path,
    image_path: Path,
    video_out: Path,
    ffmpeg_bin: str,
    *,
    video_width: int,
    width_frac: float = 0.52,
    margin_bottom_px: int = 140,
    x264_preset: str = "slow",
    x264_crf: int = 18,
) -> None:
    """Composite a PNG/JPEG centered horizontally near the bottom (letterboxed Shorts branding)."""
    if not image_path.is_file():
        raise RuntimeError(f"overlay image not found: {image_path}")
    video_out.parent.mkdir(parents=True, exist_ok=True)
    target_w = max(64, int(float(video_width) * float(width_frac)))
    mb = max(0, int(margin_bottom_px))
    fc = (
        f"[1:v]scale={target_w}:-1[lg];"
        f"[0:v][lg]overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h-{mb}:format=auto[outv]"
    )
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video_in.resolve()),
        "-i",
        str(image_path.resolve()),
        "-filter_complex",
        fc,
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        x264_preset,
        "-crf",
        str(x264_crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(video_out.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg image overlay failed\n" + (proc.stderr or proc.stdout or "")[-8000:]
        )


def pick_writable_mp4_output(preferred: Path) -> Path:
    """Return a path ffmpeg can create.

    If ``preferred`` already exists but is locked (player / IDE preview), use sibling
    ``name_render2.mp4``, ``name_render3.mp4``, … instead of failing with Permission denied.
    """
    preferred = preferred.expanduser().resolve()
    preferred.parent.mkdir(parents=True, exist_ok=True)
    if not preferred.exists():
        return preferred
    try:
        preferred.unlink()
        return preferred
    except OSError:
        pass
    stem = preferred.stem
    suf = preferred.suffix or ".mp4"
    parent = preferred.parent
    for i in range(2, 10000):
        cand = parent / f"{stem}_render{i}{suf}"
        if not cand.exists():
            return cand
    return parent / f"{stem}_render_{os.getpid()}{suf}"


def probe_video_dimensions(video_path: Path, ffmpeg_bin: str) -> Tuple[int, int]:
    """Return ``(width, height)`` of the first video stream via ffprobe."""
    cand = Path(ffmpeg_bin)
    ffprobe = shutil.which("ffprobe") or str(cand.parent / "ffprobe.exe")
    if not Path(ffprobe).exists():
        raise RuntimeError(
            "ffprobe not found (install FFmpeg with ffprobe or place ffprobe beside ffmpeg)."
        )
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video_path.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffprobe failed\n" + (proc.stderr or proc.stdout or "")[-4000:]
        )
    line = (proc.stdout or "").strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) < 2:
        raise RuntimeError(f"unexpected ffprobe output: {proc.stdout!r}")
    return int(parts[0]), int(parts[1])


def _ass_ts(seconds: float) -> str:
    """ASS ``H:MM:SS.cc`` from seconds (centisecond precision)."""
    if math.isnan(seconds) or seconds < 0:
        seconds = 0.0
    cs_total = int(round(seconds * 100))
    h = cs_total // 360000
    cs_total %= 360000
    m = cs_total // 6000
    cs_total %= 6000
    s = cs_total // 100
    cs = cs_total % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_ASS_ESC = re.compile(r"([\\{}])")


def _ass_escape_token(s: str) -> str:
    return _ASS_ESC.sub(r"\\\1", s)


def _ass_fg_block(style_colour: str) -> str:
    """Opening colour override ``{\\1c&H…&}`` before a token."""
    s = style_colour.strip()
    if not s.startswith("&H"):
        s = "&HFFFFFF"
    if not s.endswith("&"):
        s += "&"
    return "{\\1c" + s + "}"


def _metas_rows_clamped(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in metas:
        tok = (m.get("word") or "").strip()
        if not tok:
            continue
        rows.append({"word": tok, "start": float(m["start"]), "end": float(m["end"])})
    for i in range(len(rows) - 1):
        nxt_s = rows[i + 1]["start"]
        rows[i]["end"] = min(rows[i]["end"], nxt_s)
    return rows


def _static_one_line_phrase(
    rows: List[Dict[str, Any]],
    *,
    highlight_idx: Optional[int],
    uppercase: bool,
    highlight_colour: str,
    base_colour: str,
) -> str:
    """One horizontal row of words; ``highlight_idx is None`` ⇒ every word ``base_colour``."""
    parts: List[str] = []
    for j, row in enumerate(rows):
        disp = row["word"].upper() if uppercase else row["word"]
        esc = _ass_escape_token(disp)
        col = highlight_colour if highlight_idx is not None and j == highlight_idx else base_colour
        parts.append(_ass_fg_block(col) + esc)
    return " ".join(parts)


def _dialogues_for_short_block(
    metas: List[Dict[str, Any]],
    *,
    uppercase: bool,
    highlight_colour: str,
    base_colour: str,
    gap_fill_min_sec: float = 0.03,
    word_min_sec: float = 0.03,
    line_open: str = "",
) -> List[str]:
    """Separate ``Dialogue`` rows per millisecond-window so libass shows exactly **one** green word."""
    rows = _metas_rows_clamped(metas)
    if not rows:
        return []

    events: List[Tuple[float, float, str]] = []

    for i in range(len(rows)):
        ws = rows[i]["start"]
        we = max(rows[i]["end"], ws + word_min_sec)
        if i + 1 < len(rows):
            we = min(we, rows[i + 1]["start"])
        we = max(we, ws + 0.02)
        body = _static_one_line_phrase(
            rows,
            highlight_idx=i,
            uppercase=uppercase,
            highlight_colour=highlight_colour,
            base_colour=base_colour,
        )
        events.append((ws, we, body))

    for i in range(len(rows) - 1):
        gs = rows[i]["end"]
        ge = rows[i + 1]["start"]
        if ge - gs >= gap_fill_min_sec:
            body = _static_one_line_phrase(
                rows,
                highlight_idx=None,
                uppercase=uppercase,
                highlight_colour=highlight_colour,
                base_colour=base_colour,
            )
            events.append((gs, ge, body))

    events.sort(key=lambda t: (t[0], t[1]))
    lines_out: List[str] = []
    for ws, we, body in events:
        if we <= ws:
            we = ws + word_min_sec
        lines_out.append(f"Dialogue: 0,{_ass_ts(ws)},{_ass_ts(we)},KaraokeTop,,0,0,0,,{line_open}{body}")
    return lines_out


def words_to_ass_karaoke(
    words: List[Dict[str, Any]],
    *,
    play_res_x: int,
    play_res_y: int,
    max_chars_per_line: int = 36,
    max_lines: int = 1,
    max_words_per_cue: int = 5,
    max_block_seconds: float = 2.0,
    gap_seconds: float = 0.08,
    margin_v_from_top_ratio: float = 0.22,
    font_name: str = "Arial Black",
    font_size: Optional[int] = None,
    karaoke_primary_colour: str = "&H0000FF00",
    karaoke_secondary_colour: str = "&H00FFFFFF",
    karaoke_outline: int = 2,
    karaoke_shadow: int = 6,
    uppercase_words: bool = True,
) -> str:
    """Short **one-row** cues (~``max_words_per_cue`` words). Only **one word is green** at a time."""
    if not words:
        return ""

    fs = font_size if font_size is not None else max(36, int(round(play_res_y * 0.029)))
    margin_v = max(24, int(round(play_res_y * margin_v_from_top_ratio)))

    lines_eff = 1
    mw = max(1, min(int(max_words_per_cue), 8))

    blocks: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    buf: List[Dict[str, Any]] = []
    b_start = 0.0
    b_end = 0.0
    buf_text_len = 0

    def flush_block() -> None:
        nonlocal buf, buf_text_len
        if not buf:
            return
        blocks.append((b_start, max(b_end, b_start + gap_seconds), list(buf)))
        buf = []
        buf_text_len = 0

    for w in words:
        word = (w.get("word") or "").strip()
        if not word:
            continue
        ws = float(w["start"])
        we = float(w["end"])
        add_len = len(word) + (1 if buf else 0)

        if buf and len(buf) >= mw:
            flush_block()

        duration_ok = not buf or (we - b_start) <= max_block_seconds
        length_ok = not buf or (buf_text_len + add_len <= max_chars_per_line * lines_eff)

        if buf and (not duration_ok or not length_ok):
            flush_block()

        if not buf:
            b_start = ws
            buf = [{"word": word, "start": ws, "end": we}]
            buf_text_len = len(word)
            b_end = we
        else:
            buf.append({"word": word, "start": ws, "end": we})
            buf_text_len += add_len
            b_end = we

    flush_block()

    pri = karaoke_secondary_colour
    sec = karaoke_secondary_colour
    outline_col = "&H00000000"
    # BackColour / shadow fill: opaque black (ASS &HAABBGGRR — AA 00 = opaque in VSFilter/libass).
    shadow_col = "&H00000000"
    ko = max(1, int(karaoke_outline))
    ks = max(0, int(karaoke_shadow))
    x_shad = max(2, min(8, ko + 2))
    y_shad = max(4, ks)

    # Hormozi-style: thin black stroke, opaque black block shadow down + slight right, no blur.
    line_open = f"{{\\blur0\\bord{ko}\\xshad{x_shad}\\yshad{y_shad}\\4c&H00000000&\\3c&H00000000&}}"

    dialogue_lines: List[str] = []
    for _s, _e, metas in blocks:
        dialogue_lines.extend(
            _dialogues_for_short_block(
                metas,
                uppercase=uppercase_words,
                highlight_colour=karaoke_primary_colour,
                base_colour=karaoke_secondary_colour,
                line_open=line_open,
            )
        )

    if not dialogue_lines:
        return ""

    header = f"""[Script Info]
Title: Karaoke captions
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KaraokeTop,{font_name},{fs},{pri},{sec},{outline_col},{shadow_col},-1,0,0,0,100,100,0,0,1,{ko},0,8,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header + "\n".join(dialogue_lines) + "\n"


def transcribe_and_burn(
    video_path: Path,
    work_dir: Path,
    video_out: Path,
    ffmpeg_bin: str,
    *,
    language_code: str = "en-US",
    timeout_sec: float = 600.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Full pipeline: extract WAV → Speech-to-Text → SRT → burned MP4."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    wav = work_dir / f"{stem}_speech_16k.wav"
    srt = work_dir / f"{stem}_speech.srt"

    extract_linear16_wav_mono16k(video_path, ffmpeg_bin, wav)
    words, transcript = transcribe_google_long_wav(
        wav,
        language_code=language_code,
        timeout_sec=timeout_sec,
        api_key=api_key,
    )
    srt_body = words_to_srt(words)
    if not srt_body.strip():
        srt_body = (
            "1\n00:00:00,000 --> 00:00:03,000\n(no speech detected)\n"
            if not transcript
            else f"1\n00:00:00,000 --> 00:00:05,000\n{sanitize_srt_text(transcript[:500])}\n"
        )

    srt.write_text(sanitize_srt_text(srt_body), encoding="utf-8")
    burn_subtitles_ffmpeg(video_path, srt, video_out, ffmpeg_bin, cwd_for_filter=srt.parent)

    try:
        wav.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "transcript": transcript,
        "word_count": len(words),
        "srt_path": str(srt),
        "speech_auth": "api_key" if (api_key or "").strip() else "adc",
    }


def transcribe_and_burn_karaoke(
    video_path: Path,
    work_dir: Path,
    video_out: Path,
    ffmpeg_bin: str,
    *,
    backend: str = "whisper",
    whisper_model: str = "small",
    language_code: str = "en-US",
    timeout_sec: float = 600.0,
    api_key: Optional[str] = None,
    margin_v_from_top_ratio: float = 0.22,
    overlay_image: Optional[Path] = None,
    overlay_width_frac: float = 0.52,
    overlay_margin_bottom_px: int = 140,
    karaoke_primary_colour: str = "&H0000FF00",
    karaoke_secondary_colour: str = "&H00FFFFFF",
    karaoke_outline: int = 2,
    karaoke_shadow: int = 6,
    karaoke_uppercase_words: bool = True,
    encode_preset: str = "medium",
    encode_crf: int = 20,
) -> Dict[str, Any]:
    """Extract WAV → Speech-to-Text → ASS karaoke → burned MP4 (top-centered highlight).

    ``backend``: ``\"google\"`` (Cloud Speech + API key / ADC) or ``\"whisper\"`` (local OpenAI Whisper).

    Optional ``overlay_image``: composite branding PNG/JPEG centered near the bottom after subtitles.

    ``encode_preset`` / ``encode_crf``: libx264 settings for subtitle burn and overlay passes (default faster than ``slow``/18).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    wav = work_dir / f"{stem}_speech_16k.wav"
    ass_path = work_dir / f"{stem}_karaoke.ass"

    play_x, play_y = probe_video_dimensions(video_path, ffmpeg_bin)

    extract_linear16_wav_mono16k(video_path, ffmpeg_bin, wav)

    backend_norm = (backend or "whisper").strip().lower()
    if backend_norm == "whisper":
        from speech_whisper_captions import normalize_language_for_whisper, transcribe_whisper_wav

        wl = normalize_language_for_whisper(language_code)
        words, transcript = transcribe_whisper_wav(wav, model_size=whisper_model, language=wl)
        speech_auth = f"whisper:{whisper_model}"
    elif backend_norm == "google":
        try:
            words, transcript = transcribe_google_long_wav(
                wav,
                language_code=language_code,
                timeout_sec=timeout_sec,
                api_key=api_key,
            )
        except RuntimeError as exc:
            msg = str(exc)
            used_key = (api_key or "").strip()
            if used_key and (
                "401" in msg
                or "API keys are not supported" in msg
                or "UNAUTHENTICATED" in msg
                or "CREDENTIALS_MISSING" in msg
            ):
                print(
                    "[speech] REST Speech failed with API key; retrying with Application Default Credentials "
                    "(``google-cloud-speech`` + ``GOOGLE_APPLICATION_CREDENTIALS`` or "
                    "``gcloud auth application-default login``).",
                    flush=True,
                )
                words, transcript = transcribe_google_long_wav(
                    wav,
                    language_code=language_code,
                    timeout_sec=timeout_sec,
                    api_key=None,
                )
            else:
                raise
        speech_auth = "api_key" if (api_key or "").strip() else "adc"
    else:
        raise ValueError(f"Unknown karaoke backend {backend!r}; use \"google\" or \"whisper\".")

    ass_body = words_to_ass_karaoke(
        words,
        play_res_x=play_x,
        play_res_y=play_y,
        margin_v_from_top_ratio=margin_v_from_top_ratio,
        karaoke_primary_colour=karaoke_primary_colour,
        karaoke_secondary_colour=karaoke_secondary_colour,
        karaoke_outline=karaoke_outline,
        karaoke_shadow=karaoke_shadow,
        uppercase_words=karaoke_uppercase_words,
    )
    if not ass_body.strip():
        fallback = transcript.strip() or "(no speech detected)"
        hold = min(5.0, max(2.0, len(fallback) * 0.06))
        ass_body = words_to_ass_karaoke(
            [{"word": fallback[:500], "start": 0.0, "end": hold}],
            play_res_x=play_x,
            play_res_y=play_y,
            margin_v_from_top_ratio=margin_v_from_top_ratio,
            karaoke_primary_colour=karaoke_primary_colour,
            karaoke_secondary_colour=karaoke_secondary_colour,
            karaoke_outline=karaoke_outline,
            karaoke_shadow=karaoke_shadow,
            uppercase_words=karaoke_uppercase_words,
        )

    ass_path.write_text(ass_body, encoding="utf-8")

    out_mp4 = pick_writable_mp4_output(video_out)
    pref_resolved = video_out.expanduser().resolve()
    if out_mp4.resolve() != pref_resolved:
        print(
            "[karaoke] Output file is in use or not replaceable — writing to:\n"
            f"  {out_mp4}\n"
            "(Close VLC/media preview/Cursor tab showing the MP4, then delete *_render*.mp4 or rename.)",
            flush=True,
        )

    overlay_resolved = overlay_image.resolve() if overlay_image else None
    if overlay_resolved is not None and overlay_resolved.is_file():
        tmp_sub = work_dir / f"{stem}_subs_tmp.mp4"
        try:
            burn_ass_subtitles_ffmpeg(
                video_path,
                ass_path,
                tmp_sub,
                ffmpeg_bin,
                cwd_for_filter=ass_path.parent,
                x264_preset=encode_preset,
                x264_crf=encode_crf,
            )
            overlay_image_bottom_center_ffmpeg(
                tmp_sub,
                overlay_resolved,
                out_mp4,
                ffmpeg_bin,
                video_width=play_x,
                width_frac=overlay_width_frac,
                margin_bottom_px=overlay_margin_bottom_px,
                x264_preset=encode_preset,
                x264_crf=encode_crf,
            )
        finally:
            try:
                tmp_sub.unlink(missing_ok=True)
            except OSError:
                pass
    else:
        burn_ass_subtitles_ffmpeg(
            video_path,
            ass_path,
            out_mp4,
            ffmpeg_bin,
            cwd_for_filter=ass_path.parent,
            x264_preset=encode_preset,
            x264_crf=encode_crf,
        )

    try:
        wav.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "transcript": transcript,
        "word_count": len(words),
        "ass_path": str(ass_path),
        "speech_auth": speech_auth,
        "backend": backend_norm,
        "play_res": [play_x, play_y],
        "overlay_image": str(overlay_resolved) if overlay_resolved and overlay_resolved.is_file() else "",
        "video_out": str(out_mp4.resolve()),
    }
