"""Clip-audio hype analysis: RMS loudness timeline + caster hype-keyword detection.

Extracted from ``live_stream_highlight_pipeline`` so the audio-only highlight
scoring signals (loudness spikes, hype phrases in the transcript) are reusable
and unit-testable in isolation. Only numpy and the stdlib are required.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np

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
