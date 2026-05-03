"""Local Whisper transcription with word-level timings (OpenAI ``whisper`` package).

The `WhisperFlow <https://pypi.org/project/whisperflow/>`_ library wraps the same underlying Whisper
models for **streaming** use cases. For batch clips (karaoke burn-in), calling Whisper directly avoids
extra services and dependency pins while still producing ``{"word","start","end"}`` rows compatible
with ``speech_google_captions.words_to_ass_karaoke`` / ``words_to_srt``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def normalize_language_for_whisper(language_code: str) -> Optional[str]:
    """Map ``en-US`` → ``en``; empty → ``None`` (auto-detect)."""
    lc = (language_code or "").strip().replace("_", "-")
    if not lc:
        return None
    base = lc.split("-", 1)[0].lower()
    return base if len(base) >= 2 else None


def transcribe_whisper_wav(
    wav_path: Path,
    *,
    model_size: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run Whisper on a 16 kHz mono WAV; returns word dicts and full transcript text."""
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install openai-whisper (``pip install openai-whisper``) for local Whisper captions."
        ) from exc

    load_kw: Dict[str, Any] = {}
    if device:
        load_kw["device"] = device
    cache_key = (str(model_size).strip().lower(), str(device or "").strip())
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = whisper.load_model(model_size, **load_kw)
            _MODEL_CACHE[cache_key] = model

    transcribe_kw: Dict[str, Any] = {"word_timestamps": True, "verbose": False}
    if language:
        transcribe_kw["language"] = language

    result = model.transcribe(str(wav_path), **transcribe_kw)
    transcript = (result.get("text") or "").strip()

    words_out: List[Dict[str, Any]] = []
    for seg in result.get("segments") or []:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                tok = (w.get("word") or "").strip()
                if not tok:
                    continue
                words_out.append(
                    {
                        "word": tok,
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                    }
                )
            continue
        line = (seg.get("text") or "").strip()
        if line:
            words_out.append(
                {
                    "word": line,
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", max(seg.get("start", 0.0) + 0.5, 0.5))),
                }
            )

    if not words_out and transcript:
        words_out = [
            {"word": transcript, "start": 0.0, "end": max(2.0, len(transcript) * 0.08)},
        ]

    return words_out, transcript
