"""ffmpeg/ffprobe invocation and media-duration probing.

Duration is read two ways because neither is universally reliable: ffprobe
reports the container's declared duration, which is missing or wrong for some
stream recordings, so a fallback parses ffmpeg's own ``Duration:`` header line.
Both return ``None`` rather than raising, since callers treat an unknown
duration as "analyse the whole clip".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

FFPROBE_TIMEOUT_SEC = 90
FFMPEG_HEADER_TIMEOUT_SEC = 120

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d+)")


def run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command, raising RuntimeError with full output on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def resolve_ffprobe_bin(ffmpeg_bin: str | None) -> str | None:
    """Locate ``ffprobe`` on PATH or next to ``ffmpeg`` (Windows-friendly)."""
    found = shutil.which("ffprobe")
    if found:
        return found
    if ffmpeg_bin:
        parent = Path(ffmpeg_bin).resolve().parent
        for name in ("ffprobe.exe", "ffprobe"):
            candidate = parent / name
            if candidate.is_file():
                return str(candidate)
    return None


def ffprobe_duration_sec(media_path: Path, ffmpeg_bin: str | None) -> float | None:
    """Return container duration in seconds, or None if unknown."""
    exe = resolve_ffprobe_bin(ffmpeg_bin)
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SEC)
        if proc.returncode != 0:
            return None
        lines = (proc.stdout or "").strip().splitlines()
        if not lines:
            return None
        val = float(lines[0])
        if val <= 0 or val != val:  # NaN check
            return None
        return val
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None


def ffmpeg_demuxer_duration_sec(media_path: Path, ffmpeg_bin: str | None) -> float | None:
    """Parse ``Duration:`` from ``ffmpeg -i`` stderr (header read only; no full decode)."""
    if not ffmpeg_bin:
        return None
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-i", str(media_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_HEADER_TIMEOUT_SEC)
    except (OSError, subprocess.TimeoutExpired):
        return None
    blob = (proc.stderr or "") + (proc.stdout or "")
    match = _DURATION_RE.search(blob)
    if not match:
        return None
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), int(match.group(3))
    frac = match.group(4)
    try:
        sub = int(frac) / (10 ** len(frac))
    except ValueError:
        return None
    val = hours * 3600 + minutes * 60 + seconds + sub
    return val if val > 0 else None


def clip_duration_for_analysis(media_path: Path, ffmpeg_bin: str | None) -> float | None:
    """Best-effort clip duration: ffprobe first, ffmpeg header parse as fallback."""
    duration = ffprobe_duration_sec(media_path, ffmpeg_bin)
    if duration is not None:
        return duration
    return ffmpeg_demuxer_duration_sec(media_path, ffmpeg_bin)
