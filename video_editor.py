"""Portrait export for horizontal gameplay clips (e.g. 16:9 → 9:16).

Creates a tall canvas with sharp gameplay centered and blurred continuation on the top/bottom
(TikTok/Reels-style letterboxing filled with blurred video).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def apply_portrait_blur(
    input_video: str | Path,
    output_video: str | Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    width: int = 1080,
    height: int = 1920,
    blur_luma_radius: int = 20,
    blur_chroma_radius: int = 8,
    crf: int = 18,
    preset: str = "slow",
    audio_bitrate: str = "160k",
    fps: float | None = 30.0,
) -> None:
    """Blur-filled portrait (9:16): scaled/blurred background + sharp foreground centered.

    Args:
        input_video: Source clip (typically landscape).
        output_video: Destination MP4 path.
        ffmpeg_bin: ffmpeg executable name or path.
        width / height: Output frame size (default 1080×1920).
        blur_*: ``boxblur`` luma/chroma radii for top/bottom (and side) fill behind letterbox.
        crf / preset / audio_bitrate: libx264 / AAC encoding.
        fps: Output frame rate (``None`` = ffmpeg default / stream).

    Raises:
        RuntimeError: If ffmpeg exits non-zero.
    """
    inp = Path(input_video)
    outp = Path(output_video)
    outp.parent.mkdir(parents=True, exist_ok=True)

    fc = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur={blur_luma_radius}:{blur_chroma_radius}[bg];"
        f"[0:v]scale={width}:-2:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    )
    cmd: list[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(inp),
        "-filter_complex",
        fc,
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
    ]
    if fps is not None:
        cmd.extend(["-r", str(fps)])
    cmd.append(str(outp))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg portrait blur failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr[-8000:]}"
        )


if __name__ == "__main__":
    pass
