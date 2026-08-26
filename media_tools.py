"""Cross-platform ffmpeg utilities for OCR-grade video enhancement.

Ported from the original PowerShell tooling so the same commands run on
Windows, Linux, and macOS and so the filter chains can be unit-tested without
invoking ffmpeg.

Command construction is kept pure (``build_*_command``) and separated from
execution (``run_*``), which is what makes the filter graphs testable.

CLI::

    python media_tools.py ocr-optimize --input clip.mp4 --scale-factor 2 --binarize
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from errors import ToolNotFoundError

VALID_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


def resolve_ffmpeg(ffmpeg_bin: str = "ffmpeg") -> str:
    """Return an executable ffmpeg path, or raise ToolNotFoundError."""
    found = shutil.which(ffmpeg_bin)
    if not found:
        raise ToolNotFoundError(ffmpeg_bin, hint="required for video processing")
    return found


def default_ocr_output_path(input_video: Path, *, lossless: bool = False) -> Path:
    """Sibling output path: ``clip.mp4`` -> ``clip.ocr-max.mp4`` (``.mkv`` when lossless)."""
    suffix = ".ocr-max.mkv" if lossless else ".ocr-max.mp4"
    return input_video.with_name(f"{input_video.stem}{suffix}")


def build_ocr_filter_chain(scale_factor: int = 2, *, binarize: bool = False, threshold: int = 150) -> str:
    """OCR-focused filter chain: denoise, contrast, sharpen, Lanczos upscale, optional threshold."""
    if not 1 <= scale_factor <= 4:
        raise ValueError(f"scale_factor must be between 1 and 4, got {scale_factor}")
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be between 0 and 255, got {threshold}")

    parts = [
        "hqdn3d=1.2:1.2:6:6",
        "eq=contrast=1.55:brightness=0.02:gamma=0.95",
        "unsharp=7:7:2.2:7:7:0.0",
        f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos+accurate_rnd+full_chroma_int",
        "unsharp=5:5:1.4:5:5:0.0",
    ]
    if binarize:
        # Strong thresholding for high-contrast text; tune threshold for your footage.
        parts.append(f"lutyuv=y='if(gte(val,{threshold}),255,0)'")
    return ",".join(parts)


def build_ocr_optimize_command(
    input_video: Path,
    output_video: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    scale_factor: int = 2,
    crf: int = 18,
    preset: str = "slow",
    codec: str = "h264",
    lossless: bool = False,
    binarize: bool = False,
    threshold: int = 150,
) -> list[str]:
    """Build the full ffmpeg argv for an OCR-optimized encode."""
    if codec not in ("h264", "h265"):
        raise ValueError(f"codec must be 'h264' or 'h265', got {codec!r}")
    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}, got {preset!r}")
    if not 0 <= crf <= 51:
        raise ValueError(f"crf must be between 0 and 51, got {crf}")

    vf = build_ocr_filter_chain(scale_factor, binarize=binarize, threshold=threshold)
    cmd: list[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        vf,
    ]

    if lossless:
        cmd += [
            "-c:v", "ffv1",
            "-level", "3",
            "-coder", "1",
            "-context", "1",
            "-g", "1",
            "-slices", "24",
            "-slicecrc", "1",
        ]
    elif codec == "h265":
        cmd += [
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",
            "-movflags", "+faststart",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

    cmd.append(str(output_video))
    return cmd


def run_ocr_optimize(
    input_video: Path,
    output_video: Path | None = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
    **kwargs,
) -> Path:
    """Run the OCR-optimizing encode; returns the output path."""
    input_video = Path(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    out = Path(output_video) if output_video else default_ocr_output_path(
        input_video, lossless=bool(kwargs.get("lossless", False))
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ocr_optimize_command(input_video, out, ffmpeg_bin=ffmpeg, **kwargs)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg OCR optimize failed (exit {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{(proc.stderr or '')[-8000:]}"
        )
    return out


def escape_subtitles_path(path: Path) -> str:
    """Escape a path for ffmpeg's ``subtitles=`` filter argument.

    The filter parser treats ``\\`` and ``:`` specially, so Windows paths such as
    ``C:\\clips\\a.srt`` must become ``C\\:/clips/a.srt``.
    """
    text = str(Path(path).resolve()).replace("\\", "/")
    return text.replace(":", "\\:")


def build_vertical_layout_filter(
    canvas_width: int = 1080,
    canvas_height: int = 1920,
    *,
    blur_strength: int = 26,
    background_darken: float = 0.12,
) -> str:
    """Blurred-background vertical layout: darkened blurred fill + centered source."""
    if not 720 <= canvas_width <= 2160:
        raise ValueError(f"canvas_width must be 720..2160, got {canvas_width}")
    if not 1280 <= canvas_height <= 3840:
        raise ValueError(f"canvas_height must be 1280..3840, got {canvas_height}")
    if not 4 <= blur_strength <= 80:
        raise ValueError(f"blur_strength must be 4..80, got {blur_strength}")
    if not 0.0 <= background_darken <= 0.8:
        raise ValueError(f"background_darken must be 0.0..0.8, got {background_darken}")

    return (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={canvas_width}:{canvas_height},"
        f"boxblur={blur_strength}:{blur_strength},eq=brightness=-{background_darken}[bg];"
        f"[fgsrc]scale={canvas_width}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase]"
    )


def build_vertical_caption_command(
    input_video: Path,
    output_video: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    captions_file: Path | None = None,
    canvas_width: int = 1080,
    canvas_height: int = 1920,
    blur_strength: int = 26,
    background_darken: float = 0.12,
    caption_font_size: int = 64,
    caption_font: str = "Montserrat ExtraBold",
) -> list[str]:
    """Build the ffmpeg argv for a vertical edit with optional burned-in captions."""
    if not 24 <= caption_font_size <= 120:
        raise ValueError(f"caption_font_size must be 24..120, got {caption_font_size}")

    layout = build_vertical_layout_filter(
        canvas_width,
        canvas_height,
        blur_strength=blur_strength,
        background_darken=background_darken,
    )

    if captions_file is not None:
        style = (
            f"FontName={caption_font},FontSize={caption_font_size},Alignment=8,MarginV=120,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
            "Bold=1,BorderStyle=1,Outline=3,Shadow=0"
        )
        subtitle_filter = (
            f"[vbase]subtitles='{escape_subtitles_path(captions_file)}':force_style='{style}'[vout]"
        )
        filter_complex = f"{layout};{subtitle_filter}"
    else:
        filter_complex = f"{layout};[vbase]null[vout]"

    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]


def run_vertical_caption_edit(
    input_video: Path,
    output_video: Path | None = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
    captions_file: Path | None = None,
    **kwargs,
) -> Path:
    """Run the vertical (9:16) edit; returns the output path."""
    input_video = Path(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if captions_file is not None:
        captions_file = Path(captions_file)
        if not captions_file.is_file():
            raise FileNotFoundError(f"Captions file not found: {captions_file}")

    ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    out = Path(output_video) if output_video else input_video.with_name(
        f"{input_video.stem}.vertical.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_vertical_caption_command(
        input_video, out, ffmpeg_bin=ffmpeg, captions_file=captions_file, **kwargs
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg vertical edit failed (exit {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{(proc.stderr or '')[-8000:]}"
        )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ffmpeg video utilities for OCR-grade enhancement.")
    sub = parser.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr-optimize", help="Enhance a video for downstream OCR/killfeed reading.")
    ocr.add_argument("--input", required=True, type=Path, help="Source video path.")
    ocr.add_argument("--output", type=Path, help="Destination path (default: <input>.ocr-max.mp4).")
    ocr.add_argument("--scale-factor", type=int, default=2, choices=range(1, 5), help="Upscale multiplier.")
    ocr.add_argument("--crf", type=int, default=18, help="Quality for compressed modes (lower is better).")
    ocr.add_argument("--preset", default="slow", choices=VALID_PRESETS, help="Encoder speed/compression.")
    ocr.add_argument("--codec", default="h264", choices=("h264", "h265"), help="Compressed-mode codec.")
    ocr.add_argument("--lossless", action="store_true", help="Encode FFV1/MKV instead (very large files).")
    ocr.add_argument("--binarize", action="store_true", help="Hard black/white text mode.")
    ocr.add_argument("--threshold", type=int, default=150, help="Threshold used with --binarize.")
    ocr.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path.")

    vert = sub.add_parser("vertical-edit", help="Reframe to 9:16 with blurred fill and optional captions.")
    vert.add_argument("--input", required=True, type=Path, help="Source video path.")
    vert.add_argument("--output", type=Path, help="Destination path (default: <input>.vertical.mp4).")
    vert.add_argument("--captions", type=Path, help="Subtitle file (.srt/.ass) to burn in.")
    vert.add_argument("--canvas-width", type=int, default=1080, help="Output width.")
    vert.add_argument("--canvas-height", type=int, default=1920, help="Output height.")
    vert.add_argument("--blur-strength", type=int, default=26, help="Background boxblur radius.")
    vert.add_argument("--background-darken", type=float, default=0.12, help="Background brightness reduction.")
    vert.add_argument("--caption-font-size", type=int, default=64, help="Burned-in caption size.")
    vert.add_argument("--caption-font", default="Montserrat ExtraBold", help="Burned-in caption font name.")
    vert.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "ocr-optimize":
        out = run_ocr_optimize(
            args.input,
            args.output,
            ffmpeg_bin=args.ffmpeg_bin,
            scale_factor=args.scale_factor,
            crf=args.crf,
            preset=args.preset,
            codec=args.codec,
            lossless=args.lossless,
            binarize=args.binarize,
            threshold=args.threshold,
        )
        print(f"OCR-optimized video written to: {out}")
    elif args.command == "vertical-edit":
        out = run_vertical_caption_edit(
            args.input,
            args.output,
            ffmpeg_bin=args.ffmpeg_bin,
            captions_file=args.captions,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            blur_strength=args.blur_strength,
            background_darken=args.background_darken,
            caption_font_size=args.caption_font_size,
            caption_font=args.caption_font,
        )
        print(f"Vertical edit written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
