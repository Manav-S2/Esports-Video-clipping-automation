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
    return 0


if __name__ == "__main__":
    sys.exit(main())
