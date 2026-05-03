#!/usr/bin/env python3
"""Burn top-centered karaoke captions (Speech → ASS → ffmpeg).

Transcription backends:

- **whisper** (default): local OpenAI Whisper via ``openai-whisper`` (same models used by the
  WhisperFlow streaming stack). No Google API key.

- **google**: Cloud Speech-to-Text (key resolution matches ``live_stream_highlight_pipeline``).

Examples::

    py -3.14 burn_karaoke_captions.py ^
      --video "C:\\Coding\\Projects\\ca\\CAPTIONS\\round_03_..._portrait_final.mp4"

    py -3.14 burn_karaoke_captions.py --video ... --overlay-image "C:\\path\\brand.png"

    py -3.14 burn_karaoke_captions.py --video ... --no-overlay

    py -3.14 burn_karaoke_captions.py --video ... --backend google --config live_pipeline_config.json

Caption style: neon green word highlight, white other words, ALL CAPS, thick outline.
If ``Screenshot 2026-05-01 164644.png`` sits next to the video, it is overlaid bottom-center automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


def _resolve_speech_api_key(cfg: Dict[str, Any], config_path: Path) -> str:
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


def _default_output(video: Path) -> Path:
    stem = video.stem
    if stem.endswith("_final"):
        stem = stem[: -len("_final")]
    return video.with_name(f"{stem}_karaoke.mp4")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_cfg = script_dir / "live_pipeline_config.json"

    ap = argparse.ArgumentParser(description="Burn karaoke captions via Whisper or Google Speech + ffmpeg.")
    ap.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Coding\Projects\ca\CAPTIONS\round_03_2026-05-01_12-36-03_portrait_final.mp4"),
        help="Input MP4 (default: CAPTIONS portrait_final sample path)",
    )
    ap.add_argument("--output", type=Path, default=None, help="Output MP4 (default: sibling *_karaoke.mp4)")
    ap.add_argument("--config", type=Path, default=default_cfg, help="Pipeline JSON (for Google Speech keys)")
    ap.add_argument("--work-dir", type=Path, default=None, help="WAV/ASS scratch dir (default: video folder)")
    ap.add_argument("--ffmpeg", default="", help="ffmpeg binary path (default: PATH)")
    ap.add_argument("--language", default="en-US", help="Language hint (Google: full code; Whisper: base e.g. en)")
    ap.add_argument(
        "--backend",
        choices=("whisper", "google"),
        default="whisper",
        help="Transcription backend (default: whisper — local, no API key)",
    )
    ap.add_argument(
        "--whisper-model",
        default="small",
        metavar="SIZE",
        help="OpenAI Whisper model name: tiny, base, small, medium, large, etc. (default: small)",
    )
    ap.add_argument(
        "--margin-top-ratio",
        type=float,
        default=0.22,
        help="Vertical placement: fraction of frame height from top (0.22 ≈ upper blurred band)",
    )
    ap.add_argument(
        "--overlay-image",
        type=Path,
        default=None,
        help="PNG/JPEG centered near bottom (default: Video folder Screenshot 2026-05-01 164644.png if present)",
    )
    ap.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable bottom branding overlay",
    )
    ap.add_argument(
        "--overlay-width-frac",
        type=float,
        default=0.52,
        help="Overlay width as fraction of frame width (default: 0.52)",
    )
    ap.add_argument(
        "--overlay-margin-bottom",
        type=int,
        default=140,
        help="Pixels above bottom edge for overlay placement (default: 140; larger = higher on frame)",
    )
    ap.add_argument(
        "--use-adc",
        action="store_true",
        help="With --backend google: use Application Default Credentials instead of API key",
    )
    ap.add_argument(
        "--encode-preset",
        default="medium",
        metavar="PRESET",
        help="libx264 -preset for karaoke burns (default: medium; use slow for max quality)",
    )
    ap.add_argument(
        "--encode-crf",
        type=int,
        default=20,
        metavar="N",
        help="libx264 -crf for karaoke burns (default: 20; lower = bigger files / sharper)",
    )
    args = ap.parse_args()

    video_in = args.video.resolve()
    if not video_in.is_file():
        print(f"[karaoke] input not found: {video_in}", file=sys.stderr)
        return 2

    overlay_img: Path | None = None
    if not args.no_overlay:
        if args.overlay_image is not None:
            overlay_img = args.overlay_image.resolve()
            if not overlay_img.is_file():
                print(f"[karaoke] overlay image not found: {overlay_img}", file=sys.stderr)
                return 2
        else:
            cand = video_in.parent / "Screenshot 2026-05-01 164644.png"
            if cand.is_file():
                overlay_img = cand

    cfg_path = args.config.resolve()
    cfg: Dict[str, Any] = {}
    if args.backend == "google":
        if not cfg_path.is_file():
            print(f"[karaoke] config not found: {cfg_path}", file=sys.stderr)
            return 2
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    api_key: str | None
    if args.backend == "whisper":
        api_key = None
    elif args.use_adc:
        api_key = None
    else:
        api_key = _resolve_speech_api_key(cfg, cfg_path)
        if not api_key:
            print(
                "[karaoke] No Speech API key: set GOOGLE_SPEECH_API_KEY, speech_api_key.local.json, "
                "or keys in config — or pass --use-adc.",
                file=sys.stderr,
            )
            return 2

    ffmpeg_bin = args.ffmpeg.strip() or shutil.which("ffmpeg") or ""
    if not ffmpeg_bin:
        print("[karaoke] ffmpeg not found on PATH", file=sys.stderr)
        return 2

    video_out = (args.output or _default_output(video_in)).resolve()
    work_dir = (args.work_dir or video_in.parent).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(script_dir))
    try:
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        DefaultCredentialsError = None  # type: ignore[misc, assignment]
    from speech_google_captions import transcribe_and_burn_karaoke

    print(f"[karaoke] in : {video_in}", flush=True)
    print(f"[karaoke] out: {video_out}", flush=True)
    print(f"[karaoke] backend: {args.backend}", flush=True)
    if overlay_img:
        print(f"[karaoke] overlay: {overlay_img}", flush=True)
    try:
        info = transcribe_and_burn_karaoke(
            video_in,
            work_dir,
            video_out,
            ffmpeg_bin,
            backend=args.backend,
            whisper_model=args.whisper_model,
            language_code=args.language,
            api_key=api_key,
            margin_v_from_top_ratio=args.margin_top_ratio,
            overlay_image=overlay_img,
            overlay_width_frac=args.overlay_width_frac,
            overlay_margin_bottom_px=args.overlay_margin_bottom,
            encode_preset=str(args.encode_preset),
            encode_crf=int(args.encode_crf),
        )
    except Exception as exc:
        if DefaultCredentialsError is not None and isinstance(exc, DefaultCredentialsError):
            print(f"[karaoke] {exc}", file=sys.stderr)
            print(
                "[karaoke] Set up ADC or use --backend whisper for local captions.",
                file=sys.stderr,
            )
            return 4
        raise
    print(f"[karaoke] words={info.get('word_count')} transcript_len={len(info.get('transcript') or '')}", flush=True)
    print(f"[karaoke] ass={info.get('ass_path')}", flush=True)
    vo = info.get("video_out")
    if vo:
        print(f"[karaoke] wrote: {vo}", flush=True)
    print("[karaoke] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
