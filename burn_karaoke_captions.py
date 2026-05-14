#!/usr/bin/env python3
"""Burn top-centered karaoke captions (Google Cloud Speech → ASS → ffmpeg).

Transcription uses Cloud Speech-to-Text (key resolution matches ``live_stream_highlight_pipeline``).

Examples::

    py -3.14 burn_karaoke_captions.py ^
      --video "C:\\Coding\\Projects\\ca\\CAPTIONS\\round_03_..._portrait_final.mp4"

    py -3.14 burn_karaoke_captions.py --video ... --overlay-image "C:\\path\\brand.png"

    py -3.14 burn_karaoke_captions.py --video ... --no-overlay

    py -3.14 burn_karaoke_captions.py --video ... --config live_pipeline_config.json

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

    ap = argparse.ArgumentParser(description="Burn karaoke captions via Google Cloud Speech + ffmpeg.")
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
    ap.add_argument("--language", default="en-US", help="Cloud Speech language code (e.g. en-US)")
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
        help="Use Application Default Credentials instead of API key for Cloud Speech",
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
    ap.add_argument(
        "--speech-recognition-timeout-sec",
        type=float,
        default=None,
        metavar="SEC",
        help="Speech-to-text timeout for long clips (default: speech_recognition_timeout_sec from pipeline JSON or 600)",
    )
    ap.add_argument(
        "--karaoke-caption-time-offset-sec",
        type=float,
        default=None,
        metavar="SEC",
        help="Uniform seconds added to karaoke word timings after mux alignment (negative = captions earlier)",
    )
    ap.add_argument(
        "--karaoke-disable-av-mux-timing-fix",
        action="store_true",
        help="Disable ffprobe A/V stream start_time shift (speech vs video PTS alignment)",
    )
    ap.add_argument(
        "--karaoke-invert-mux-timing",
        action="store_true",
        help="Invert audio-minus-video mux correction if sync is systematically wrong",
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
    if not cfg_path.is_file():
        print(f"[karaoke] config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))

    if args.speech_recognition_timeout_sec is not None:
        speech_timeout = float(args.speech_recognition_timeout_sec)
    else:
        try:
            speech_timeout = float(cfg.get("speech_recognition_timeout_sec") or 600)
        except (TypeError, ValueError):
            speech_timeout = 600.0
    speech_timeout = float(max(60.0, speech_timeout))

    api_key: str | None = None if args.use_adc else _resolve_speech_api_key(cfg, cfg_path)
    if api_key == "":
        api_key = None

    def _truthy_json(v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

    if args.karaoke_caption_time_offset_sec is not None:
        karaoke_offset_sec = float(args.karaoke_caption_time_offset_sec)
    else:
        raw_off = cfg.get("karaoke_caption_time_offset_sec")
        karaoke_offset_sec = 0.0
        if raw_off is not None:
            try:
                karaoke_offset_sec = float(raw_off)
            except (TypeError, ValueError):
                karaoke_offset_sec = 0.0

    disable_mux_fix = bool(args.karaoke_disable_av_mux_timing_fix) or _truthy_json(
        cfg.get("karaoke_disable_av_mux_timing_fix")
    )
    invert_mux_fix = bool(args.karaoke_invert_mux_timing) or _truthy_json(
        cfg.get("karaoke_google_invert_mux_timing_fix")
    ) or _truthy_json(cfg.get("karaoke_vertex_invert_mux_timing_fix"))

    if not args.use_adc and not api_key:
        print(
            "[karaoke] No Speech credentials: pass --use-adc with ADC set up, "
            "or set GOOGLE_SPEECH_API_KEY / speech_api_key.local.json / keys in pipeline JSON.",
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
    print(f"[karaoke] backend: google_cloud_speech", flush=True)
    if overlay_img:
        print(f"[karaoke] overlay: {overlay_img}", flush=True)
    try:
        info = transcribe_and_burn_karaoke(
            video_in,
            work_dir,
            video_out,
            ffmpeg_bin,
            language_code=args.language,
            timeout_sec=speech_timeout,
            api_key=api_key,
            margin_v_from_top_ratio=args.margin_top_ratio,
            overlay_image=overlay_img,
            overlay_width_frac=args.overlay_width_frac,
            overlay_margin_bottom_px=args.overlay_margin_bottom,
            encode_preset=str(args.encode_preset),
            encode_crf=int(args.encode_crf),
            karaoke_caption_time_offset_sec=karaoke_offset_sec,
            karaoke_disable_av_mux_timing_fix=disable_mux_fix,
            karaoke_invert_mux_timing_fix=invert_mux_fix,
        )
    except Exception as exc:
        if DefaultCredentialsError is not None and isinstance(exc, DefaultCredentialsError):
            print(f"[karaoke] {exc}", file=sys.stderr)
            print("[karaoke] Set up ADC (``gcloud auth application-default login`` or a service-account JSON).", file=sys.stderr)
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
