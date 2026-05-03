#!/usr/bin/env python3
"""One-off: send a clip contact sheet to NVIDIA vision API with CS2_Highlights.docx rules context."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from live_stream_highlight_pipeline import (
    _chat_text,
    _extract_docx_text,
    _extract_json,
    _file_data_url,
    _json_post,
)


def _run_ffmpeg(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def _extract_frames(clip: Path, frame_dir: Path, ffmpeg: str) -> List[Path]:
    pattern = frame_dir / "frame_%03d.jpg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(clip),
        "-vf",
        "fps=1/8,scale=960:-2",
        "-frames:v",
        "10",
        "-q:v",
        "3",
        str(pattern),
    ]
    _run_ffmpeg(cmd)
    return sorted(frame_dir.glob("frame_*.jpg"))


def _contact_sheet(frames: List[Path], out: Path) -> Path:
    selected = frames[:6]
    thumbs: List[Image.Image] = []
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
    sheet.save(out, "JPEG", quality=85, optimize=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask NVIDIA vision whether a CS2 clip is a highlight vs rules DOCX.")
    parser.add_argument(
        "--clip",
        type=Path,
        default=Path(r"C:\Coding\Projects\ca\Esports-Video-clipping-automation\Original Clips\Clip4.mp4"),
        help="Path to MP4 clip",
    )
    parser.add_argument(
        "--rules-docx",
        type=Path,
        default=None,
        help="Path to CS2_Highlights.docx (default: beside live_pipeline_config / cwd)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="NVIDIA API key (else CLIP4_NVIDIA_API_KEY or NVIDIA_API_KEY env)",
    )
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--model", default="meta/llama-3.2-90b-vision-instruct")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write parsed JSON to this path (UTF-8). Default: beside clip as <stem>_nvidia_analysis.json",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = (os.getenv("CLIP4_NVIDIA_API_KEY") or "").strip()
    if not api_key:
        api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        print(
            "Missing NVIDIA API key: pass --api-key or set NVIDIA_API_KEY or CLIP4_NVIDIA_API_KEY.",
            file=sys.stderr,
        )
        return 2

    rules_path = args.rules_docx
    if rules_path is None:
        candidates = [
            repo / "CS2_Highlights.docx",
            repo / "Original Clips" / "CS2_Highlights.docx",
        ]
        rules_path = next((p for p in candidates if p.exists()), candidates[0])

    rules_text = ""
    if rules_path.exists():
        rules_text = _extract_docx_text(rules_path)
        print(f"[rules] loaded {rules_path} ({len(rules_text)} chars)")
    else:
        print(f"[rules] WARNING: not found at {rules_path}; continuing with empty rules_context.")

    clip = args.clip.resolve()
    if not clip.exists():
        print(f"Clip not found: {clip}", file=sys.stderr)
        return 2

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found in PATH", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="nv_clip_") as tmp:
        td = Path(tmp)
        frames = _extract_frames(clip, td, ffmpeg)
        if not frames:
            print("No frames extracted", file=sys.stderr)
            return 3
        sheet_path = td / "contact_sheet.jpg"
        _contact_sheet(frames, sheet_path)

        prompt = (
            "OUTPUT RULES: Reply with ONLY a single JSON object. No markdown, no **bold**, no headings, "
            "no preamble or trailing text. First character must be { and last must be }.\n\n"
            "You are judging a CS2 gameplay clip from a contact sheet of sampled frames (left-to-right, top-to-bottom "
            "is time order).\n"
            "First describe what is happening in the round from the visuals: map area or site if inferable, "
            "Buy/round phase if visible, bomb plant/defuse/skills/duels/kill feed/trades, and how the sequence evolves. "
            "If something is unclear, say so; do not invent stats you cannot see.\n"
            "Then use ONLY the highlight criteria in rules_context for the yes/no highlight decision.\n\n"
            "JSON schema (fill every key):\n"
            '{"round_description": string, "is_highlight": boolean, "confidence": number 0-1, '
            '"why_highlight": [string], "why_not_highlight": [string], '
            '"rules_matched": [string], "final_reason": string}\n\n'
            "round_description: STRICT maximum ~120 words / 6 sentences. "
            "Present tense; grounded in frames only. Do not repeat ideas or pad.\n"
            "rules_matched must quote or paraphrase which bullet/rule lines from rules_context justify your decision "
            "(empty array if not a highlight).\n"
            "final_reason must briefly explain yes/no and cite rules_context.\n\n"
            f"Clip file (context): {clip.name}\n\n"
            "rules_context:\n"
            f"{rules_text[:12000]}"
        )

        payload: Dict[str, Any] = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _file_data_url(sheet_path)}},
                    ],
                }
            ],
            # Vision JSON can be verbose; truncation breaks parse (must fit full object).
            "max_tokens": 8192,
            "temperature": 0.0,
            # OpenAI-compatible: bias model toward valid JSON (omit if API rejects).
            "response_format": {"type": "json_object"},
        }

        print(f"[nvidia] POST {args.base_url}/chat/completions model={args.model}")
        resp = _json_post(f"{args.base_url}/chat/completions", api_key, payload, timeout_sec=300)
        raw_text = _chat_text(resp)
        print("\n--- Raw model text ---\n")
        print(raw_text[:8000])
        print("\n--- Parsed JSON ---\n")
        try:
            parsed = _extract_json(raw_text)
            print(json.dumps(parsed, indent=2))
            out_path = args.output
            if out_path is None:
                out_path = clip.with_name(f"{clip.stem}_nvidia_analysis.json")
            out_path = out_path.resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            print(f"\n[saved] {out_path}", file=sys.stderr)
        except Exception as exc:
            print(f"(parse failed: {exc})", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
