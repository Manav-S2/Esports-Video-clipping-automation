#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from detect_cs2_highlight import _extract_sample_frames, _generate_multimodal_json


def main() -> int:
    root = Path(__file__).resolve().parent
    clip = root / "Original Clips" / "Clip5.mp4"
    tmp = root / "Original Clips" / "_tmp_clip5_obs"
    tmp.mkdir(parents=True, exist_ok=True)

    frames = _extract_sample_frames(clip, tmp, sample_seconds=5.0, max_frames=10)

    prompt = (
        "Analyze this CS2 clip and return ONLY strict JSON: "
        "{"
        "\"map_guess\": string,"
        "\"score_t\": number|null,"
        "\"score_ct\": number|null,"
        "\"round_guess\": number|null,"
        "\"notable_players\": [string],"
        "\"events\": [{\"attacker\": string, \"victim\": string, \"weapon\": string}],"
        "\"is_highlight\": boolean,"
        "\"confidence\": number,"
        "\"reason\": string"
        "}."
    )

    vertex_project_id = (os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    vertex_api_key = (os.getenv("VERTEX_API_KEY") or "").strip()
    if not vertex_project_id or not vertex_api_key:
        raise RuntimeError("Set VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) and VERTEX_API_KEY.")

    data = _generate_multimodal_json(
        prompt=prompt,
        image_paths=frames,
        ai_backend="vertex",
        model="gemini-2.5-flash",
        gemini_api_key=None,
        vertex_project_id=vertex_project_id,
        vertex_location=os.getenv("VERTEX_LOCATION", "us-central1").strip() or "us-central1",
        vertex_api_key=vertex_api_key,
    )

    out = root / "clip5_gemini_observation.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    for p in tmp.glob("*.jpg"):
        p.unlink(missing_ok=True)
    tmp.rmdir()

    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
