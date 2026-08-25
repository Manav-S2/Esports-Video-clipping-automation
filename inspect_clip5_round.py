#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from detect_cs2_highlight import (
    _apply_per_map_highlight_cap,
    _assign_round_numbers,
    _extract_sample_frames,
    _generate_multimodal_json,
    _load_kill_events,
    _load_round_start_events,
    _normalize_kill_time_column,
    _score_rounds,
)


def _canon(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _collect_round_kill_pairs(kills_df) -> dict[int, list[tuple[str, str]]]:
    pairs: dict[int, list[tuple[str, str]]] = {}
    if kills_df is None or len(kills_df) == 0 or "_round_number" not in kills_df.columns:
        return pairs

    cols = list(kills_df.columns)
    attacker_col = None
    victim_col = None
    for c in cols:
        lc = c.lower()
        if attacker_col is None and lc in {"attacker_name", "attacker", "killer_name", "killer"}:
            attacker_col = c
        if victim_col is None and lc in {"user_name", "victim_name", "user", "victim"}:
            victim_col = c

    if attacker_col is None or victim_col is None:
        return pairs

    for _, r in kills_df.iterrows():
        rnd = int(r.get("_round_number", 0))
        if rnd <= 0:
            continue
        a = _canon(str(r.get(attacker_col, "")))
        v = _canon(str(r.get(victim_col, "")))
        if not a and not v:
            continue
        pairs.setdefault(rnd, []).append((a, v))
    return pairs


def _extract_clip_observations(clip_path: Path, key: str, project_id: str, location: str, model: str) -> dict[str, Any]:
    tmp = clip_path.parent / "_tmp_clip5_frames"
    if tmp.exists():
        for p in tmp.glob("*.jpg"):
            p.unlink(missing_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    frames = _extract_sample_frames(clip_path, tmp, sample_seconds=4.0, max_frames=12)
    if not frames:
        return {"error": "No frames extracted from clip."}

    prompt = (
        "You are analyzing a CS2 broadcast clip. Return ONLY strict JSON with schema: "
        "{"
        "\"map_guess\": string,"
        "\"score_t\": number|null,"
        "\"score_ct\": number|null,"
        "\"round_guess\": number|null,"
        "\"events\": [{\"attacker\": string, \"victim\": string}],"
        "\"is_highlight\": boolean,"
        "\"confidence\": number,"
        "\"reason\": string"
        "}."
        "Use killfeed/HUD cues if visible; if unclear, set null."
    )

    try:
        obs = _generate_multimodal_json(
            prompt=prompt,
            image_paths=frames,
            ai_backend="vertex",
            model=model,
            gemini_api_key=None,
            vertex_project_id=project_id,
            vertex_location=location,
            vertex_api_key=key,
        )
    finally:
        for p in tmp.glob("*.jpg"):
            p.unlink(missing_ok=True)
        tmp.rmdir()
    return obs


def main() -> int:
    root = Path(__file__).resolve().parent
    clip = root / "Original Clips" / "Clip5.mp4"
    demo_dir = root / "blast-open-rotterdam-2026-parivision-vs-falcons-bo3-jB8BGzhBMGFhbYcN6o5bu2"
    key = (os.getenv("VERTEX_API_KEY") or "").strip()
    project = (os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = os.getenv("VERTEX_LOCATION", "us-central1").strip() or "us-central1"
    if not key or not project:
        raise RuntimeError("Set VERTEX_API_KEY and VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT).")
    model = "gemini-2.5-flash"

    obs = _extract_clip_observations(clip, key, project, location, model)

    observed_pairs: list[tuple[str, str]] = []
    for e in obs.get("events", []) if isinstance(obs, dict) else []:
        if not isinstance(e, dict):
            continue
        observed_pairs.append((_canon(str(e.get("attacker", ""))), _canon(str(e.get("victim", "")))) )

    map_guess = _canon(str(obs.get("map_guess", ""))) if isinstance(obs, dict) else ""

    candidates: list[dict[str, Any]] = []
    round_rows_all: list[dict[str, object]] = []

    for demo in sorted(demo_dir.glob("*.dem")):
        kills_df = _load_kill_events(demo)
        rs = _load_round_start_events(demo)
        kills_df, _ = _normalize_kill_time_column(kills_df)
        kills_df, _ = _assign_round_numbers(kills_df, rs)

        pair_by_round = _collect_round_kill_pairs(kills_df)
        rows, _ = _score_rounds(kills_df, demo.name)
        round_rows_all.extend(rows)

        map_bonus = 0.0
        if map_guess and map_guess in _canon(demo.name):
            map_bonus = 2.0

        for rnd, pairs in pair_by_round.items():
            score = map_bonus
            for oa, ov in observed_pairs:
                if not oa and not ov:
                    continue
                for da, dv in pairs:
                    if oa and ov and oa == da and ov == dv:
                        score += 3.0
                        break
                    if oa and oa == da:
                        score += 1.0
                    if ov and ov == dv:
                        score += 1.0
            candidates.append({"demo": demo.name, "round": rnd, "match_score": score})

    candidates.sort(key=lambda x: float(x["match_score"]), reverse=True)
    best = candidates[0] if candidates else {"demo": "", "round": 0, "match_score": 0.0}

    rounds_scored, _, _ = _apply_per_map_highlight_cap(round_rows_all, max_highlights_per_map=5)
    round_index = {(str(r.get("demo")), int(r.get("round", 0))): r for r in rounds_scored}
    chosen_row = round_index.get((str(best.get("demo", "")), int(best.get("round", 0))), {})

    out = {
        "clip": str(clip),
        "gemini_observation": obs,
        "likely_demo": best.get("demo", ""),
        "likely_round": int(best.get("round", 0)),
        "match_score": float(best.get("match_score", 0.0)),
        "round_analysis": {
            "round_score": float(chosen_row.get("round_score", 0.0)) if chosen_row else 0.0,
            "is_round_highlight": bool(chosen_row.get("is_round_highlight", False)) if chosen_row else False,
            "kills_total": float(chosen_row.get("kills_total", 0.0)) if chosen_row else 0.0,
            "headshot_ratio": float(chosen_row.get("headshot_ratio", 0.0)) if chosen_row else 0.0,
            "max_multikill_by_player": float(chosen_row.get("max_multikill_by_player", 0.0)) if chosen_row else 0.0,
        },
        "top_candidates": candidates[:10],
    }

    out_path = root / "clip5_round_inspection.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
