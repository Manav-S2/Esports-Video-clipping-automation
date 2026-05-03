#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from google import genai  # type: ignore

from detect_cs2_highlight import (
    _apply_per_map_highlight_cap,
    _assign_round_numbers,
    _load_kill_events,
    _load_round_start_events,
    _normalize_kill_time_column,
    _score_rounds,
)


def _canon(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Model did not return JSON: {text[:400]}")
    return json.loads(text[start : end + 1])


def _collect_round_kill_pairs(kills_df) -> Dict[int, List[Tuple[str, str]]]:
    pairs: Dict[int, List[Tuple[str, str]]] = {}
    if kills_df is None or len(kills_df) == 0 or "_round_number" not in kills_df.columns:
        return pairs

    cols = list(kills_df.columns)
    attacker_col = next((c for c in cols if c.lower() in {"attacker_name", "attacker", "killer_name", "killer"}), None)
    victim_col = next((c for c in cols if c.lower() in {"user_name", "victim_name", "user", "victim"}), None)
    if attacker_col is None or victim_col is None:
        return pairs

    for _, r in kills_df.iterrows():
        rnd = int(r.get("_round_number", 0))
        if rnd <= 0:
            continue
        a = _canon(str(r.get(attacker_col, "")))
        v = _canon(str(r.get(victim_col, "")))
        if a or v:
            pairs.setdefault(rnd, []).append((a, v))
    return pairs


def _analyze_full_video_with_gemini(clip_path: Path, api_key: str, model: str) -> Dict[str, Any]:
    root = clip_path.resolve().parent.parent
    demo_path = Path(
        os.getenv(
            "CLIP_DEMO_PATH",
            str(
                root
                / "blast-open-rotterdam-2026-parivision-vs-falcons-bo3-jB8BGzhBMGFhbYcN6o5bu2"
                / "parivision-vs-falcons-m1-mirage.dem"
            ),
        )
    ).resolve()
    rules_docx_path = Path(os.getenv("HIGHLIGHT_RULES_DOCX", str(root / "CS2_Highlights.docx"))).resolve()
    round_state_path = root / "gemini_round_sequence_state.json"

    if not demo_path.exists():
        raise FileNotFoundError(f"Required demo file not found: {demo_path}")
    if not rules_docx_path.exists():
        raise FileNotFoundError(
            "Required highlight-rules Word file not found: "
            f"{rules_docx_path}. Place CS2_Highlights.docx in workspace root or set HIGHLIGHT_RULES_DOCX."
        )

    expected_round_hint = "unknown"
    if round_state_path.exists():
        try:
            state = json.loads(round_state_path.read_text(encoding="utf-8"))
            expected_round_hint = str(int(state.get("last_round", 0)) + 1)
        except Exception:
            expected_round_hint = "unknown"

    print(f"[full-video] Initializing Gemini client for {clip_path.name}", flush=True)
    client = genai.Client(api_key=api_key)
    print("[full-video] Uploading full clip to Gemini Files API...", flush=True)
    uploaded = client.files.upload(file=str(clip_path))
    print(f"[full-video] Uploaded file name: {uploaded.name}", flush=True)
    print("[full-video] Uploading demo file for context...", flush=True)
    uploaded_demo = client.files.upload(file=str(demo_path), config={"mime_type": "application/octet-stream"})
    print(f"[full-video] Uploaded demo file name: {uploaded_demo.name}", flush=True)
    print("[full-video] Uploading highlight rules DOCX...", flush=True)
    uploaded_rules = client.files.upload(
        file=str(rules_docx_path),
        config={"mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    )
    print(f"[full-video] Uploaded rules file name: {uploaded_rules.name}", flush=True)

    # Wait until file becomes available for inference.
    for _ in range(120):
        current = client.files.get(name=uploaded.name)
        state = str(getattr(current, "state", ""))
        print(f"[full-video] File state: {state}", flush=True)
        if "ACTIVE" in state:
            uploaded = current
            break
        if "FAILED" in state:
            raise RuntimeError(f"Gemini file processing failed: {state}")
        time.sleep(2)
    else:
        raise RuntimeError("Timed out waiting for Gemini file processing.")

    prompt = (
        "Analyze this full CS2 clip and return ONLY strict JSON with schema: "
        "{"
        "\"map_guess\": string,"
        "\"score_t\": number|null,"
        "\"score_ct\": number|null,"
        "\"round_guess\": number|null,"
        "\"events\": [{\"attacker\": string, \"victim\": string, \"weapon\": string}],"
        "\"is_highlight\": boolean,"
        "\"confidence\": number,"
        "\"reason\": string"
        "}."
        "Use killfeed/HUD/gameplay context from the full clip. "
        "ALWAYS use the attached demo file and attached highlight-rules DOCX as source-of-truth context. "
        f"Round sequence hint: this clip is likely the next round in sequence, expected around round {expected_round_hint}."
    )

    try:
        print("[full-video] Sending full-video inference request...", flush=True)
        response = client.models.generate_content(model=model, contents=[uploaded, uploaded_demo, uploaded_rules, prompt])
        text = getattr(response, "text", "") or ""
        print("[full-video] Received model response.", flush=True)
        parsed = _extract_json(text)
    finally:
        try:
            client.files.delete(name=uploaded.name)
            print("[full-video] Deleted uploaded Gemini file.", flush=True)
        except Exception:
            pass
        try:
            client.files.delete(name=uploaded_demo.name)
            print("[full-video] Deleted uploaded demo file.", flush=True)
        except Exception:
            pass
        try:
            client.files.delete(name=uploaded_rules.name)
            print("[full-video] Deleted uploaded rules file.", flush=True)
        except Exception:
            pass

    return parsed


def _match_round_from_events(root: Path, obs: Dict[str, Any]) -> Dict[str, Any]:
    demo_dir = root / "blast-open-rotterdam-2026-parivision-vs-falcons-bo3-jB8BGzhBMGFhbYcN6o5bu2"
    observed_pairs: List[Tuple[str, str]] = []
    for e in obs.get("events", []) if isinstance(obs, dict) else []:
        if not isinstance(e, dict):
            continue
        observed_pairs.append((_canon(str(e.get("attacker", ""))), _canon(str(e.get("victim", "")))) )

    map_guess = _canon(str(obs.get("map_guess", ""))) if isinstance(obs, dict) else ""

    candidates: List[Dict[str, Any]] = []
    round_rows_all: List[Dict[str, object]] = []

    for demo in sorted(demo_dir.glob("*.dem")):
        kills_df = _load_kill_events(demo)
        rs = _load_round_start_events(demo)
        kills_df, _ = _normalize_kill_time_column(kills_df)
        kills_df, _ = _assign_round_numbers(kills_df, rs)

        pair_by_round = _collect_round_kill_pairs(kills_df)
        rows, _ = _score_rounds(kills_df, demo.name)
        round_rows_all.extend(rows)

        map_bonus = 2.0 if (map_guess and map_guess in _canon(demo.name)) else 0.0

        for rnd, pairs in pair_by_round.items():
            score = map_bonus
            for oa, ov in observed_pairs:
                if not oa and not ov:
                    continue
                exact_hit = False
                for da, dv in pairs:
                    if oa and ov and oa == da and ov == dv:
                        score += 3.0
                        exact_hit = True
                        break
                if exact_hit:
                    continue
                for da, dv in pairs:
                    if oa and oa == da:
                        score += 1.0
                    if ov and ov == dv:
                        score += 1.0
            candidates.append({"demo": demo.name, "round": int(rnd), "match_score": float(score)})

    candidates.sort(key=lambda x: float(x["match_score"]), reverse=True)
    best = candidates[0] if candidates else {"demo": "", "round": 0, "match_score": 0.0}

    rounds_scored, _, _ = _apply_per_map_highlight_cap(round_rows_all, max_highlights_per_map=5)
    index = {(str(r.get("demo")), int(r.get("round", 0))): r for r in rounds_scored}
    chosen = index.get((str(best.get("demo", "")), int(best.get("round", 0))), {})

    return {
        "likely_demo": best.get("demo", ""),
        "likely_round": int(best.get("round", 0)),
        "match_score": float(best.get("match_score", 0.0)),
        "round_analysis": {
            "round_score": float(chosen.get("round_score", 0.0)) if chosen else 0.0,
            "is_round_highlight": bool(chosen.get("is_round_highlight", False)) if chosen else False,
            "kills_total": float(chosen.get("kills_total", 0.0)) if chosen else 0.0,
            "headshot_ratio": float(chosen.get("headshot_ratio", 0.0)) if chosen else 0.0,
            "max_multikill_by_player": float(chosen.get("max_multikill_by_player", 0.0)) if chosen else 0.0,
        },
        "top_candidates": candidates[:10],
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    clip = root / "Original Clips" / "Clip5.mp4"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"[full-video] Using model: {model}", flush=True)

    obs = _analyze_full_video_with_gemini(clip, api_key=api_key, model=model)
    print("[full-video] Matching observed events against demo rounds...", flush=True)
    match = _match_round_from_events(root, obs)

    out = {
        "clip": str(clip),
        "analysis_mode": "full_video_upload",
        "gemini_observation": obs,
        **match,
    }

    # Persist round sequence continuity for next clip inference hint.
    state_path = root / "gemini_round_sequence_state.json"
    try:
        state_path.write_text(
            json.dumps(
                {
                    "last_demo": match.get("likely_demo", ""),
                    "last_round": int(match.get("likely_round", 0)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    out_path = root / "clip5_full_video_round_inspection.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
