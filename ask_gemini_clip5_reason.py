#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import re
import zipfile
from pathlib import Path
from google import genai  # type: ignore

from detect_cs2_highlight import (
    _assign_round_numbers,
    _load_kill_events,
    _load_round_start_events,
    _normalize_kill_time_column,
    _score_rounds,
)


def _get_round_details(demo_path: Path, round_number: int) -> dict:
    kills_df = _load_kill_events(demo_path)
    rs_df = _load_round_start_events(demo_path)
    kills_df, _ = _normalize_kill_time_column(kills_df)
    kills_df, _ = _assign_round_numbers(kills_df, rs_df)

    cols = list(kills_df.columns)
    attacker_col = next((c for c in cols if c.lower() in {"attacker_name", "attacker", "killer_name", "killer"}), None)
    victim_col = next((c for c in cols if c.lower() in {"user_name", "victim_name", "user", "victim"}), None)
    weapon_col = next((c for c in cols if c.lower() in {"weapon", "weapon_name"}), None)

    kill_events = []
    rk = kills_df[kills_df["_round_number"] == int(round_number)]
    for _, row in rk.iterrows():
        kill_events.append(
            {
                "attacker": str(row.get(attacker_col, "")) if attacker_col else "",
                "victim": str(row.get(victim_col, "")) if victim_col else "",
                "weapon": str(row.get(weapon_col, "")) if weapon_col else "",
            }
        )

    round_rows, _ = _score_rounds(kills_df, demo_path.name)
    scored_row = next((r for r in round_rows if int(r.get("round", 0)) == int(round_number)), {})
    return {
        "round": int(round_number),
        "kills": kill_events,
        "round_score": float(scored_row.get("round_score", 0.0)) if scored_row else 0.0,
        "kills_total": float(scored_row.get("kills_total", 0.0)) if scored_row else 0.0,
        "headshot_ratio": float(scored_row.get("headshot_ratio", 0.0)) if scored_row else 0.0,
        "max_multikill_by_player": float(scored_row.get("max_multikill_by_player", 0.0)) if scored_row else 0.0,
    }


def _build_demo_context_text(demo_path: Path, round_number: int) -> str:
    kills_df = _load_kill_events(demo_path)
    rs_df = _load_round_start_events(demo_path)
    kills_df, _ = _normalize_kill_time_column(kills_df)
    kills_df, _ = _assign_round_numbers(kills_df, rs_df)
    rows, _ = _score_rounds(kills_df, demo_path.name)

    cols = list(kills_df.columns)
    attacker_col = next((c for c in cols if c.lower() in {"attacker_name", "attacker", "killer_name", "killer"}), None)
    victim_col = next((c for c in cols if c.lower() in {"user_name", "victim_name", "user", "victim"}), None)
    weapon_col = next((c for c in cols if c.lower() in {"weapon", "weapon_name"}), None)

    lines = [
        "Demo context extracted from .dem file",
        f"demo_file={demo_path.name}",
        f"focus_round={round_number}",
        "",
        "Round scoreboard summary:",
    ]
    for r in sorted(rows, key=lambda x: int(x.get("round", 0))):
        lines.append(
            "round={round} score={score} kills={kills} hs_ratio={hs} max_multi={mm}".format(
                round=int(r.get("round", 0)),
                score=float(r.get("round_score", 0.0)),
                kills=float(r.get("kills_total", 0.0)),
                hs=float(r.get("headshot_ratio", 0.0)),
                mm=float(r.get("max_multikill_by_player", 0.0)),
            )
        )

    lines.append("")
    lines.append("Kill events for focus round:")
    rk = kills_df[kills_df["_round_number"] == int(round_number)]
    for _, row in rk.iterrows():
        a = str(row.get(attacker_col, "")) if attacker_col else ""
        v = str(row.get(victim_col, "")) if victim_col else ""
        w = str(row.get(weapon_col, "")) if weapon_col else ""
        lines.append(f"{a} -> {v} ({w})")

    return "\n".join(lines)


def _extract_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path, "r") as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    text = re.sub(r"\n{3,}", "\n\n", xml)
    return text.strip()


def main() -> int:
    root = Path(r"c:\Coding\Projects\ca\Esports-Video-clipping-automation")
    clip = str(root / "Original Clips" / "Clip5.mp4")
    demo = Path(
        r"c:\Coding\Projects\ca\Esports-Video-clipping-automation\blast-open-rotterdam-2026-parivision-vs-falcons-bo3-jB8BGzhBMGFhbYcN6o5bu2\parivision-vs-falcons-m1-mirage.dem"
    )
    rules_docx = root / "CS2_Highlights.docx"
    if not rules_docx.exists():
        rules_docx = root / "Original Clips" / "CS2_Highlights.docx"
    state_path = root / "gemini_round_sequence_state.json"

    if not demo.exists():
        raise FileNotFoundError(f"Required demo file not found: {demo}")
    if not rules_docx.exists():
        raise FileNotFoundError(
            f"Required highlight-rules Word file not found: {rules_docx}. "
            "Place CS2_Highlights.docx in workspace root."
        )

    expected_round_hint = "unknown"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_round_hint = str(int(state.get("last_round", 0)) + 1)
        except Exception:
            expected_round_hint = "unknown"

    round_for_context = int(expected_round_hint) if expected_round_hint.isdigit() else 1
    round_details = _get_round_details(demo, round_for_context)
    demo_context_text = _build_demo_context_text(demo, round_for_context)
    demo_context_path = root / "clip5_demo_context.txt"
    demo_context_path.write_text(demo_context_text, encoding="utf-8")
    rules_text_path = root / "clip5_rules_context.txt"
    rules_text_path.write_text(_extract_docx_text(rules_docx), encoding="utf-8")

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to run this script.")

    client = genai.Client(api_key=api_key)
    print("[reason] uploading full clip...", flush=True)
    uploaded = client.files.upload(file=clip)
    print(f"[reason] uploaded: {uploaded.name}", flush=True)
    print("[reason] uploading parsed demo context text...", flush=True)
    uploaded_demo = client.files.upload(file=str(demo_context_path), config={"mime_type": "text/plain"})
    print(f"[reason] uploaded demo: {uploaded_demo.name}", flush=True)
    print("[reason] uploading highlight rules docx...", flush=True)
    uploaded_rules = client.files.upload(
        file=str(rules_text_path),
        config={"mime_type": "text/plain"},
    )
    print(f"[reason] uploaded rules: {uploaded_rules.name}", flush=True)

    for _ in range(120):
        current = client.files.get(name=uploaded.name)
        state = str(getattr(current, "state", ""))
        print(f"[reason] state: {state}", flush=True)
        if "ACTIVE" in state:
            uploaded = current
            break
        if "FAILED" in state:
            raise RuntimeError(f"Gemini file processing failed: {state}")
        time.sleep(2)
    else:
        raise RuntimeError("Timed out waiting for Gemini file processing")

    prompt = (
        "Analyze this full CS2 clip and explain ONLY why you classify it as highlight or non-highlight. "
        "Use the attached demo file and attached highlight-rules DOCX as mandatory context. "
        f"Round sequence hint: this clip is likely around round {expected_round_hint}. "
        f"Parsed demo round details to use as additional hard context: {json.dumps(round_details)}. "
        "Return strict JSON with schema: "
        '{"is_highlight": boolean, "confidence": number, "why_highlight": [string], '
        '"why_not_highlight": [string], "final_reason": string}. '
        "No markdown."
    )

    try:
        print("[reason] sending inference request...", flush=True)
        response = None
        last_error = None
        model_candidates = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        for model_name in model_candidates:
            for attempt in range(1, 4):
                try:
                    print(f"[reason] trying model={model_name} attempt={attempt}", flush=True)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[uploaded, uploaded_demo, uploaded_rules, prompt],
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[reason] model call failed: {exc}", flush=True)
                    time.sleep(4 * attempt)
            if response is not None:
                break

        if response is None:
            raise RuntimeError(f"All model attempts failed: {last_error}")
        text = getattr(response, "text", "") or ""
        out = r"c:\Coding\Projects\ca\Esports-Video-clipping-automation\clip5_gemini_reason_raw.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[reason] wrote raw response: {out}", flush=True)
        print(text, flush=True)
    finally:
        try:
            client.files.delete(name=uploaded.name)
            print("[reason] deleted uploaded file", flush=True)
        except Exception:
            pass
        try:
            client.files.delete(name=uploaded_demo.name)
            print("[reason] deleted uploaded demo", flush=True)
        except Exception:
            pass
        try:
            client.files.delete(name=uploaded_rules.name)
            print("[reason] deleted uploaded rules", flush=True)
        except Exception:
            pass
        try:
            demo_context_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            rules_text_path.unlink(missing_ok=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
