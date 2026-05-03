#!/usr/bin/env python3
"""CS2 highlight detector for automation pipelines.

This script supports combining two sources:
1) Demo-derived match signals (.dem, .rar, or a folder containing .dem files)
2) Gemini vision review of sampled clip frames

Use cases:
- Fast demo-only screening
- Video-only screening via Gemini
- Hybrid score (recommended) where demo + Gemini are fused
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from video_editor import apply_portrait_blur


@dataclass
class DetectionResult:
    input_path: str
    demos_analyzed: List[str]
    score: float
    is_highlight: bool
    confidence: float
    demo_score: Optional[float]
    vision_score: Optional[float]
    features: Dict[str, float]
    vision: Dict[str, float]
    rounds_scored: List[Dict[str, object]]
    top_rounds: List[Dict[str, object]]
    round_highlight_cutoffs: Dict[str, float]
    highlight_round_details: List[Dict[str, Any]]
    notes: List[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "input_path": self.input_path,
                "demos_analyzed": self.demos_analyzed,
                "score": round(self.score, 3),
                "is_highlight": self.is_highlight,
                "confidence": round(self.confidence, 3),
                "demo_score": None if self.demo_score is None else round(self.demo_score, 3),
                "vision_score": None if self.vision_score is None else round(self.vision_score, 3),
                "features": {k: round(v, 3) for k, v in self.features.items()},
                "vision": {k: round(v, 3) for k, v in self.vision.items()},
                "rounds_scored": self.rounds_scored,
                "top_rounds": self.top_rounds,
                "round_highlight_cutoffs": {
                    k: round(v, 3) for k, v in self.round_highlight_cutoffs.items()
                },
                "highlight_round_details": self.highlight_round_details,
                "notes": self.notes,
            },
            indent=2,
        )


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    column_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in column_map:
            return column_map[name.lower()]
    return None


def _extract_demo_from_rar(rar_path: Path) -> Path:
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if not seven_zip:
        raise RuntimeError(
            "7-Zip CLI was not found in PATH. Install 7-Zip and ensure 7z.exe is available."
        )

    extract_dir = Path(tempfile.mkdtemp(prefix="cs2_demo_extract_"))
    cmd = [seven_zip, "x", str(rar_path), f"-o{extract_dir}", "-y"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to extract RAR archive.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    demos = sorted(extract_dir.rglob("*.dem"), key=lambda p: p.stat().st_size, reverse=True)
    if not demos:
        raise RuntimeError("RAR extracted successfully, but no .dem file was found inside.")
    return demos[0]


def _resolve_demo_paths(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        demos = sorted(input_path.glob("*.dem"), key=lambda p: p.stat().st_size, reverse=True)
        if not demos:
            raise ValueError(f"No .dem files found in directory: {input_path}")
        return demos

    suffix = input_path.suffix.lower()
    if suffix == ".dem":
        return [input_path]
    if suffix == ".rar":
        return [_extract_demo_from_rar(input_path)]
    raise ValueError("Demo input must be a .dem, .rar, or a directory containing .dem files.")


def _load_kill_events(demo_path: Path):
    try:
        from demoparser2 import DemoParser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "demoparser2 is not installed. Install it with: pip install demoparser2"
        ) from exc

    parser = DemoParser(str(demo_path))

    # demoparser2 versions expose slightly different method names.
    errors: List[str] = []
    for method_name in ("parse_event", "parse_events", "parse_ticks"):
        method = getattr(parser, method_name, None)
        if not method:
            continue
        try:
            if method_name == "parse_event":
                return method("player_death")
            if method_name == "parse_events":
                events = method(["player_death"])
                if isinstance(events, dict) and "player_death" in events:
                    return events["player_death"]
                return events
            if method_name == "parse_ticks":
                # Fallback in case only tick parsing is exposed.
                return method(["attacker_name", "user_name", "is_headshot", "weapon", "event_name"])
        except Exception as exc:  # pragma: no cover - defensive compatibility
            errors.append(f"{method_name}: {exc}")

    raise RuntimeError(
        "Unable to read kill events from demoparser2. "
        "Tried methods parse_event/parse_events/parse_ticks. Errors: "
        + " | ".join(errors)
    )


def _load_round_start_events(demo_path: Path):
    try:
        from demoparser2 import DemoParser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "demoparser2 is not installed. Install it with: pip install demoparser2"
        ) from exc

    parser = DemoParser(str(demo_path))
    errors: List[str] = []
    for method_name in ("parse_event", "parse_events"):
        method = getattr(parser, method_name, None)
        if not method:
            continue
        try:
            if method_name == "parse_event":
                return method("round_start")
            events = method(["round_start"])
            if isinstance(events, dict) and "round_start" in events:
                return events["round_start"]
            return events
        except Exception as exc:  # pragma: no cover - defensive compatibility
            errors.append(f"{method_name}: {exc}")

    raise RuntimeError(
        "Unable to read round_start events from demoparser2. "
        "Tried methods parse_event/parse_events. Errors: "
        + " | ".join(errors)
    )


def _load_round_end_events(demo_path: Path):
    try:
        from demoparser2 import DemoParser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "demoparser2 is not installed. Install it with: pip install demoparser2"
        ) from exc

    parser = DemoParser(str(demo_path))
    errors: List[str] = []
    for method_name in ("parse_event", "parse_events"):
        method = getattr(parser, method_name, None)
        if not method:
            continue
        try:
            if method_name == "parse_event":
                return method("round_end")
            events = method(["round_end"])
            if isinstance(events, dict) and "round_end" in events:
                return events["round_end"]
            return events
        except Exception as exc:  # pragma: no cover - defensive compatibility
            errors.append(f"{method_name}: {exc}")

    raise RuntimeError(
        "Unable to read round_end events from demoparser2. "
        "Tried methods parse_event/parse_events. Errors: "
        + " | ".join(errors)
    )


def _load_item_purchase_events(demo_path: Path):
    try:
        from demoparser2 import DemoParser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "demoparser2 is not installed. Install it with: pip install demoparser2"
        ) from exc

    parser = DemoParser(str(demo_path))
    errors: List[str] = []
    for method_name in ("parse_event", "parse_events"):
        method = getattr(parser, method_name, None)
        if not method:
            continue
        try:
            if method_name == "parse_event":
                return method("item_purchase")
            events = method(["item_purchase"])
            if isinstance(events, dict) and "item_purchase" in events:
                return events["item_purchase"]
            return events
        except Exception as exc:  # pragma: no cover - defensive compatibility
            errors.append(f"{method_name}: {exc}")

    raise RuntimeError(
        "Unable to read item_purchase events from demoparser2. "
        "Tried methods parse_event/parse_events. Errors: "
        + " | ".join(errors)
    )


def _load_team_ticks(demo_path: Path):
    try:
        from demoparser2 import DemoParser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "demoparser2 is not installed. Install it with: pip install demoparser2"
        ) from exc

    parser = DemoParser(str(demo_path))
    return parser.parse_ticks(["steamid", "team_name", "tick"])


def _normalize_kill_time_column(kills_df):
    notes: List[str] = []
    if kills_df is None or len(kills_df) == 0:
        return kills_df, notes

    columns = list(kills_df.columns)
    if "_time_seconds" in columns:
        return kills_df, notes

    time_col = _pick_column(columns, ["time", "seconds", "game_time", "clock_time", "tick"])
    if not time_col:
        return kills_df, ["No event time column found; time-based features disabled."]

    kd = kills_df.copy()
    if str(time_col).lower() == "tick":
        kd["_time_seconds"] = kd[time_col].astype(float) / 64.0
        notes.append("Used tick/64 fallback for event timing.")
    else:
        kd["_time_seconds"] = kd[time_col].astype(float)
    return kd, notes


def _assign_round_numbers(kills_df, round_start_df):
    if kills_df is None or len(kills_df) == 0:
        return kills_df, ["No kill events available for round assignment."]

    notes: List[str] = []
    if round_start_df is None or len(round_start_df) == 0:
        notes.append("No round_start events found; round-by-round scoring disabled.")
        return kills_df, notes

    kill_columns = list(kills_df.columns)
    round_columns = list(round_start_df.columns)
    kill_tick_col = _pick_column(kill_columns, ["tick"])
    rs_tick_col = _pick_column(round_columns, ["tick"])
    rs_round_col = _pick_column(round_columns, ["round", "round_num", "round_number"])

    if not kill_tick_col or not rs_tick_col or not rs_round_col:
        notes.append("Required tick/round columns missing; round-by-round scoring disabled.")
        return kills_df, notes

    import pandas as pd  # type: ignore

    rs = round_start_df[[rs_tick_col, rs_round_col]].copy()
    rs[rs_tick_col] = rs[rs_tick_col].astype(float)
    rs = rs.sort_values(rs_tick_col).drop_duplicates(subset=[rs_tick_col], keep="last")
    rs = rs.rename(columns={rs_tick_col: "_rs_tick", rs_round_col: "_round_number"})

    kd = kills_df.copy()
    kd[kill_tick_col] = kd[kill_tick_col].astype(float)
    kd = kd.sort_values(kill_tick_col)
    kd = kd.rename(columns={kill_tick_col: "_kill_tick"})

    merged = pd.merge_asof(
        kd,
        rs,
        left_on="_kill_tick",
        right_on="_rs_tick",
        direction="backward",
    )
    merged["_round_number"] = merged["_round_number"].fillna(0).astype(int)
    merged = merged.rename(columns={"_kill_tick": kill_tick_col})
    return merged, notes


def _compute_features(kills_df) -> Tuple[Dict[str, float], List[str]]:
    notes: List[str] = []

    if kills_df is None or len(kills_df) == 0:
        return {
            "kills_total": 0.0,
            "kills_per_minute": 0.0,
            "headshot_ratio": 0.0,
            "multi_kill_burst_score": 0.0,
            "unique_killers": 0.0,
        }, ["No kill events found in demo."]

    columns = list(kills_df.columns)

    attacker_col = _pick_column(columns, ["attacker_name", "attacker", "killer_name", "killer"])
    time_col = _pick_column(columns, ["_time_seconds", "time", "seconds", "game_time", "clock_time", "tick"]) 
    hs_col = _pick_column(columns, ["is_headshot", "headshot", "head_shot"])

    kills_total = float(len(kills_df))

    if attacker_col:
        unique_killers = float(kills_df[attacker_col].fillna("unknown").nunique())
    else:
        unique_killers = 0.0
        notes.append("Attacker column not found; unique killer feature disabled.")

    if hs_col:
        headshot_ratio = float(kills_df[hs_col].astype(float).mean())
    else:
        headshot_ratio = 0.0
        notes.append("Headshot column not found; headshot feature disabled.")

    kills_per_minute = 0.0
    multi_kill_burst_score = 0.0

    if time_col:
        times = kills_df[time_col].astype(float).sort_values().tolist()
        if times:
            demo_seconds = max(times) - min(times)
            if demo_seconds > 0:
                kills_per_minute = kills_total / (demo_seconds / 60.0)

        if attacker_col:
            local_df = kills_df[[attacker_col, time_col]].copy()
            local_df[attacker_col] = local_df[attacker_col].fillna("unknown")
            local_df[time_col] = local_df[time_col].astype(float)
            local_df = local_df.sort_values(time_col)

            # Burst score: repeated kills by same player within 8 seconds.
            burst_points = 0.0
            for _, grp in local_df.groupby(attacker_col):
                t = grp[time_col].tolist()
                if len(t) < 2:
                    continue
                left = 0
                for right in range(len(t)):
                    while t[right] - t[left] > 8.0:
                        left += 1
                    window_kills = right - left + 1
                    if window_kills >= 2:
                        burst_points += (window_kills - 1) * 0.6
            multi_kill_burst_score = burst_points
        else:
            notes.append("Cannot compute multikill bursts without attacker column.")
    else:
        notes.append("No event time column found; time-based features disabled.")

    features = {
        "kills_total": kills_total,
        "kills_per_minute": kills_per_minute,
        "headshot_ratio": headshot_ratio,
        "multi_kill_burst_score": multi_kill_burst_score,
        "unique_killers": unique_killers,
    }
    return features, notes


def _score_demo_highlight(features: Dict[str, float]) -> Tuple[float, bool, float]:
    # Weighted heuristic tuned for quick highlight screening.
    score = 0.0
    score += min(features["kills_total"], 12.0) * 0.55
    score += min(features["kills_per_minute"], 10.0) * 0.45
    score += features["headshot_ratio"] * 2.0
    score += min(features["multi_kill_burst_score"], 8.0) * 0.8
    score += min(features["unique_killers"], 8.0) * 0.2

    # Default threshold: 7.5 works well as initial high-recall filter.
    threshold = 7.5
    is_highlight = score >= threshold

    # Confidence rises with distance from threshold.
    confidence = 1.0 / (1.0 + math.exp(-(score - threshold)))
    return score, is_highlight, confidence


def _score_rounds(kills_df, demo_name: str) -> Tuple[List[Dict[str, object]], List[str]]:
    notes: List[str] = []
    if kills_df is None or len(kills_df) == 0:
        return [], [f"{demo_name}: No kills available for round scoring."]

    if "_round_number" not in kills_df.columns:
        return [], [f"{demo_name}: _round_number missing; round scoring skipped."]

    columns = list(kills_df.columns)
    attacker_col = _pick_column(columns, ["attacker_name", "attacker", "killer_name", "killer"])
    hs_col = _pick_column(columns, ["is_headshot", "headshot", "head_shot"])
    time_col = _pick_column(columns, ["_time_seconds", "time", "seconds", "game_time", "clock_time", "tick"])
    wipe_col = _pick_column(columns, ["wipe"])
    round_end_reason_col = _pick_column(columns, ["_round_end_reason"])

    round_rows: List[Dict[str, object]] = []
    for round_num, grp in kills_df.groupby("_round_number"):
        if int(round_num) <= 0:
            continue

        g = grp.copy()
        kills_total = float(len(g))
        headshot_ratio = float(g[hs_col].astype(float).mean()) if hs_col else 0.0
        unique_killers = float(g[attacker_col].fillna("unknown").nunique()) if attacker_col else 0.0
        wipe_events = float(g[wipe_col].astype(float).sum()) if wipe_col else 0.0
        round_end_reason = ""
        if round_end_reason_col and len(g[round_end_reason_col].dropna()) > 0:
            round_end_reason = str(g[round_end_reason_col].dropna().iloc[0])

        burst_score = 0.0
        max_multikill_by_player = 0.0
        clutch_cue_score = 0.0
        if attacker_col and time_col:
            local = g[[attacker_col, time_col]].copy()
            local[attacker_col] = local[attacker_col].fillna("unknown")
            local[time_col] = local[time_col].astype(float)
            attacker_counts = local.groupby(attacker_col).size().tolist()
            if attacker_counts:
                max_multikill_by_player = float(max(attacker_counts))

            for _, tgrp in local.groupby(attacker_col):
                t = sorted(tgrp[time_col].tolist())
                if len(t) < 2:
                    continue
                left = 0
                for right in range(len(t)):
                    while t[right] - t[left] > 6.0:
                        left += 1
                    window_kills = right - left + 1
                    if window_kills >= 2:
                        burst_score += (window_kills - 1) * 1.0

            if kills_total > 0 and max_multikill_by_player >= 3:
                share = max_multikill_by_player / kills_total
                if share >= 0.5:
                    clutch_cue_score = max_multikill_by_player * share

        round_pressure_score = 0.0
        if round_end_reason in {"bomb_exploded", "bomb_defused"}:
            round_pressure_score = 0.8

        round_score = 0.0
        round_score += min(kills_total, 6.0) * 1.0
        round_score += headshot_ratio * 1.4
        round_score += min(max_multikill_by_player, 4.0) * 0.9
        round_score += min(burst_score, 5.0) * 0.9
        round_score += min(clutch_cue_score, 4.0) * 1.2
        round_score += min(unique_killers, 5.0) * 0.2
        round_score += min(wipe_events, 2.0) * 1.0
        round_score += round_pressure_score

        round_rows.append(
            {
                "demo": demo_name,
                "round": int(round_num),
                "round_score": round(round_score, 3),
                "kills_total": round(kills_total, 3),
                "headshot_ratio": round(headshot_ratio, 3),
                "max_multikill_by_player": round(max_multikill_by_player, 3),
                "burst_score": round(burst_score, 3),
                "clutch_cue_score": round(clutch_cue_score, 3),
                "unique_killers": round(unique_killers, 3),
                "wipe_events": round(wipe_events, 3),
                "round_end_reason": round_end_reason,
                "round_pressure_score": round(round_pressure_score, 3),
            }
        )

    if not round_rows:
        notes.append(f"{demo_name}: No valid rounds were scored.")
    return round_rows, notes


def _map_score_from_rounds(round_rows: Sequence[Dict[str, object]]) -> Optional[float]:
    if not round_rows:
        return None
    ordered = sorted(round_rows, key=lambda r: float(r["round_score"]), reverse=True)
    top_n = ordered[:3]
    if not top_n:
        return None
    return sum(float(r["round_score"]) for r in top_n) / len(top_n)


def _build_round_bounds_seconds(round_start_df, round_end_df) -> Dict[int, Tuple[float, float]]:
    bounds: Dict[int, Tuple[float, float]] = {}
    if round_start_df is None or len(round_start_df) == 0:
        return bounds

    rs_cols = list(round_start_df.columns)
    rs_tick_col = _pick_column(rs_cols, ["tick"])
    rs_round_col = _pick_column(rs_cols, ["round", "round_num", "round_number"])
    if not rs_tick_col or not rs_round_col:
        return bounds

    starts: Dict[int, float] = {}
    for r, t in zip(round_start_df[rs_round_col].tolist(), round_start_df[rs_tick_col].tolist()):
        rn = int(r)
        ts = float(t) / 64.0
        if rn not in starts or ts < starts[rn]:
            starts[rn] = ts

    ends: Dict[int, float] = {}
    if round_end_df is not None and len(round_end_df) > 0:
        re_cols = list(round_end_df.columns)
        re_tick_col = _pick_column(re_cols, ["tick"])
        re_round_col = _pick_column(re_cols, ["round", "round_num", "round_number"])
        if re_tick_col and re_round_col:
            for r, t in zip(round_end_df[re_round_col].tolist(), round_end_df[re_tick_col].tolist()):
                ends[int(r)] = float(t) / 64.0

    sorted_rounds = sorted(starts.keys())
    for i, rn in enumerate(sorted_rounds):
        st = starts[rn]
        if rn in ends:
            en = ends[rn]
        elif i + 1 < len(sorted_rounds):
            en = max(st + 1.0, starts[sorted_rounds[i + 1]] - 0.01)
        else:
            en = st + 20.0
        if en <= st:
            en = st + 1.0
        bounds[rn] = (st, en)
    return bounds


def _infer_eco_loss_rounds(demo_path: Path, round_start_df, round_end_df) -> Dict[int, Dict[str, Any]]:
    """Heuristic eco-loss detector using purchase spend on losing side per round."""
    info: Dict[int, Dict[str, Any]] = {}
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return info

    purchases = _load_item_purchase_events(demo_path)
    if purchases is None or len(purchases) == 0:
        return info

    p_cols = list(purchases.columns)
    p_tick_col = _pick_column(p_cols, ["tick"])
    p_cost_col = _pick_column(p_cols, ["cost"])
    p_name_col = _pick_column(p_cols, ["item_name", "weapon", "name"])
    p_steam_col = _pick_column(p_cols, ["steamid"])
    if not p_tick_col or not p_cost_col or not p_name_col or not p_steam_col:
        return info

    # Assign purchase events to round numbers using round_start ticks.
    rs_cols = list(round_start_df.columns)
    rs_tick_col = _pick_column(rs_cols, ["tick"])
    rs_round_col = _pick_column(rs_cols, ["round", "round_num", "round_number"])
    if not rs_tick_col or not rs_round_col:
        return info

    rs = round_start_df[[rs_tick_col, rs_round_col]].copy()
    rs[rs_tick_col] = rs[rs_tick_col].astype(float)
    rs = rs.sort_values(rs_tick_col).drop_duplicates(subset=[rs_tick_col], keep="last")
    rs = rs.rename(columns={rs_tick_col: "_rs_tick", rs_round_col: "_round_number"})

    p = purchases[[p_tick_col, p_cost_col, p_name_col, p_steam_col]].copy()
    p[p_tick_col] = p[p_tick_col].astype(float)
    p[p_cost_col] = p[p_cost_col].astype(float)
    p = p.sort_values(p_tick_col).rename(columns={p_tick_col: "_purchase_tick"})
    p = pd.merge_asof(p, rs, left_on="_purchase_tick", right_on="_rs_tick", direction="backward")
    p["_round_number"] = p["_round_number"].fillna(0).astype(int)

    # Map each purchase (steamid,tick) to side using parse_ticks snapshot.
    team_ticks = _load_team_ticks(demo_path)
    t_cols = list(team_ticks.columns)
    t_tick_col = _pick_column(t_cols, ["tick"])
    t_team_col = _pick_column(t_cols, ["team_name"])
    t_steam_col = _pick_column(t_cols, ["steamid"])
    if not t_tick_col or not t_team_col or not t_steam_col:
        return info

    t = team_ticks[[t_tick_col, t_team_col, t_steam_col]].copy()
    t[t_tick_col] = t[t_tick_col].astype(float)
    t[t_steam_col] = t[t_steam_col].astype(str)
    t = t.sort_values([t_steam_col, t_tick_col], kind="mergesort").reset_index(drop=True)
    t = t.rename(columns={t_tick_col: "_team_tick", t_team_col: "_team_name", t_steam_col: "_steamid"})

    p = p.rename(columns={p_steam_col: "_steamid", p_name_col: "_item_name", p_cost_col: "_cost"})
    p["_steamid"] = p["_steamid"].astype(str)
    p = p.sort_values(["_steamid", "_purchase_tick"], kind="mergesort").reset_index(drop=True)
    merged_parts = []
    for sid, p_grp in p.groupby("_steamid", sort=False):
        t_grp = t[t["_steamid"] == sid]
        if len(t_grp) == 0:
            p_local = p_grp.copy()
            p_local["_team_tick"] = float("nan")
            p_local["_team_name"] = ""
            merged_parts.append(p_local)
            continue

        p_local = p_grp.sort_values("_purchase_tick", kind="mergesort").reset_index(drop=True)
        t_local = t_grp.sort_values("_team_tick", kind="mergesort").reset_index(drop=True)
        p_local = pd.merge_asof(
            p_local,
            t_local[["_team_tick", "_team_name"]],
            left_on="_purchase_tick",
            right_on="_team_tick",
            direction="backward",
        )
        p_local["_steamid"] = sid
        merged_parts.append(p_local)

    p = pd.concat(merged_parts, ignore_index=True) if merged_parts else p

    # Determine losing side by round from round_end winner side.
    loser_by_round: Dict[int, str] = {}
    if round_end_df is not None and len(round_end_df) > 0:
        re_cols = list(round_end_df.columns)
        re_round_col = _pick_column(re_cols, ["round", "round_num", "round_number"])
        re_winner_col = _pick_column(re_cols, ["winner"])
        if re_round_col and re_winner_col:
            for r, w in zip(round_end_df[re_round_col].tolist(), round_end_df[re_winner_col].tolist()):
                winner = str(w).upper()
                loser = "CT" if winner == "T" else "TERRORIST"
                loser_by_round[int(r)] = loser

    rifle_tokens = {
        "ak47", "m4a1", "m4a1_silencer", "famas", "galilar", "sg556", "aug", "awp", "scar20", "g3sg1"
    }

    for rnd, grp in p.groupby("_round_number"):
        rn = int(rnd)
        if rn <= 0 or rn not in loser_by_round:
            continue

        loser_side = loser_by_round[rn]
        side_grp = grp[grp["_team_name"].astype(str).str.upper() == loser_side]
        if len(side_grp) == 0:
            continue

        spend = float(side_grp["_cost"].sum())
        items = [str(x).lower() for x in side_grp["_item_name"].fillna("").tolist()]
        rifle_count = sum(1 for it in items if any(tok in it for tok in rifle_tokens))

        is_full_eco = spend <= 4500.0 and rifle_count == 0
        info[rn] = {
            "loser_side": loser_side,
            "loser_side_spend": spend,
            "loser_rifle_purchases": float(rifle_count),
            "is_eco_loss": bool(is_full_eco),
        }

    return info


def _extract_interval_frames(video_path: Path, out_dir: Path, start_sec: float, end_sec: float, max_frames: int) -> List[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available in PATH.")

    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(1.0, end_sec - start_sec)
    fps = max(0.2, min(2.0, max_frames / duration))
    pattern = out_dir / "round_frame_%03d.jpg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-vf",
        f"fps={fps}",
        "-q:v",
        "3",
        str(pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to extract interval frames for Gemini review.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    frames = sorted(out_dir.glob("round_frame_*.jpg"))
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    return frames


def _gemini_confirm_exceptional_round(
    video_path: Path,
    api_key: Optional[str],
    model: str,
    start_sec: float,
    end_sec: float,
    ai_backend: str,
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
    max_frames: int = 6,
) -> Tuple[bool, str, float]:
    with tempfile.TemporaryDirectory(prefix="gemini_round_check_") as td:
        frames = _extract_interval_frames(video_path, Path(td), start_sec, end_sec, max_frames=max_frames)
        if not frames:
            return False, "No frames extracted for round review.", 0.0
        prompt = (
            "You are validating whether an eco-loss CS2 round should still be posted as highlight. "
            "Return ONLY strict JSON with schema: "
            '{"allow_post": boolean, "flickshot_detected": boolean, "strange_event_detected": boolean, '
            '"confidence": number 0-1, "reason": string}. '
            "Set allow_post=true only if there is clear exceptional play (flick, insane transfer, unusual event)."
        )

        parsed = _generate_multimodal_json(
            prompt=prompt,
            image_paths=frames,
            ai_backend=ai_backend,
            model=model,
            gemini_api_key=api_key,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            vertex_api_key=vertex_api_key,
        )
        allow_post = bool(parsed.get("allow_post", False))
        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", "Gemini round review."))
        return allow_post, reason, confidence


def _apply_eco_guard_with_gemini(
    rounds_scored: Sequence[Dict[str, object]],
    eco_info_by_demo: Dict[str, Dict[int, Dict[str, Any]]],
    round_bounds_by_demo: Dict[str, Dict[int, Tuple[float, float]]],
    video_path: Optional[Path],
    gemini_api_key: Optional[str],
    gemini_model: str,
    ai_backend: str,
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
    clip_start_demo_seconds: float,
    video_duration_seconds: float,
) -> Tuple[List[Dict[str, object]], List[str]]:
    notes: List[str] = []
    updated: List[Dict[str, object]] = []

    for row in rounds_scored:
        r = dict(row)
        demo = str(r.get("demo", ""))
        rnd = int(r.get("round", 0))
        eco_info = eco_info_by_demo.get(demo, {}).get(rnd, {})
        is_eco_loss = bool(eco_info.get("is_eco_loss", False))

        r["eco_loss_round"] = is_eco_loss
        r["loser_side"] = eco_info.get("loser_side", "")
        r["loser_side_spend"] = float(eco_info.get("loser_side_spend", 0.0))
        r["loser_rifle_purchases"] = float(eco_info.get("loser_rifle_purchases", 0.0))
        r["posting_allowed"] = bool(r.get("is_round_highlight", False))
        r["eco_guard_reason"] = ""
        r["gemini_review_confidence"] = 0.0

        if bool(r.get("is_round_highlight", False)) and is_eco_loss:
            r["posting_allowed"] = False
            r["eco_guard_reason"] = "Blocked: opposition full eco lost round."

            if video_path and _vision_backend_available(ai_backend, gemini_api_key, vertex_project_id, vertex_api_key):
                bounds = round_bounds_by_demo.get(demo, {}).get(rnd)
                if bounds:
                    try:
                        clip_start = bounds[0] - clip_start_demo_seconds
                        clip_end = bounds[1] - clip_start_demo_seconds

                        if video_duration_seconds > 0:
                            if clip_end < 0.0 or clip_start > video_duration_seconds:
                                r["eco_guard_reason"] = (
                                    "Round outside clip window for Gemini review; blocked by eco guard."
                                )
                                updated.append(r)
                                continue
                            clip_start = max(0.0, min(clip_start, video_duration_seconds))
                            clip_end = max(0.0, min(clip_end, video_duration_seconds))

                        if clip_end <= clip_start:
                            clip_end = clip_start + 1.0

                        allow_post, reason, conf = _gemini_confirm_exceptional_round(
                            video_path=video_path,
                            api_key=gemini_api_key,
                            model=gemini_model,
                            start_sec=clip_start,
                            end_sec=clip_end,
                            ai_backend=ai_backend,
                            vertex_project_id=vertex_project_id,
                            vertex_location=vertex_location,
                            vertex_api_key=vertex_api_key,
                        )
                        r["gemini_review_confidence"] = conf
                        r["eco_guard_reason"] = reason
                        if allow_post:
                            r["posting_allowed"] = True
                    except Exception as exc:
                        r["eco_guard_reason"] = f"Gemini review failed: {exc}"
                else:
                    r["eco_guard_reason"] = "Round bounds unavailable for Gemini review; blocked by eco guard."
            else:
                if ai_backend == "vertex":
                    r["eco_guard_reason"] = (
                        "Blocked by eco guard. Provide --video-path and --vertex-project-id to allow exceptional-play override."
                    )
                else:
                    r["eco_guard_reason"] = (
                        "Blocked by eco guard. Provide --video-path and GEMINI_API_KEY to allow exceptional-play override."
                    )

        updated.append(r)

    blocked = sum(1 for r in updated if bool(r.get("is_round_highlight", False)) and not bool(r.get("posting_allowed", False)))
    if blocked > 0:
        notes.append(f"Eco guard blocked {blocked} highlight rounds pending/denied Gemini exceptional-play confirmation.")
    return updated, notes


def _apply_per_map_highlight_cap(
    round_rows: Sequence[Dict[str, object]],
    max_highlights_per_map: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float], List[str]]:
    notes: List[str] = []
    if max_highlights_per_map <= 0:
        return [dict(r, is_round_highlight=False) for r in round_rows], {}, [
            "max_highlights_per_map <= 0, so no rounds are marked as highlights."
        ]

    by_demo: Dict[str, List[Dict[str, object]]] = {}
    for row in round_rows:
        demo = str(row.get("demo", "unknown"))
        by_demo.setdefault(demo, []).append(dict(row))

    cutoffs: Dict[str, float] = {}
    updated: List[Dict[str, object]] = []

    for demo, rows in by_demo.items():
        ordered = sorted(rows, key=lambda r: float(r["round_score"]), reverse=True)
        for i, row in enumerate(ordered):
            row["is_round_highlight"] = i < max_highlights_per_map
        if ordered:
            nth_idx = min(max_highlights_per_map, len(ordered)) - 1
            cutoffs[demo] = float(ordered[nth_idx]["round_score"]) if nth_idx >= 0 else 0.0
            notes.append(
                f"{demo}: capped to top {max_highlights_per_map} highlight rounds (cutoff {cutoffs[demo]:.3f})."
            )
        updated.extend(ordered)

    # Restore deterministic order by demo then round.
    updated = sorted(updated, key=lambda r: (str(r.get("demo", "")), int(r.get("round", 0))))
    return updated, cutoffs, notes


def _apply_attention_gate(
    rounds_scored: Sequence[Dict[str, object]],
    kill_events_by_demo_round: Dict[str, Dict[int, List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Block low-attention rounds unless they show standout highlight traits.

    Allowed traits (any one is enough):
    - clutch cue (from round scoring)
    - multi-kill by a single player (>=3)
    - no-scope kill
    - deagle/r8 highlight kill
    - flick/quick-scope cue inferred by event flags (best-effort)
    """
    notes: List[str] = []
    updated: List[Dict[str, object]] = []

    blocked = 0
    for row in rounds_scored:
        r = dict(row)
        if not bool(r.get("is_round_highlight", False)):
            updated.append(r)
            continue

        demo = str(r.get("demo", ""))
        rnd = int(r.get("round", 0))
        events = kill_events_by_demo_round.get(demo, {}).get(rnd, [])

        has_clutch = float(r.get("clutch_cue_score", 0.0)) >= 2.0
        has_multikill = float(r.get("max_multikill_by_player", 0.0)) >= 3.0
        has_noscope = any(bool(e.get("noscope", False)) for e in events)
        has_deagle = any(
            any(tok in str(e.get("weapon", "")).lower() for tok in ("deagle", "revolver"))
            for e in events
        )

        # Best-effort quick-scope / flick heuristics from available event flags.
        has_quickscope = any(
            ("awp" in str(e.get("weapon", "")).lower() and not bool(e.get("noscope", False)))
            for e in events
        )
        has_flick_like = any(
            bool(e.get("attacker_in_air", False)) or bool(e.get("attacker_blind", False))
            for e in events
        )

        attention_ok = any([has_clutch, has_multikill, has_noscope, has_deagle, has_quickscope, has_flick_like])
        r["attention_gate_passed"] = bool(attention_ok)

        if not attention_ok:
            blocked += 1
            r["posting_allowed"] = False
            prior_reason = str(r.get("eco_guard_reason", "")).strip()
            gate_reason = (
                "Blocked: low-attention round (no clutch, no 3K+ single-player multi-kill, "
                "no flick/quick-scope cue, no no-scope, no deagle highlight)."
            )
            r["eco_guard_reason"] = f"{prior_reason} | {gate_reason}" if prior_reason else gate_reason

        updated.append(r)

    if blocked > 0:
        notes.append(f"Attention gate blocked {blocked} rounds that lacked standout highlight traits.")
    return updated, notes


def _extract_sample_frames(video_path: Path, out_dir: Path, sample_seconds: float, max_frames: int) -> List[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available in PATH.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%04d.jpg"
    vf = f"fps=1/{sample_seconds}"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-vf",
        vf,
        "-q:v",
        "3",
        str(pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to extract sample frames with ffmpeg.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) > max_frames:
        step = len(frames) / float(max_frames)
        sampled: List[Path] = []
        idx = 0.0
        while len(sampled) < max_frames and int(idx) < len(frames):
            sampled.append(frames[int(idx)])
            idx += step
        frames = sampled
    return frames


def _extract_json_block(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Gemini response did not contain JSON.")
    return json.loads(match.group(0))


def _vision_backend_available(
    ai_backend: str,
    gemini_api_key: Optional[str],
    vertex_project_id: Optional[str],
    vertex_api_key: Optional[str],
) -> bool:
    if ai_backend == "vertex":
        return bool(vertex_project_id or vertex_api_key)
    return bool(gemini_api_key)


def _generate_multimodal_json(
    prompt: str,
    image_paths: Sequence[Path],
    ai_backend: str,
    model: str,
    gemini_api_key: Optional[str],
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
) -> Dict[str, object]:
    if ai_backend == "vertex" and vertex_api_key:
        if not vertex_project_id:
            raise RuntimeError("Vertex API key mode requires --vertex-project-id.")

        model_resource = model.strip()
        if model_resource.startswith("projects/"):
            endpoint = (
                f"https://{vertex_location}-aiplatform.googleapis.com/v1/"
                f"{model_resource}:generateContent"
            )
        else:
            endpoint = (
                f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project_id}/"
                f"locations/{vertex_location}/publishers/google/models/{model_resource}:generateContent"
            )
        endpoint = f"{endpoint}?key={urllib.parse.quote(vertex_api_key)}"

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image_path in image_paths:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                    }
                }
            )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vertex API key request failed ({exc.code}): {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Vertex API key request failed: {exc}") from exc

        data = json.loads(body)
        text_parts: List[str] = []
        for cand in data.get("candidates", []):
            content = cand.get("content", {}) if isinstance(cand, dict) else {}
            for part in content.get("parts", []) if isinstance(content, dict) else []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        combined_text = "\n".join(text_parts).strip()
        if not combined_text:
            raise RuntimeError(f"Vertex response missing text payload: {body}")
        return _extract_json_block(combined_text)

    try:
        from google import genai as genai_sdk  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "google-genai is not installed. Install with: pip install google-genai"
        ) from exc

    if ai_backend == "vertex":
        client_kwargs: Dict[str, Any] = {"vertexai": True}
        if vertex_api_key:
            client_kwargs["api_key"] = vertex_api_key
            client_kwargs["location"] = vertex_location
        else:
            if not vertex_project_id:
                raise RuntimeError(
                    "Vertex backend requires --vertex-project-id/GOOGLE_CLOUD_PROJECT or --vertex-api-key."
                )
            client_kwargs["project"] = vertex_project_id
            client_kwargs["location"] = vertex_location
        client = genai_sdk.Client(
            **client_kwargs,
        )
    else:
        if not gemini_api_key:
            raise RuntimeError("Gemini backend requires --gemini-api-key or GEMINI_API_KEY.")
        client = genai_sdk.Client(api_key=gemini_api_key)

    contents: List[object] = [prompt]
    for image_path in image_paths:
        contents.append(
            genai_types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type="image/jpeg",
            )
        )

    response = client.models.generate_content(model=model, contents=contents)
    response_text = getattr(response, "text", "") or ""
    return _extract_json_block(response_text)


def _get_video_duration_seconds(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    try:
        return float((proc.stdout or "").strip())
    except Exception:
        return 0.0


def _score_vision_with_gemini(
    video_path: Path,
    api_key: Optional[str],
    model: str,
    sample_seconds: float,
    max_frames: int,
    ai_backend: str,
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
) -> Tuple[Dict[str, float], List[str]]:
    with tempfile.TemporaryDirectory(prefix="gemini_frames_") as td:
        frames = _extract_sample_frames(video_path, Path(td), sample_seconds, max_frames)
        if not frames:
            raise RuntimeError("No frames were extracted for Gemini analysis.")

        prompt = (
            "You are scoring a CS2 clip for highlight potential from sampled frames only. "
            "Return ONLY strict JSON with this exact schema: "
            '{"vision_score": number 0-10, "is_highlight": boolean, "confidence": number 0-1, '
            '"signals": {"action_intensity": number 0-10, "combat_presence": number 0-10, '
            '"clutch_or_multikill_cues": number 0-10, "broadcast_hype_cues": number 0-10}, '
            '"reasons": [string, string]}. '
            "No markdown, no extra text."
        )

        parsed = _generate_multimodal_json(
            prompt=prompt,
            image_paths=frames,
            ai_backend=ai_backend,
            model=model,
            gemini_api_key=api_key,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            vertex_api_key=vertex_api_key,
        )

        signals = parsed.get("signals", {}) if isinstance(parsed, dict) else {}
        vision = {
            "vision_score": float(parsed.get("vision_score", 0.0)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "action_intensity": float(signals.get("action_intensity", 0.0)) if isinstance(signals, dict) else 0.0,
            "combat_presence": float(signals.get("combat_presence", 0.0)) if isinstance(signals, dict) else 0.0,
            "clutch_or_multikill_cues": float(signals.get("clutch_or_multikill_cues", 0.0)) if isinstance(signals, dict) else 0.0,
            "broadcast_hype_cues": float(signals.get("broadcast_hype_cues", 0.0)) if isinstance(signals, dict) else 0.0,
        }
        backend_label = "Vertex" if ai_backend == "vertex" else "Gemini"
        notes = [f"{backend_label} analyzed {len(frames)} sampled frames using model {model}."]
        for reason in parsed.get("reasons", []) if isinstance(parsed, dict) else []:
            notes.append(f"{backend_label}: {reason}")
        return vision, notes


def _collect_round_kill_events(kills_df) -> Dict[int, List[Dict[str, Any]]]:
    if kills_df is None or len(kills_df) == 0 or "_round_number" not in kills_df.columns:
        return {}

    columns = list(kills_df.columns)
    attacker_col = _pick_column(columns, ["attacker_name", "attacker", "killer_name", "killer"])
    victim_col = _pick_column(columns, ["user_name", "victim_name", "user", "victim"])
    weapon_col = _pick_column(columns, ["weapon", "weapon_name"])
    hs_col = _pick_column(columns, ["is_headshot", "headshot", "head_shot"])
    time_col = _pick_column(columns, ["_time_seconds", "time", "seconds", "game_time", "clock_time", "tick"])
    ns_col = _pick_column(columns, ["noscope"])
    smoke_col = _pick_column(columns, ["thrusmoke", "through_smoke"])
    pen_col = _pick_column(columns, ["penetrated"])
    blind_col = _pick_column(columns, ["attackerblind"])
    air_col = _pick_column(columns, ["attackerinair"])

    rows: Dict[int, List[Dict[str, Any]]] = {}
    for _, row in kills_df.iterrows():
        rnd = int(row.get("_round_number", 0))
        if rnd <= 0:
            continue
        evt = {
            "attacker": "" if not attacker_col else str(row.get(attacker_col, "")),
            "victim": "" if not victim_col else str(row.get(victim_col, "")),
            "weapon": "" if not weapon_col else str(row.get(weapon_col, "")),
            "headshot": False if not hs_col else bool(row.get(hs_col, False)),
            "time_seconds": float(row.get(time_col, 0.0)) if time_col else 0.0,
            "noscope": False if not ns_col else bool(row.get(ns_col, False)),
            "through_smoke": False if not smoke_col else bool(row.get(smoke_col, False)),
            "penetrated": 0.0 if not pen_col else float(row.get(pen_col, 0.0)),
            "attacker_blind": False if not blind_col else bool(row.get(blind_col, False)),
            "attacker_in_air": False if not air_col else bool(row.get(air_col, False)),
        }
        rows.setdefault(rnd, []).append(evt)
    return rows


def _extract_single_frame(video_path: Path, timestamp_sec: float, out_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and out_path.exists()


def _gemini_round_visual_feedback(
    video_path: Path,
    api_key: Optional[str],
    model: str,
    round_events: List[Dict[str, Any]],
    clip_start_demo_seconds: float,
    video_duration_seconds: float,
    ai_backend: str,
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
    max_frames: int = 4,
) -> Dict[str, Any]:
    # Prioritize visually notable events.
    scored = []
    for e in round_events:
        s = 0.0
        if e.get("headshot"):
            s += 1.0
        if e.get("noscope"):
            s += 2.0
        if e.get("through_smoke"):
            s += 1.2
        if float(e.get("penetrated", 0.0)) > 0:
            s += 1.0
        if e.get("attacker_blind"):
            s += 1.0
        if e.get("attacker_in_air"):
            s += 1.0
        scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [e for _, e in scored[:max_frames]]

    with tempfile.TemporaryDirectory(prefix="round_visual_feedback_") as td:
        td_path = Path(td)
        frame_files: List[Path] = []
        for i, e in enumerate(chosen, 1):
            demo_t = float(e.get("time_seconds", 0.0))
            clip_t = demo_t - clip_start_demo_seconds
            if clip_t < 0.0 or (video_duration_seconds > 0 and clip_t > video_duration_seconds):
                continue
            fpath = td_path / f"evt_{i:02d}.jpg"
            if _extract_single_frame(video_path, clip_t, fpath):
                frame_files.append(fpath)

        if not frame_files:
            return {
                "visual_score": 0.0,
                "highlight_worthy_shot": False,
                "flickshot_detected": False,
                "strange_event_detected": False,
                "summary": "No mappable round frames extracted for Gemini review. Adjust clip-start mapping.",
                "confidence": 0.0,
            }

        event_text = "\n".join(
            [
                f"- {e.get('attacker')} killed {e.get('victim')} with {e.get('weapon')} "
                f"(HS={e.get('headshot')}, noScope={e.get('noscope')}, smoke={e.get('through_smoke')}, pen={e.get('penetrated')})"
                for e in chosen
            ]
        )

        prompt = (
            "You are reviewing a CS2 round for highlight-worthiness. "
            "Use the provided frame images and kill-event metadata.\n"
            f"Kill events:\n{event_text}\n"
            "Return ONLY strict JSON with schema: "
            '{"visual_score": number 0-10, "highlight_worthy_shot": boolean, '
            '"flickshot_detected": boolean, "strange_event_detected": boolean, '
            '"summary": string, "confidence": number 0-1}. '
            "No markdown."
        )

        try:
            parsed = _generate_multimodal_json(
                prompt=prompt,
                image_paths=frame_files,
                ai_backend=ai_backend,
                model=model,
                gemini_api_key=api_key,
                vertex_project_id=vertex_project_id,
                vertex_location=vertex_location,
                vertex_api_key=vertex_api_key,
            )
            return {
                "visual_score": float(parsed.get("visual_score", 0.0)),
                "highlight_worthy_shot": bool(parsed.get("highlight_worthy_shot", False)),
                "flickshot_detected": bool(parsed.get("flickshot_detected", False)),
                "strange_event_detected": bool(parsed.get("strange_event_detected", False)),
                "summary": str(parsed.get("summary", "Gemini visual review completed.")),
                "confidence": float(parsed.get("confidence", 0.0)),
            }
        except Exception as exc:
            return {
                "visual_score": 0.0,
                "highlight_worthy_shot": False,
                "flickshot_detected": False,
                "strange_event_detected": False,
                "summary": f"Gemini visual review failed: {exc}",
                "confidence": 0.0,
            }


def _combine_scores(
    demo_score: Optional[float],
    vision_score: Optional[float],
    demo_weight: float,
    vision_weight: float,
) -> Tuple[float, bool, float, List[str]]:
    notes: List[str] = []

    if demo_score is None and vision_score is None:
        return 0.0, False, 0.0, ["No scoring sources were available."]

    if demo_score is not None and vision_score is not None:
        total_w = demo_weight + vision_weight
        if total_w <= 0:
            total_w = 1.0
            demo_weight = 0.5
            vision_weight = 0.5
        combined = (demo_score * demo_weight + vision_score * vision_weight) / total_w
        notes.append(f"Combined score using demo_weight={demo_weight} and vision_weight={vision_weight}.")
    elif demo_score is not None:
        combined = demo_score
        notes.append("Used demo score only (vision score unavailable).")
    else:
        combined = float(vision_score)
        notes.append("Used Gemini vision score only (demo score unavailable).")

    threshold = 7.0
    is_highlight = combined >= threshold
    confidence = 1.0 / (1.0 + math.exp(-(combined - threshold)))
    return combined, is_highlight, confidence, notes


def detect_highlight(
    demo_input: Optional[Path],
    video_path: Optional[Path],
    gemini_api_key: Optional[str],
    gemini_model: str,
    ai_backend: str,
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
    sample_seconds: float,
    max_frames: int,
    demo_weight: float,
    vision_weight: float,
    top_rounds_count: int,
    max_highlights_per_map: int,
    clip_start_demo_seconds: float,
    visual_feedback_max_rounds: int,
) -> DetectionResult:
    notes: List[str] = []
    demos: List[Path] = []

    demo_score: Optional[float] = None
    demo_features: Dict[str, float] = {
        "kills_total": 0.0,
        "kills_per_minute": 0.0,
        "headshot_ratio": 0.0,
        "multi_kill_burst_score": 0.0,
        "unique_killers": 0.0,
    }
    rounds_scored: List[Dict[str, object]] = []
    round_highlight_cutoffs: Dict[str, float] = {}
    eco_info_by_demo: Dict[str, Dict[int, Dict[str, Any]]] = {}
    round_bounds_by_demo: Dict[str, Dict[int, Tuple[float, float]]] = {}
    kill_events_by_demo_round: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    if demo_input is not None:
        if not demo_input.exists():
            raise FileNotFoundError(f"Demo input not found: {demo_input}")

        demos = _resolve_demo_paths(demo_input)
        map_scores: List[float] = []
        map_feature_list: List[Dict[str, float]] = []
        for demo in demos:
            kills_df = _load_kill_events(demo)
            round_start_df = _load_round_start_events(demo)
            round_end_df = _load_round_end_events(demo)
            round_bounds_by_demo[demo.name] = _build_round_bounds_seconds(round_start_df, round_end_df)
            eco_info_by_demo[demo.name] = _infer_eco_loss_rounds(demo, round_start_df, round_end_df)

            kills_df, normalize_notes = _normalize_kill_time_column(kills_df)
            notes.extend([f"{demo.name}: {n}" for n in normalize_notes])

            kills_df, round_assign_notes = _assign_round_numbers(kills_df, round_start_df)
            notes.extend([f"{demo.name}: {n}" for n in round_assign_notes])

            kill_events_by_demo_round[demo.name] = _collect_round_kill_events(kills_df)

            # Attach round_end reason to each kill row for context-aware round scoring.
            if round_end_df is not None and len(round_end_df) > 0 and "_round_number" in kills_df.columns:
                re_cols = list(round_end_df.columns)
                re_round_col = _pick_column(re_cols, ["round", "round_num", "round_number"])
                re_reason_col = _pick_column(re_cols, ["reason"])
                if re_round_col and re_reason_col:
                    reason_map = {
                        int(r): str(v)
                        for r, v in zip(round_end_df[re_round_col].tolist(), round_end_df[re_reason_col].tolist())
                    }
                    kills_df = kills_df.copy()
                    kills_df["_round_end_reason"] = kills_df["_round_number"].map(reason_map)

            features, map_notes = _compute_features(kills_df)
            round_rows, round_notes = _score_rounds(kills_df, demo.name)
            notes.extend(round_notes)
            rounds_scored.extend(round_rows)

            score = _map_score_from_rounds(round_rows)
            if score is None:
                score, _, _ = _score_demo_highlight(features)
            map_scores.append(score)
            map_feature_list.append(features)
            notes.extend([f"{demo.name}: {n}" for n in map_notes])

        if map_scores:
            demo_score = max(map_scores)
            notes.append("Demo score uses max map score to capture peak highlight moments.")

        # Aggregate display features across maps.
        if map_feature_list:
            keys = list(demo_features.keys())
            for k in keys:
                demo_features[k] = sum(m[k] for m in map_feature_list) / len(map_feature_list)

    vision: Dict[str, float] = {
        "vision_score": 0.0,
        "confidence": 0.0,
        "action_intensity": 0.0,
        "combat_presence": 0.0,
        "clutch_or_multikill_cues": 0.0,
        "broadcast_hype_cues": 0.0,
    }
    vision_score: Optional[float] = None
    if video_path is not None:
        if not video_path.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        if _vision_backend_available(ai_backend, gemini_api_key, vertex_project_id, vertex_api_key):
            try:
                vision, vision_notes = _score_vision_with_gemini(
                    video_path=video_path,
                    api_key=gemini_api_key,
                    model=gemini_model,
                    sample_seconds=sample_seconds,
                    max_frames=max_frames,
                    ai_backend=ai_backend,
                    vertex_project_id=vertex_project_id,
                    vertex_location=vertex_location,
                    vertex_api_key=vertex_api_key,
                )
                vision_score = vision["vision_score"]
                notes.extend(vision_notes)
            except Exception as exc:
                backend_label = "Vertex" if ai_backend == "vertex" else "Gemini"
                notes.append(f"{backend_label} vision scoring failed and was skipped: {exc}")
        else:
            if ai_backend == "vertex":
                notes.append("Video provided but Vertex project was not set, so vision scoring was skipped.")
            else:
                notes.append("Video provided but GEMINI_API_KEY was not set, so vision scoring was skipped.")

    combined_score, is_highlight, confidence, combine_notes = _combine_scores(
        demo_score=demo_score,
        vision_score=vision_score,
        demo_weight=demo_weight,
        vision_weight=vision_weight,
    )
    notes.extend(combine_notes)

    rounds_scored, round_highlight_cutoffs, cap_notes = _apply_per_map_highlight_cap(
        round_rows=rounds_scored,
        max_highlights_per_map=max_highlights_per_map,
    )
    notes.extend(cap_notes)

    video_duration_seconds = 0.0
    if video_path is not None and video_path.exists():
        video_duration_seconds = _get_video_duration_seconds(video_path)

    rounds_scored, eco_notes = _apply_eco_guard_with_gemini(
        rounds_scored=rounds_scored,
        eco_info_by_demo=eco_info_by_demo,
        round_bounds_by_demo=round_bounds_by_demo,
        video_path=video_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        ai_backend=ai_backend,
        vertex_project_id=vertex_project_id,
        vertex_location=vertex_location,
        vertex_api_key=vertex_api_key,
        clip_start_demo_seconds=clip_start_demo_seconds,
        video_duration_seconds=video_duration_seconds,
    )
    notes.extend(eco_notes)

    rounds_scored, attention_notes = _apply_attention_gate(
        rounds_scored=rounds_scored,
        kill_events_by_demo_round=kill_events_by_demo_round,
    )
    notes.extend(attention_notes)

    # Add per-round event mapping and optional Gemini visual feedback.
    highlight_round_details: List[Dict[str, Any]] = []
    visual_review_budget = max(0, int(visual_feedback_max_rounds))
    for r in sorted(rounds_scored, key=lambda x: float(x["round_score"]), reverse=True):
        if not bool(r.get("is_round_highlight", False)):
            continue
        detail: Dict[str, Any] = {
            "demo": r.get("demo", ""),
            "round": int(r.get("round", 0)),
            "round_score": float(r.get("round_score", 0.0)),
            "posting_allowed": bool(r.get("posting_allowed", False)),
            "eco_loss_round": bool(r.get("eco_loss_round", False)),
        }

        demo_name = str(detail["demo"])
        round_num = int(detail["round"])
        events = kill_events_by_demo_round.get(demo_name, {}).get(round_num, [])
        detail["kill_events"] = events

        if (
            visual_review_budget > 0
            and video_path is not None
            and _vision_backend_available(ai_backend, gemini_api_key, vertex_project_id, vertex_api_key)
        ):
            feedback = _gemini_round_visual_feedback(
                video_path=video_path,
                api_key=gemini_api_key,
                model=gemini_model,
                round_events=events,
                clip_start_demo_seconds=clip_start_demo_seconds,
                video_duration_seconds=video_duration_seconds,
                ai_backend=ai_backend,
                vertex_project_id=vertex_project_id,
                vertex_location=vertex_location,
                vertex_api_key=vertex_api_key,
            )
            detail["vertex_visual_feedback"] = feedback
            visual_review_budget -= 1
        else:
            detail["vertex_visual_feedback"] = {
                "visual_score": 0.0,
                "highlight_worthy_shot": False,
                "flickshot_detected": False,
                "strange_event_detected": False,
                "summary": "Visual feedback skipped (missing api key/video path or budget).",
                "confidence": 0.0,
            }

        highlight_round_details.append(detail)

    top_rounds = sorted(rounds_scored, key=lambda r: float(r["round_score"]), reverse=True)[:top_rounds_count]
    if rounds_scored:
        highlight_rounds = sum(1 for r in rounds_scored if bool(r.get("posting_allowed", False)))
        notes.append(
            f"Round scoring enabled: {len(rounds_scored)} rounds scored across demos; {highlight_rounds} rounds allowed for posting after guards."
        )
    else:
        notes.append("Round scoring produced no rows; fallback map-level scoring used where needed.")

    notes.insert(0, "Classified as highlight candidate." if is_highlight else "Classified as non-highlight.")

    resolved_input = demo_input or video_path
    return DetectionResult(
        input_path=str(resolved_input) if resolved_input else "",
        demos_analyzed=[str(p) for p in demos],
        score=combined_score,
        is_highlight=is_highlight,
        confidence=confidence,
        demo_score=demo_score,
        vision_score=vision_score,
        features=demo_features,
        vision=vision,
        rounds_scored=rounds_scored,
        top_rounds=top_rounds,
        round_highlight_cutoffs=round_highlight_cutoffs,
        highlight_round_details=highlight_round_details,
        notes=notes,
    )


def _generate_text_json(
    prompt: str,
    ai_backend: str,
    model: str,
    gemini_api_key: Optional[str],
    vertex_project_id: Optional[str],
    vertex_location: str,
    vertex_api_key: Optional[str],
) -> Dict[str, Any]:
    """Generates a JSON response from a text-only prompt."""
    if ai_backend == "vertex" and vertex_api_key:
        if not vertex_project_id:
            raise RuntimeError("Vertex API key mode requires --vertex-project-id.")
        endpoint = (
            f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project_id}/"
            f"locations/{vertex_location}/publishers/google/models/{model}:generateContent"
            f"?key={urllib.parse.quote(vertex_api_key)}"
        )
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json_block(text)

    try:
        from google import genai as genai_sdk
    except ImportError:
        raise RuntimeError("google-genai is not installed.")

    if ai_backend == "vertex":
        client = genai_sdk.Client(vertexai=True, project=vertex_project_id, location=vertex_location)
    else:
        client = genai_sdk.Client(api_key=gemini_api_key)

    response = client.models.generate_content(model=model, contents=[prompt])
    return _extract_json_block(response.text or "")


class LiveStreamMonitor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stream_url = args.live_url
        self.api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.model = args.gemini_model
        self.backend = args.ai_backend
        
        self.output_root = Path(args.output_dir or "live_pipeline_output")
        self.screens_dir = self.output_root / "screenshots"
        self.clips_dir = self.output_root / "raw_clips"
        self.processed_dir = self.output_root / "processed"
        
        for d in [self.screens_dir, self.clips_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.current_round = None
        self.recording_proc = None
        self.recording_path = None
        self.last_screenshot_time = 0

    def _resolve_stream(self) -> str:
        if "twitch.tv" in self.stream_url or "youtube.com" in self.stream_url:
            res = subprocess.run(["streamlink", "--stream-url", self.stream_url, "best"], capture_output=True, text=True)
            if res.returncode == 0:
                return res.stdout.strip()
        return self.stream_url

    def capture_screenshot(self) -> Optional[Path]:
        path = self.screens_dir / f"live_{int(time.time())}.jpg"
        stream = self._resolve_stream()
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", stream, "-frames:v", "1", "-q:v", "2", str(path)]
        if subprocess.run(cmd).returncode == 0:
            return path
        return None

    def detect_round(self, image_path: Path) -> Optional[int]:
        prompt = (
            "Analyze this CS2 stream screenshot. Identify the current round number from the HUD. "
            "Return ONLY a JSON object: {\"round\": number_or_null}"
        )
        try:
            res = _generate_multimodal_json(
                prompt, [image_path], self.backend, self.model, self.api_key,
                self.args.vertex_project_id, self.args.vertex_location, self.args.vertex_api_key
            )
            return res.get("round")
        except Exception as e:
            print(f"Round detection error: {e}")
            return None

    def start_recording(self, round_num: int):
        self.current_round = round_num
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = self.clips_dir / f"round_{round_num}_{timestamp}.mp4"
        stream = self._resolve_stream()
        
        # Record at 1080p
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", stream,
            "-vf", "scale=1920:1080", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k", str(self.recording_path)
        ]
        self.recording_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print(f"Started recording Round {round_num} -> {self.recording_path.name}")

    def stop_recording(self):
        if self.recording_proc:
            # Send 'q' to ffmpeg to stop gracefully
            try:
                self.recording_proc.communicate(input=b'q', timeout=5)
            except:
                self.recording_proc.terminate()
            self.recording_proc = None
            print(f"Stopped recording Round {self.current_round}")
            self.process_clip(self.recording_path, self.current_round)

    def process_clip(self, clip_path: Path, round_num: int):
        print(f"Analyzing Round {round_num} for highlights...")
        
        # 1. Classify Highlight via Gemini
        prompt = (
            "Is this CS2 round a highlight? Look for multi-kills, clutches, or exceptional aim. "
            "Return ONLY JSON: {\"is_highlight\": boolean, \"reason\": \"string\", \"title\": \"string\", \"seo_keywords\": [\"string\"]}"
        )
        try:
            analysis = _generate_multimodal_json(
                prompt, [clip_path], self.backend, self.model, self.api_key,
                self.args.vertex_project_id, self.args.vertex_location, self.args.vertex_api_key
            )
        except Exception as e:
            print(f"Highlight analysis failed: {e}")
            return

        if not analysis.get("is_highlight"):
            print(f"Round {round_num} is not a highlight. Skipping.")
            return

        print(f"HIGHLIGHT DETECTED: {analysis.get('reason')}")

        # 2. Edit Video: portrait 9:16 with blurry top/bottom (shared implementation)
        portrait_path = self.processed_dir / f"round_{round_num}_portrait.mp4"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        apply_portrait_blur(
            clip_path,
            portrait_path,
            ffmpeg_bin="ffmpeg",
            blur_chroma_radius=8,
            crf=18,
            preset="slow",
            fps=30.0,
        )

        # 3. Add Captions (Riverside Script)
        captioned_path = self.processed_dir / f"round_{round_num}_captioned.mp4"
        ps_cmd = [
            "powershell", "-ExecutionPolicy", "Bypass", "-File", "riverside_vertical_caption_edit.ps1",
            "-InputVideo", str(portrait_path), "-OutputVideo", str(captioned_path)
        ]
        subprocess.run(ps_cmd)

        # 4. Generate Title & SEO
        print(f"Generated Title: {analysis.get('title')}")
        print(f"SEO Keywords: {', '.join(analysis.get('seo_keywords', []))}")

        # 5. Post to Instagram (Placeholder)
        if self.args.instagram_post:
            self.post_to_instagram(captioned_path, analysis)

    def post_to_instagram(self, video_path: Path, meta: Dict[str, Any]):
        print(f"Posting to Instagram: {video_path.name}")
        # Note: Requires instagrapi
        try:
            from instagrapi import Client
            cl = Client()
            cl.login(os.getenv("INSTA_USER"), os.getenv("INSTA_PASS"))
            caption = f"{meta.get('title')}\n\n" + " ".join([f"#{k.replace(' ', '')}" for k in meta.get("seo_keywords", [])])
            cl.clip_upload(str(video_path), caption=caption)
            print("Successfully posted to Instagram!")
        except ImportError:
            print("instagrapi not installed. Skipping Instagram post.")
        except Exception as e:
            print(f"Instagram post failed: {e}")

    def run(self):
        print(f"Starting live monitor for: {self.stream_url}")
        try:
            while True:
                now = time.time()
                if now - self.last_screenshot_time >= 2:
                    self.last_screenshot_time = now
                    shot = self.capture_screenshot()
                    if shot:
                        round_num = self.detect_round(shot)
                        shot.unlink() # Cleanup
                        
                        if round_num is not None:
                            if self.current_round is None:
                                self.start_recording(round_num)
                            elif round_num > self.current_round:
                                self.stop_recording()
                                self.start_recording(round_num)
                        
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop_recording()
            print("Monitor stopped.")


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Detect whether a CS2 clip is a highlight candidate using demo data and optional Gemini vision."
    )
    parser.add_argument(
        "--demo-input",
        required=False,
        help="Path to .dem, .rar, or a directory containing .dem files",
    )
    parser.add_argument(
        "--video-path",
        required=False,
        help="Path to clip video file for Gemini vision scoring",
    )
    parser.add_argument(
        "--gemini-api-key",
        required=False,
        help="Gemini API key (optional if GEMINI_API_KEY env var is set)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Model name for vision scoring (Gemini API or Vertex Gemini model)",
    )
    parser.add_argument(
        "--ai-backend",
        choices=["gemini", "vertex"],
        default="gemini",
        help="Vision backend: gemini (API key) or vertex (project/location with ADC)",
    )
    parser.add_argument(
        "--vertex-project-id",
        required=False,
        help="GCP project id for Vertex backend (or set GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--vertex-location",
        default="us-central1",
        help="Vertex location for model calls",
    )
    parser.add_argument(
        "--vertex-api-key",
        required=False,
        help="Optional API key for Vertex backend (can replace ADC in some setups)",
    )
    parser.add_argument(
        "--frame-sample-seconds",
        type=float,
        default=2.0,
        help="Sample one frame every N seconds for Gemini",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum sampled frames sent to Gemini",
    )
    parser.add_argument(
        "--demo-weight",
        type=float,
        default=0.55,
        help="Weight for demo score when combining with Gemini",
    )
    parser.add_argument(
        "--vision-weight",
        type=float,
        default=0.45,
        help="Weight for Gemini score when combining with demo score",
    )
    parser.add_argument(
        "--top-rounds-count",
        type=int,
        default=5,
        help="Number of highest-scored rounds to surface in output",
    )
    parser.add_argument(
        "--max-highlights-per-map",
        type=int,
        default=5,
        help="Maximum number of rounds flagged as highlights per map",
    )
    parser.add_argument(
        "--clip-start-demo-seconds",
        type=float,
        default=0.0,
        help="Demo timeline second that corresponds to clip video t=0. Used to map demo kill times to clip frames.",
    )
    parser.add_argument(
        "--visual-feedback-max-rounds",
        type=int,
        default=6,
        help="Maximum number of highlighted rounds to send for Gemini visual feedback.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        help="Optional path to save JSON result. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Deprecated alias for --demo-input",
    )
    parser.add_argument(
        "--live-url",
        help="URL of the live stream to monitor (Twitch/YouTube/Direct)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable live stream monitoring mode",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store pipeline outputs",
    )
    parser.add_argument(
        "--instagram-post",
        action="store_true",
        help="Automatically post detected highlights to Instagram",
    )

    args = parser.parse_args()

    if args.monitor:
        if not args.live_url:
            raise ValueError("--live-url is required for monitoring mode")
        monitor = LiveStreamMonitor(args)
        monitor.run()
        return 0


    demo_input_raw = args.demo_input or args.input
    demo_input = Path(demo_input_raw).expanduser().resolve() if demo_input_raw else None
    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else None

    if demo_input is None and video_path is None:
        raise ValueError("Provide at least one source: --demo-input or --video-path")

    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    vertex_project_id = args.vertex_project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    vertex_api_key = args.vertex_api_key or os.getenv("VERTEX_API_KEY") or os.getenv("GOOGLE_API_KEY")

    result = detect_highlight(
        demo_input=demo_input,
        video_path=video_path,
        gemini_api_key=gemini_api_key,
        gemini_model=args.gemini_model,
        ai_backend=args.ai_backend,
        vertex_project_id=vertex_project_id,
        vertex_location=args.vertex_location,
        vertex_api_key=vertex_api_key,
        sample_seconds=args.frame_sample_seconds,
        max_frames=args.max_frames,
        demo_weight=args.demo_weight,
        vision_weight=args.vision_weight,
        top_rounds_count=args.top_rounds_count,
        max_highlights_per_map=args.max_highlights_per_map,
        clip_start_demo_seconds=args.clip_start_demo_seconds,
        visual_feedback_max_rounds=args.visual_feedback_max_rounds,
    )
    payload = result.to_json()

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Saved: {output_path}")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
