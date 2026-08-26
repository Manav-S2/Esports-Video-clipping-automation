# Esports Video Clipping Automation

> **Project type:** Python application pipeline (media automation / applied ML).
> Standalone — no containers, orchestration, or infrastructure tooling involved;
> it runs directly from a virtual environment on Windows, Linux, or macOS.

End-to-end automation that turns **live CS2 (Counter-Strike 2) esports streams into ready-to-post
vertical highlight Reels** — recording rounds off a live stream, scoring them for highlight value
with demo-file analytics and multimodal LLM review, reframing them to 9:16 with blurred
letterboxing, burning karaoke-style captions from Google Cloud Speech word timestamps, and
optionally publishing to Instagram.

Built and battle-tested against real broadcast footage (BLAST Open Rotterdam 2026, NAVI vs G2,
PARIVISION vs Falcons) — see `highlight_result_*.json` for real scored runs.

## What it does

```
live stream ──▶ round detection ──▶ per-round recording ──▶ highlight scoring ──▶ portrait edit ──▶ karaoke captions ──▶ post
  (Twitch/       (Gemini vision      (ffmpeg/streamlink)     (demo .dem stats      (9:16 blur       (Cloud Speech STT     (Instagram,
   YouTube/       on HUD                                      + Gemini/Vertex       letterbox)       → ASS karaoke →       optional)
   direct URL)    screenshots)                                + audio RMS hype)                      ffmpeg burn-in)
```

## Repository map

| Path | Role |
| --- | --- |
| `live_stream_highlight_pipeline.py` | Main orchestrator: live monitoring, round recording, Vertex/Gemini highlight scoring, portrait export, caption hooks, SEO text, Instagram posting, crash-safe state |
| `detect_cs2_highlight.py` | Highlight detector: CS2 demo (`.dem`) parsing via demoparser2, per-round kill/eco features, weighted scoring, Gemini vision fusion, eco-round guard, per-map highlight cap |
| `speech_google_captions.py` | Google Cloud Speech-to-Text (REST + chunked long-audio + ADC/LRO paths) → SRT / ASS karaoke subtitles, audio-vs-video timeline drift correction, ffmpeg burn-in |
| `burn_karaoke_captions.py` | CLI wrapper: one command from finished clip to caption-burned output |
| `video_editor.py` | 16:9 → 9:16 portrait export with blurred background fill (Reels/TikTok framing) |
| `stream_recorder.py` | Standalone stream round-recorder (screenshot → Gemini round read → per-round capture) |
| `unified_pipeline.py` | Minimal single-file pipeline variant (record → classify → portrait edit) |
| `ask_gemini_clip5_reason.py`, `clip5_*.py`, `inspect_clip5_round.py`, `ask_nvidia_clip_highlight.py` | Model-evaluation harnesses used to tune the highlight prompts (Gemini / NVIDIA NIM) |
| `media_tools.py`, `extract_killfeed_snapshots.ps1`, `rebuild_every_second_8k.ps1`, `optimize_frames_numpy_pandas.py` | OCR-grade video enhancement + killfeed ROI extraction toolchain (earlier OCR-based detection approach) |
| `speech_google_captions.py` + `CAPTIONS/` | Caption styling assets and outputs |
| `scripts/bootstrap.ps1`, `scripts/bootstrap.sh` | One-command environment setup (locked install + test run + tool check) |
| `tests/` | Unit tests for the pure scoring/parsing/caption logic (`python -m unittest discover tests`) |
| `docs/` | [Architecture](docs/ARCHITECTURE.md) · [Setup](docs/SETUP.md) · [Tool reference](docs/TOOLS.md) |

## Quick start

```powershell
# 1. Environment — or just run: .\scripts\bootstrap.ps1  (Linux/macOS: ./scripts/bootstrap.sh)
python -m venv .venv
.\.venv\Scripts\activate
uv sync                                # reproducible install from committed uv.lock
# (or: pip install -r requirements-lock.txt / requirements.txt)
# ffmpeg + streamlink must be on PATH

# 2. Credentials (never committed — see SECURITY.md)
copy .env.example .env                                             # env-var reference
copy live_pipeline_config.example.json live_pipeline_config.json   # fill in keys
copy speech_api_key.local.example.json speech_api_key.local.json

# 3. Run the live pipeline
$env:GEMINI_API_KEY = "YOUR_KEY"
python .\live_stream_highlight_pipeline.py --config .\live_pipeline_config.json
```

Offline highlight screening of an existing clip + demo:

```powershell
python .\detect_cs2_highlight.py --demo-input "C:\demos" --video-path "C:\clip.mp4" --output-json result.json
```

Burn karaoke captions onto a finished clip:

```powershell
py -3.14 burn_karaoke_captions.py --video "C:\path\to\clip_portrait.mp4"
```

## Highlight scoring model

Per-round features extracted from the demo file (kills, kills/minute, headshot ratio,
multi-kill burst score, unique killers) are combined by a weighted heuristic with a logistic
confidence curve, then fused with a Gemini/Vertex vision pass over sampled frames (or
audio + written editorial rules in `highlight_vertex_audio_only` mode). Guards prevent
common false positives: eco-round wins are re-verified by Gemini before qualifying, HUD
misreads are clamped by `max_round_jump`, and a per-map cap keeps output volume editorial.

## Testing

```powershell
python -m pytest            # or: python -m unittest discover tests -v
```

Tests cover the deterministic core: demo-feature scoring and thresholds, LLM JSON
repair/extraction (markdown fences, unquoted keys, trailing commas, truncated blobs),
config time parsing, timeline slicing, SRT/ASS timestamp formatting, and caption line
wrapping. Network, ffmpeg, and model calls are intentionally out of unit-test scope.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline stages, data flow, failure handling
- [docs/SETUP.md](docs/SETUP.md) — environment setup, credentials, configuration reference
- [docs/TOOLS.md](docs/TOOLS.md) — detailed CLI reference for every tool in the repo
- [SECURITY.md](SECURITY.md) — secret handling policy
- [CHANGELOG.md](CHANGELOG.md) — development history

## License

Proprietary — see [LICENSE](LICENSE). Evaluation for code review permitted; all other rights reserved.
