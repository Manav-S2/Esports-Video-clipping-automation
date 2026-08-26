# Setup

## Requirements

* Python 3.12+ (3.14 works; `py -3.14` launcher paths appear in some examples)
* `ffmpeg` on PATH (recording, editing, caption burn-in)
* `streamlink` on PATH for Twitch/YouTube page URLs (a portable Windows build under
  `streamlink_portable/` is detected automatically)
* Windows 10/11, Linux, or macOS — the pipeline is standalone (no containers required)

## One-command setup (recommended)

```powershell
.\scripts\bootstrap.ps1          # Windows
```

```bash
./scripts/bootstrap.sh            # Linux / macOS
```

The bootstrap script creates an isolated virtual environment from the committed `uv.lock`
(falling back to `requirements-lock.txt` when `uv` is absent), runs the test suite to prove
the install works, and reports whether `ffmpeg` and `streamlink` are on PATH.

## Manual setup

```powershell
uv sync                           # locked install from uv.lock
```

Or with plain venv + pip:

```powershell
python -m venv .venv
.\.venv\Scripts\activate         # Linux/macOS: source .venv/bin/activate
pip install -r requirements-lock.txt
```

Both paths create `.venv/` in the repository root; no global installs are performed.

## External tools

`ffmpeg` and `streamlink` are invoked as subprocesses and must be on PATH — they are not
vendored. On Windows a portable streamlink build under `streamlink_portable/` is detected
automatically; on Linux/macOS install them with your package manager or
`pip install streamlink`.

Launch the pipeline with `python run_pipeline.py` — it resolves a CA bundle for TLS and
forwards any extra arguments to the pipeline.

## Credentials

All secrets live in git-ignored local files or environment variables — see
[SECURITY.md](../SECURITY.md).

| Secret | Where it goes |
| --- | --- |
| Gemini API key | `GEMINI_API_KEY` env var or `gemini_api_key` in `live_pipeline_config.json` |
| Google Cloud Speech | `speech_api_key.local.json` (copy from `speech_api_key.local.example.json`) or ADC via `GOOGLE_APPLICATION_CREDENTIALS` |
| AWS (optional S3 upload) | `aws_credentials.local.json` (copy from `aws_credentials.local.example.json`) |
| Instagram (optional) | `INSTA_USER` / `INSTA_PASS` env vars or config fields |

## Configuration

Copy `live_pipeline_config.example.json` to `live_pipeline_config.json`. Key fields:

| Field | Meaning |
| --- | --- |
| `stream_url` | Direct media URL, Twitch URL, or YouTube URL |
| `screenshot_interval_sec` | Monitoring cadence (2 recommended live) |
| `round_detection_min_confidence` | Gemini confidence needed to accept a round read |
| `max_round_jump` | Rejects HUD misreads that jump too many rounds ahead |
| `highlight_vertex_audio_only` | `true` = score with clip audio + rules docx only |
| `caption_cmd_template` | Optional external caption hook (`{input}` / `{output}`) |
| `instagram_enabled` | Keep `false` until dry-runs look good |
| `output_root` | Where `screens/ round_raw/ round_edited/ round_final/ meta/` are created |

Times accept numbers, numeric strings, or clock strings (`"1:23:45"`, `"12:30"`).

## Recommended first run

1. `instagram_enabled: false`, empty `caption_cmd_template`.
2. Verify `round_raw/` and `round_edited/` outputs look correct.
3. Test a caption hook that copies `{input}` to `{output}`.
4. Enable real captions, then Instagram.
