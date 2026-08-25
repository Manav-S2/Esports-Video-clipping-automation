# Setup

## Requirements

* Python 3.12+ (3.14 works; `py -3.14` launcher paths appear in some examples)
* `ffmpeg` on PATH (recording, editing, caption burn-in)
* `streamlink` on PATH for Twitch/YouTube page URLs (a portable Windows build is supported —
  see `run_live_pipeline.ps1`)
* Windows 10/11 or Docker (Linux image provided)

## Windows

```powershell
.\setup_windows_env.ps1          # creates venv + installs requirements
# or manually:
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`install_requirements.ps1` / `install_requirements.bat` are thin wrappers for machines
without profile scripts enabled.

## Docker

```bash
docker compose up --build        # see docker-compose.yml for mounts
```

The image bundles ffmpeg and the Python deps; configs and output folders are mounted from
the host so recordings survive container restarts.

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
