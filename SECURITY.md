# Security

## Secret handling

No credentials are committed to this repository — only `*.example.json` placeholders.
Real secrets live in git-ignored files or environment variables:

| Secret | Local file (git-ignored) | Env alternative |
| --- | --- | --- |
| Gemini API key | `live_pipeline_config.json` | `GEMINI_API_KEY` |
| Google Cloud Speech key | `speech_api_key.local.json` | `GOOGLE_SPEECH_API_KEY` / ADC (`GOOGLE_APPLICATION_CREDENTIALS`) |
| AWS credentials | `aws_credentials.local.json` | standard AWS env vars / profile |
| Instagram login | `live_pipeline_config.json` | `INSTA_USER` / `INSTA_PASS` |

`.gitignore` excludes every secret file, virtualenv, and generated media artifact, so
credentials never enter git history.

## Network surface

* All Google/NVIDIA API calls go over HTTPS with explicitly constructed SSL contexts
  (`_make_ssl_context`, `_google_https_ssl_context`) using `certifi` roots.
* The pipeline makes outbound requests only (stream ingest, model APIs, optional
  Instagram/S3 upload); it opens no listening ports.
* Instagram posting is opt-in (`instagram_enabled`, default off in the example config).

## Reporting

This is a personal project; report issues to the repository owner via GitHub.
