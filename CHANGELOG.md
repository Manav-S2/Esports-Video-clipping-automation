# Changelog

Development history of the project (newest first). See `git log` for full detail.

## 2026-08 — Optional container sandbox
- Reinstated `Dockerfile`/`docker-compose.yml` as *optional* application packaging:
  the image bundles ffmpeg and streamlink and defaults to running the test suite,
  so the project can be exercised in isolation without host tooling.
- CI gained a `docker-build` job that builds the image and runs the suite inside it.

## 2026-08 — Cross-platform tooling
- Ported all PowerShell utilities to tested Python CLIs (`media_tools.py`,
  `run_pipeline.py`); removed the obsolete Windows installer scripts.
- `boto3` moved to an optional `aws` extra.
- **Audit policy:** the `pip-audit` CI job is now blocking — a known vulnerability
  in a pinned dependency fails the build rather than being reported and ignored.

## 2026-08 — Standalone pipeline (Docker removed)
- Removed `Dockerfile`, `docker-compose.yml` and `.dockerignore`; the pipeline now runs
  directly from a virtual environment on Windows, Linux, or macOS.
- Added `scripts/bootstrap.ps1` / `scripts/bootstrap.sh` for one-command setup
  (locked install, test-suite verification, ffmpeg/streamlink PATH check).
- Docs and in-source hints reworded away from container assumptions.

## 2026-08 — Quality tooling
- uv lockfile with a CI job proving fresh-clone install; ruff (I/UP/W/B) and mypy gates;
  typed error hierarchy with loguru logging; test suite grown to 111 tests.

## 2026-08 — Review packaging
- Added LICENSE, structured documentation (`docs/`), SECURITY and CONTRIBUTING notes,
  changelog, and a unit-test suite for the deterministic core logic.
- Restructured README into a project overview; detailed CLI reference moved to
  `docs/TOOLS.md`.

## 2026-05-22 — Caption robustness
- Improved karaoke fallback paths and Google Speech retry robustness.

## 2026-05-14 — Karaoke captions as default
- Preferred Google Cloud Speech karaoke captions in the live pipeline.
- Fixed audio/video mux timing drift for long clips
  (`adjust_speech_words_to_video_timeline`).

## 2026-05-12 — Vertex highlights
- Live pipeline: Vertex-based highlight scoring, karaoke caption paths,
  CAPTIONS/Docker parity.

## 2026-05-06 — Docker packaging
- Added Dockerfile / docker-compose packaging for VM deployment.
- Expanded the live highlight pipeline; updated example config and dependencies.

## 2026-05-03 — Repository hygiene
- Tracked pipeline sources; dropped `.venv` from the repo; broadened `.gitignore`
  for secrets and large assets.

## 2026-03 — 2026-04 — Foundations
- OCR-grade video enhancement toolchain (`ocr_max_optimize.ps1`), killfeed ROI
  extraction, 1 fps / 8K frame experiments (NumPy + pandas optimization).
- CS2 demo-based highlight detection (`detect_cs2_highlight.py`) with Gemini vision
  fusion, eco-round guard, and per-map caps; model-evaluation harnesses
  (`clip5_*`, `ask_*`) against BLAST/NAVI broadcast footage.
- Live stream monitoring, per-round recording, and 9:16 portrait export.
