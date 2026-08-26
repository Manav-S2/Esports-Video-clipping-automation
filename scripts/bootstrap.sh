#!/usr/bin/env bash
# One-command setup for the highlight pipeline on Linux/macOS.
#
# Creates an isolated virtual environment from the committed uv.lock (or
# requirements-lock.txt as a fallback), runs the test suite, and verifies that
# ffmpeg and streamlink are reachable. Run from anywhere:
#
#     ./scripts/bootstrap.sh
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "==> Setting up environment in $repo_root"

if command -v uv >/dev/null 2>&1; then
    echo "==> uv found; syncing locked environment (uv.lock)"
    uv sync --frozen
else
    echo "==> uv not found; falling back to venv + requirements-lock.txt"
    [ -d .venv ] || python3 -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
    ./.venv/bin/python -m pip install -r requirements-lock.txt
fi

python_bin="./.venv/bin/python"

echo "==> Verifying the test suite runs"
"$python_bin" -m pytest -q

echo "==> Checking external media tools"
for tool in ffmpeg streamlink; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "    OK      $tool: $(command -v "$tool")"
    else
        echo "    MISSING $tool (needed for live capture; see docs/SETUP.md)"
    fi
done

cat <<'NEXT'

Setup complete. Next steps:
  1. cp .env.example .env                                     # fill in API keys
  2. cp live_pipeline_config.example.json live_pipeline_config.json
  3. ./.venv/bin/python live_stream_highlight_pipeline.py --config ./live_pipeline_config.json
NEXT
