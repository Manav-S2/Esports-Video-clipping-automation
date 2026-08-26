"""Cross-platform launcher for the live highlight pipeline.

Replaces run_live_pipeline.ps1. Resolves the CA bundle so TLS verification works
on machines without a system trust store configured, then execs the pipeline
with any extra arguments forwarded.

Usage::

    python run_pipeline.py                      # uses live_pipeline_config.json
    python run_pipeline.py --config other.json  # extra args are forwarded
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "live_pipeline_config.json"
PIPELINE_ENTRY = "live_stream_highlight_pipeline.py"


def resolve_ca_bundle() -> str | None:
    """Return a CA bundle path: existing SSL_CERT_FILE, else certifi's bundle."""
    existing = (os.environ.get("SSL_CERT_FILE") or "").strip()
    if existing and Path(existing).is_file():
        return existing
    try:
        import certifi
    except ImportError:
        return None
    where = certifi.where()
    return where if Path(where).is_file() else None


def build_environment(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the child process: unbuffered output plus a CA bundle."""
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONUNBUFFERED"] = "1"
    ca = resolve_ca_bundle()
    if ca:
        env["SSL_CERT_FILE"] = ca
    return env


def build_command(extra_args: list[str], *, python_bin: str | None = None) -> list[str]:
    """Build the child argv, injecting --config only when the caller omitted it."""
    cmd = [python_bin or sys.executable, str(REPO_ROOT / PIPELINE_ENTRY)]
    if "--config" not in extra_args:
        cmd += ["--config", str(REPO_ROOT / DEFAULT_CONFIG)]
    cmd += extra_args
    return cmd


def main(argv: list[str] | None = None) -> int:
    extra_args = list(sys.argv[1:] if argv is None else argv)
    cmd = build_command(extra_args)
    return subprocess.run(cmd, env=build_environment()).returncode


if __name__ == "__main__":
    sys.exit(main())
