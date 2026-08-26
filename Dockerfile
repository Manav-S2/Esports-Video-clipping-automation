# syntax=docker/dockerfile:1
#
# Reproducible sandbox for the CS2 highlight application.
#
# The pipeline shells out to ffmpeg and streamlink; this image supplies both so
# the project runs from a fresh clone without touching the host toolchain.
# Packaging only — this is an application image, not infrastructure-as-code.
#
#   docker build -t cs2-highlight .
#   docker run --rm cs2-highlight                      # runs the test suite
#   docker run --rm -it cs2-highlight bash             # shell
#   docker compose run --rm pipeline                   # live run (see compose file)

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ffmpeg: recording, editing, caption burn-in. libsndfile1: audio analysis.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv drives the locked install so the image matches uv.lock exactly.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first: only re-resolves when the manifests change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# streamlink is a Python dependency, so it lands on PATH with the venv.
# Smoke check: the default command proves the image can run the suite.
CMD ["python", "-m", "pytest", "-q"]
