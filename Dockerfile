# syntax=docker/dockerfile:1
# Esports video clipping / live highlight pipeline (Linux).
# Requires at runtime: live_pipeline_config.json, CS2_Highlights.docx, and API credentials (env or mounted JSON).
#
# BuildKit: if builds fail on COPY --link etc., use Docker 23+ and run:
#   set DOCKER_BUILDKIT=1   (PowerShell / CMD before docker build)
#
# If ``apt-get`` fails with ``Failed to fetch http://deb.debian.org/...`` or
# ``Temporary failure resolving 'deb.debian.org'``, the **build VM** cannot reach Debian mirrors.
# Fix on the host: stable internet, disable VPN briefly, retry; Docker Desktop → Settings → reset / update;
# corporate firewall may block apt — build on another network or use a Debian HTTP(S) proxy (build-time only).
#
# Build:  docker build -t esports-video-clipping:latest .
# Run:    docker compose run --rm pipeline
# Shell:  docker compose run --rm pipeline bash

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=600

RUN set -eux; \
    printf '%s\n' \
        'Acquire::Retries "8";' \
        'Acquire::http::Timeout "120";' \
        'Acquire::https::Timeout "120";' \
        'Acquire::ForceIPv4 "true";' \
        > /etc/apt/apt.conf.d/99docker-retry; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# ``IncompleteRead`` / ``ProtocolError`` mid-download = flaky VPN/Wi‑Fi/firewall or CDN reset — retry ``docker compose build``.
RUN set -eu; \
    for n in 1 2 3 4 5 6; do \
        if pip install --upgrade pip setuptools wheel \
            && pip install --retries 25 -r requirements.txt; then \
            exit 0; \
        fi; \
        echo "PyPI deps failed (attempt $n), retrying in $((n * 10))s..." >&2; \
        sleep $((n * 10)); \
    done; \
    exit 1

COPY . .

ENTRYPOINT ["python", "-u"]
CMD ["live_stream_highlight_pipeline.py", "--help"]
