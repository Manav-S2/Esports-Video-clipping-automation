"""Focused modules extracted from the live highlight orchestrator.

``live_stream_highlight_pipeline`` grew to hold CLI parsing, subprocess
orchestration, network calls and scoring in one file. Cohesive pieces are being
moved here one at a time, each with its own tests, while the orchestrator keeps
importing them so behaviour is unchanged.
"""

from __future__ import annotations
