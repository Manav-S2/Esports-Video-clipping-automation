"""Schema validation for ``live_pipeline_config.json``.

The pipeline previously read its config with bare ``cfg.get(...)`` calls, so a
typo or a wrong type surfaced hours into a live run — or silently changed
behaviour. ``validate_config`` checks types, ranges and enums up front and
raises :class:`errors.PipelineConfigError` describing every problem at once.

Validation is intentionally non-destructive: unknown keys are reported as
warnings rather than errors, so a config written for a newer build still runs.

Usage::

    from config_schema import load_config
    cfg = load_config(Path("live_pipeline_config.json"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from errors import PipelineConfigError

# name -> (python type, minimum, maximum); None bound means unbounded.
_NUMERIC_FIELDS: dict[str, tuple[type, float | None, float | None]] = {
    "screenshot_interval_sec": (float, 0.1, 3600.0),
    "min_round_record_sec": (float, 0.0, 7200.0),
    "max_round_record_sec": (float, 1.0, 21600.0),
    "clip_start_offset_sec": (float, -600.0, 600.0),
    "round_detection_min_confidence": (float, 0.0, 1.0),
    "max_round_jump": (int, 1, 30),
    "stable_round_reads_to_start": (int, 1, 20),
    "round_transition_confirmations": (int, 1, 20),
    "highlight_parallel_workers": (int, 1, 32),
    "screenshot_4k_width": (int, 320, 7680),
    "screenshot_4k_height": (int, 240, 4320),
    "portrait_blur_crf": (int, 0, 51),
    "portrait_blur_width": (int, 240, 4320),
    "portrait_blur_height": (int, 240, 7680),
    "karaoke_ffmpeg_crf": (int, 0, 51),
    "caption_hook_timeout_sec": (float, 1.0, 86400.0),
    "speech_recognition_timeout_sec": (float, 1.0, 86400.0),
    "streamlink_resolve_timeout_sec": (float, 1.0, 3600.0),
    "numpy_contrast_clip_percent": (float, 0.0, 10.0),
    "numpy_saturation_boost": (float, 0.0, 5.0),
    "numpy_unsharp_radius": (float, 0.0, 10.0),
    "numpy_unsharp_amount": (float, 0.0, 10.0),
    "karaoke_vertex_inline_video_max_mb": (float, 1.0, 2048.0),
    "karaoke_margin_top_ratio": (float, 0.0, 1.0),
    "karaoke_overlay_width_frac": (float, 0.0, 1.0),
    "karaoke_overlay_margin_bottom_px": (int, 0, 4320),
}

# Region-of-interest ratios are fractions of the frame.
_ROI_FIELDS = ("round_roi_x", "round_roi_y", "round_roi_w", "round_roi_h")

_BOOL_FIELDS = (
    "highlight_yield_to_hud_vision",
    "highlight_vertex_audio_only",
    "round_started_required",
    "require_consecutive_round_increments",
    "process_partial_on_max_duration",
    "record_suspend_while_hud_idle",
    "karaoke_use_adc",
    "karaoke_no_overlay",
    "karaoke_vertex_send_full_video",
    "karaoke_vertex_audio_only",
    "karaoke_disable_av_mux_timing_fix",
    "karaoke_async",
    "instagram_enabled",
)

_STRING_FIELDS = (
    "stream_url",
    "api_provider",
    "highlight_api_provider",
    "aws_rekognition_region",
    "gemini_api_key",
    "gemini_model",
    "nvidia_api_key",
    "nvidia_base_url",
    "nvidia_model",
    "vertex_project_id",
    "vertex_location",
    "demo_file",
    "rules_docx",
    "output_root",
    "karaoke_vertex_roster_path",
    "portrait_blur_preset",
    "caption_cmd_template",
    "caption_provider",
    "speech_language_code",
    "karaoke_overlay_image",
    "karaoke_ffmpeg_preset",
    "instagram_username",
    "instagram_password",
)

_LIST_FIELDS = (
    "vertex_api_keys",
    "streamlink_extra_args",
    "streamlink_twitch_extra_args",
)

_ENUM_FIELDS: dict[str, tuple[str, ...]] = {
    "api_provider": ("rekognition", "gemini", "nvidia", "vertex"),
    "highlight_api_provider": ("vertex", "gemini", "nvidia"),
    # Documented on PipelineConfig.caption_provider in live_stream_highlight_pipeline.
    "caption_provider": (
        "auto",
        "none",
        "google_speech",
        "shell",
        "karaoke_google",
        "karaoke_vertex",
        "karaoke_whisper",
    ),
    "portrait_blur_preset": (
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    ),
    "karaoke_ffmpeg_preset": (
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    ),
}

REQUIRED_FIELDS = ("stream_url",)

_KNOWN_FIELDS = (
    set(_NUMERIC_FIELDS)
    | set(_ROI_FIELDS)
    | set(_BOOL_FIELDS)
    | set(_STRING_FIELDS)
    | set(_LIST_FIELDS)
    | {"stream_input_seek_hms", "stream_input_seek_sec", "karaoke_caption_time_offset_sec"}
)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating a config mapping."""

    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_numeric(cfg: dict[str, Any], errors: list[str]) -> None:
    for name, (expected, low, high) in _NUMERIC_FIELDS.items():
        if name not in cfg:
            continue
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"{name}: expected {expected.__name__}, got {type(value).__name__}")
            continue
        if expected is int and not float(value).is_integer():
            errors.append(f"{name}: expected a whole number, got {value}")
            continue
        if low is not None and value < low:
            errors.append(f"{name}: {value} is below the minimum {low}")
        if high is not None and value > high:
            errors.append(f"{name}: {value} is above the maximum {high}")


def _check_roi(cfg: dict[str, Any], errors: list[str]) -> None:
    for name in _ROI_FIELDS:
        if name not in cfg:
            continue
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"{name}: expected a number between 0 and 1, got {type(value).__name__}")
            continue
        if not 0.0 <= float(value) <= 1.0:
            errors.append(f"{name}: {value} is not a frame fraction between 0 and 1")

    x, w = cfg.get("round_roi_x"), cfg.get("round_roi_w")
    if isinstance(x, int | float) and isinstance(w, int | float) and float(x) + float(w) > 1.0:
        errors.append(f"round_roi_x + round_roi_w = {float(x) + float(w):.3f} exceeds the frame width")
    y, h = cfg.get("round_roi_y"), cfg.get("round_roi_h")
    if isinstance(y, int | float) and isinstance(h, int | float) and float(y) + float(h) > 1.0:
        errors.append(f"round_roi_y + round_roi_h = {float(y) + float(h):.3f} exceeds the frame height")


def _check_simple_types(cfg: dict[str, Any], errors: list[str]) -> None:
    for name in _BOOL_FIELDS:
        if name in cfg and not isinstance(cfg[name], bool):
            errors.append(f"{name}: expected true/false, got {type(cfg[name]).__name__}")
    for name in _STRING_FIELDS:
        if name in cfg and not isinstance(cfg[name], str):
            errors.append(f"{name}: expected a string, got {type(cfg[name]).__name__}")
    for name in _LIST_FIELDS:
        if name not in cfg:
            continue
        value = cfg[name]
        if not isinstance(value, list):
            errors.append(f"{name}: expected a list, got {type(value).__name__}")
        elif not all(isinstance(item, str) for item in value):
            errors.append(f"{name}: every entry must be a string")


def _check_enums(cfg: dict[str, Any], errors: list[str]) -> None:
    for name, allowed in _ENUM_FIELDS.items():
        value = cfg.get(name)
        if isinstance(value, str) and value and value not in allowed:
            errors.append(f"{name}: {value!r} is not one of {', '.join(allowed)}")


def _check_consistency(cfg: dict[str, Any], errors: list[str]) -> None:
    lo, hi = cfg.get("min_round_record_sec"), cfg.get("max_round_record_sec")
    if isinstance(lo, int | float) and isinstance(hi, int | float) and float(lo) > float(hi):
        errors.append(
            f"min_round_record_sec ({lo}) is greater than max_round_record_sec ({hi})"
        )

    if cfg.get("instagram_enabled") is True:
        has_user = bool(str(cfg.get("instagram_username", "") or "").strip())
        has_pass = bool(str(cfg.get("instagram_password", "") or "").strip())
        if not (has_user and has_pass):
            errors.append(
                "instagram_enabled is true but instagram_username/instagram_password are empty "
                "(set them in config or via INSTA_USER / INSTA_PASS)"
            )


def validate_config(cfg: Any) -> ValidationReport:
    """Validate a parsed config mapping; returns errors and warnings."""
    if not isinstance(cfg, dict):
        return ValidationReport(
            errors=[f"config root must be a JSON object, got {type(cfg).__name__}"],
            warnings=[],
        )

    errors: list[str] = []
    for name in REQUIRED_FIELDS:
        if not str(cfg.get(name, "") or "").strip():
            errors.append(f"{name} is required and must be non-empty")

    _check_numeric(cfg, errors)
    _check_roi(cfg, errors)
    _check_simple_types(cfg, errors)
    _check_enums(cfg, errors)
    _check_consistency(cfg, errors)

    warnings = [f"unknown config key: {key}" for key in sorted(set(cfg) - _KNOWN_FIELDS)]
    return ValidationReport(errors=errors, warnings=warnings)


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate a pipeline config file.

    Raises:
        PipelineConfigError: if the file is missing, unparsable, or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise PipelineConfigError(f"config file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineConfigError(f"config file is not valid JSON ({path}): {exc}") from exc
    except OSError as exc:
        raise PipelineConfigError(f"could not read config file ({path}): {exc}") from exc

    report = validate_config(raw)
    if not report.ok:
        bullets = "\n".join(f"  - {problem}" for problem in report.errors)
        raise PipelineConfigError(f"invalid config ({path}):\n{bullets}")
    return raw
