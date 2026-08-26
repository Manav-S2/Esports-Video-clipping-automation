# Architecture

## Overview

The system is a staged media pipeline. Every stage writes durable artifacts to disk and
records state in `meta/`, so a crash or restart resumes without losing recorded rounds.

```
            +------------------------------------------------------------+
            |                live_stream_highlight_pipeline.py            |
            |                                                            |
 stream --->| 1. MONITOR    screenshot every N sec (ffmpeg/streamlink)    |
            |       |        Gemini vision reads round number from HUD    |
            |       v        (confidence threshold + max_round_jump guard)|
            | 2. RECORD     per-round 1080p capture -> round_raw/         |
            |       v                                                    |
            | 3. SCORE      detect_cs2_highlight fusion:                  |
            |                 - demo .dem features (demoparser2)          |
            |                 - Gemini/Vertex vision on sampled frames    |
            |                 - or audio-RMS + editorial rules (docx)     |
            |                 - eco-guard re-check, per-map cap           |
            |       v                                                    |
            | 4. EDIT       video_editor.apply_portrait_blur -> 9:16     |
            |       v        round_edited/                                |
            | 5. CAPTION    speech_google_captions: STT -> ASS karaoke ->|
            |       v        ffmpeg burn-in -> round_final/              |
            | 6. PUBLISH    title/SEO text gen; optional Instagram post  |
            +------------------------------------------------------------+
```

## Stage detail

### 1. Round monitoring
`ffmpeg` (direct URLs) or `streamlink` (Twitch/YouTube page URLs) grabs a JPEG every
`screenshot_interval_sec`. Gemini reads the scoreboard round number. Two guards reject bad
reads: `round_detection_min_confidence` and `max_round_jump` (a HUD misread that jumps
several rounds ahead is discarded).

### 2. Recording
Each detected round boundary rotates the recorder: the finished round's file is closed and
queued for scoring while the next round records. Recording subprocesses run at reduced OS
priority (`_subprocess_creationflags_low_priority`, `_os_suspend_pid`/`_os_resume_pid`) so
scoring never starves the capture.

### 3. Highlight scoring (`detect_cs2_highlight.py`)
* **Demo path** — demoparser2 loads kill / round_start / round_end / item_purchase events.
  `_compute_features` derives kills_total, kills/minute, headshot ratio, multi-kill burst
  score and unique killers; `_score_demo_highlight` applies a weighted heuristic with a
  logistic confidence curve around a 7.5 threshold. `_score_rounds` +
  `_map_score_from_rounds` rank rounds within a map (mean of top-3 round scores).
* **Vision path** — `_extract_sample_frames` samples frames; Gemini returns structured JSON
  (`_extract_json_block`), fused with the demo score by `--demo-weight` / `--vision-weight`.
* **Guards** — `_infer_eco_loss_rounds` + `_apply_eco_guard_with_gemini` demote wins over
  eco rounds unless Gemini confirms the play is exceptional; `_apply_per_map_highlight_cap`
  bounds output volume; `_apply_attention_gate` requires visual attention cues.
* **Audio-only mode** — `highlight_vertex_audio_only` scores from clip audio RMS spikes
  (`_mono16_wav_rms_timeline`, `_summarize_rms_spikes`), caster-hype keywords
  (`_hype_hits_in_text`) and the editorial rules document, with no image upload.

### 4. Portrait edit (`video_editor.py`)
Single ffmpeg filter graph: blurred scaled copy fills the 1080x1920 canvas, sharp source is
centered. CRF/preset/fps are parameters, not constants.

### 5. Captions (`speech_google_captions.py`)
Three auth/transport paths: REST `speech:recognize` with API key (up to ~58 s), automatic
~52 s ffmpeg chunking for longer clips on the same key, or ADC + `google-cloud-speech` LRO.
Word timestamps follow the *decoded audio* timeline while ffmpeg burns on the *video*
timeline; `adjust_speech_words_to_video_timeline` measures the ffprobe audio-minus-video
`start_time` delta and shifts words so karaoke highlighting stays in sync. Output is an
ASS file (per-word colour switch, ALL-CAPS, thick outline) burned by ffmpeg; SRT export
(`words_to_srt`, `_wrap_lines`) is available for portable subtitles.

### 6. Robustness / LLM output handling
All model responses go through defensive JSON recovery (`_extract_json`,
`_sanitize_llm_json_blob`, `_close_unbalanced_curly`): markdown fences stripped, bare keys
quoted, trailing commas removed, truncated responses salvaged back to the last complete
element. Network calls classify retryable failures (`_is_retryable_urllib_failure`) and
honour Gemini's structured retry delays (`_google_gemini_retry_delay`).

## Design decisions

* **Filesystem as the contract** — every stage's output is a plain file in a well-named
  folder (`round_raw/` → `round_edited/` → `round_final/`, `meta/` for JSON logs/state).
  Stages can be re-run independently and inspected by hand.
* **Stdlib-first HTTP** — Gemini/Vertex/Speech REST calls use `urllib` with explicit SSL
  contexts rather than heavy SDKs in the hot path, keeping the dependency footprint
  small; official SDKs are used only where required (LRO speech).
* **Standalone, no containers** — the pipeline runs directly from a virtual environment
  on Windows (with portable streamlink fallback), Linux, or macOS. External media tools
  (ffmpeg, streamlink) are resolved from PATH rather than baked into an image.
* **Evaluation harnesses kept in-repo** — `clip5_*` and `ask_*` scripts document how the
  prompts and model choices were validated against a known round, which is why the main
  prompts look the way they do.
