# OCR-Max Video Optimization

This workspace includes a high-quality OCR enhancement pipeline for videos.

## What it does

The script applies an OCR-focused filter chain:

- Preserves color while enhancing luma contrast for text focus
- Denoises to reduce OCR confusion from noise
- Increases local contrast and readability
- Sharpens text edges
- Upscales with Lanczos for better character definition
- Encodes with OCR-friendly compressed H.264 by default for much smaller files
- Supports optional lossless FFV1 mode when you need maximum fidelity

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\ocr_max_optimize.ps1 -InputVideo "D:\path\to\your\video.mp4"
```

Optional parameters:

```powershell
-OutputVideo "D:\path\to\output.ocr-max.mp4"   # custom output path
-ScaleFactor 2                                     # 1 to 4
-Crf 18                                            # compressed mode quality (lower is higher quality)
-Preset slow                                       # ultrafast..veryslow (speed vs compression)
-Codec h264                                        # h264 (faster) or h265 (smaller, slower)
-Lossless                                          # switch to FFV1 MKV output (very large files)
-Binarize                                          # hard black/white text mode
-Threshold 150                                     # threshold used with -Binarize
```

## Highest clarity mode (compressed, recommended default)

```powershell
powershell -ExecutionPolicy Bypass -File .\ocr_max_optimize.ps1 -InputVideo "D:\path\to\your\video.mp4" -ScaleFactor 2 -Crf 18 -Preset slow -Binarize -Threshold 155
```

If text strokes break, reduce threshold (for example `-Threshold 135`).
If background bleed remains, raise threshold (for example `-Threshold 170`).

## True lossless archival mode (largest files)

```powershell
powershell -ExecutionPolicy Bypass -File .\ocr_max_optimize.ps1 -InputVideo "D:\path\to\your\video.mp4" -ScaleFactor 2 -Lossless -Binarize -Threshold 155
```

Use this mode only when you need mathematically lossless output for repeated processing.

## Smaller compressed files with H.265

```powershell
powershell -ExecutionPolicy Bypass -File .\ocr_max_optimize.ps1 -InputVideo "D:\path\to\your\video.mp4" -ScaleFactor 2 -Codec h265 -Crf 20 -Preset slow -Binarize -Threshold 155
```

H.265 is usually slower to encode than H.264, but often produces smaller files at similar visual quality.

## Extract every second, upscale to 8K, and rebuild video

Use this script when you want one image per second from a clip, run NumPy+pandas color optimization and sharpening (no grayscale), then upscale each image to 8K, then combine all images back into one video.

Current default profile is tuned for natural-looking output (clean profile):

- NumPy+pandas stage is disabled by default
- JPG extraction quality defaults to 2 (higher quality)
- Final encode defaults to H.264 CRF 14, preset slow

Install Python dependencies once:

```powershell
pip install numpy pandas pillow
```

```powershell
powershell -ExecutionPolicy Bypass -File .\rebuild_every_second_8k.ps1 -InputVideo "E:\27 (1).mp4"
```

Optional parameters:

```powershell
-WorkingDir "D:\temp\every_second_work"            # where extracted/upscaled frames are stored
-OutputVideo "D:\output\clip.every-second.8k.mp4"  # final combined video path
-Codec h264|h265                                     # encoder for final 8K video
-Crf 14                                               # quality (lower = better quality)
-Preset slow                                          # speed vs compression
-JpegQuality 2                                        # frame JPG quality (2 best, 31 lowest)
-UseNumpyPandasOptimization                           # opt in to NumPy+pandas enhancement stage
-SkipNumpyPandasOptimization                          # force-disable NumPy stage when opt-in is set
-PythonExe "C:\Python313\python.exe"               # optional explicit Python path
-SharpenAmount 1.4                                    # NumPy unsharp detail boost
-SaturationBoost 1.07                                 # color boost, no grayscale conversion
```

Default behavior:

- Extracts at exactly 1 FPS (`fps=1`), so each second of source video contributes one image.
  -Skips NumPy+pandas stage unless `-UseNumpyPandasOptimization` is explicitly passed.
  -Adds FFmpeg upscaling and sharpening while preserving color.
- Upscales optimized frames to 7680x4320 with Lanczos.
- Rebuilds output as a 1 FPS video from the 8K frames.

## Killfeed ROI extraction (clear OCR snapshots)

Use this script when you want cleaner killfeed detection and snapshots with a folder layout similar to your existing runs.

```powershell
powershell -ExecutionPolicy Bypass -File .\extract_killfeed_snapshots.ps1 -InputVideo "D:\path\to\your\video.mp4"
```

Optional parameters:

```powershell
-OutputDir "D:\path\killfeed_snapshots_YYYY-MM-DD_HH-mm-ss"  # run folder root
-SampleFps 8                                                     # OCR scan sampling rate
-MinTextLength 8                                                 # lower for short names
-Threshold 155                                                   # binarization threshold
-RoiX 0.50 -RoiY 0.00 -RoiW 0.50 -RoiH 0.50                    # top-right quarter crop ratios
```

The run folder is recreated with this structure each run:

```text
killfeed_snapshots_YYYY-MM-DD_HH-mm-ss/
	frames_raw/           # full-frame PNG samples
	frames_killfeed_roi/  # cropped killfeed ROI PNGs
	frames_ocr_ready/     # thresholded OCR-ready ROI PNGs
	snapshots/            # detected events: *_full.png and *_roi.png
	meta/
		ocr_scan.csv
		killfeed_events.csv
		killfeed_extracted_summary.txt
```

## CS2 highlight detection (demo + Gemini)

Use this script to classify whether a clip is a highlight candidate by combining:

- Demo-derived match signals from `.dem`, `.rar`, or a directory of `.dem` files
- Gemini vision analysis from sampled video frames

Script:

```text
detect_cs2_highlight.py
```

Recommended setup in a standard Python venv:

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\activate
pip install -r requirements.txt
```

Run with both demo folder and clip video:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
python .\detect_cs2_highlight.py `
  --demo-input "C:\path\to\demo_folder" `
  --video-path "C:\path\to\clip.mp4" `
  --output-json "C:\path\to\result.json"
```

Useful options:

```text
--gemini-model gemini-2.0-flash
--frame-sample-seconds 2.0
--max-frames 10
--demo-weight 0.55
--vision-weight 0.45
```

## Live CS2 clip bot

Use this pipeline when you want to monitor a live stream, record each detected round, classify highlights with Gemini, export a 1080x1920 portrait Reel with blurred top/bottom background, optionally run a caption command, generate title/SEO text, and optionally post with Instagram.

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
python .\live_stream_highlight_pipeline.py --config .\live_pipeline_config.example.json
```

Important config fields:

- `stream_url`: direct media URL, Twitch URL, or YouTube URL. Twitch/YouTube page URLs require `streamlink`.
- `screenshot_interval_sec`: screenshot cadence for round monitoring. Use `2` for the intended live loop.
- `round_detection_min_confidence`: Gemini confidence threshold before accepting a detected round.
- `max_round_jump`: guards against bad HUD reads that jump too many rounds ahead.
- `highlight_vertex_audio_only`: when `true`, post-round Vertex scoring uses **clip audio + `rules_docx` text only** (semantic match to your Word rules; no JPEG snapshots or contact sheet). When `false`, Vertex gets the default **nine thumbnails + optional audio**.
- `caption_cmd_template`: optional command hook. It receives `{input}` and `{output}` placeholders and should create the captioned output file.
- `instagram_enabled`: set to `true` only after local dry-runs look good. Credentials can come from config or `INSTA_USER` / `INSTA_PASS`.

Output folders are created under `output_root`:

```text
screens/        # temporary monitoring screenshots
round_raw/      # 1920x1080 per-round recordings
round_edited/   # portrait blur edits
round_final/    # final captioned or ready-to-post videos
meta/           # detection logs, result JSON, caption hook logs, state
```

Start with `instagram_enabled: false` and an empty `caption_cmd_template`. After the raw and final videos look correct, test a caption hook that copies `{input}` to `{output}`, then swap in Riverside/local caption automation.
