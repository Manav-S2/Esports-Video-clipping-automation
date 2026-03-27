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
-Crf 18                                               # quality (lower = better quality)
-Preset slow                                          # speed vs compression
-JpegQuality 3                                        # frame JPG quality (2 best, 31 lowest)
-UseNumpyPandasOptimization                           # enabled by default
-PythonExe "C:\Python313\python.exe"               # optional explicit Python path
-SharpenAmount 1.4                                    # NumPy unsharp detail boost
-SaturationBoost 1.07                                 # color boost, no grayscale conversion
```

Default behavior:

- Extracts at exactly 1 FPS (`fps=1`), so each second of source video contributes one image.
- Optimizes extracted frames with NumPy+pandas and writes per-frame metrics to CSV.
- Adds sharpening in both NumPy stage and FFmpeg stage while preserving color.
- Upscales optimized frames to 7680x4320 with Lanczos.
- Rebuilds output as a 1 FPS video from the 8K frames.
