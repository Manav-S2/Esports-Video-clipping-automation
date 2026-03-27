import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


def compute_metrics(arr: np.ndarray) -> tuple[float, float, float]:
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    brightness = float(np.mean(luminance))
    contrast = float(np.std(luminance))

    gx = np.diff(luminance, axis=1)
    gy = np.diff(luminance, axis=0)
    sharpness = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    return brightness, contrast, sharpness


def clip_percentile_stretch(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    out = arr.copy()
    for c in range(3):
        channel = out[:, :, c]
        low = np.percentile(channel, low_pct)
        high = np.percentile(channel, high_pct)
        if high - low < 1e-5:
            continue
        channel = (channel - low) * (255.0 / (high - low))
        out[:, :, c] = np.clip(channel, 0, 255)
    return out


def adjust_saturation(arr: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount - 1.0) < 1e-6:
        return arr

    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    luminance = luminance[:, :, None]
    adjusted = luminance + amount * (arr - luminance)
    return np.clip(adjusted, 0, 255)


def process_frame(input_path: Path, output_path: Path, sharpen_amount: float, saturation: float) -> dict:
    image = Image.open(input_path).convert("RGB")
    original = np.asarray(image, dtype=np.float32)

    contrast_stretched = clip_percentile_stretch(original, low_pct=1.0, high_pct=99.0)

    base_img = Image.fromarray(contrast_stretched.astype(np.uint8), mode="RGB")
    denoised_img = base_img.filter(ImageFilter.MedianFilter(size=3))
    blurred_img = denoised_img.filter(ImageFilter.GaussianBlur(radius=1.0))

    denoised = np.asarray(denoised_img, dtype=np.float32)
    blurred = np.asarray(blurred_img, dtype=np.float32)
    detail = denoised - blurred
    sharpened = denoised + sharpen_amount * detail

    colored = adjust_saturation(sharpened, saturation)
    final = np.clip(colored, 0, 255).astype(np.uint8)

    brightness, contrast, sharpness = compute_metrics(final.astype(np.float32))

    Image.fromarray(final, mode="RGB").save(output_path, format="JPEG", quality=95, subsampling=0)

    return {
        "frame": input_path.name,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def process_one(args: tuple[Path, Path, float, float]) -> dict:
    input_path, output_path, sharpen_amount, saturation = args
    return process_frame(input_path, output_path, sharpen_amount, saturation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR-oriented color optimization using NumPy + pandas (no grayscale)."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with input JPG frames")
    parser.add_argument("--output-dir", required=True, help="Directory for optimized JPG frames")
    parser.add_argument(
        "--metrics-csv",
        required=True,
        help="Path to CSV report containing per-frame quality metrics",
    )
    parser.add_argument(
        "--sharpen-amount",
        type=float,
        default=1.4,
        help="Unsharp detail amplification amount",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.07,
        help="Color saturation multiplier",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip frames that already exist in output-dir",
    )
    parser.add_argument(
        "--errors-csv",
        default="",
        help="Optional CSV path to record per-frame processing errors",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metrics_csv = Path(args.metrics_csv)
    errors_csv = Path(args.errors_csv) if args.errors_csv else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input frames folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(input_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(input_dir.glob("frame_*.png"))
    if not frame_files:
        raise RuntimeError("No input frames found.")

    jobs: list[tuple[Path, Path, float, float]] = []
    for frame_file in frame_files:
        out_file = output_dir / frame_file.with_suffix(".jpg").name
        if args.resume and out_file.exists():
            continue
        jobs.append((frame_file, out_file, args.sharpen_amount, args.saturation))

    rows = []
    errors = []
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    errors.append({"error": str(exc)})
                if i % 100 == 0:
                    print(f"Processed {i}/{len(jobs)}")

    if args.resume:
        existing_outputs = sorted(output_dir.glob("frame_*.jpg"))
        missing = {p.name for p in existing_outputs}
        for frame_file in frame_files:
            out_name = frame_file.with_suffix(".jpg").name
            if out_name in missing and not any(r["frame"] == out_name for r in rows):
                image = Image.open(output_dir / out_name).convert("RGB")
                arr = np.asarray(image, dtype=np.float32)
                brightness, contrast, sharpness = compute_metrics(arr)
                rows.append(
                    {
                        "frame": out_name,
                        "brightness": brightness,
                        "contrast": contrast,
                        "sharpness": sharpness,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(metrics_csv, index=False)

    if errors_csv:
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors).to_csv(errors_csv, index=False)

    print(f"Optimized frames : {len(sorted(output_dir.glob('frame_*.jpg')))}")
    print(f"Output folder    : {output_dir}")
    print(f"Metrics CSV      : {metrics_csv}")
    print(
        "Sharpness mean   : "
        f"{df['sharpness'].mean():.3f} | "
        f"Contrast mean: {df['contrast'].mean():.3f}"
    )
    print(f"Frame errors     : {len(errors)}")


if __name__ == "__main__":
    main()