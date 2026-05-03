import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import shutil

# Make sure you have these installed:
# pip install google-generativeai streamlink PyPDF2 Pillow pandas openpyxl
# You also need ffmpeg and streamlink installed on your system PATH.

# Only import google.genai if it's installed, to avoid breaking if not needed
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Warning: google-generativeai not installed. Round detection will not work.")
    genai = None
    genai_types = None


class StreamRoundRecorder:
    def __init__(self, stream_url: str, gemini_api_key: str, gemini_model: str, output_root: Path):
        self.stream_url = stream_url
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.output_root = output_root

        self.screens_dir = self.output_root / "screenshots"
        self.raw_clips_dir = self.output_root / "raw_clips"

        self.screens_dir.mkdir(parents=True, exist_ok=True)
        self.raw_clips_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        if genai and self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)

        self.current_round: int | None = None
        self.recording_process: subprocess.Popen | None = None
        self.current_recording_path: Path | None = None
        self._resolved_stream_url: str | None = None

    def _resolve_stream_input(self) -> str:
        """Resolves page URLs (e.g., Twitch, YouTube) to a direct playable stream URL using streamlink."""
        if self._resolved_stream_url:
            return self._resolved_stream_url

        src = self.stream_url.strip()
        lower = src.lower()

        if "twitch.tv/" in lower or "youtube.com/" in lower or "youtu.be/" in lower:
            streamlink_path = shutil.which("streamlink")
            if not streamlink_path:
                raise RuntimeError(
                    "streamlink is required for Twitch/YouTube URLs. Install it with: pip install streamlink"
                    " and ensure it's in your system PATH."
                )
            print(f"Resolving stream URL for {src}...")
            proc = subprocess.run(
                [streamlink_path, "--stream-url", src, "best"],
                capture_output=True,
                text=True,
                check=True,
            )
            resolved = (proc.stdout or "").strip().splitlines()[-1].strip()
            if not resolved:
                raise RuntimeError("streamlink returned an empty stream URL.")
            self._resolved_stream_url = resolved
            return resolved

        self._resolved_stream_url = src
        return src

    def capture_screenshot(self) -> Path | None:
        """Captures a screenshot from the live stream using ffmpeg."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.screens_dir / f"screen_{timestamp}.jpg"

        try:
            stream_input = self._resolve_stream_input()
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError("ffmpeg not found in system PATH.")

            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                stream_input,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(screenshot_path),
            ]
            print(f"Capturing screenshot: {screenshot_path.name}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            if screenshot_path.exists():
                return screenshot_path
            print(f"Error: Screenshot file not created at {screenshot_path}")
            return None
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg screenshot failed: {e.stderr}")
            return None
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None

    def detect_round_from_screenshot(self, image_path: Path) -> int | None:
        """Uses Gemini to detect the current round number from a screenshot."""
        if not self.client:
            print("Gemini client not initialized. Cannot detect rounds.")
            return None

        prompt = (
            "Detect the current round number from this CS2 gameplay screenshot. "
            "Return only the round number as an integer, or 'null' if not found."
        )

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            contents = [
                genai_types.Part.from_text(prompt),
                genai_types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
            ]

            response = self.client.models.generate_content(model=self.gemini_model, contents=contents)
            response_text = response.text.strip()

            if response_text.isdigit():
                return int(response_text)
            try:
                parsed = json.loads(response_text.replace("'", '"'))
                if parsed.get("round") is not None:
                    return int(parsed["round"])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            print(f"Gemini returned non-numeric/unparsable response: {response_text}")
            return None
        except Exception as e:
            print(f"Gemini round detection failed: {e}")
            return None
        finally:
            image_path.unlink(missing_ok=True)

    def start_recording_round(self, round_num: int):
        """Starts screen recording the current round in 1080p."""
        if self.recording_process:
            print("Warning: Recording already in progress. Stopping previous recording.")
            self.stop_recording()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording_path = self.raw_clips_dir / f"round_{round_num:02d}_{timestamp}.mp4"

        try:
            stream_input = self._resolved_stream_url
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError("ffmpeg not found in system PATH.")

            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                stream_input,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-vf",
                "scale=1920:1080",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(self.current_recording_path),
            ]

            print(f"Starting recording for Round {round_num}: {self.current_recording_path.name}")
            self.recording_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.current_round = round_num
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.current_recording_path = None
            self.current_round = None

    def stop_recording(self):
        """Stops the current recording process."""
        if self.recording_process:
            print(f"Stopping recording for Round {self.current_round}...")
            self.recording_process.communicate(input=b"q")
            self.recording_process.wait()
            self.recording_process = None
            self.current_round = None
            print("Recording stopped.")
        self.current_recording_path = None

    def run_monitor(self):
        """Main loop to monitor the stream, detect rounds, and record clips."""
        print(f"Starting live stream monitor for: {self.stream_url}")

        try:
            self._resolve_stream_input()
            print(f"Resolved stream input: {self._resolved_stream_url}")

            while True:
                screenshot = self.capture_screenshot()
                if screenshot:
                    detected_round = self.detect_round_from_screenshot(screenshot)

                    if detected_round is not None:
                        if self.current_round is None:
                            self.start_recording_round(detected_round)
                        elif detected_round > self.current_round:
                            print(f"Detected new round: {detected_round}. Previous was {self.current_round}.")
                            self.stop_recording()
                            self.start_recording_round(detected_round)
                        else:
                            if self.current_round != detected_round:
                                print(
                                    f"Warning: Detected round {detected_round}, but expected {self.current_round} "
                                    "or higher. Still recording current round."
                                )
                    else:
                        print("Could not detect round number from screenshot. Continuing to monitor.")

                time.sleep(2)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user (Ctrl+C).")
        except Exception as e:
            print(f"An error occurred in the monitoring loop: {e}")
        finally:
            self.stop_recording()
            print("Stream recorder gracefully shut down.")


def main():
    parser = argparse.ArgumentParser(description="Live stream CS2 round recorder.")
    parser.add_argument(
        "--stream-url",
        required=True,
        help="URL of the live stream (e.g., Twitch, YouTube, or direct media URL).",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Your Gemini API key (or set GEMINI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-pro-vision",
        help="Gemini Vision model to use for round detection.",
    )
    parser.add_argument(
        "--output-dir",
        default="stream_recordings",
        help="Directory to save screenshots and recorded clips.",
    )

    args = parser.parse_args()

    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key and genai:
        raise ValueError(
            "Gemini API key is required for round detection. Provide with --gemini-api-key or set GEMINI_API_KEY."
        )
    elif not genai:
        print("Warning: google-generativeai not installed. Round detection will not work.")
        gemini_api_key = ""

    output_path = Path(args.output_dir)

    recorder = StreamRoundRecorder(
        stream_url=args.stream_url,
        gemini_api_key=gemini_api_key or "",
        gemini_model=args.gemini_model,
        output_root=output_path,
    )
    recorder.run_monitor()


if __name__ == "__main__":
    main()
