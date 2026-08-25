import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types as genai_types
from loguru import logger

from video_editor import apply_portrait_blur

# Configure Loguru
logger.add("pipeline.log", rotation="500 MB")

class HighlightPipeline:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

        self.client = genai.Client(api_key=self.config['gemini_api_key'])
        self.stream_url = self.config['stream_url']
        self.output_dir = Path("highlights_output")
        self.output_dir.mkdir(exist_ok=True)

        self.current_round = None
        self.is_recording = False
        self.record_proc = None
        self.current_video_path = None

    def capture_screenshot(self):
        """Captures a screenshot from the live stream."""
        screenshot_path = "temp_screenshot.jpg"
        # Using streamlink to get the direct URL if needed, or ffmpeg directly
        cmd = [
            'ffmpeg', '-y', '-i', self.stream_url,
            '-frames:v', '1', '-q:v', '2', screenshot_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None

    def detect_round(self, screenshot_path):
        """Uses Gemini to detect the current round from a screenshot."""
        prompt = (
            "Detect the current round number from this CS2 gameplay screenshot. "
            "Return only the round number as an integer, or 'null' if not found."
        )

        with open(screenshot_path, "rb") as f:
            image_data = f.read()

        try:
            response = self.client.models.generate_content(
                model=self.config["gemini_model"],
                contents=[
                    prompt,
                    genai_types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                ],
            )
            text = response.text.strip()
            return int(text) if text.isdigit() else None
        except Exception as e:
            logger.error(f"Gemini round detection failed: {e}")
            return None

    def start_recording(self, round_num):
        """Starts screen recording the current round."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_path = self.output_dir / f"round_{round_num}_{timestamp}_raw.mp4"

        # Use ffmpeg to record the stream
        cmd = [
            'ffmpeg', '-y', '-i', self.stream_url,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
            '-c:a', 'aac', '-b:a', '128k',
            str(self.current_video_path)
        ]
        self.record_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.is_recording = True
        self.current_round = round_num
        logger.info(f"Started recording round {round_num}")

    def stop_recording(self):
        """Stops the current recording."""
        if self.record_proc:
            self.record_proc.terminate()
            try:
                self.record_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.record_proc.kill()
            self.is_recording = False
            logger.info(f"Stopped recording round {self.current_round}")

    def classify_highlight(self, video_path):
        """Uses Gemini to classify if the recorded round is a highlight."""
        logger.info(f"Uploading {video_path} to Gemini for classification...")

        # Upload file to Gemini
        video_file = self.client.files.upload(file=str(video_path))

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            logger.error("Video processing failed on Gemini.")
            return False

        prompt = "Analyze this CS2 round. Is it a highlight (e.g., multi-kill, clutch, insane shots)? Respond with a JSON: {'is_highlight': boolean, 'reason': string, 'title': string, 'seo_keywords': [string]}"

        try:
            response = self.client.models.generate_content(
                model=self.config['gemini_model'],
                contents=[video_file, prompt]
            )
            # Simple JSON extraction
            result = json.loads(response.text.replace("```json", "").replace("```", "").strip())
            return result
        except Exception as e:
            logger.error(f"Gemini classification failed: {e}")
            return None

    def process_highlight(self, video_path, analysis):
        """Edits the video, adds captions (placeholder), and prepares for posting."""
        portrait_video = video_path.with_name(video_path.stem.replace("_raw", "") + "_portrait.mp4")

        # 1. Apply portrait blur (1080p)
        apply_portrait_blur(str(video_path), str(portrait_video))

        # 2. Add captions (Placeholder for Riverside automation or alternative)
        # For now, we assume the portrait video is ready for the next step.

        logger.info(f"Highlight processed: {portrait_video}")
        logger.info(f"Title: {analysis.get('title')}")
        logger.info(f"Keywords: {analysis.get('seo_keywords')}")

        # 3. Post to Instagram (Placeholder)
        # self.post_to_instagram(portrait_video, analysis)

    def run(self):
        logger.info("Starting pipeline...")
        while True:
            screenshot = self.capture_screenshot()
            if screenshot:
                detected_round = self.detect_round(screenshot)
                os.remove(screenshot)

                if detected_round is not None:
                    if not self.is_recording:
                        self.start_recording(detected_round)
                    elif detected_round > self.current_round:
                        # New round started, stop current and process
                        self.stop_recording()
                        raw_video = self.current_video_path

                        # Run classification in background or sequentially
                        analysis = self.classify_highlight(raw_video)
                        if analysis and analysis.get('is_highlight'):
                            self.process_highlight(raw_video, analysis)

                        # Start next round
                        self.start_recording(detected_round)

            time.sleep(2) # Monitor every 2 seconds

if __name__ == "__main__":
    # Create a dummy config if not exists or use existing
    config_file = "live_pipeline_config.json"
    pipeline = HighlightPipeline(config_file)
    pipeline.run()
