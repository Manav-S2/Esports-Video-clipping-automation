import os
import subprocess
import time
from datetime import datetime


def capture_screenshots(twitch_url, output_dir="every_second_screenshots", interval_seconds=2):
    """
    Captures screenshots from a Twitch live stream every `interval_seconds`.

    Args:
        twitch_url (str): The URL of the Twitch live stream.
        output_dir (str): Directory to save the screenshots (relative to script location).
        interval_seconds (int): Time interval between screenshots in seconds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the HLS stream URL using Streamlink
    try:
        stream_process = subprocess.run(
            ["streamlink", "--stream-url", twitch_url, "best"],
            capture_output=True, text=True, check=True
        )
        stream_url = stream_process.stdout.strip()
        if not stream_url:
            print("Error: Could not get stream URL from Streamlink. Is the stream live or URL correct?")
            return
        print(f"Stream URL obtained: {stream_url}")
    except subprocess.CalledProcessError as e:
        print(f"Error calling Streamlink: {e}")
        print(f"Stderr: {e.stderr}")
        return
    except FileNotFoundError:
        print("Error: streamlink command not found. Make sure Streamlink is installed and in your PATH.")
        return

    print(f"Starting screenshot capture from {twitch_url} every {interval_seconds} seconds. Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filepath = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")

            # Use ffmpeg to take a single screenshot
            # -i: input
            # -ss: seek to position (not needed here, but useful for specific times)
            # -vframes 1: take only one video frame
            # -q:v 2: quality (2 is good, 1 is best, 31 is worst)
            ffmpeg_command = [
                "ffmpeg",
                "-i", stream_url,
                "-ss", "00:00:01", # Seek to 1 second to ensure a frame is available
                "-vframes", "1",
                "-q:v", "2",
                output_filepath
            ]

            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                print(f"Captured {output_filepath}")
            except subprocess.CalledProcessError as e:
                print(f"Error capturing screenshot with FFmpeg: {e}")
                print(f"Stderr: {e.stderr.decode()}") # Decode stderr for better readability
            except FileNotFoundError:
                print("Error: ffmpeg command not found. Make sure FFmpeg is installed and in your PATH.")
                return

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nScreenshot capture stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    TWITCH_URL = "https://www.twitch.tv/fooya"
    SCREENSHOT_INTERVAL = 2  # seconds
    # The output directory will be created relative to where this script is run
    OUTPUT_DIRECTORY = "every_second_screenshots"

    capture_screenshots(TWITCH_URL, OUTPUT_DIRECTORY, SCREENSHOT_INTERVAL)
