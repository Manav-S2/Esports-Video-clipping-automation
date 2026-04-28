import cv2
import numpy as np
import subprocess
import os

def apply_portrait_blur(input_video, output_video):
    """
    Applies blurry stripes on top and bottom in portrait mode (9:16) for a 16:9 input.
    Input: 1920x1080
    Output: 1080x1920
    """
    # Using ffmpeg for efficiency and high quality
    # Filter complex explanation:
    # 1. Scale input to 1080x1920 to fill the background, crop to center, blur it
    # 2. Scale input to 1080x(variable) to fit width, maintaining aspect ratio
    # 3. Overlay the scaled input on top of the blurred background
    
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-filter_complex', 
        '[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg];'
        '[0:v]scale=1080:-2[fg];'
        '[bg][fg]overlay=(W-w)/2:(H-h)/2',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
        '-c:a', 'copy',
        output_video
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Portrait video saved to {output_video}")

if __name__ == "__main__":
    # Example usage
    # apply_portrait_blur("input.mp4", "output_portrait.mp4")
    pass
