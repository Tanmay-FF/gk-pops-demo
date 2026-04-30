# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Video I/O — read, write, and re-encode helpers.

Uses GPU-accelerated NVENC encoding when available, falls back to CPU libx264.
"""
import os
import subprocess
import tempfile

import cv2
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Detect NVENC support once at import time
_NVENC_AVAILABLE = None

def _check_nvenc() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        r = subprocess.run(
            [FFMPEG_EXE, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in r.stdout
    except Exception:
        _NVENC_AVAILABLE = False
    if _NVENC_AVAILABLE:
        print("[INFO] NVENC GPU encoding available — using h264_nvenc")
    else:
        print("[INFO] NVENC not available — using CPU libx264")
    return _NVENC_AVAILABLE


def open_video(path: str):
    """Open video and return (cap, w, h, fps, total_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, fps, total


def create_writer(w: int, h: int, fps: int):
    """Create an AVI writer in tempdir. Returns (writer, avi_path)."""
    avi_path = os.path.join(tempfile.gettempdir(), "pops_demo_raw.avi")
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    return writer, avi_path


def reencode_to_mp4(avi_path: str) -> str:
    """Re-encode AVI to H.264 MP4. Uses NVENC if available, else CPU."""
    out_path = os.path.join(tempfile.gettempdir(), "pops_demo_output.mp4")
    if os.path.exists(out_path):
        os.remove(out_path)

    if _check_nvenc():
        cmd = [
            FFMPEG_EXE, "-y", "-i", avi_path,
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
            "-cq", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
    else:
        cmd = [
            FFMPEG_EXE, "-y", "-i", avi_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]

    subprocess.run(cmd, capture_output=True)
    if os.path.exists(avi_path):
        os.remove(avi_path)
    return out_path
