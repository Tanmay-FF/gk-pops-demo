# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Video I/O — read, write, and re-encode helpers.

Uses GPU-accelerated NVENC encoding when available, falls back to CPU libx264.
"""
import os
import shutil
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
        print("[INFO] NVENC GPU encoding available - using h264_nvenc")
    else:
        print("[INFO] NVENC not available - using CPU libx264")
    return _NVENC_AVAILABLE


def open_video(path: str):
    """Open video and return (cap, w, h, fps, total_frames).

    The capture is closed if it cannot be handed back. Raising with it still
    open leaked the handle to nobody — the caller never received it, so the
    caller cannot release it — and under Windows an unreleased capture keeps the
    file locked, which turns "this clip has no readable metadata" into "this
    clip cannot be re-uploaded until the app restarts".
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except BaseException:
        cap.release()
        raise
    return cap, w, h, fps, total


#: How many of this process's finished run directories to keep on disk. The
#: newest is the one the UI is serving; the rest are kept only so a user who
#: still has an older tab open does not get a 404.
_KEEP_RUN_DIRS = 3

#: Monotonic per-process run counter, so two runs in one process cannot land in
#: the same directory even within the same clock second.
_run_seq = 0


def _run_dir() -> str:
    """A fresh directory for one run's video files.

    Both files used to be FIXED paths in tempdir — `pops_demo_raw.avi` and
    `pops_demo_output.mp4`. Two runs that overlap by even a second then share
    them, which fails two ways: the writer of run B truncates the AVI run A is
    still encoding (`PermissionError: being used by another process` on Windows,
    observed), and where it does not fail it silently serves run B's MP4 as run
    A's result. Overlap is not hypothetical here — the case report is a separate
    Gradio event that can still be running when the next Run Analysis starts,
    and two app instances on one machine share this same tempdir.

    The BASENAMES stay the same so a downloaded file is still
    `pops_demo_output.mp4`; only the directory is per-run.
    """
    global _run_seq
    _run_seq += 1
    base = tempfile.gettempdir()
    path = os.path.join(base, f"pops_run_{os.getpid()}_{_run_seq}")
    os.makedirs(path, exist_ok=True)
    _prune_run_dirs(base)
    return path


def _prune_run_dirs(base: str) -> None:
    """Drop this process's oldest run directories. Never raises, and never
    touches another process's — a second app instance is serving those."""
    prefix = f"pops_run_{os.getpid()}_"
    try:
        mine = sorted(
            (int(name[len(prefix):]), os.path.join(base, name))
            for name in os.listdir(base)
            if name.startswith(prefix) and name[len(prefix):].isdigit()
        )
    except OSError:
        return
    if len(mine) <= _KEEP_RUN_DIRS:
        return
    for _seq, path in mine[:-_KEEP_RUN_DIRS]:
        shutil.rmtree(path, ignore_errors=True)


def create_writer(w: int, h: int, fps: int):
    """Create an AVI writer in a per-run tempdir. Returns (writer, avi_path)."""
    avi_path = os.path.join(_run_dir(), "pops_demo_raw.avi")
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    return writer, avi_path


def discard_run_output(avi_path: str) -> None:
    """Throw away the partial video a cancelled run had started writing.

    The AVI lives in its own per-run directory (see _run_dir()), so a cancelled
    run leaves a truncated file that nothing consumes and that only the pruner
    would eventually clear. Removing it here keeps a cancel from accumulating
    dead frames in tempdir at ~1 MB/s of cancelled footage.

    Best-effort by design: the caller is already on its way out with a
    RunCancelled, and failing to tidy up must not replace that with an OSError.
    On Windows the writer has to have been released first, or the file is still
    locked — the caller does that before calling this.
    """
    if not avi_path:
        return
    run_dir = os.path.dirname(avi_path)
    try:
        if os.path.exists(avi_path):
            os.remove(avi_path)
        # Only if empty: the MP4 re-encode never ran for a cancelled run, but a
        # rmtree here would be a wide blast radius for a tidy-up.
        if run_dir and os.path.isdir(run_dir) and not os.listdir(run_dir):
            os.rmdir(run_dir)
    except OSError as e:
        print(f"[WARN] could not remove the cancelled run's partial video: {e}")


def _encode_through_hook(avi_path: str, out_path: str, frame_hook):
    """Decode the AVI, run every frame through `frame_hook`, pipe to ffmpeg.

    Returns (returncode, stderr_text) so the caller's existing failure
    reporting handles both encode paths identically.

    stderr goes to a temp FILE rather than a pipe on purpose: nothing reads the
    pipe until after stdin is closed, and a chatty ffmpeg filling the OS pipe
    buffer would block it forever while we sat waiting to finish writing
    frames. A file cannot deadlock.
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen the tracked video ({avi_path})")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
    ] + _encoder_args() + [out_path]

    err_file = tempfile.TemporaryFile()
    try:
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=err_file)
        except OSError as e:
            raise RuntimeError(
                f"could not run ffmpeg ({FFMPEG_EXE}): {e}") from e
        idx = 0
        try:
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    idx += 1
                    # 1-based to match the frame loop that wrote this AVI, so a
                    # hook can index the same numbers the findings carry.
                    frame_hook(frame, idx)
                    proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                # ffmpeg died mid-stream. Its exit code and stderr below say
                # why; re-raising here would replace that with a less useful
                # error.
                pass
        except BaseException:
            # A hook that raises is a bug worth surfacing, but ffmpeg is still
            # sitting on a half-written file waiting for stdin. Kill it here or
            # it outlives the run holding a lock on its own output.
            proc.kill()
            proc.wait()
            raise
        finally:
            cap.release()
            try:
                proc.stdin.close()
            except OSError:
                pass
        rc = proc.wait()
        err_file.seek(0)
        return rc, err_file.read().decode("utf-8", "replace")
    finally:
        err_file.close()


def _encoder_args() -> list[str]:
    """The codec half of the ffmpeg command, shared by both encode paths."""
    if _check_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def reencode_to_mp4(avi_path: str, frame_hook=None) -> str:
    """Re-encode AVI to H.264 MP4. Uses NVENC if available, else CPU.

    The MP4 lands beside its own AVI — see _run_dir() for why that is not a
    fixed path any more.

    `frame_hook(frame, frame_idx)` — when given, every decoded frame is passed
    through it (mutated in place, 1-based index to match the frame loop's
    numbering) before being encoded. This exists for overlays that can only be
    computed once the whole run is known, the operational rule findings being
    the case in hand. Frames are piped straight into ffmpeg rather than written
    to a second AVI: an intermediate would cost a whole extra XVID generation
    on the only copy of the run's video.
    """
    out_path = os.path.join(os.path.dirname(avi_path), "pops_demo_output.mp4")
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError as e:
            # Windows refuses to unlink a file the browser is still streaming,
            # and ffmpeg's own -y would hit the same wall. Say which file and
            # let the caller fail with that instead of with ffmpeg's exit code,
            # which does not name the problem.
            raise RuntimeError(
                f"cannot replace the previous output video ({out_path}): {e}. "
                f"Close any tab still playing it and run again.") from e

    if frame_hook is not None:
        rc, stderr_text = _encode_through_hook(avi_path, out_path, frame_hook)
    else:
        cmd = [FFMPEG_EXE, "-y", "-i", avi_path] + _encoder_args() + [out_path]
        # The return code was ignored here, and the AVI deleted regardless. A
        # failed encode therefore destroyed the only copy of the run's video and
        # handed back a path to a file that does not exist — the UI then
        # rendered an empty player on a run that otherwise succeeded, with
        # nothing in the log to say why. Check, keep the source, name the
        # reason.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            raise RuntimeError(
                f"could not run ffmpeg ({FFMPEG_EXE}): {e}") from e
        rc, stderr_text = proc.returncode, proc.stderr

    if rc != 0 or not os.path.exists(out_path):
        tail = (stderr_text or "").strip().splitlines()[-4:]
        print(f"[ERROR] ffmpeg failed (exit {rc}); the raw AVI is "
              f"kept at {avi_path}")
        for line in tail:
            print(f"[ERROR]   {line}")
        raise RuntimeError(
            "video encoding failed (ffmpeg exit "
            f"{rc}): {' / '.join(tail) or 'no output'}")

    # Only once the MP4 is known to exist: until then the AVI is the run's only
    # video.
    if os.path.exists(avi_path):
        try:
            os.remove(avi_path)
        except OSError as e:
            # Not fatal — the MP4 is what the UI serves, and the run
            # directories are pruned. Worth a line so a growing tempdir has an
            # explanation.
            print(f"[WARN] could not remove the intermediate AVI: {e}")
    return out_path
