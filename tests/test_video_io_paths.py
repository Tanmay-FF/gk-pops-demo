# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Two overlapping runs must not share one output file.

Run with:  python tests/test_video_io_paths.py

WHAT BROKE
----------
`create_writer()` and `reencode_to_mp4()` both used FIXED paths in the system
tempdir — `pops_demo_raw.avi` and `pops_demo_output.mp4`. Runs overlap in this
app: the case report is a separate Gradio event that can still be running when
the next Run Analysis starts, and a second app instance on the same machine
shares the same tempdir. Two runs then write one file, which fails two ways:

  PermissionError: [WinError 32] The process cannot access the file because it
  is being used by another process: '...\\pops_demo_raw.avi'

— observed while a second run was encoding — and, where it does not raise, the
quieter one: run B's MP4 is served as run A's result.

Each run now gets its own directory. The basenames are unchanged, so a
downloaded file is still `pops_demo_output.mp4`.

No GPU, no ffmpeg, no video decode — the encode itself is stubbed.

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import video_io

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


class _Sandbox:
    """Point video_io's tempdir at a scratch directory and reset its counter."""

    def __enter__(self):
        self.base = tempfile.mkdtemp(prefix="pops_vio_test_")
        self._real_gettempdir = video_io.tempfile.gettempdir
        self._real_seq = video_io._run_seq
        video_io.tempfile.gettempdir = lambda: self.base
        video_io._run_seq = 0
        return self.base

    def __exit__(self, *exc):
        video_io.tempfile.gettempdir = self._real_gettempdir
        video_io._run_seq = self._real_seq
        shutil.rmtree(self.base, ignore_errors=True)
        return False


def _fake_ffmpeg(returncode=0, stderr=""):
    """A stand-in for subprocess.run that behaves like ffmpeg: on success it
    writes the output file named last on the command line."""
    class _Proc:
        pass

    def run(cmd, *a, **k):
        proc = _Proc()
        proc.returncode = returncode
        proc.stderr = stderr
        proc.stdout = ""
        if returncode == 0:
            open(cmd[-1], "wb").close()
        return proc

    return run


def main():
    section("Each run writes to its own directory")
    with _Sandbox() as base:
        first = video_io._run_dir()
        second = video_io._run_dir()
        check("two runs get different directories", first != second,
              f"{os.path.basename(first)} vs {os.path.basename(second)}")
        check("both live under the tempdir",
              os.path.dirname(first) == base and os.path.dirname(second) == base)
        check("both directories exist",
              os.path.isdir(first) and os.path.isdir(second))
        check("the directory names carry this process's pid",
              os.path.basename(first).startswith(f"pops_run_{os.getpid()}_"),
              os.path.basename(first))

    section("The basenames the user sees do not change")
    with _Sandbox():
        # A real writer would need cv2 and a codec; the path is what matters, so
        # stub the writer out.
        real_writer = video_io.cv2.VideoWriter
        video_io.cv2.VideoWriter = lambda *a, **k: object()
        try:
            _w1, avi1 = video_io.create_writer(64, 48, 20)
            _w2, avi2 = video_io.create_writer(64, 48, 20)
        finally:
            video_io.cv2.VideoWriter = real_writer
        check("the AVI is still called pops_demo_raw.avi",
              os.path.basename(avi1) == "pops_demo_raw.avi", avi1)
        check("two writers do not share a path", avi1 != avi2)

        real_run = video_io.subprocess.run
        real_nvenc = video_io._check_nvenc
        # A CompletedProcess-shaped stub that writes the output file, because
        # reencode_to_mp4 now checks both the exit code and the file: a stub
        # returning None would have it fail on returncode instead of testing
        # paths.
        video_io.subprocess.run = _fake_ffmpeg(returncode=0)
        video_io._check_nvenc = lambda: False
        try:
            open(avi1, "wb").close()
            out1 = video_io.reencode_to_mp4(avi1)
        finally:
            video_io.subprocess.run = real_run
            video_io._check_nvenc = real_nvenc
        check("the MP4 is still called pops_demo_output.mp4",
              os.path.basename(out1) == "pops_demo_output.mp4", out1)
        check("the MP4 lands beside its own AVI",
              os.path.dirname(out1) == os.path.dirname(avi1))
        check("the AVI is cleaned up after the encode",
              not os.path.exists(avi1))

    section("A failed encode keeps the AVI and says why")
    with _Sandbox():
        # The old code ignored ffmpeg's exit code and deleted the AVI anyway, so
        # a failed encode destroyed the run's only video and returned a path to
        # a file that did not exist — the UI showed an empty player on a run
        # that had otherwise succeeded.
        real_writer = video_io.cv2.VideoWriter
        video_io.cv2.VideoWriter = lambda *a, **k: object()
        try:
            _w, avi = video_io.create_writer(64, 48, 20)
        finally:
            video_io.cv2.VideoWriter = real_writer
        open(avi, "wb").close()

        real_run = video_io.subprocess.run
        real_nvenc = video_io._check_nvenc
        video_io.subprocess.run = _fake_ffmpeg(
            returncode=1, stderr="x264: no such encoder\nConversion failed!")
        video_io._check_nvenc = lambda: False
        raised = None
        try:
            video_io.reencode_to_mp4(avi)
        except RuntimeError as e:
            raised = str(e)
        finally:
            video_io.subprocess.run = real_run
            video_io._check_nvenc = real_nvenc

        check("a non-zero ffmpeg exit raises", raised is not None)
        check("the error names ffmpeg's exit code",
              bool(raised) and "exit 1" in raised, raised or "")
        check("the error carries ffmpeg's own last words",
              bool(raised) and "Conversion failed!" in raised, raised or "")
        check("the source AVI is kept for diagnosis", os.path.exists(avi))

        # And a missing ffmpeg is named as such rather than surfacing as a
        # FileNotFoundError from deep inside subprocess.
        def _no_ffmpeg(*a, **k):
            raise OSError("The system cannot find the file specified")

        video_io.subprocess.run = _no_ffmpeg
        video_io._check_nvenc = lambda: False
        raised = None
        try:
            video_io.reencode_to_mp4(avi)
        except RuntimeError as e:
            raised = str(e)
        finally:
            video_io.subprocess.run = real_run
            video_io._check_nvenc = real_nvenc
        check("a missing ffmpeg is reported as such",
              bool(raised) and "could not run ffmpeg" in raised, raised or "")

    section("Old directories are pruned, other processes' are not")
    with _Sandbox() as base:
        foreign = os.path.join(base, "pops_run_999999_1")
        os.makedirs(foreign)
        made = [video_io._run_dir() for _ in range(video_io._KEEP_RUN_DIRS + 2)]
        surviving = {d for d in made if os.path.isdir(d)}
        check(f"at most {video_io._KEEP_RUN_DIRS} of this process's run dirs remain",
              len(surviving) <= video_io._KEEP_RUN_DIRS, f"{len(surviving)}")
        check("the newest run dir is one of the survivors", made[-1] in surviving)
        check("the oldest run dir is gone", made[0] not in surviving)
        check("another process's run dir is untouched", os.path.isdir(foreign))

    section("Pruning never takes the run down with it")
    with _Sandbox() as base:
        shutil.rmtree(base)  # tempdir vanished under us
        try:
            video_io._prune_run_dirs(base)
            ok = True
        except Exception as e:
            ok = False
            print(f"    raised {e!r}")
        check("a missing tempdir is not an error", ok)

    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        for name in _FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
