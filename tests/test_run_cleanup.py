"""A run that dies must give its video handles back.

`cap` and `writer` used to be released at exactly two points in the frame loop:
the end of the video, and the cancel branch. Every other way out leaked both —
and the leak is not academic. The RuntimeError that started this investigation

    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                  (torch.FloatTensor) should be the same

is raised from the classifier INSIDE the loop, so each failed run left an open
VideoCapture and a VideoWriter still holding a lock on its AVI. Under Windows
that lock is what turns one bad run into "the next run's writer cannot open its
file".

These tests drive the real cleanup path with a stubbed pipeline: no clip, no
weights, no GPU, so they run anywhere.
"""
import sys
import os
import io
import contextlib
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.cancellation import CancelToken, RunCancelled
from engine.tracker import TrackingEngine


class FakeHandle:
    """Stands in for a cv2.VideoCapture / VideoWriter: counts releases, and can
    fail one, because a release that raises must not stop the other."""

    def __init__(self, fail=False):
        self.releases = 0
        self.fail = fail

    def release(self):
        self.releases += 1
        if self.fail:
            raise RuntimeError("release blew up")


def _engine():
    e = TrackingEngine.__new__(TrackingEngine)
    e.device = "cpu"
    e._gpu_lock = threading.Lock()
    e._cancel = CancelToken()
    e._pending_case_reports = []
    e._run_handles = None
    return e


def _registered(e, avi_path=""):
    cap, writer = FakeHandle(), FakeHandle()
    e._run_handles = (cap, writer, avi_path)
    return cap, writer


def _partial_avi():
    """A run directory with a partial AVI in it, like a dead run leaves."""
    run_dir = tempfile.mkdtemp(prefix="pops_run_test_")
    avi = os.path.join(run_dir, "pops_demo_raw.avi")
    with open(avi, "wb") as f:
        f.write(b"partial frames")
    return run_dir, avi


def test_a_crash_in_the_pipeline_still_releases_the_handles():
    e = _engine()
    run_dir, avi = _partial_avi()
    cap = writer = None

    def exploding_pipeline(*a, **kw):
        nonlocal cap, writer
        cap, writer = _registered(e, avi)          # the loop got this far
        raise RuntimeError("Input type (torch.cuda.FloatTensor) and weight "
                           "type (torch.FloatTensor) should be the same")

    e._process_video = exploding_pipeline
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            e.process_video("clip")
    except RuntimeError as err:
        assert "should be the same" in str(err), (
            f"the real error was replaced by the cleanup: {err}")
    else:
        raise AssertionError("the pipeline error was swallowed")

    assert cap.releases == 1, f"capture not released: {cap.releases}"
    assert writer.releases == 1, f"writer not released: {writer.releases}"
    assert e._run_handles is None, "the registration outlived the run"
    assert not os.path.exists(avi), "the partial video was left behind"
    assert not os.path.isdir(run_dir), "the empty run directory was left behind"


def test_a_cancelled_run_releases_the_handles_too():
    """The cancel branch no longer releases anything itself — the wrapper's
    finally owns it, so both paths cannot drift apart."""
    e = _engine()
    run_dir, avi = _partial_avi()
    cap = writer = None

    def cancelled_pipeline(*a, **kw):
        nonlocal cap, writer
        cap, writer = _registered(e, avi)
        raise RunCancelled("cancelled at frame 10 of 415")

    e._process_video = cancelled_pipeline
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            e.process_video("clip")
    except RunCancelled:
        pass
    else:
        raise AssertionError("expected RunCancelled")

    assert (cap.releases, writer.releases) == (1, 1)
    assert not os.path.exists(avi)


def test_a_successful_run_keeps_its_output():
    """The finally must not bin the video of a run that worked. `_process_video`
    deregisters once it has released the handles itself, and that is what the
    finally checks."""
    e = _engine()
    run_dir, avi = _partial_avi()

    def good_pipeline(*a, **kw):
        cap, writer = _registered(e, avi)
        cap.release()
        writer.release()
        e._run_handles = None                      # what the real path does
        return "result"

    e._process_video = good_pipeline
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert e.process_video("clip") == "result"
    assert os.path.exists(avi), "the finally deleted a finished run's video"
    os.remove(avi)
    os.rmdir(run_dir)


def test_a_release_that_raises_does_not_hide_the_real_error():
    """Cleanup runs while an exception is already propagating. A failure in it
    must not replace the error worth reading, and must not stop the other
    handle from being released."""
    e = _engine()
    run_dir, avi = _partial_avi()
    cap = FakeHandle(fail=True)
    writer = FakeHandle()

    def exploding_pipeline(*a, **kw):
        e._run_handles = (cap, writer, avi)
        raise ValueError("the real problem")

    e._process_video = exploding_pipeline
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            e.process_video("clip")
    except ValueError as err:
        assert str(err) == "the real problem", err
    else:
        raise AssertionError("the pipeline error was swallowed")

    assert writer.releases == 1, "a failed capture release skipped the writer"
    assert "releasing the video capture failed" in buf.getvalue(), buf.getvalue()
    assert not os.path.exists(avi), "the partial video survived a failed release"


def test_releasing_twice_is_harmless():
    """The helper is called from a finally that can run after `_process_video`
    already cleaned up, and from tests. It must be idempotent."""
    e = _engine()
    cap, writer = _registered(e, "")
    e._release_run_handles()
    e._release_run_handles()
    assert (cap.releases, writer.releases) == (1, 1)
    assert e._run_handles is None


def test_open_video_does_not_leak_a_capture_it_cannot_return():
    """`open_video` raises when a file has no readable metadata. Raising with
    the capture still open leaks it to nobody — the caller never got it — and
    under Windows that keeps the file locked."""
    from engine import video_io

    released = []

    class FakeCapture:
        def isOpened(self):
            return False

        def release(self):
            released.append(True)

        def get(self, _prop):
            return 0

    real = video_io.cv2.VideoCapture
    video_io.cv2.VideoCapture = lambda *a, **kw: FakeCapture()
    try:
        try:
            video_io.open_video("nonexistent.mp4")
        except ValueError:
            pass
        else:
            raise AssertionError("expected the unreadable-video error")
    finally:
        video_io.cv2.VideoCapture = real

    assert released == [True], "the capture was leaked on the failure path"


def test_a_failed_encode_does_not_lose_the_rest_of_the_run():
    """`reencode_to_mp4` raises now rather than returning a path it failed to
    write. Everything after the encode — POPS reconciliation, analytics, the
    heat-map, the case report — is worth having without a playable video, so the
    engine swallows that one exception and returns no video instead of no run."""
    import inspect
    from engine.tracker import TrackingEngine

    # A wiring check, not a behavioural one: reaching this line for real needs a
    # clip, the weights and a working writer, which is test_golden_clip's
    # territory. What it pins is the part that can be lost in an edit — the
    # `try:` being the statement immediately enclosing the call, and the handler
    # leaving out_path unset rather than re-raising.
    src = inspect.getsource(TrackingEngine._process_video)
    call = "out_path = reencode_to_mp4(avi_path)"
    i = src.index(call)
    assert src[:i].rstrip().endswith("try:"), (
        "the encode is not inside a try: " + src[:i].rstrip()[-120:])
    after = src[i:]
    assert "out_path = None" in after[:1500], (
        "a failed encode must leave out_path None rather than propagate")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")
