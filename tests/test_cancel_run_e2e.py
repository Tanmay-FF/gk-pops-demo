"""A cancelled run must not poison the next one.

The unit file (tests/test_cancel_run.py) pins the cancel mechanics with fakes.
This one drives the REAL pipeline on the golden clip: a full run, a run cancelled
partway, then a full run that has to reproduce the first one frame for frame.

That third run is the whole point. A cancel breaks out of the frame loop with
`persist=True` tracker state, a live predictor, a partially built `_linker`, and
half-populated per-cart histories — none of which any run had ever inherited
before, because until now the only way out of that loop was the end of the video.
`_reset()` and `_reset_trackers()` are supposed to clear all of it; this is the
test that says so on real pixels rather than by reading the code.

Needs the golden clip and the detection weights; skips with a reason when either
is missing, exactly like tests/test_golden_clip.py.
"""
import sys
import os
import io
import contextlib
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.cancellation import RunCancelled

from test_golden_clip import (  # noqa: E402
    Skip, load_baseline, find_clip, run_pipeline, compare,
)


def _capture(engine_obj, clip, settings):
    """One real run. Returns (per-frame counts and display ids, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        frames = run_pipeline(clip, settings, engine_obj=engine_obj)
    return frames, out.getvalue()


def _assert_same(later, earlier, label):
    diffs = compare(later, {"frames": earlier})
    assert not diffs, (
        f"{label} differs from the earlier run in the same process on "
        f"{len(diffs)} frame(s):\n  " + "\n  ".join(diffs[:15])
    )


def _press_cancel_on_frame(engine_obj, frame_no):
    """Press Cancel once the loop has really processed `frame_no` frames.

    Tied to progress, not to a clock. A timer press is a coin flip on this
    pipeline: run 1 of the golden clip measured 26.6 s and 40.8 s in the same
    session, and the cold weight load happens before the first frame, so a fixed
    delay can land before the loop starts (the run then completes and the test
    fails for a reason that has nothing to do with cancellation) or after it ends.

    Counts `model.track` calls because that is exactly once per frame, and
    restores the attribute afterwards so the engine is left as it was found.
    """
    real_track = engine_obj.model.track
    state = {"n": 0, "pressed_at": None}

    def counting_track(*a, **kw):
        state["n"] += 1
        if state["n"] == frame_no:
            state["pressed_at"] = state["n"]
            engine_obj.request_cancel()
        return real_track(*a, **kw)

    engine_obj.model.track = counting_track

    def restore():
        try:
            del engine_obj.model.track
        except AttributeError:
            engine_obj.model.track = real_track

    return state, restore


def test_a_cancelled_run_leaves_the_next_run_unchanged():
    """The one that matters. Run 2 is cancelled mid-loop; run 3 must reproduce
    run 1 exactly."""
    baseline = load_baseline()
    clip = find_clip(baseline)
    settings = baseline["settings"]

    import engine.tracker as T
    engine_obj = T.TrackingEngine()

    frames1, _log1 = _capture(engine_obj, clip, settings)
    assert frames1, "run 1 captured no frames"
    total = len(frames1)
    press_at = max(2, min(10, total // 4))

    state, restore = _press_cancel_on_frame(engine_obj, press_at)
    out = io.StringIO()
    cancelled = False
    try:
        with contextlib.redirect_stdout(out):
            run_pipeline(clip, settings, engine_obj=engine_obj)
    except RunCancelled as e:
        cancelled = True
        reason = str(e)
    finally:
        restore()
    log2 = out.getvalue()

    assert state["pressed_at"] == press_at, (
        f"the loop never reached frame {press_at}, so nothing was cancelled: "
        f"{state}")
    assert cancelled, (
        "run 2 finished instead of being cancelled — the frame loop is not "
        "looking at the flag. Log tail:\n" + log2[-600:])

    # It has to stop PROMPTLY. Read the frame it stopped on out of the engine's
    # own log rather than timing it: a cancel only honoured at the end of the
    # clip is indistinguishable from no cancel, and a frame count says so
    # without depending on how fast this machine is.
    m = re.search(r"\[CANCEL\] run \d+ stopped at frame (\d+) of (\d+)", log2)
    assert m, "no cancel line in the log:\n" + log2[-600:]
    stopped_at, total_frames = int(m.group(1)), int(m.group(2))
    assert stopped_at < total_frames, (
        f"the run ran to the end of the clip ({stopped_at}/{total_frames}) "
        f"despite a cancel at frame {press_at}")
    assert stopped_at <= press_at + 2, (
        f"cancel pressed at frame {press_at} but the loop ran to "
        f"{stopped_at} — more than one frame of latency")
    assert "at frame" in reason, reason

    frames3, log3 = _capture(engine_obj, clip, settings)
    _assert_same(frames3, frames1, "the run after a cancelled run")
    assert "duplicate tracking callback" not in log3, (
        "the cancelled run left a duplicate tracking callback:\n" + log3)
    assert "models not on" not in log3, (
        "the cancelled run left a model on the wrong device:\n" + log3)


def test_a_cancel_pressed_between_runs_does_not_kill_the_next_one():
    """A press with nothing in flight is a no-op that clears itself. Without
    that, an idle Cancel would arm a trap for whatever the user ran next."""
    baseline = load_baseline()
    clip = find_clip(baseline)
    settings = baseline["settings"]

    import engine.tracker as T
    engine_obj = T.TrackingEngine()

    frames1, _ = _capture(engine_obj, clip, settings)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        engine_obj.request_cancel()          # idle press, targets run 1
    frames2, _ = _capture(engine_obj, clip, settings)
    _assert_same(frames2, frames1, "the run after an idle Cancel press")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for t in tests:
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
        else:
            passed += 1
            print(f"PASS {t.__name__}")
    print(f"\n{passed} passed, {skipped} skipped")
    sys.exit(0)
