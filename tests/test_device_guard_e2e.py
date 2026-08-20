"""End-to-end: a failed local-VLM case report must not degrade the next run.

The unit file (tests/test_device_guard.py) pins the guard's logic with fakes.
This one drives the REAL pipeline on the golden clip, twice in one process, with
a real GPU offload round-trip and a deliberately broken VLM in between, and
requires run 2 to reproduce run 1 frame for frame.

Run-against-run, deliberately not against the committed baseline: whether the
pipeline still agrees with the baseline is tests/test_golden_clip.py's job, and
coupling the two files would make every deliberate scoring or linking change
surface here as a phantom device-drift failure.

It covers the two Phase 6 scenarios that do not actually need a live Qwen pass:

  * two consecutive runs in one process with a case report between them
    (the duplicate-tracking-callback regression showed up here as frame 52
    dropping from 3 people to 1, which per-frame counts catch), and
  * a case report that fails — including one whose VLM unload fails, which is
    the path that used to leave `_offloaded` True forever and permanently
    disable the start-of-run guard.

Needs the golden clip and the detection weights; skips with a reason when
either is missing, exactly like tests/test_golden_clip.py.
"""
import sys
import os
import io
import contextlib
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from test_golden_clip import (  # noqa: E402
    Skip, load_baseline, find_clip, run_pipeline, compare,
)

HAS_CUDA = torch.cuda.is_available()


def _fake_pending(engine_obj):
    """The smallest `_pending_case_report` payload that reaches the VLM.

    Built by hand rather than harvested from the run: whether a real run
    captures any incident frames depends on the clip, and this test is about
    the offload round-trip, not about the scoring.
    """
    return {
        "captures": [{"image_bytes": b"", "frame_idx": 0}],
        "full_json": {"video_info": {}},
        "event_log": [],
        "peak_snapshots": [],
        "vlm_backend": "Qwen3-VL-2B (local)",   # "Claude" not in it -> offloads
        "vlm_api_key": "",
        "analytics_result": None,
    }


class _BrokenVLM:
    """Fails at analysis AND at unload — the worst case, and the one that used
    to poison the process. `unload_model` raising skipped the restore, so the
    stack stayed on the CPU with `device == "cuda"` and `_offloaded == True`
    made the start-of-run guard a permanent no-op."""

    def __init__(self, **kw):
        pass

    def analyze_incident(self, **kw):
        raise RuntimeError("induced VLM failure")

    def unload_model(self):
        raise RuntimeError("induced unload failure")


def _run_failing_case_report(engine_obj):
    """Run the real _run_case_report path — real release/restore, broken VLM —
    and return everything it printed."""
    import engine.tracker as T

    # Appended, and appended LAST after clearing: a real run with captures has
    # already queued its own payload, and finalize_case_report() pops the
    # OLDEST (Phase 9.4.2). Without the clear this ran the real payload and the
    # fake one was never reached, so the induced-failure assertions below were
    # passing for the wrong reason.
    engine_obj._pending_case_reports.clear()
    engine_obj._pending_case_reports.append(_fake_pending(engine_obj))
    real = T.VLMAnalyzer
    T.VLMAnalyzer = _BrokenVLM
    buf = io.StringIO()
    try:
        # stderr too: finalize_case_report() traceback.print_exc()s the induced
        # failure by design, and a real traceback in passing test output reads
        # like a real failure.
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            html, path = engine_obj.finalize_case_report()
    finally:
        T.VLMAnalyzer = real
    # finalize_case_report swallows the error into an inline banner by design.
    assert path is None and "failed" in html, (html, path)
    return buf.getvalue()


def _capture(engine_obj, clip, settings):
    """One real run. Returns (per-frame counts and display ids, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        frames = run_pipeline(clip, settings, engine_obj=engine_obj)
    return frames, out.getvalue()


def _assert_same(later, earlier, label):
    """Require two runs of THIS process to agree, using test_golden_clip's
    comparison so the diff reads the same way."""
    diffs = compare(later, {"frames": earlier})
    assert not diffs, (
        f"{label} differs from the earlier run in the same process on "
        f"{len(diffs)} frame(s):\n  " + "\n  ".join(diffs[:15])
    )


def test_a_run_started_during_a_case_report_is_correct_anyway():
    """The reported crash, on real pixels: a second Run Analysis clicked while a
    local-VLM case report is still generating.

    The two are separate Gradio dependencies over one engine (app_poc_v2.py:1508
    and :1567), so the run really can start mid-report. Before Phase 9.4.1 it
    ran its frame loop against the CPU-parked detection stack while
    `_classifier.device` still said cuda, and died on the first classify:

        RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                      (torch.FloatTensor) should be the same

    The fake VLM here holds the GPU for a second — long enough that a run which
    is NOT serialised gets well into its frame loop before the restore. The
    assertion is not just "no exception": the run has to reproduce run 1 frame
    for frame, which also catches the quieter failure where an overlapping run
    loses the predictor and silently halves its detections.
    """
    baseline = load_baseline()
    clip = find_clip(baseline)
    settings = baseline["settings"]

    import engine.tracker as T
    engine_obj = T.TrackingEngine()

    frames1, log1 = _capture(engine_obj, clip, settings)
    assert frames1, "run 1 captured no frames"

    if not HAS_CUDA:
        raise Skip("no CUDA — the offload the overlap races is a no-op on cpu")

    generating = threading.Event()

    class _SlowVLM:
        """Holds the card the way a local Qwen pass does."""

        def __init__(self, **kw):
            pass

        def analyze_incident(self, **kw):
            generating.set()
            time.sleep(1.0)
            return {"summary": "held the GPU for a while"}

        def unload_model(self):
            pass

    engine_obj._pending_case_reports.clear()
    engine_obj._pending_case_reports.append(_fake_pending(engine_obj))

    report_out = {}

    def report():
        # Deliberately NOT wrapped in redirect_stdout. contextlib's redirect
        # swaps the process-global sys.stdout, so a redirect entered on this
        # thread and exited while the main thread is inside its own restores
        # the wrong stream and silently swallows the rest of the file's output.
        # Nothing here prints anyway: the VLM and the HTML builder are stubs.
        report_out["result"] = engine_obj.finalize_case_report()

    real_vlm = T.VLMAnalyzer
    real_build = T.build_case_report_html
    T.VLMAnalyzer = _SlowVLM
    T.build_case_report_html = lambda *a, **kw: ("<p>ok</p>", "<html/>")
    try:
        thread = threading.Thread(target=report)
        thread.start()
        assert generating.wait(30), "the fake VLM never started generating"
        assert engine_obj._offloaded is True, (
            "the detection stack was not parked — this test would then prove "
            "nothing about an overlapping run")
        # The click that used to crash. It must WAIT here, not run.
        frames2, log2 = _capture(engine_obj, clip, settings)
        thread.join(120)
        assert not thread.is_alive(), "the case-report thread hung"
    finally:
        T.VLMAnalyzer = real_vlm
        T.build_case_report_html = real_build

    _assert_same(frames2, frames1, "the run that overlapped a case report")
    assert "duplicate tracking callback" not in log2, (
        f"the overlapping run lost the predictor:\n{log2}")
    assert "still parked on the CPU at the start of a run" not in log2, (
        f"the run saw a stranded offload flag:\n{log2}")
    assert engine_obj._offloaded is False, "the restore did not run"
    assert report_out["result"][0], "the case report produced nothing"


def test_two_runs_across_a_failed_case_report_agree():
    """The whole failure, end to end, on real pixels."""
    baseline = load_baseline()
    clip = find_clip(baseline)
    settings = baseline["settings"]

    import engine.tracker as T
    engine_obj = T.TrackingEngine()

    # --- run 1: the reference every later run in this process must match -----
    frames1, log1 = _capture(engine_obj, clip, settings)
    assert frames1, "run 1 captured no frames"
    assert "models not on" not in log1, f"run 1 saw drift it should not:\n{log1}"

    # --- a case report that fails at analysis AND at unload ------------------
    log_cr = _run_failing_case_report(engine_obj)
    if HAS_CUDA:
        assert "unloading the VLM failed" in log_cr, log_cr
        # The contract: an unload that raised must NOT skip the restore.
        assert engine_obj._offloaded is False, (
            "_offloaded stayed True after a failed unload — the start-of-run "
            "guard is now a permanent no-op and every later run will raise the "
            "device-mismatch error"
        )
        for name, mod in (("detector", engine_obj.model),
                          ("pose", engine_obj._pose_model),
                          ("quality", engine_obj._classifier._quality_model),
                          ("fill", engine_obj._classifier._fill_model)):
            dev = T.TrackingEngine._param_device(mod)
            assert dev in (None, "cuda"), f"{name} left on {dev} after restore"
        assert engine_obj._held_predictor is None, "predictor still parked"
        assert getattr(engine_obj.model, "predictor", None) is not None, (
            "the detector lost its predictor — the next .track() will register "
            "a duplicate tracking callback and silently halve detections"
        )

    # --- run 2: same process, same clip, same answer ------------------------
    frames2, log2 = _capture(engine_obj, clip, settings)
    _assert_same(frames2, frames1, "run 2")
    assert "models not on" not in log2, (
        f"run 2 had to repair a device after a clean restore:\n{log2}")


def test_a_lost_predictor_does_not_degrade_the_next_run():
    """The reported failure: "the cart and the person are not even getting
    detected".

    Losing the predictor with nothing parked is what the offload contract exists
    to prevent, but it only has to happen once — a case report and the next Run
    Analysis are separate Gradio events and can overlap — and the damage is
    permanent for the process: Model.track() APPENDS a second tracking callback,
    so the tracker updates twice per frame from then on.

    Measured on FF1763940475070 …INSIDE.mp4 before the dedupe went in: run 2 of
    the same process fell from 4 carts / 15 people to 2 carts / 7 people, which
    is exactly what the operator saw. Asserted as full per-frame parity rather
    than as a count, because a partial degradation is the same bug.
    """
    baseline = load_baseline()
    clip = find_clip(baseline)
    settings = baseline["settings"]

    import engine.tracker as T
    from engine import ultralytics_compat

    engine_obj = T.TrackingEngine()
    frames1, _log1 = _capture(engine_obj, clip, settings)
    assert frames1, "run 1 captured no frames"
    assert ultralytics_compat.tracking_callback_counts(engine_obj.model) == {
        "on_predict_start": 1, "on_predict_postprocess_end": 1}, (
        "run 1 already had duplicate tracking callbacks")

    # Every .to() nulls `predictor`; this is that state with nothing parked to
    # put back, i.e. the one the guards cannot repair.
    engine_obj.model.predictor = None
    engine_obj._held_predictor = None

    frames2, log2 = _capture(engine_obj, clip, settings)
    assert "duplicate tracking callback" in log2, (
        "run 2 did not report the re-registration — either ultralytics stopped "
        "appending (then this test is obsolete) or the first-frame check is gone")
    assert ultralytics_compat.tracking_callback_counts(engine_obj.model) == {
        "on_predict_start": 1, "on_predict_postprocess_end": 1}, (
        "duplicate tracking callbacks survived run 2")
    _assert_same(frames2, frames1, "run 2 after a lost predictor")


def test_a_stranded_classifier_is_repaired_and_changes_nothing():
    """The poisoned state itself: the quality model left on the CPU while
    `device` still says cuda. Before the guard this raised

        RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
        (torch.FloatTensor) should be the same

    on every run until the process restarted, because
    CartClassifier.load_quality early-returns on an unchanged pt_path and so
    never reloads it. Now it costs one warning and nothing else.
    """
    if not HAS_CUDA:
        print("  (skipped: no CUDA)")
        return

    baseline = load_baseline()
    clip = find_clip(baseline)
    settings = baseline["settings"]

    import engine.tracker as T
    engine_obj = T.TrackingEngine()
    # Load the classifiers by doing one real run, then strand one of them the
    # way a half-completed .to("cuda") does.
    frames_before, _ = _capture(engine_obj, clip, settings)
    assert engine_obj._classifier._quality_model is not None, (
        "the warm-up run did not load the quality model")
    engine_obj._classifier._quality_model.to("cpu")

    frames_after, log = _capture(engine_obj, clip, settings)
    _assert_same(frames_after, frames_before, "the run after stranding")
    warns = [l for l in log.splitlines() if "models not on" in l]
    assert len(warns) == 1, f"expected exactly one drift warning, got {warns}"
    assert "quality" in warns[0], warns[0]
    assert "start of run" in warns[0], warns[0]
    assert T.TrackingEngine._param_device(
        engine_obj._classifier._quality_model) == "cuda"


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
    print(f"\n{passed} passed, {skipped} skipped (cuda={HAS_CUDA})")
    sys.exit(0)
