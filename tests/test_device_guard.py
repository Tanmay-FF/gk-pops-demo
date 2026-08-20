"""Device-drift guard and VLM unload ordering.

Covers the failure that produced, on every run after a local-VLM case report:
    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
    (torch.FloatTensor) should be the same

Runs on CPU-only machines: the CUDA-specific assertions are skipped there, the
ordering and mixed-device assertions are not.
"""
import sys
import os
import contextlib
import io
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine.cancellation import CancelToken
from engine.tracker import TrackingEngine

HAS_CUDA = torch.cuda.is_available()


class FakeClassifier:
    def __init__(self, device, quality, fill):
        self.device = device
        self._quality_model = quality
        self._fill_model = fill


class FakeYOLO:
    """Stands in for ultralytics Model: wraps an nn.Module in `.model` and
    nulls `predictor` on every `.to()`, the way Model._apply() does."""

    def __init__(self, module):
        self.model = module
        self.predictor = object()
        self.fail_next_to = False

    def to(self, dev):
        self.predictor = None
        if self.fail_next_to:
            self.fail_next_to = False
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        self.model.to(dev)
        return self


class FakeTracker:
    def __init__(self):
        self.resets = 0
        self.fail = False

    def reset(self):
        if self.fail:
            raise AttributeError("this ultralytics has no BYTETracker.reset")
        self.resets += 1


class FakePredictor:
    def __init__(self, trackers=None):
        self.trackers = trackers if trackers is not None else [FakeTracker()]


def _engine(device, *, detector_dev="cpu", quality_dev="cpu"):
    # __new__, so __init__ never runs: every attribute the methods under test
    # touch has to be set here by hand.
    e = TrackingEngine.__new__(TrackingEngine)
    e.device = device
    e._pose_model = None
    e._held_predictor = None
    e._offloaded = False
    e._gpu_lock = threading.Lock()
    e._pending_case_reports = []
    e._cancel = CancelToken()
    e._run_handles = None
    e.model = FakeYOLO(torch.nn.Conv2d(3, 4, 3).to(detector_dev))
    e._classifier = FakeClassifier(
        device, torch.nn.Conv2d(3, 4, 3).to(quality_dev), None)
    return e


def test_a_deliberate_offload_is_not_repaired():
    """The case report runs as a separate .then()-chained Gradio event, so a
    second run can begin while the stack is parked on the CPU on purpose.
    Pulling it back then would OOM the VLM mid-pass."""
    e = _engine("cuda")
    e._offloaded = True
    assert e._ensure_on_device("start of run") == []
    assert TrackingEngine._param_device(e.model) == "cpu", "moved anyway"


def test_no_drift_is_a_noop():
    e = _engine("cpu")
    assert e._ensure_on_device("t") == []


def test_mixed_module_is_detected():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    assert TrackingEngine._param_device(m) == "cpu"
    if HAS_CUDA:
        m[0].to("cuda")
        assert TrackingEngine._param_device(m) == "mixed"


def test_absent_model_is_not_drift():
    assert TrackingEngine._param_device(None) is None


def test_drifted_models_are_named_and_moved_back():
    if not HAS_CUDA:
        print("  (skipped: no CUDA)")
        return
    e = _engine("cuda")
    assert sorted(e._ensure_on_device("t")) == ["detector", "quality"]
    assert TrackingEngine._param_device(e.model) == "cuda"
    assert TrackingEngine._param_device(e._classifier._quality_model) == "cuda"
    assert e._ensure_on_device("t") == []


def test_failed_repair_falls_back_to_cpu_not_a_mixed_stack():
    if not HAS_CUDA:
        print("  (skipped: no CUDA)")
        return
    e = _engine("cuda")
    e.model.fail_next_to = True
    e._ensure_on_device("t")
    assert e.device == "cpu"
    assert e._classifier.device == "cpu", "classifier still sends inputs to CUDA"
    assert TrackingEngine._param_device(e.model) == "cpu"
    assert TrackingEngine._param_device(e._classifier._quality_model) == "cpu"


def test_predictor_survives_a_move_that_raised():
    """A .to() that raises has already nulled `predictor`; the retry has to
    take the parked one or the next .track() registers a duplicate tracking
    callback."""
    e = _engine("cpu")
    parked = e.model.predictor
    e._held_predictor = parked
    e.model.predictor = None      # as if a failed .to() had run
    e._move_detector("cpu")
    assert e.model.predictor is parked
    assert e._held_predictor is None


def test_the_predictor_survives_a_repair_that_failed_into_the_cpu_fallback():
    """The full retry path: a restore that raised parked the predictor, the
    start-of-run guard tries to repair, and that move fails too. The CPU
    fallback must still end up with a predictor — Model._apply() nulls it
    before it can raise, so nothing else is left to put it back, and losing it
    means the next .track() registers a duplicate tracking callback and
    silently halves detections for the rest of the process."""
    if not HAS_CUDA:
        print("  (skipped: no CUDA)")
        return
    e = _engine("cuda")
    parked = FakePredictor()
    e._held_predictor = parked
    e.model.predictor = None          # as a .to() that raised leaves it
    e.model.fail_next_to = True       # ... and the repair fails too
    e._ensure_on_device("start of run")
    assert e.device == "cpu", "expected the visible CPU fallback"
    assert e.model.predictor is parked, "the predictor was dropped"


def _fake_report_engine(calls):
    """An engine whose offload is a log line, for the _run_case_report tests.

    The stubs model the real contract, not just the call: the release RAISES
    `_offloaded` (Phase 9.4.3 moved that assignment to before its first move, so
    the flag means "a release was attempted"), and the restore lowers it. That
    matters because `_run_case_report` now gates the restore on the flag — a
    stub that skipped it would make these tests assert against a state the real
    code never produces.
    """
    def release():
        calls.append("release")
        e._offloaded = True

    def restore():
        calls.append("restore")
        e._offloaded = False

    e = TrackingEngine.__new__(TrackingEngine)
    e.device = "cuda"
    e._offloaded = False
    e._gpu_lock = threading.Lock()
    e._pending_case_reports = []
    e._cancel = CancelToken()
    e._run_handles = None
    e._release_detection_gpu_memory = release
    e._restore_detection_gpu_memory = restore
    return e


def test_vlm_is_unloaded_before_the_detection_stack_is_restored():
    """_run_case_report must unload the VLM in its finally — including when
    analyze_incident raised — and unload before restoring, or the restore
    allocates while the VLM's weights are still resident."""
    import engine.tracker as tracker_mod

    calls = []

    class FakeVLM:
        def __init__(self, **kw):
            pass

        def analyze_incident(self, **kw):
            calls.append("analyze")
            raise RuntimeError("VLM blew up")

        def unload_model(self):
            calls.append("unload")

    e = _fake_report_engine(calls)

    real_vlm = tracker_mod.VLMAnalyzer
    tracker_mod.VLMAnalyzer = FakeVLM
    try:
        raised = False
        try:
            e._run_case_report(captures=[{}], full_json={"video_info": {}},
                               event_log=[], peak_snapshots=[],
                               vlm_backend="Qwen3-VL-2B (local)", vlm_api_key="",
                               analytics_result=None)
        except RuntimeError:
            raised = True
    finally:
        tracker_mod.VLMAnalyzer = real_vlm

    assert raised, "the original error must not be swallowed here"
    assert calls == ["release", "analyze", "unload", "restore"], calls


def test_constructor_failure_does_not_break_the_finally():
    """VLMAnalyzer(...) itself can raise (a missing transformers version);
    the finally must not trip over an unbound `vlm`."""
    import engine.tracker as tracker_mod

    calls = []

    def boom(**kw):
        raise RuntimeError("Qwen3-VL requires transformers>=4.57.0")

    e = _fake_report_engine(calls)

    real_vlm = tracker_mod.VLMAnalyzer
    tracker_mod.VLMAnalyzer = boom
    try:
        try:
            e._run_case_report(captures=[{}], full_json={"video_info": {}},
                               event_log=[], peak_snapshots=[],
                               vlm_backend="Qwen3-VL-2B (local)", vlm_api_key="",
                               analytics_result=None)
        except RuntimeError as err:
            assert "transformers" in str(err)
        else:
            raise AssertionError("expected the constructor error")
    finally:
        tracker_mod.VLMAnalyzer = real_vlm

    assert calls == ["release", "restore"], calls


def test_buffers_count_as_drift_not_just_parameters():
    """A module whose parameters all moved but whose BatchNorm running_mean did
    not raises the same conv2d device error. Module._apply() moves children,
    then parameters, then buffers, so this is the tail of a partial move."""
    m = torch.nn.BatchNorm2d(4)
    assert TrackingEngine._param_device(m) == "cpu"
    if not HAS_CUDA:
        print("  (partly skipped: no CUDA)")
        return
    m.running_mean = m.running_mean.to("cuda")
    assert TrackingEngine._param_device(m) == "mixed", (
        "a stranded buffer read as clean — the guard would never repair it")


def test_real_models_report_one_device_not_mixed():
    """The buffer scan must not invent phantom drift on the real stack: every
    buffer there has to track .to(). If this fails, _ensure_on_device() logs a
    warning and does a pointless repair on EVERY run."""
    if not HAS_CUDA:
        print("  (skipped: no CUDA)")
        return
    import os
    from engine.config import (MODEL_PATH, POSE_MODEL_PATH,
                               QUALITY_WEIGHT_PATH, FILL_WEIGHT_PATH)
    from engine.classifier import CartClassifier
    from ultralytics import YOLO

    missing = [p for p in (MODEL_PATH, POSE_MODEL_PATH, QUALITY_WEIGHT_PATH,
                           FILL_WEIGHT_PATH) if not os.path.exists(p)]
    if missing:
        print(f"  (skipped: missing weights {missing})")
        return

    cls = CartClassifier(device="cuda")
    cls.load_quality(QUALITY_WEIGHT_PATH)
    cls.load_fill(FILL_WEIGHT_PATH)
    for name, mod in (("detector", YOLO(MODEL_PATH).to("cuda")),
                      ("pose", YOLO(POSE_MODEL_PATH).to("cuda")),
                      ("quality", cls._quality_model),
                      ("fill", cls._fill_model)):
        assert TrackingEngine._param_device(mod) == "cuda", (
            f"{name} reads as {TrackingEngine._param_device(mod)} while clean "
            f"on cuda — some buffer of it does not follow .to()")


def test_a_parked_predictor_is_reattached_even_without_drift():
    """A restore that raised after the detector's own move already succeeded
    leaves the drift check clean and `predictor` None — and Model.track() reacts
    to a missing predictor by registering a SECOND tracking callback, which is
    the silent detection-degradation regression. _ensure_on_device() is the last
    place that can put it back."""
    e = _engine("cpu")
    parked = FakePredictor()
    e._held_predictor = parked
    e.model.predictor = None
    assert e._ensure_on_device("t") == [], "no drift expected here"
    assert e.model.predictor is parked, "the parked predictor was dropped"
    assert e._held_predictor is None


def test_a_parked_predictor_is_left_alone_while_offloaded():
    """While the stack is deliberately on the CPU for a local-VLM pass the
    predictor is parked on purpose; re-attaching there is how Model.track()
    would run against CPU weights mid-VLM."""
    e = _engine("cuda")
    parked = FakePredictor()
    e._offloaded = True
    e._held_predictor = parked
    e.model.predictor = None
    assert e._ensure_on_device("start of run") == []
    assert e.model.predictor is None
    assert e._held_predictor is parked


def test_an_unload_that_raises_still_restores_the_detection_stack():
    """The regression Phase 5b introduced: unload_model() raising inside the
    finally skipped the restore, so _offloaded stayed True for the life of the
    process — and _ensure_on_device() early-returns on that flag, which turns
    the start-of-run guard into a permanent no-op and makes the device-mismatch
    error permanent."""
    import engine.tracker as tracker_mod

    calls = []

    class FakeVLM:
        def __init__(self, **kw):
            pass

        def analyze_incident(self, **kw):
            calls.append("analyze")
            raise RuntimeError("VLM blew up")

        def unload_model(self):
            calls.append("unload")
            raise RuntimeError("unload blew up too")

    e = _fake_report_engine(calls)

    real_vlm = tracker_mod.VLMAnalyzer
    tracker_mod.VLMAnalyzer = FakeVLM
    try:
        try:
            e._run_case_report(captures=[{}], full_json={"video_info": {}},
                               event_log=[], peak_snapshots=[],
                               vlm_backend="Qwen3-VL-2B (local)", vlm_api_key="",
                               analytics_result=None)
        except RuntimeError as err:
            assert "VLM blew up" in str(err), (
                f"the unload error masked the real one: {err}")
        else:
            raise AssertionError("expected the analysis error")
    finally:
        tracker_mod.VLMAnalyzer = real_vlm

    assert calls == ["release", "analyze", "unload", "restore"], calls


def test_offloaded_is_cleared_by_a_restore_whose_moves_all_failed():
    """Phase 3 swallows the move failure; the flag must still come down or the
    guard stays disabled."""
    e = _engine("cuda" if HAS_CUDA else "cpu")
    if not HAS_CUDA:
        print("  (skipped: no CUDA — the offload path early-returns on cpu)")
        return
    e._offloaded = True
    e.model.fail_next_to = True
    e._restore_detection_gpu_memory()
    assert e._offloaded is False, "the guard is now a permanent no-op"


def test_trackers_are_reset_between_runs():
    """Stale BoTSORT state across runs is a wrong-output bug: measured on the
    golden clip, run 2 in the same process lost a cart on frame 116 until this
    reset landed."""
    e = _engine("cpu")
    t = FakeTracker()
    e.model.predictor = FakePredictor([t])
    e._reset_trackers()
    assert t.resets == 1


def test_trackers_are_reset_through_a_parked_predictor():
    """While the stack is offloaded for a local-VLM pass, .to("cpu") has already
    nulled self.model.predictor — so an overlapping run would find nothing to
    reset and inherit the previous run's tracks."""
    e = _engine("cuda")
    t = FakeTracker()
    e._offloaded = True
    e._held_predictor = FakePredictor([t])
    e.model.predictor = None
    e._reset_trackers()
    assert t.resets == 1
    assert e.model.predictor is None, "must not un-park the predictor"
    assert e._held_predictor is not None


def test_reset_trackers_survives_a_cold_engine_and_an_old_ultralytics():
    """First run has no predictor at all; an ultralytics without
    BYTETracker.reset() must degrade to stale state, not abort the run."""
    e = _engine("cpu")
    e.model.predictor = None
    e._held_predictor = None
    e._reset_trackers()                      # no predictor at all: no-op

    e.model.predictor = object()             # predictor with no .trackers
    e._reset_trackers()

    t = FakeTracker()
    t.fail = True
    e.model.predictor = FakePredictor([t])
    e._reset_trackers()                      # raises inside; must be swallowed
    assert t.resets == 0


# ----------------------------------------------------------------------
# Phase 9 — two Gradio events, one engine, one GPU.
# See docs/device_drift_fix_plan.md section 9.
# ----------------------------------------------------------------------

def test_a_run_waits_for_a_case_report_instead_of_using_parked_weights():
    """The reported crash. `process_video` and `finalize_case_report` are
    separate Gradio dependencies over one engine; a local-VLM report parks the
    detection stack on the CPU, and a run that started during that window
    classified CUDA tensors against CPU weights:

        RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                      (torch.FloatTensor) should be the same

    Asserted on the lock directly rather than through a real offload:
    `_release_detection_gpu_memory()` early-returns on a non-cuda device, so an
    overlap test driven through the real path would pass on a CPU-only box
    without exercising anything.
    """
    e = _engine("cuda" if HAS_CUDA else "cpu")
    order = []
    report_started = threading.Event()
    let_report_finish = threading.Event()

    def fake_pipeline(*a, **kw):
        # Stands in for the whole frame loop. The assertion that matters is
        # that nothing here ever observes a parked stack.
        order.append("run")
        assert not e._offloaded, (
            "the frame loop started while the detection stack was parked on "
            "the CPU — this is the device-mismatch crash")
        return "run-result"

    def fake_report():
        e._offloaded = True
        order.append("report-start")
        report_started.set()
        let_report_finish.wait(5)
        order.append("report-end")
        e._offloaded = False
        return ("html", None)

    e._process_video = fake_pipeline
    e._finalize_case_report_locked = fake_report

    report = threading.Thread(target=e.finalize_case_report)
    report.start()
    assert report_started.wait(5), "the fake report never started"

    run = threading.Thread(target=lambda: order.append(e.process_video("clip")))
    run.start()
    # The run must be BLOCKED, not merely slow: give it room to misbehave.
    time.sleep(0.2)
    assert order == ["report-start"], f"the run did not wait: {order}"

    let_report_finish.set()
    run.join(5)
    report.join(5)
    assert not run.is_alive() and not report.is_alive(), "a thread hung"
    assert order == ["report-start", "report-end", "run", "run-result"], order


def test_each_finalize_consumes_the_payload_of_its_own_run():
    """Phase 9.4.2. `_pending_case_report` used to be one slot cleared by
    `_reset()`, so run 2 starting before run 1's chained finalize event either
    served run 1 run 2's payload or served it nothing at all — silently, as
    "No case report for this clip" on a run that did have captures."""
    e = _engine("cpu")
    e._run_case_report = lambda **kw: (kw["full_json"]["tag"], None)
    for tag in ("run-1", "run-2"):
        e._pending_case_reports.append(
            {"captures": [{}], "full_json": {"tag": tag}, "event_log": [],
             "peak_snapshots": {}, "vlm_backend": "Claude (API)",
             "vlm_api_key": "", "analytics_result": None})

    seen = [e.finalize_case_report()[0], e.finalize_case_report()[0]]
    assert seen == ["run-1", "run-2"], seen
    # A third call has nothing left, and says so rather than repeating a report.
    assert e.finalize_case_report() == ("", None)


def test_orphaned_pending_reports_are_dropped_oldest_first():
    """The queue is bounded. A chained finalize event that never fires — a
    browser reload mid-run — otherwise leaves its payload behind forever, and
    each one pins a full JSON document plus the captured frames. Dropping the
    OLDEST is what keeps the newest runs paired with their own reports."""
    from engine.config import PENDING_CASE_REPORTS_MAX as CAP

    e = _engine("cpu")
    e._run_case_report = lambda **kw: (kw["full_json"]["tag"], None)

    def stash(tag):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            e._stash_pending_case_report(
                {"captures": [{}], "full_json": {"tag": tag}, "event_log": [],
                 "peak_snapshots": {}, "vlm_backend": "Claude (API)",
                 "vlm_api_key": "", "analytics_result": None})
        return buf.getvalue()

    logs = [stash(f"run-{i}") for i in range(CAP + 2)]
    assert len(e._pending_case_reports) == CAP, e._pending_case_reports
    kept = [p["full_json"]["tag"] for p in e._pending_case_reports]
    assert kept == [f"run-{i}" for i in range(2, CAP + 2)], kept
    assert sum("dropped an orphaned" in ln for ln in logs) == 2, logs
    # The next finalize gets the oldest SURVIVING payload, not a dropped one.
    assert e.finalize_case_report()[0] == "run-2"


def test_a_failed_report_is_still_consumed_and_not_served_twice():
    """The old slot was cleared before the try precisely so a failure could not
    be re-run; popping has to keep that property."""
    e = _engine("cpu")

    def boom(**kw):
        raise RuntimeError("induced report failure")

    e._run_case_report = boom
    e._pending_case_reports.append(
        {"captures": [{}], "full_json": {}, "event_log": [],
         "peak_snapshots": {}, "vlm_backend": "Claude (API)", "vlm_api_key": "",
         "analytics_result": None})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        html, path = e.finalize_case_report()
    assert path is None and "failed" in html, (html, path)
    assert e._pending_case_reports == [], "the failed payload was left queued"


def test_a_release_that_raises_is_still_owed_a_restore():
    """Phase 9.4.3. The release now runs INSIDE `_run_case_report`'s try, so it
    can raise partway — and `_offloaded` is set before its first move so the
    finally can still tell that a restore is owed."""
    import engine.tracker as tracker_mod

    calls = []
    e = _engine("cuda" if HAS_CUDA else "cpu")

    def release_that_raises():
        calls.append("release")
        e._offloaded = True                  # the real order: flag, then move
        raise RuntimeError("OOM before anything moved")

    e._release_detection_gpu_memory = release_that_raises
    e._restore_detection_gpu_memory = lambda: calls.append("restore")

    real_vlm = tracker_mod.VLMAnalyzer

    def no_vlm(**kw):
        raise AssertionError("the VLM must not be constructed after a failed "
                             "release")

    tracker_mod.VLMAnalyzer = no_vlm
    try:
        try:
            e._run_case_report(captures=[{}], full_json={"video_info": {}},
                               event_log=[], peak_snapshots=[],
                               vlm_backend="Qwen3-VL-2B (local)", vlm_api_key="",
                               analytics_result=None)
        except RuntimeError as err:
            assert "OOM before anything moved" in str(err), err
        else:
            raise AssertionError("expected the release error")
    finally:
        tracker_mod.VLMAnalyzer = real_vlm

    assert calls == ["release", "restore"], calls


def test_the_offload_flag_goes_up_before_the_first_move():
    """The real `_release_detection_gpu_memory`, not a stub. The flag is set
    before the first `.to("cpu")` so it means "a release was ATTEMPTED": with it
    set at the end instead, a release that OOMs partway leaves the stack half on
    the CPU with the flag down, `_run_case_report`'s finally skips the restore
    it is owed, and nothing puts the detector back."""
    e = _engine("cuda" if HAS_CUDA else "cpu")
    if not HAS_CUDA:
        print("  (skipped: no CUDA — the release early-returns on cpu)")
        return
    e.model.fail_next_to = True               # OOM on the very first move
    try:
        e._release_detection_gpu_memory()
    except torch.cuda.OutOfMemoryError:
        pass
    else:
        raise AssertionError("expected the induced OOM")
    assert e._offloaded is True, (
        "a release that raised partway must still leave a restore owed")
    assert e._held_predictor is not None, "the predictor was not parked"


def test_a_restore_is_skipped_when_no_release_ever_ran():
    """The other half of the gate, and the reason it exists. An unconditional
    restore is NOT a no-op when nothing was parked: it calls
    `self.model.to(self.device)`, and `Model._apply()` nulls `predictor` on
    every `.to()` including a same-device one. With no release there is no
    parked predictor to put back, `_ensure_on_device()` sees no drift, and the
    next `.track()` appends a second tracking callback — the silent
    detection-degradation regression."""
    import engine.tracker as tracker_mod

    calls = []
    e = _engine("cuda" if HAS_CUDA else "cpu")
    parked = e.model.predictor
    assert parked is not None

    # A device-mismatched release is a no-op that leaves the flag DOWN — the
    # same state a release that raised on its very first line leaves behind.
    e._release_detection_gpu_memory = lambda: calls.append("release-noop")
    e._restore_detection_gpu_memory = lambda: calls.append("restore")

    class FakeVLM:
        def __init__(self, **kw):
            pass

        def analyze_incident(self, **kw):
            return {"summary": "ok"}

        def unload_model(self):
            calls.append("unload")

    real_vlm = tracker_mod.VLMAnalyzer
    real_build = tracker_mod.build_case_report_html
    tracker_mod.VLMAnalyzer = FakeVLM
    tracker_mod.build_case_report_html = lambda *a, **kw: ("<p>ok</p>", "<html/>")
    try:
        e._run_case_report(captures=[{}], full_json={"video_info": {}},
                           event_log=[], peak_snapshots=[],
                           vlm_backend="Qwen3-VL-2B (local)", vlm_api_key="",
                           analytics_result=None)
    finally:
        tracker_mod.VLMAnalyzer = real_vlm
        tracker_mod.build_case_report_html = real_build

    assert calls == ["release-noop", "unload"], calls
    assert e.model.predictor is parked, "the predictor was dropped"


def test_a_stranded_offload_flag_is_repaired_not_obeyed():
    """Phase 9.4.5. With `_gpu_lock` held, `_offloaded` True at the start of a
    run cannot mean "parked right now" — it means a restore was skipped and the
    flag was stranded. Left set, `_ensure_on_device()` early-returns forever and
    the device-mismatch error becomes permanent for the process."""
    e = _engine("cuda" if HAS_CUDA else "cpu")
    if not HAS_CUDA:
        print("  (skipped: no CUDA — the drift repair is a no-op on cpu)")
        return
    e._offloaded = True
    e._classifier._quality_model.to("cpu")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # The two steps _process_video() takes before touching a frame.
        if e._offloaded:
            e._offloaded = False
        drifted = e._ensure_on_device(context="start of run")
    assert e._offloaded is False
    assert "quality" in drifted, drifted
    assert TrackingEngine._param_device(e._classifier._quality_model) == "cuda"


def test_the_stranded_flag_repair_is_wired_into_the_run():
    """The repair above is only worth anything if `_process_video` actually does
    it before the guard runs. Asserted on the source because reaching that line
    for real needs a clip, weights and a GPU."""
    import inspect
    src = inspect.getsource(TrackingEngine._process_video)
    head = src[:src.index('self._ensure_on_device(context="start of run")')]
    assert "self._offloaded = False" in head, (
        "the stranded-flag repair must run BEFORE the start-of-run guard, or "
        "the guard early-returns on the flag and repairs nothing")


def test_classifier_inputs_follow_the_weights_not_the_requested_device():
    """Phase 9.4.4, defence in depth: with the quality model parked on the CPU
    and `classifier.device` still "cuda", `classify()` must produce a result
    rather than raising the conv2d device error. Derived per model, because a
    restore that OOM'd partway leaves quality and fill on different devices —
    one answer for both would move the crash from stage 1 to stage 2."""
    from engine.classifier import CartClassifier

    assert CartClassifier._weights_device(
        torch.nn.Conv2d(3, 4, 3), "cuda") == torch.device("cpu")
    assert CartClassifier._weights_device(None, "cuda") == "cuda"

    if not HAS_CUDA:
        print("  (skipped the mixed and end-to-end cases: no CUDA)")
        return

    # A module split across devices has no right answer; CPU is the one that
    # cannot raise, and the engine repairs the module at the start of the run.
    mixed = torch.nn.Conv2d(3, 4, 3).to("cuda")
    mixed.bias = torch.nn.Parameter(mixed.bias.detach().cpu())
    assert CartClassifier._weights_device(mixed, "cuda") == torch.device("cpu")

    import numpy as np

    class Quality(torch.nn.Module):
        """Two-class head, like the real quality model."""

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            assert x.device == self.w.device, (
                f"input on {x.device}, weights on {self.w.device} — this is "
                f"the reported RuntimeError")
            return torch.zeros(x.shape[0], 2, device=self.w.device) + self.w

    class Fill(torch.nn.Module):
        """(fill_logits, bag_logits), like the real fill model."""

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            assert x.device == self.w.device, (
                f"input on {x.device}, weights on {self.w.device} — stage 2 "
                f"needs its OWN device, not stage 1's")
            return (torch.zeros(x.shape[0], 3, device=self.w.device) + self.w,
                    torch.zeros(x.shape[0], 2, device=self.w.device) + self.w)

    c = CartClassifier("cuda")            # asked for cuda...
    c._quality_model = Quality()          # ...but parked on the cpu,
    c._fill_model = Fill().to("cuda")     # and disagreeing with stage 1.
    c._quality_threshold = 0.0
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    out = c.classify(frame, (100, 100, 300, 400))
    assert out["quality"] == "valid_cart", out
    res = c.classify_batch(frame, [(100, 100, 300, 400)], ["Cart 1"])
    assert res["Cart 1"]["quality"] == "valid_cart", res


def test_the_pose_predictor_is_not_handed_a_device_the_weights_are_not_on():
    """Phase 9.4.4, third site. `predict(device="cuda")` while the pose module
    is parked on the CPU makes ultralytics re-initialise AutoBackend on the card
    and take VRAM back from a generating VLM. Passing the module's real device,
    NOT dropping the argument: `predict()` with no device reaches
    `select_device("")`, which auto-selects the first available GPU."""
    import inspect
    src = inspect.getsource(TrackingEngine._process_video)
    call = src[src.index("pres = self._pose_model.predict("):]
    call = call[:call.index(")")]
    assert "device=self.device" not in call, (
        "the pose predict still pins self.device: " + call)
    assert "device=" in call, (
        "dropping device= does not make predict() follow the module — "
        "select_device('') auto-selects the first GPU: " + call)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed (cuda={HAS_CUDA})")
