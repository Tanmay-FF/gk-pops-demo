"""The Cancel button: a run that is stopped must actually stop, and must not
poison the next one.

Gradio's own `cancels=` cannot do this job — its docstring says a function
already running "will be allowed to finish" — so cancellation is cooperative:
the frame loop tests a flag every frame and the local VLM tests it per generated
token. These tests pin the parts that can go wrong quietly:

  * a cancel that reaches across into an EARLIER run's pending case report,
    which would be the same class of silent wrong-output bug as
    docs/device_drift_fix_plan.md section 9.4.2;
  * a cancelled run leaving tracker state behind, so the NEXT run disagrees
    with the baseline (test_cancel_run_e2e.py drives that one on real pixels);
  * a cancel arriving while the run is queued behind a case report, which is
    the longest window in the whole chain and the one a naive
    "check at the top of the loop" misses.

CPU-only safe: nothing here needs CUDA.
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

from engine.cancellation import CancelToken, RunCancelled
from engine.tracker import TrackingEngine


def _engine():
    """Enough of a TrackingEngine for the wrapper and the report queue.

    `__new__`, so `__init__` never runs: the pipeline itself is stubbed out in
    every test here, and building a real engine would load weights.
    """
    e = TrackingEngine.__new__(TrackingEngine)
    e.device = "cpu"
    e._gpu_lock = threading.Lock()
    e._cancel = CancelToken()
    e._pending_case_reports = []
    e._offloaded = False
    return e


# ----------------------------------------------------------------------
# CancelToken
# ----------------------------------------------------------------------

def test_a_cancel_stops_the_run_it_was_aimed_at():
    t = CancelToken()
    seq = t.begin()
    assert not t.is_cancelled(seq)
    assert t.request() == seq
    assert t.is_cancelled(seq)
    try:
        t.raise_if_cancelled(seq, "in the frame loop")
    except RunCancelled as e:
        assert "in the frame loop" in str(e), e
    else:
        raise AssertionError("expected RunCancelled")


def test_a_cancel_does_not_reach_backwards_into_earlier_work():
    """The bug this design exists to prevent. Run 1 finishes and its case report
    is still pending; run 2 starts and the user cancels IT. Run 1's report is
    not the thing that was cancelled and must survive."""
    t = CancelToken()
    run1 = t.begin()
    run2 = t.begin()
    t.request()
    assert t.is_cancelled(run2), "the cancelled run kept going"
    assert not t.is_cancelled(run1), (
        "cancelling run 2 also discarded run 1's pending case report")


def test_a_cancel_with_nothing_in_flight_clears_itself():
    """Pressing Cancel when nothing is running must not arm a trap for the next
    run — the flag would otherwise kill it before it read a frame."""
    t = CancelToken()
    assert t.request() is None, "nothing has begun; there is nothing to target"
    stale = t.begin()
    assert not t.is_cancelled(stale)

    # And the same after a real run: cancel late, then start again.
    t.request()
    nxt = t.begin()
    assert not t.is_cancelled(nxt), "a stale cancel killed the next run"


def test_untracked_work_is_never_cancelled():
    """`run_seq=None` is what a direct _process_video call (tests, headless
    callers) passes. It must not be affected by a token it never joined."""
    t = CancelToken()
    t.begin()
    t.request()
    assert not t.is_cancelled(None)
    t.raise_if_cancelled(None)          # must not raise


def test_the_token_is_safe_to_use_from_two_threads():
    """`request()` is called from the Cancel button's own Gradio event while the
    run holds every other lock, so it must never block on the engine."""
    t = CancelToken()
    seqs = []
    barrier = threading.Barrier(4)

    def begin_many():
        barrier.wait()
        for _ in range(200):
            seqs.append(t.begin())

    threads = [threading.Thread(target=begin_many) for _ in range(3)]
    for th in threads:
        th.start()
    barrier.wait()
    t.request()
    for th in threads:
        th.join(10)
        assert not th.is_alive(), "begin() deadlocked"
    assert len(seqs) == 600
    assert len(set(seqs)) == 600, "two units of work got the same sequence"


# ----------------------------------------------------------------------
# The engine wrapper
# ----------------------------------------------------------------------

def test_a_cancel_while_the_run_waits_for_a_case_report_is_honoured():
    """The longest window in the chain, and the one a loop-only check misses: a
    run can sit on `_gpu_lock` for the whole of a local-VLM pass without reading
    a single frame. The sequence is claimed BEFORE the wait for exactly this."""
    e = _engine()
    started = []
    e._process_video = lambda *a, **kw: started.append("ran") or "result"

    e._gpu_lock.acquire()                       # stand in for a case report
    try:
        run = threading.Thread(target=lambda: started.append(
            _run_and_catch(e, "clip")))
        run.start()
        time.sleep(0.2)
        assert started == [], f"the run did not wait: {started}"
        e.request_cancel()
        time.sleep(0.05)
    finally:
        e._gpu_lock.release()
    run.join(5)
    assert not run.is_alive(), "the run hung"
    assert started == ["cancelled"], (
        f"a cancel during the queue wait did not stop the run: {started}")


def _run_and_catch(engine_obj, *args):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            engine_obj.process_video(*args)
    except RunCancelled:
        return "cancelled"
    return "completed"


def test_a_run_that_was_not_cancelled_still_runs():
    """The obvious regression: the checkpoint must not fire on its own."""
    e = _engine()
    e._process_video = lambda *a, **kw: "result"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert e.process_video("clip") == "result"


def test_request_cancel_reports_when_there_is_nothing_to_cancel():
    e = _engine()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert e.request_cancel() is None
    assert "nothing to cancel" in buf.getvalue(), buf.getvalue()


def test_the_run_sequence_reaches_the_pipeline():
    """`run_seq` has to arrive at `_process_video`, or the per-frame check tests
    a sequence nobody ever cancels."""
    e = _engine()
    seen = {}
    e._process_video = lambda *a, **kw: seen.update(kw) or "result"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e.process_video("clip")
    assert seen.get("run_seq") == 1, seen


# ----------------------------------------------------------------------
# The case report
# ----------------------------------------------------------------------

def _payload(tag, run_seq):
    return {"captures": [{}], "full_json": {"tag": tag}, "event_log": [],
            "peak_snapshots": {}, "vlm_backend": "Claude (API)",
            "vlm_api_key": "", "analytics_result": None, "run_seq": run_seq}


def test_a_cancelled_run_does_not_take_an_earlier_report_with_it():
    """End to end through the engine, not just the token: run 1's report is
    queued, run 2 is cancelled, and run 1's report must still be generated."""
    e = _engine()
    e._run_case_report = lambda **kw: (kw["full_json"]["tag"], None)
    run1 = e._cancel.begin()
    e._pending_case_reports.append(_payload("run-1", run1))
    run2 = e._cancel.begin()
    e._cancel.request()                          # cancels run 2

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        html, path = e.finalize_case_report()
    assert html == "run-1", (
        f"cancelling run {run2} discarded run {run1}'s report: {html}")


def test_a_cancelled_runs_own_report_is_dropped():
    e = _engine()
    e._run_case_report = lambda **kw: (_ for _ in ()).throw(
        AssertionError("the VLM must not be started for a cancelled run"))
    seq = e._cancel.begin()
    e._pending_case_reports.append(_payload("run-1", seq))
    e._cancel.request()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        html, path = e.finalize_case_report()
    assert "cancelled" in html.lower(), html
    assert path is None
    assert "[CANCEL] dropped the case report" in buf.getvalue(), buf.getvalue()


# ----------------------------------------------------------------------
# The VLM
# ----------------------------------------------------------------------

def test_the_vlm_checks_the_flag_before_every_model_call():
    """One check at the single dispatch point covers every backend, including
    the Claude API path where an in-flight HTTPS request cannot be interrupted
    at all — so the granularity there is one frame, not one token."""
    from engine.vlm_analyzer import VLMAnalyzer

    cancelled = {"v": False}
    v = VLMAnalyzer(backend="Claude (API)", should_cancel=lambda: cancelled["v"])
    v._call_claude = lambda *a, **kw: "described"
    assert v._call_vlm(None, "prompt", 10) == "described"

    cancelled["v"] = True
    try:
        v._call_vlm(None, "prompt", 10)
    except RunCancelled:
        pass
    else:
        raise AssertionError("a cancelled analyzer still called the model")


def test_the_stopping_criteria_matches_what_transformers_expects():
    """Verified against the installed transformers rather than assumed: the
    criteria protocol has moved between versions, and a criteria that silently
    never fires would leave a Cancel waiting out a whole summary generation."""
    from engine.vlm_analyzer import VLMAnalyzer

    cancelled = {"v": False}
    v = VLMAnalyzer(backend="Qwen3-VL-2B (local)",
                    should_cancel=lambda: cancelled["v"])
    criteria = v._cancel_criteria()
    ids = torch.zeros((1, 4), dtype=torch.long)
    scores = torch.zeros((1, 8))
    assert not bool(criteria(ids, scores).any()), "stopped before any cancel"
    cancelled["v"] = True
    assert bool(criteria(ids, scores).all()), "a cancel did not stop generation"


def test_an_analyzer_with_no_cancel_hook_behaves_as_before():
    """Every existing caller constructs VLMAnalyzer without should_cancel."""
    from engine.vlm_analyzer import VLMAnalyzer

    v = VLMAnalyzer()
    assert v._should_cancel() is False
    v._call_claude = lambda *a, **kw: "described"
    assert v._call_vlm(None, "prompt", 10) == "described"


def test_a_cancel_is_not_folded_into_the_report_as_an_error():
    """`analyze_incident` turns every failure into an inline "the VLM could not
    run" report, which is right for a failure and wrong for a cancel: the user
    asked for the work to stop, so it has to reach the caller."""
    from engine.vlm_analyzer import VLMAnalyzer

    v = VLMAnalyzer(backend="Claude (API)", should_cancel=lambda: True)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            # A HIGH score on purpose: the benign branch is deterministic and
            # calls no model at all, so it has nothing to cancel and would pass
            # this test without exercising anything.
            v.analyze_incident(
                captures=[{"frame_idx": 0, "timestamp": 0.0, "trigger": "t",
                           "image_bytes": b"", "cart_id": 1}],
                pops_data={"pops_summary": {"C1": {"max_score": 85}}},
                event_log=[], peak_snapshots={}, video_info={},
            )
    except RunCancelled:
        pass
    else:
        raise AssertionError("the cancel was swallowed into a degraded report")


# ----------------------------------------------------------------------
# The partial video
# ----------------------------------------------------------------------

def test_the_partial_video_of_a_cancelled_run_is_removed():
    """A cancelled run has already written frames to its AVI. Nothing consumes
    that file, and at ~1 MB/s of cancelled footage it is worth removing rather
    than leaving for the run-directory pruner."""
    import tempfile
    from engine.video_io import discard_run_output

    run_dir = tempfile.mkdtemp(prefix="pops_run_test_")
    avi = os.path.join(run_dir, "pops_demo_raw.avi")
    with open(avi, "wb") as f:
        f.write(b"partial")
    discard_run_output(avi)
    assert not os.path.exists(avi), "the partial video was left behind"
    assert not os.path.isdir(run_dir), "the empty run directory was left behind"

    # Best-effort, and must never raise: the caller is on its way out with a
    # RunCancelled and a tidy-up failure must not replace it.
    discard_run_output(avi)
    discard_run_output("")
    discard_run_output(os.path.join(run_dir, "gone.avi"))

    # A directory with other files in it is left alone — a tidy-up has no
    # business with a wide blast radius.
    run_dir2 = tempfile.mkdtemp(prefix="pops_run_test_")
    avi2 = os.path.join(run_dir2, "pops_demo_raw.avi")
    keep = os.path.join(run_dir2, "pops_demo_output.mp4")
    for path in (avi2, keep):
        with open(path, "wb") as f:
            f.write(b"x")
    discard_run_output(avi2)
    assert not os.path.exists(avi2)
    assert os.path.exists(keep), "the tidy-up removed a file it does not own"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")
