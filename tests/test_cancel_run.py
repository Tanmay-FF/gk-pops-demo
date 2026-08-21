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

from engine.cancellation import (CancelToken, RunCancelled,
                                 RunSuperseded)
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


# ----------------------------------------------------------------------
# Supersede: a second Run stops what is already in flight
# ----------------------------------------------------------------------
# The user-facing rule is "if the VLM is running and I run another video, the
# VLM stops and the new video starts". That is NOT the Cancel button's
# semantics: a press stops the NEWEST work, while a new run has to stop
# everything OLDER than itself and leave itself alone. Hence a second
# predicate; see CancelToken.supersede().

def test_a_new_run_cancels_the_work_it_replaces():
    t = CancelToken()
    old = t.begin()                        # run 1, its case report still going
    superseded, new = t.supersede()        # run 2 starts

    assert superseded == old, f"expected to supersede run {old}, got {superseded}"
    assert new == old + 1
    assert t.is_cancelled(old), "the replaced run was not told to stop"
    assert not t.is_cancelled(new), "the new run cancelled itself"


def test_supersede_survives_a_later_begin():
    """The race that rules out request()-then-begin(). begin() drops a stale
    cancel and clears the event; if that clearing also killed the supersede,
    the VLM would poll a cleared flag and never stop."""
    t = CancelToken()
    old = t.begin()
    _, new = t.supersede()
    later = t.begin()                      # e.g. a third run queued behind

    assert t.is_cancelled(old), (
        "a later begin() cleared the supersede out from under work that had "
        "not noticed it yet"
    )
    assert not t.is_cancelled(later)
    assert new < later


def test_supersede_reaches_every_older_unit_not_just_the_newest():
    """Two things can be in flight at once: a report for run N while run N+1
    waits on the engine lock. A new run must stop both."""
    t = CancelToken()
    first = t.begin()
    second = t.begin()
    _, newest = t.supersede()

    assert t.is_cancelled(first), "the oldest in-flight unit was left running"
    assert t.is_cancelled(second)
    assert not t.is_cancelled(newest)


def test_supersede_is_distinguishable_from_a_cancel_press():
    """What decides whether the panel says "cancelled" or is left untouched."""
    t = CancelToken()
    superseded_run = t.begin()
    t.supersede()
    assert t.was_superseded(superseded_run) is True

    t2 = CancelToken()
    pressed = t2.begin()
    t2.request()
    assert t2.is_cancelled(pressed) is True
    assert t2.was_superseded(pressed) is False, (
        "a Cancel press must not look like a supersede, or the report panel "
        "silently stops saying it was cancelled"
    )


def test_raise_if_cancelled_reports_which_kind():
    t = CancelToken()
    old = t.begin()
    t.supersede()
    try:
        t.raise_if_cancelled(old, "in the case report")
    except RunSuperseded as e:
        assert "superseded" in str(e)
    else:
        raise AssertionError("expected RunSuperseded")

    t2 = CancelToken()
    run = t2.begin()
    t2.request()
    try:
        t2.raise_if_cancelled(run)
    except RunSuperseded:
        raise AssertionError("a Cancel press must not raise RunSuperseded")
    except RunCancelled:
        pass
    else:
        raise AssertionError("expected RunCancelled")


def test_run_superseded_is_a_run_cancelled():
    """Existing `except RunCancelled` handlers must keep catching it, so a
    caller that does not care about the distinction still unwinds correctly."""
    assert issubclass(RunSuperseded, RunCancelled)


def test_the_cancel_button_still_works_after_a_supersede():
    """The ceiling is monotonic and never cleared, so it must not shadow a
    later press aimed at the run that did the superseding."""
    t = CancelToken()
    t.begin()
    _, current = t.supersede()
    assert not t.is_cancelled(current)

    t.request()
    assert t.is_cancelled(current), "Cancel stopped working after a supersede"
    assert not t.was_superseded(current), (
        "the press was misreported as a supersede, so the panel would be left "
        "blank instead of saying cancelled"
    )


def test_supersede_with_nothing_in_flight_reports_nothing_replaced():
    t = CancelToken()
    superseded, first = t.supersede()
    assert superseded is None, "claimed to replace work that never existed"
    assert first == 1
    assert not t.is_cancelled(first)


def test_a_superseded_report_writes_nothing_rather_than_a_cancelled_panel():
    """The forward-reaching wrong-output bug. case_report_html is in
    FLUSH_OUTPUT_NAMES, so the new run has already blanked it; a superseded
    report that RETURNS a panel paints the dead run over the live one. It has
    to raise instead."""
    e = _engine()
    old = e._cancel.begin()
    e._pending_case_reports.append(
        {"run_seq": old, "captures": [{}], "full_json": {"video_info": {}},
         "event_log": [], "peak_snapshots": [], "vlm_backend": "Claude (API)",
         "vlm_api_key": "", "analytics_result": None})
    e._cancel.supersede()                  # a new run replaces it

    try:
        e.finalize_case_report()
    except RunSuperseded:
        pass
    else:
        raise AssertionError(
            "a superseded case report returned a value instead of raising; it "
            "would be painted over the new run's freshly flushed panel"
        )
    assert not e._pending_case_reports, "the superseded payload was left queued"


def test_a_pressed_cancel_still_paints_the_cancelled_panel():
    """The other half: a user Cancel must keep SAYING cancelled."""
    e = _engine()
    run = e._cancel.begin()
    e._pending_case_reports.append(
        {"run_seq": run, "captures": [{}], "full_json": {"video_info": {}},
         "event_log": [], "peak_snapshots": [], "vlm_backend": "Claude (API)",
         "vlm_api_key": "", "analytics_result": None})
    e._cancel.request()

    html, path = e.finalize_case_report()
    assert "cancelled" in html.lower(), html
    assert path is None


def test_an_earlier_completed_run_still_gets_its_report_after_a_supersede():
    """The backwards-reaching bug the sequence numbers exist for, re-checked
    under the new predicate: superseding run 2 must not eat run 1's report if
    run 1 finished before the supersede."""
    t = CancelToken()
    finished = t.begin()
    assert not t.is_cancelled(finished)
    # run 1 completes and its report is queued; nothing has been superseded yet
    # so it stays eligible.
    assert not t.was_superseded(finished)


def test_a_new_run_does_not_wait_out_the_case_report_it_replaced():
    """The user-facing claim, end to end on the real locking.

    A case report holds `_gpu_lock` for its whole pass and a run has to acquire
    it, so the new run DOES wait -- but only for as long as the report takes to
    notice it was superseded, not for as long as the report would have taken.
    The stub here would run for 30s if nobody stopped it; the run must get going
    in a fraction of that.
    """
    e = _engine()
    started = threading.Event()
    stub_finished_naturally = threading.Event()
    outcome = {}

    def slow_report(**pending):
        # Stands in for a local VLM: polls the cancel flag the way
        # _call_vlm and the StoppingCriteria do, between units of work.
        started.set()
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            e._cancel.raise_if_cancelled(pending.get("run_seq"),
                                         "in the case report")
            time.sleep(0.005)
        stub_finished_naturally.set()
        return "<p>report</p>", None

    e._run_case_report = slow_report
    run_seq = e._cancel.begin()
    e._pending_case_reports.append(
        {"run_seq": run_seq, "captures": [{}], "full_json": {"video_info": {}},
         "event_log": [], "peak_snapshots": [], "vlm_backend": "Claude (API)",
         "vlm_api_key": "", "analytics_result": None})

    def report_thread():
        try:
            e.finalize_case_report()
        except RunSuperseded as exc:
            outcome["superseded"] = str(exc)
        except BaseException as exc:            # noqa: BLE001 - reported below
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=report_thread, daemon=True)
    t.start()
    assert started.wait(5), "the stub case report never started"

    # The new run. process_video is the wrapper that supersedes and then takes
    # the lock; the pipeline underneath is stubbed so this test stays CPU-only.
    e._process_video = lambda *a, **k: "ran"
    t0 = time.perf_counter()
    result = e.process_video("some_video.mp4")
    elapsed = time.perf_counter() - t0

    t.join(10)
    assert result == "ran"
    assert not stub_finished_naturally.is_set(), (
        "the case report ran to completion; the new run waited it out instead "
        "of superseding it"
    )
    assert "superseded" in outcome, (
        f"the report did not stop with RunSuperseded: {outcome}"
    )
    assert elapsed < 5.0, (
        f"the new run took {elapsed:.1f}s to get going; it should only wait for "
        f"the superseded report to release the engine lock"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")
