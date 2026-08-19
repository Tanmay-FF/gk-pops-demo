"""Device-drift guard and VLM unload ordering.

Covers the failure that produced, on every run after a local-VLM case report:
    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
    (torch.FloatTensor) should be the same

Runs on CPU-only machines: the CUDA-specific assertions are skipped there, the
ordering and mixed-device assertions are not.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

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


def _engine(device, *, detector_dev="cpu", quality_dev="cpu"):
    e = TrackingEngine.__new__(TrackingEngine)
    e.device = device
    e._pose_model = None
    e._held_predictor = None
    e._offloaded = False
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

    e = TrackingEngine.__new__(TrackingEngine)
    e.device = "cuda"
    e._release_detection_gpu_memory = lambda: calls.append("release")
    e._restore_detection_gpu_memory = lambda: calls.append("restore")

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

    e = TrackingEngine.__new__(TrackingEngine)
    e.device = "cuda"
    e._release_detection_gpu_memory = lambda: calls.append("release")
    e._restore_detection_gpu_memory = lambda: calls.append("restore")

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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed (cuda={HAS_CUDA})")
