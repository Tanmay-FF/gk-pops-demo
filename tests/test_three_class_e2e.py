"""End to end on the 3-class detector backend (docs/three_class_detector_plan.md).

On the primary golden clip, through the real engine:
  * the run completes and the tracking JSON says which backend produced it;
  * per-frame cart records carry the detector's call, pops_summary carries the
    per-cart read counts, and the JSON is strict (no NaN / Infinity, which
    json.dumps would otherwise emit for a NaN P(non-empty));
  * EMPTY reads really come from the detector (source == "detector");
  * a local-VLM style GPU offload round-trip (release -> restore) leaves the
    next run frame-for-frame identical — the property the device-drift work
    protects on the original path, re-proved for the DirectTrackingModel.

Run-against-run, not against a committed baseline: the 3-class baselines are
tests/test_golden_clip.py's job. Needs CUDA, the YOLOv5 repo and weights at the
config paths, and the golden clip; skips otherwise.
"""
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import torch  # noqa: E402

from test_golden_clip import Skip, load_baseline, find_clip, run_pipeline, compare  # noqa: E402

BACKEND = "yolov5_3class"


def _require_environment():
    from engine import config as C

    if not torch.cuda.is_available():
        raise Skip("no CUDA")
    for path in (C.V5_WEIGHTS_PATH, os.path.join(C.V5_REPO_PATH, "models", "common.py")):
        if not os.path.exists(path):
            raise Skip(f"missing {path}")
    baseline = load_baseline()
    return find_clip(baseline), dict(baseline["settings"], detector_backend=BACKEND)


def _strict_json(text):
    def refuse(token):
        raise AssertionError(f"tracking JSON contains {token}")
    return json.loads(text, parse_constant=refuse)


def _run(engine_obj, clip, settings):
    """One real run; returns (frames from the HUD spy, tracking JSON dict)."""
    import engine.tracker as T

    holder = {}
    real = T.TrackingEngine._process_video

    def keep_json(self, *a, **k):
        out = real(self, *a, **k)
        holder["json"] = out[2]
        return out

    T.TrackingEngine._process_video = keep_json
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            frames = run_pipeline(clip, settings, engine_obj=engine_obj)
    finally:
        T.TrackingEngine._process_video = real
    return frames, _strict_json(holder["json"])


def test_three_class_run_reports_detector_state_and_survives_offload():
    clip, settings = _require_environment()
    import engine.tracker as T

    engine_obj = T.TrackingEngine(detector_backend=BACKEND)
    assert engine_obj._uses_cart_state, "3-class backend did not report cart state"

    frames1, doc = _run(engine_obj, clip, settings)
    assert frames1, "run 1 captured no frames"
    assert doc["processing_info"]["detector_backend"] == BACKEND
    assert "detector_state" in doc["field_notes"]

    cart_objs = [c for f in doc["frames"].values() for c in f["carts"].values()]
    assert cart_objs, "no cart was ever detected on the golden clip"
    states = {c["detector"]["state"] for c in cart_objs}
    assert states <= {"empty", "non_empty", None}, states
    assert states & {"empty", "non_empty"}, "no frame carried a detector call"
    for c in cart_objs:
        p = c["detector"]["p_non_empty"]
        assert p is None or 0.0 <= p <= 1.0, p
        assert c["classification"]["source"] in ("detector", "classifier")

    summaries = list(doc["pops_summary"].values())
    assert summaries and all("detector_reads" in s for s in summaries)
    total_reads = sum(sum(s["detector_reads"].values()) for s in summaries)
    assert total_reads > 0, "no cart was ever read"
    empties = sum(s["detector_reads"].get("empty", 0) for s in summaries)
    if empties:
        assert any(c["classification"]["source"] == "detector" for c in cart_objs), (
            "EMPTY reads were counted but no frame shows a detector-sourced read")

    # GPU offload round-trip, as a local-VLM case report does it.
    engine_obj._release_detection_gpu_memory()
    assert engine_obj._param_device(engine_obj.model) == "cpu"
    engine_obj._restore_detection_gpu_memory()
    assert engine_obj._param_device(engine_obj.model) == "cuda"

    frames2, _ = _run(engine_obj, clip, settings)
    diffs = compare(frames2, {"frames": frames1})
    assert not diffs, (
        f"run 2 after the offload round-trip differs on {len(diffs)} frame(s):\n  "
        + "\n  ".join(diffs[:15]))


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
