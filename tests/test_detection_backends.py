"""engine/detection_backends.py and CartClassifier.classify_loaded().

Pins:
  * both cart classes are merged BEFORE NMS (one box per cart), persons are
    not merged into carts, and P(non-empty) survives YOLOv5's NMS exactly;
  * the tracking driver keeps ids across frames, carries each detection's
    P(non-empty) to the right track through the tracker's `idx`, reports an
    untracked frame as id None, and survives `predictor = None`;
  * importing YOLOv5 refuses a foreign top-level `utils` and restores the
    process-wide state it changes;
  * classify_loaded() can never answer empty, renormalises over the loaded
    classes, and leaves classify() untouched;
  * the trajectory-cache key changes with the backend.

No GPU and no detector weights needed. The NMS integration test needs the
YOLOv5 repository at config.V5_REPO_PATH and skips without it.
"""
import math
import os
import sys
import types
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine.tracker  # noqa: E402,F401 — installs the Retail tracker subclasses
from engine import config as C  # noqa: E402
from engine import detection_backends as DB  # noqa: E402


class Skip(Exception):
    pass


# ---------------------------------------------------------------------------
# Class merge + NMS
# ---------------------------------------------------------------------------
def _pred(rows):
    """rows: (cx, cy, w, h, obj, p_person, p_empty, p_non_empty)"""
    return torch.tensor([rows], dtype=torch.float32)


def test_merge_keeps_best_cart_score_and_share():
    m = DB.merge_cart_classes(_pred([(50, 50, 20, 20, 0.9, 0.1, 0.2, 0.6)]), 0, 1, 2)
    assert m.shape == (1, 1, 9)
    row = m[0, 0].tolist()
    assert math.isclose(row[6], 0.6, rel_tol=1e-6), "cart score must be max(p_empty, p_non_empty)"
    assert math.isclose(row[7], 0.75, rel_tol=1e-6), "share must be p_ne / (p_e + p_ne)"
    assert row[8] == 1.0


def test_merge_survives_zero_cart_probabilities():
    m = DB.merge_cart_classes(_pred([(5, 5, 2, 2, 0.9, 0.9, 0.0, 0.0)]), 0, 1, 2)
    assert torch.isfinite(m).all()


def test_merge_rejects_bad_indices_and_shapes():
    for args in ((_pred([(0,) * 8]), 0, 1, 9), (torch.zeros(3, 8), 0, 1, 2)):
        try:
            DB.merge_cart_classes(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args[1:]}")


def test_recover_share_undoes_objectness_scaling():
    obj = np.array([0.8, 0.3, 0.0])
    share = np.array([0.25, 1.0, 0.5])
    got = DB.recover_share(share * obj, obj)
    assert np.allclose(got[:2], [0.25, 1.0]) and math.isnan(got[2])


def _yolov5_nms():
    repo = os.path.abspath(C.V5_REPO_PATH)
    if not os.path.isfile(os.path.join(repo, "utils", "general.py")):
        raise Skip(f"YOLOv5 repo not at {repo}")
    with DB._V5_IMPORT_LOCK, DB._yolov5_on_path(DB.Path(repo)):
        from utils.general import non_max_suppression
    return non_max_suppression


def test_one_box_per_cart_after_nms_and_share_is_exact():
    nms = _yolov5_nms()
    # Two boxes on the same cart, one called empty, one non-empty (per-class
    # NMS would keep both), plus a person standing inside the cart's box.
    pred = _pred([
        (100, 100, 60, 80, 0.90, 0.01, 0.10, 0.80),   # cart, mostly non-empty
        (101, 101, 60, 80, 0.85, 0.01, 0.70, 0.20),   # same cart, empty-ish
        (100, 95, 30, 70, 0.80, 0.90, 0.02, 0.02),    # person inside it
    ])
    merged = DB.merge_cart_classes(pred, 0, 1, 2)
    out = nms(merged, 0.25, 0.45, nm=2)[0].numpy()
    carts = out[out[:, 5] == DB.CART_CLS]
    persons = out[out[:, 5] == DB.PERSON_CLS]
    assert len(carts) == 1, f"two cart boxes survived NMS: {out}"
    assert len(persons) == 1, "the person inside the cart was suppressed"
    share = DB.recover_share(carts[:, 6], carts[:, 7])[0]
    assert math.isclose(share, 0.8 / 0.9, rel_tol=1e-5), share


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------
def test_detections_validate_their_shapes():
    try:
        DB.Detections(np.zeros((2, 4), np.float32), np.zeros(3, np.float32),
                      np.zeros(3, np.float32), np.zeros(3, np.float32), (10, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched shapes accepted")
    e = DB.Detections.empty((720, 1280, 3))
    assert len(e) == 0 and e.as_tracker_input().shape == (0, 6) and e.orig_shape == (720, 1280)


def test_frame_validation():
    for bad in (None, np.zeros((10, 10), np.uint8), np.zeros((0, 10, 3), np.uint8)):
        try:
            DB._check_frame(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {getattr(bad, 'shape', bad)}")


# ---------------------------------------------------------------------------
# Tracking driver
# ---------------------------------------------------------------------------
class ScriptedDetector:
    """Returns pre-scripted detections frame by frame."""

    provides_cart_state = True
    names = dict(DB.UNIFIED_NAMES)

    def __init__(self, script):
        self.script = list(script)
        self.module = torch.nn.Linear(1, 1)
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return self

    def detect(self, frame, imgsz=None):
        boxes = self.script.pop(0)
        if not boxes:
            return DB.Detections.empty(frame.shape)
        a = np.array(boxes, np.float32)
        return DB.Detections(a[:, :4], a[:, 4], a[:, 5], a[:, 6], tuple(frame.shape[:2]))


FRAME = np.zeros((480, 640, 3), np.uint8)
CART_A = [100, 100, 200, 220, 0.9, DB.CART_CLS, 0.95]
CART_B = [400, 100, 500, 220, 0.9, DB.CART_CLS, 0.05]
PERSON = [250, 50, 300, 300, 0.9, DB.PERSON_CLS, float("nan")]


def _track(model, n):
    return [model.track(FRAME, persist=True, tracker=C.TRACKER_CONFIG)[0] for _ in range(n)]


def test_ids_persist_and_cart_state_follows_its_track():
    # Same objects every frame, but the detector's ORDER flips each frame:
    # p_non_empty must follow the box through idx, not the position.
    script = [[CART_A, CART_B, PERSON] if i % 2 == 0 else [PERSON, CART_B, CART_A] for i in range(8)]
    model = DB.DirectTrackingModel(ScriptedDetector(script), "test")
    by_id = {}
    for res in _track(model, 8):
        assert res.boxes.id is not None
        for tid, x1, cls, p in zip(res.boxes.id.int().tolist(), res.boxes.xyxy[:, 0].tolist(),
                                   res.boxes.cls.int().tolist(), res.p_non_empty.tolist()):
            by_id.setdefault(tid, set()).add((round(x1), cls, None if math.isnan(p) else round(p, 2)))
    assert len(by_id) == 3, f"ids were not stable: {by_id}"
    seen = {next(iter(v)) for v in by_id.values() if len(v) == 1}
    assert (100, DB.CART_CLS, 0.95) in seen and (400, DB.CART_CLS, 0.05) in seen, by_id
    assert (250, DB.PERSON_CLS, None) in seen, by_id


def test_untracked_frame_reports_no_ids():
    model = DB.DirectTrackingModel(ScriptedDetector([[]]), "test")
    res = _track(model, 1)[0]
    assert res.boxes.id is None and len(res.p_non_empty) == 0


def test_boxes_are_clipped_to_the_frame():
    edge = [600, 400, 700, 520, 0.9, DB.CART_CLS, 0.5]   # runs past 640x480
    model = DB.DirectTrackingModel(ScriptedDetector([[edge]] * 3), "test")
    xyxy = _track(model, 3)[-1].boxes.xyxy.numpy()
    assert xyxy[:, 2].max() <= 640 and xyxy[:, 3].max() <= 480, xyxy


def test_predictor_none_starts_a_fresh_tracker_instead_of_crashing():
    model = DB.DirectTrackingModel(ScriptedDetector([[CART_A]] * 4), "test")
    _track(model, 2)
    old = model.predictor.trackers[0]
    model.predictor = None
    assert model.predictor is not None and model.predictor.trackers == []
    _track(model, 1)
    assert model.predictor.trackers[0] is not old
    try:
        model.predictor = object()
    except TypeError:
        pass
    else:
        raise AssertionError("a predictor without trackers was accepted")


def test_parked_predictor_round_trip_keeps_tracker_state():
    """tracker.py parks `predictor` across a GPU offload and puts it back."""
    model = DB.DirectTrackingModel(ScriptedDetector([[CART_A]] * 4), "test")
    first = _track(model, 2)[-1].boxes.id.tolist()
    held = model.predictor
    model.to("cpu")
    model.predictor = held
    assert _track(model, 1)[0].boxes.id.tolist() == first


def test_reset_through_predictor_trackers_restarts_ids():
    model = DB.DirectTrackingModel(ScriptedDetector([[CART_A]] * 6), "test")
    _track(model, 3)
    for t in model.predictor.trackers:
        t.reset()
    assert _track(model, 3)[-1].boxes.id.tolist() == [1.0]


def test_missing_tracker_config_is_refused():
    model = DB.DirectTrackingModel(ScriptedDetector([[CART_A]]), "test")
    try:
        model.track(FRAME, tracker="")
    except ValueError:
        pass
    else:
        raise AssertionError("empty tracker config accepted")
    try:
        DB._build_tracker("does/not/exist.yaml")
    except DB.DetectionBackendError:
        pass
    else:
        raise AssertionError("missing tracker yaml accepted")


def test_unknown_backend_is_refused():
    try:
        DB.build_detection_model("nope", "cpu")
    except DB.DetectionBackendError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("unknown backend accepted")


def test_cache_key_and_fingerprint_change_with_backend():
    from engine.trajectory_cache import make_video_key

    fps = {b: DB.backend_fingerprint(b) for b in C.DETECTOR_BACKENDS}
    assert len(set(fps.values())) == len(fps), fps
    path = os.path.abspath(__file__)
    keys = {make_video_key(path, b) for b in C.DETECTOR_BACKENDS}
    assert len(keys) == len(C.DETECTOR_BACKENDS)
    assert make_video_key(path) not in keys


# ---------------------------------------------------------------------------
# YOLOv5 import isolation
# ---------------------------------------------------------------------------
def test_a_foreign_top_level_utils_is_refused():
    repo = DB.Path(os.path.abspath(C.V5_REPO_PATH))
    saved = sys.modules.get("utils")
    fake = types.ModuleType("utils")
    fake.__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")
    sys.modules["utils"] = fake
    try:
        with DB._yolov5_on_path(repo):
            pass
    except DB.DetectionBackendError as e:
        assert "utils" in str(e)
    else:
        raise AssertionError("a foreign 'utils' module was accepted")
    finally:
        if saved is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = saved


def test_process_state_is_restored():
    import cv2

    before_threads = cv2.getNumThreads()
    before_filters = list(warnings.filters)
    before_np = np.get_printoptions()
    with DB._ProcessStateGuard():
        cv2.setNumThreads(0)
        warnings.filterwarnings("ignore", category=UserWarning)
        np.set_printoptions(linewidth=7)
        os.environ["OMP_NUM_THREADS"] = "123456"
    assert cv2.getNumThreads() == before_threads
    assert list(warnings.filters) == before_filters
    assert np.get_printoptions() == before_np
    assert os.environ.get("OMP_NUM_THREADS") != "123456"


# ---------------------------------------------------------------------------
# CartClassifier.classify_loaded
# ---------------------------------------------------------------------------
class _Const(torch.nn.Module):
    def __init__(self, *outs):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.outs = [torch.tensor([o], dtype=torch.float32) for o in outs]

    def forward(self, x):
        return self.outs[0] if len(self.outs) == 1 else tuple(self.outs)


def _classifier(quality_logits, fill_logits, bag_logits=(0.0, 3.0), fill_classes=("empty", "partial", "full")):
    from engine.classifier import CartClassifier

    clf = CartClassifier("cpu")
    clf._quality_model = _Const(quality_logits)
    clf._fill_model = _Const(fill_logits, bag_logits)
    clf._fill_classes = list(fill_classes)
    clf._quality_temp = clf._fill_temp = clf._bag_temp = 1.0
    clf.set_quality_threshold(0.5)
    return clf


FRAME_CLS = np.full((200, 200, 3), 127, np.uint8)
BB = (20, 20, 180, 180)


def test_classify_loaded_never_answers_empty():
    clf = _classifier((5.0, -5.0), (6.0, 1.0, 0.0))   # fill model is sure it is empty
    assert clf.classify(FRAME_CLS, BB)["fill"] == "empty"
    r = clf.classify_loaded(FRAME_CLS, BB)
    assert r["fill"] == "partial", r
    p = torch.softmax(torch.tensor([1.0, 0.0]), 0)[0].item()
    assert math.isclose(r["fill_conf"], p, rel_tol=1e-5), "fill_conf must be renormalised over loaded classes"
    assert r["bag"] == "unbagged" and r["quality"] == "valid_cart"


def test_classify_loaded_passes_an_unreadable_crop_through():
    clf = _classifier((-5.0, 5.0), (0.0, 3.0, 0.0))
    r = clf.classify_loaded(FRAME_CLS, BB)
    assert r["quality"] == "unclear" and r["is_valid"] is False


def test_classify_loaded_without_a_loaded_class_says_unclassified():
    clf = _classifier((5.0, -5.0), (1.0,), fill_classes=("empty",))
    assert clf.classify_loaded(FRAME_CLS, BB)["fill"] == "unclassified"


def test_both_entry_points_run_without_autograd():
    """A read must never build an autograd graph (memory per crop, and torch
    warns on every float() of a requires_grad tensor)."""
    clf = _classifier((5.0, -5.0), (0.0, 1.0, 4.0))
    seen = []

    def spy(module, inputs, output):
        seen.append(torch.is_grad_enabled())

    clf._quality_model.register_forward_hook(spy)
    clf.classify(FRAME_CLS, BB)
    clf.classify_loaded(FRAME_CLS, BB)
    assert seen == [False, False], seen


def test_classify_is_unchanged_for_loaded_crops():
    clf = _classifier((5.0, -5.0), (0.0, 1.0, 4.0))
    r = clf.classify(FRAME_CLS, BB)
    probs = torch.softmax(torch.tensor([0.0, 1.0, 4.0]), 0)
    assert r["fill"] == "full" and math.isclose(r["fill_conf"], probs[2].item(), rel_tol=1e-5)


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
