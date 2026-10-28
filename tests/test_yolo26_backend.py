"""The YOLOv26 3-class backend reproduces ultralytics' predict(), with the
two cart classes merged before NMS (engine/detection_backends.py,
Yolo26ThreeClassDetector).

ultralytics' Results keep only each box's top class, so the backend runs
predict()'s steps itself to get P(non-empty). These tests pin that it is the
SAME computation, on real frames of the primary golden clip:
  * person boxes and confidences are identical to predict();
  * every cart box it returns is a cart box predict() returns, same
    coordinates and confidence;
  * every cart box predict() returns that the backend dropped is a
    cross-class duplicate (IoU above the NMS threshold with a kept cart box)
    -- the empty_cart / non_empty_cart double box the merge exists to remove;
  * P(non-empty) agrees with predict()'s cart class (non_empty_cart exactly
    when P(non-empty) > 0.5).
Plus model-free checks of the shared merge and the class-name resolution.

Needs the YOLOv26 weights at config.Y26_WEIGHTS_PATH and the golden clip;
skips without them. Uses CUDA when available, CPU otherwise.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine import config as C  # noqa: E402
from engine import detection_backends as DB  # noqa: E402

N_FRAMES = 60


class Skip(Exception):
    pass


NAMES3 = {"person": C.THREE_CLASS_PERSON, "empty_cart": C.THREE_CLASS_EMPTY_CART,
          "non_empty_cart": C.THREE_CLASS_NON_EMPTY_CART}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


# ---------------------------------------------------------------------------
# Model-free
# ---------------------------------------------------------------------------
def test_merge_cart_scores_along_the_class_dim():
    probs = torch.tensor([[[0.9, 0.1], [0.2, 0.6], [0.6, 0.2]]])   # (B=1, nc=3, N=2)
    person, cart, share = DB.merge_cart_scores(probs, 0, 1, 2, dim=1)
    assert person.shape == cart.shape == share.shape == (1, 1, 2)
    assert torch.allclose(cart[0, 0], torch.tensor([0.6, 0.6]))
    assert torch.allclose(share[0, 0], torch.tensor([0.75, 0.25]))


def test_class_names_resolve_by_name_not_position():
    raw, idx = DB._three_class_indices(["non_empty_cart", "person", "empty_cart"], NAMES3, "w.pt")
    assert idx == (1, 2, 0) and raw[0] == "non_empty_cart"
    for names in (["person", "cart"], ["person", "empty_cart", "empty_cart"]):
        try:
            DB._three_class_indices(names, NAMES3, "w.pt")
        except DB.DetectionBackendError:
            continue
        raise AssertionError(f"accepted {names}")


def test_backend_is_registered():
    assert "yolo26_3class" in C.DETECTOR_BACKENDS and "yolo26_3class" in C.THREE_CLASS_BACKENDS
    assert DB.backend_fingerprint("yolo26_3class") != DB.backend_fingerprint("yolov5_3class")


# ---------------------------------------------------------------------------
# Against predict(), on real frames
# ---------------------------------------------------------------------------
def _frames():
    import cv2
    from test_golden_clip import load_baseline, find_clip, Skip as GoldenSkip

    if not os.path.isfile(C.Y26_WEIGHTS_PATH):
        raise Skip(f"missing {C.Y26_WEIGHTS_PATH}")
    try:
        clip = find_clip(load_baseline())
    except GoldenSkip as why:
        raise Skip(str(why))
    cap = cv2.VideoCapture(clip)
    out = []
    while len(out) < N_FRAMES:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    cap.release()
    if not out:
        raise Skip(f"could not read {clip}")
    return out


def test_matches_predict_on_real_frames():
    from ultralytics import YOLO

    frames = _frames()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = DB.Yolo26ThreeClassDetector(C.Y26_WEIGHTS_PATH, device, C.Y26_TRACK_CONF, C.Y26_IOU, C.Y26_IMGSZ, NAMES3)
    ref = YOLO(C.Y26_WEIGHTS_PATH)
    n_carts = n_person = n_dup = 0
    for fi, im in enumerate(frames, 1):
        r = ref.predict(im, conf=C.Y26_TRACK_CONF, iou=C.Y26_IOU, imgsz=C.Y26_IMGSZ, device=device, verbose=False)[0].boxes
        rx, rc, rk = r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(int)
        d = det.detect(im)

        ref_p = sorted((tuple(np.round(x, 1)), round(float(c), 4)) for x, c, k in zip(rx, rc, rk) if k == 0)
        got_p = sorted((tuple(np.round(x, 1)), round(float(c), 4))
                       for x, c, k in zip(d.xyxy, d.conf, d.cls) if int(k) == DB.PERSON_CLS)
        assert got_p == ref_p, f"frame {fi}: person boxes differ\n  predict {ref_p}\n  backend {got_p}"
        n_person += len(got_p)

        ref_c = [(x, float(c), int(k)) for x, c, k in zip(rx, rc, rk) if k in (1, 2)]
        got_c = [(x, float(c), float(p)) for x, c, k, p in zip(d.xyxy, d.conf, d.cls, d.p_non_empty)
                 if int(k) == DB.CART_CLS]
        matched = set()
        for x, c, p in got_c:
            hits = [j for j, (rxj, rcj, _) in enumerate(ref_c)
                    if np.allclose(rxj, x, atol=0.05) and abs(rcj - c) < 1e-4]
            assert hits, f"frame {fi}: backend cart box {np.round(x, 1)} conf {c:.4f} not in predict()"
            j = hits[0]
            matched.add(j)
            assert 0.0 <= p <= 1.0, p
            if abs(p - 0.5) > 1e-3:
                assert (ref_c[j][2] == 2) == (p > 0.5), (
                    f"frame {fi}: predict says class {ref_c[j][2]}, backend P(non-empty) {p:.3f}")
        for j, (rxj, _, _) in enumerate(ref_c):
            if j in matched:
                continue
            assert any(_iou(rxj, x) > C.Y26_IOU for x, _, _ in got_c), (
                f"frame {fi}: predict() cart box {np.round(rxj, 1)} dropped without a merged duplicate")
            n_dup += 1
        n_carts += len(got_c)
    assert n_carts > 0 and n_person > 0, "the clip produced no detections to compare"
    print(f"  {len(frames)} frames: {n_person} person and {n_carts} cart boxes identical to predict(); "
          f"{n_dup} cross-class duplicate cart boxes merged")


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
