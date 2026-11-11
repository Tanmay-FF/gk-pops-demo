# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The 5-class backend (person / empty_cart / loaded_cart / stacked_cart /
child_in_cart) on top of the 3-class driver.

What it pins:
  * the tracker still sees exactly two classes, person and cart, so a loaded to
    empty flip, a stacked cart, or a child can never split a track;
  * every cart box carries p_non_empty and p_stacked in [0, 1]; persons carry NaN;
  * children are returned separately, never inside the tracked boxes;
  * the checkpoint's class names are resolved by name.

Needs config.Y26_5CLASS_WEIGHTS_PATH and one sample clip; skips otherwise.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine import config as C  # noqa: E402
from engine import detection_backends as DB  # noqa: E402


class Skip(Exception):
    pass


def _clip():
    for d in (C.TEST_VIDEO_DIR,):
        found = sorted(glob.glob(os.path.join(d, "*.mp4")))
        if found:
            return found[0]
    raise Skip("no sample clip")


def _model():
    if not os.path.isfile(C.Y26_5CLASS_WEIGHTS_PATH):
        raise Skip(f"missing {C.Y26_5CLASS_WEIGHTS_PATH}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return DB.build_detection_model("yolo26_5class", dev)


def test_backend_is_registered_and_fingerprinted():
    assert "yolo26_5class" in C.DETECTOR_BACKENDS
    fps = {DB.backend_fingerprint(b) for b in C.DETECTOR_BACKENDS}
    assert len(fps) == len(C.DETECTOR_BACKENDS)


def test_tracker_sees_only_person_and_cart():
    m = _model()
    assert m.provides_cart_state
    assert m.names == {0: "person", 1: "cart"}


def test_cart_boxes_carry_state_and_children_stay_out_of_tracking():
    m = _model()
    cap = cv2.VideoCapture(_clip())
    n_cart = 0
    for _ in range(40):
        ok, f = cap.read()
        if not ok:
            break
        r = m.track(f, persist=True, tracker=C.TRACKER_CONFIG, imgsz=640)[0]
        assert r.children is not None and r.children.shape[1:] == (5,)
        if r.boxes.id is None:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        assert set(cls) <= {0, 1}, f"a non person/cart class reached the tracker: {set(cls)}"
        assert len(r.p_non_empty) == len(r.p_stacked) == len(cls)
        for c, pe, ps in zip(cls, r.p_non_empty, r.p_stacked):
            if c == 1:
                assert 0.0 <= pe <= 1.0 and 0.0 <= ps <= 1.0
                n_cart += 1
            else:
                assert np.isnan(pe) and np.isnan(ps)
    assert n_cart > 0, "no cart in the first 40 frames to check"


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
