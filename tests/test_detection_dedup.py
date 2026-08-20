"""Nested duplicate detections must not become two tracks.

The detector emits a tight box and a loose box for the same cart. IoU-based NMS
structurally cannot suppress that pair — nested boxes measured IoU 0.32-0.51
against a 0.7 gate while their containment was 0.96-1.00 — so both survived,
BoTSORT tracked two objects, and one physical cart became Cart 1 AND Cart 3,
each scored its own POPS row.

These tests pin the measure (containment, not IoU), the scope (same class only)
and the outcome (one track, not two).

Runs anywhere — no GPU, no weights, no video.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import engine.tracker  # noqa: F401  — installs the tracker subclasses
from ultralytics.trackers.track import TRACKER_MAP

from engine.config import NESTED_DUP_CONTAIN_MIN
from engine.detection_dedup import (
    containment, nested_duplicate_mask, mask_for_results, xywh_to_xyxy,
)

#: The real pair, straight off the detector at frame 141 of
#: 1764092528600_B8A44F40EFB5-medium-OUTSIDE.mp4.
LOOSE = (671, 312, 846, 523)   # conf 0.874
TIGHT = (671, 311, 777, 425)   # conf 0.609

CART, PERSON = 0.0, 1.0


class Dets:
    """Duck-types the ultralytics Boxes object the tracker consumes."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Dets(self.xywh[mask], self.conf[mask], self.cls[mask])


def to_xywh(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def iou(a, b):
    ox1, oy1 = max(a[0], b[0]), max(a[1], b[1])
    ox2, oy2 = min(a[2], b[2]), min(a[3], b[3])
    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0
    i = (ox2 - ox1) * (oy2 - oy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return i / (aa + ab - i)


# ---------------------------------------------------------------------------
# Why IoU cannot do this
# ---------------------------------------------------------------------------

def test_the_real_pair_is_invisible_to_iou_but_obvious_to_containment():
    """The whole reason this module exists, asserted on real numbers."""
    assert iou(LOOSE, TIGHT) < 0.7, (
        "if IoU cleared the NMS gate, plain NMS would already have handled this"
    )
    assert containment(LOOSE, TIGHT) >= NESTED_DUP_CONTAIN_MIN, (
        f"containment {containment(LOOSE, TIGHT):.3f} should catch what "
        f"IoU {iou(LOOSE, TIGHT):.3f} cannot"
    )


def test_containment_is_one_for_a_perfectly_nested_box():
    outer = (0, 0, 100, 100)
    inner = (10, 10, 20, 20)
    assert containment(outer, inner) == 1.0
    assert containment(inner, outer) == 1.0, "containment must be symmetric"
    assert iou(outer, inner) < 0.02, "IoU collapses on a size mismatch; containment does not"


def test_containment_is_zero_for_disjoint_and_degenerate_boxes():
    assert containment((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert containment((0, 0, 10, 10), (5, 5, 5, 5)) == 0.0     # zero area
    assert containment((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------

def test_nested_duplicate_is_dropped_and_the_confident_box_survives():
    boxes = np.array([LOOSE, TIGHT], dtype=float)
    keep = nested_duplicate_mask(boxes, [0.874, 0.609], [CART, CART],
                                 NESTED_DUP_CONTAIN_MIN)
    assert list(keep) == [True, False], (
        "the more confident of a nested pair must be the one kept"
    )


def test_the_kept_box_is_chosen_by_confidence_not_by_order():
    """Same pair, reversed input order and reversed confidence."""
    boxes = np.array([LOOSE, TIGHT], dtype=float)
    keep = nested_duplicate_mask(boxes, [0.2, 0.9], [CART, CART],
                                 NESTED_DUP_CONTAIN_MIN)
    assert list(keep) == [False, True]


def test_a_different_class_inside_the_same_box_is_never_suppressed():
    """A person standing inside their cart's box is the normal case in this
    footage. Measured 8 such cross-class nested pairs on the golden clip."""
    boxes = np.array([LOOSE, TIGHT], dtype=float)
    keep = nested_duplicate_mask(boxes, [0.874, 0.609], [CART, PERSON],
                                 NESTED_DUP_CONTAIN_MIN)
    assert list(keep) == [True, True]


def test_separate_carts_are_left_alone():
    """Cart 2 on the golden clip had zero overlap with the duplicate pair across
    41 and 25 shared frames. Dedup must not touch that case."""
    far = (535, 143, 597, 219)
    boxes = np.array([LOOSE, far], dtype=float)
    keep = nested_duplicate_mask(boxes, [0.874, 0.895], [CART, CART],
                                 NESTED_DUP_CONTAIN_MIN)
    assert list(keep) == [True, True]


def test_partial_overlap_below_the_threshold_survives():
    """Two carts queueing must both survive; only near-total containment goes."""
    a = (0, 0, 100, 100)
    b = (50, 0, 150, 100)          # containment 0.5
    keep = nested_duplicate_mask(np.array([a, b], dtype=float), [0.9, 0.8],
                                 [CART, CART], NESTED_DUP_CONTAIN_MIN)
    assert list(keep) == [True, True]


def test_a_chain_of_nested_boxes_collapses_to_one_not_to_none():
    """A > B > C: comparing against KEPT boxes only must leave A standing."""
    a = (0, 0, 100, 100)
    b = (5, 5, 90, 90)
    c = (10, 10, 80, 80)
    keep = nested_duplicate_mask(np.array([a, b, c], dtype=float),
                                 [0.9, 0.8, 0.7], [CART, CART, CART],
                                 NESTED_DUP_CONTAIN_MIN)
    assert list(keep) == [True, False, False]
    assert keep.sum() == 1


def test_empty_and_single_detection_inputs_are_safe():
    for n in (0, 1):
        keep = nested_duplicate_mask(np.zeros((n, 4)), np.zeros(n), np.zeros(n),
                                     NESTED_DUP_CONTAIN_MIN)
        assert len(keep) == n and keep.all()


def test_equal_confidence_is_deterministic():
    """A stable sort keeps detector order, so the same input always gives the
    same output — a flip-flopping winner would jitter the tracked box."""
    boxes = np.array([LOOSE, TIGHT], dtype=float)
    runs = {tuple(nested_duplicate_mask(boxes, [0.5, 0.5], [CART, CART],
                                        NESTED_DUP_CONTAIN_MIN))
            for _ in range(5)}
    assert len(runs) == 1


def test_xywh_conversion_round_trips():
    got = xywh_to_xyxy([to_xywh(LOOSE)])[0]
    assert list(map(int, got)) == list(LOOSE)


def test_mask_for_results_reads_a_boxes_like_object():
    dets = Dets([to_xywh(LOOSE), to_xywh(TIGHT)], [0.874, 0.609], [CART, CART])
    assert list(mask_for_results(dets, NESTED_DUP_CONTAIN_MIN)) == [True, False]


# ---------------------------------------------------------------------------
# The outcome, through the real tracker
# ---------------------------------------------------------------------------

def _args():
    class Args:
        tracker_type = "botsort"
        track_high_thresh = 0.3
        track_low_thresh = 0.1
        new_track_thresh = 0.4
        track_buffer = 120
        match_thresh = 0.7
        fuse_score = True
        gmc_method = "none"
        proximity_thresh = 0.5
        appearance_thresh = 0.25
        with_reid = False
    return Args()


def _drift(box, step):
    x1, y1, x2, y2 = box
    return (x1 + step, y1, x2 + step, y2)


def test_a_nested_pair_produces_one_track_not_two():
    """The user-visible bug: one physical cart, two IDs."""
    trk = TRACKER_MAP["botsort"](args=_args(), frame_rate=30)
    ids = set()
    for step in range(12):
        dets = Dets(
            [to_xywh(_drift(LOOSE, step * 3)), to_xywh(_drift(TIGHT, step * 3))],
            [0.874, 0.609], [CART, CART],
        )
        rows = trk.update(dets)
        for r in rows:
            ids.add(int(r[4]))
    assert len(ids) == 1, (
        f"one cart was tracked as {len(ids)} objects (IDs {sorted(ids)}) — this "
        f"is the defect that produced Cart 1 and Cart 3 for the same cart"
    )


def test_two_genuinely_separate_carts_still_get_two_tracks():
    """The dedup must not achieve its result by under-counting."""
    trk = TRACKER_MAP["botsort"](args=_args(), frame_rate=30)
    far = (300, 600, 380, 700)
    ids = set()
    for step in range(12):
        dets = Dets(
            [to_xywh(_drift(LOOSE, step * 3)), to_xywh(_drift(far, step * 3))],
            [0.874, 0.895], [CART, CART],
        )
        for r in trk.update(dets):
            ids.add(int(r[4]))
    assert len(ids) == 2, f"expected two carts, got IDs {sorted(ids)}"


def test_a_person_inside_a_cart_box_still_gets_its_own_track():
    trk = TRACKER_MAP["botsort"](args=_args(), frame_rate=30)
    ids = set()
    for step in range(12):
        dets = Dets(
            [to_xywh(_drift(LOOSE, step * 3)), to_xywh(_drift(TIGHT, step * 3))],
            [0.874, 0.8], [CART, PERSON],
        )
        for r in trk.update(dets):
            ids.add(int(r[4]))
    assert len(ids) == 2, (
        f"a person standing in their cart was suppressed as a duplicate: {sorted(ids)}"
    )


def test_dedup_runs_before_the_confidence_split():
    """A duplicate pair straddling track_high_thresh — the loose box above it,
    the tight one below — must still collapse. Filtering per-pool would miss
    this, which is why the mask is applied to update()'s whole input."""
    trk = TRACKER_MAP["botsort"](args=_args(), frame_rate=30)
    ids = set()
    for step in range(12):
        dets = Dets(
            [to_xywh(_drift(LOOSE, step * 3)), to_xywh(_drift(TIGHT, step * 3))],
            [0.9, 0.2], [CART, CART],       # 0.2 is in the low pool
        )
        for r in trk.update(dets):
            ids.add(int(r[4]))
    assert len(ids) == 1, f"cross-pool duplicate survived: {sorted(ids)}"


if __name__ == "__main__":
    import ultralytics

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed (ultralytics={ultralytics.__version__}, "
          f"NESTED_DUP_CONTAIN_MIN={NESTED_DUP_CONTAIN_MIN})")
