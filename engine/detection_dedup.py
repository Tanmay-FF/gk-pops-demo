# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Suppress nested duplicate detections that IoU-based NMS structurally cannot.

THE PROBLEM
-----------
The detector routinely emits two boxes for one cart: a tight one around the
basket and a loose one that also swallows the handle, wheels and shadow. NMS
compares them by IoU, and IoU between nested boxes is bounded by their area
ratio — it never approaches 1 no matter how completely one sits inside the
other. Measured on 1764092528600_B8A44F40EFB5-medium-OUTSIDE.mp4:

    frame 141   [671,312,846,523] conf 0.874   (175x211)
                [671,311,777,425] conf 0.609   (106x114)
                IoU 0.323   containment 0.991
    frame 150   IoU 0.512   containment 1.000
    frame 169   IoU 0.434   containment 0.964

With the default `iou=0.7` NMS gate, nothing in that range is ever suppressed.
Both boxes survive, BoTSORT quite correctly tracks two objects, and one physical
cart is reported as two — it picked up display IDs Cart 1 and Cart 3, was scored
twice (POPS 75 "HIGH PRIORITY" and POPS 75 "PUSHOUT ALERT"), and the run summary
claimed 3 carts where there were 2.

Lowering the NMS threshold is not the fix: to catch an IoU of 0.32 you would
have to merge boxes that are genuinely 68% disjoint, which would start eating
adjacent carts in a queue.

THE MEASURE
-----------
Containment — intersection over the area of the SMALLER box, sometimes called
IoMin or Intersection-over-Smaller. It is 1.0 for a perfectly nested pair
regardless of the size difference, which is exactly the case IoU is blind to.

WHAT IS DELIBERATELY NOT SUPPRESSED
-----------------------------------
Suppression is scoped to a single class. A person standing inside their cart's
bounding box is the normal case in this footage, not a duplicate, and must
survive — as must a cart inside a doorway zone, a bag inside a cart, and so on.

Two carts in a queue, one 90%+ occluded behind the other, WOULD be merged. That
is the accepted cost: a box that far occluded has an unreliable extent anyway,
and reporting one cart twice is the worse failure for loss prevention. The
threshold is a config knob (NESTED_DUP_CONTAIN_MIN) so it can be raised if a
real queue case shows up.

WHERE THIS RUNS
---------------
From RetailBOTSORT.update, on the tracker's input, BEFORE the high/low
confidence pools are split. Filtering there catches a duplicate pair that
straddles the split — one box above track_high_thresh and its twin below —
which per-pool filtering would miss. Suppressing pre-tracker also means no
second track is ever created, so no display ID, POPS row, classification vote,
cart fact or frame-JSON entry is ever written for the duplicate. Merging IDs
after the fact would have to retract all of those.
"""
import numpy as np


def containment(a, b) -> float:
    """Intersection over the area of the smaller box. 1.0 when nested.

    Both boxes are (x1, y1, x2, y2). Returns 0.0 for degenerate boxes rather
    than dividing by zero — a zero-area detection is not a duplicate of
    anything.
    """
    ox1 = max(a[0], b[0])
    oy1 = max(a[1], b[1])
    ox2 = min(a[2], b[2])
    oy2 = min(a[3], b[3])
    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0
    inter = (ox2 - ox1) * (oy2 - oy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return float(inter / smaller)


def xywh_to_xyxy(xywh):
    """Centre-form to corner-form. The tracker is handed centre-form boxes."""
    xywh = np.asarray(xywh, dtype=np.float64).reshape(-1, 4)
    half_w = xywh[:, 2] * 0.5
    half_h = xywh[:, 3] * 0.5
    return np.stack([xywh[:, 0] - half_w, xywh[:, 1] - half_h,
                     xywh[:, 0] + half_w, xywh[:, 1] + half_h], axis=1)


def nested_duplicate_mask(xyxy, conf, cls, contain_min: float) -> np.ndarray:
    """Boolean keep-mask over detections; False means "nested duplicate".

    Greedy and confidence-ordered, the same discipline as NMS: walk detections
    from most to least confident, and drop one when it is `contain_min`-or-more
    contained by an already-kept box OF THE SAME CLASS. Comparing against kept
    boxes only (rather than all boxes) means a chain A>B>C collapses to A alone
    and never to the empty set.

    Confidence order matters: it is what makes the surviving box the detector's
    best guess rather than whichever happened to be listed first.
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    cls = np.asarray(cls).reshape(-1)
    n = len(conf)

    keep = np.ones(n, dtype=bool)
    if n < 2 or contain_min <= 0:
        return keep

    # Descending confidence; np.argsort is ascending, so reverse. Stable sort so
    # equal confidences keep detector order and the result is deterministic.
    order = np.argsort(conf, kind="stable")[::-1]

    kept = []
    for i in order:
        duplicate = False
        for j in kept:
            if cls[i] != cls[j]:
                continue
            if containment(xyxy[i], xyxy[j]) >= contain_min:
                duplicate = True
                break
        if duplicate:
            keep[i] = False
        else:
            kept.append(i)
    return keep


def mask_for_results(results, contain_min: float) -> np.ndarray:
    """Keep-mask for an ultralytics Boxes-like object.

    Reads `.xywh` (or `.xywhr` for oriented boxes, whose first four columns are
    the same centre-form box), `.conf` and `.cls` — the same attributes
    BYTETracker.update itself consumes, so anything the tracker can handle this
    can filter.
    """
    if results is None or len(results) < 2:
        return np.ones(len(results) if results is not None else 0, dtype=bool)

    boxes = getattr(results, "xywhr", None)
    if boxes is None:
        boxes = results.xywh
    boxes = np.asarray(boxes, dtype=np.float64).reshape(len(results), -1)[:, :4]

    return nested_duplicate_mask(
        xywh_to_xyxy(boxes),
        np.asarray(results.conf, dtype=np.float64),
        np.asarray(results.cls),
        contain_min,
    )
