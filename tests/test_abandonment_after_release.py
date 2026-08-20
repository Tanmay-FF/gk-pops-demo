# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Abandonment must survive the link being released.

`abandoned` is the strongest term in compute_pops() — it floors an outbound
loaded cart at MERCH_REMOVED_FLOOR (75) and is the only route to PUSHOUT ALERT
below PUSHOUT_SCORE. It used to be readable ONLY while the cart was still linked,
which made it unavailable in exactly the state where the question matters: a cart
whose owner has gone. See docs/missed_pushout_fix_plan.md.

THE INVARIANT THAT MUST NOT BREAK
---------------------------------
classify_event() documents that `abandoned` is only ever True for a cart that was
LINKED first, so a cart which never had an owner — one sitting in a corral, say —
cannot reach PUSHOUT ALERT except on score alone. The fix widens WHEN the question
can be asked, not WHICH carts it can be asked about. A cart with no remembered
owner still answers False forever, and that is pinned below.

These are unit tests over the abandonment decision, expressed against the same
inputs the frame loop computes it from. They do not decode video — the end-to-end
behaviour on real pixels is pinned by tests/fixtures/golden/baseline_outside.json.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import (  # noqa: E402
    ABANDON_FRAMES, WALKAWAY_GAP_FRAC, WALKAWAY_MIN_GAP_PX, LINK_DRIFT_IOU,
    STALE_CART_FRAMES,
)
from engine.scoring import (  # noqa: E402
    compute_pops, classify_event, MERCH_REMOVED_FLOOR, HIGH_SCORE,
)

_passed = _failed = 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f" — {extra}" if extra else ""))


def section(title):
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# A faithful transcription of the abandonment decision in
# TrackingEngine.process_video's per-cart block. Kept in one place so the tests
# below read as scenarios rather than as bookkeeping.
# ---------------------------------------------------------------------------
def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _gap(a, b):
    """Shortest distance between two boxes, 0.0 when they touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


class Cart:
    """One cart's abandonment state across frames."""

    def __init__(self, bbox=(600, 300, 800, 560)):
        self.bbox = bbox
        self.last_owner = None
        self.walkaway = 0
        self.last_seen = None
        self.abandoned = False

    def step(self, frame, linked_person=None, people=None, gone=None):
        """One frame. `people` is {person: bbox}, `gone` is {person: frames_gone}."""
        people = people or {}
        gone = gone or {}

        if linked_person is not None:
            if self.last_owner != linked_person:
                self.walkaway = 0
            self.last_owner = linked_person
        if self.last_owner is not None:
            if self.last_seen is not None and frame - self.last_seen > STALE_CART_FRAMES:
                self.last_owner = None
                self.walkaway = 0
            self.last_seen = frame

        owner = linked_person if linked_person is not None else self.last_owner

        attended = any(_iou(self.bbox, pb) >= LINK_DRIFT_IOU
                       for pb in people.values())
        if attended:
            self.walkaway = 0

        person_gone = (not attended and owner is not None
                       and gone.get(owner, 0) > ABANDON_FRAMES)

        person_far = False
        if not attended and owner is not None and owner in people:
            pb = people[owner]
            diag = ((self.bbox[2] - self.bbox[0]) ** 2
                    + (self.bbox[3] - self.bbox[1]) ** 2) ** 0.5
            far_gap = max(WALKAWAY_GAP_FRAC * diag, WALKAWAY_MIN_GAP_PX)
            if _gap(self.bbox, pb) > far_gap:
                self.walkaway += 1
            else:
                self.walkaway = 0
            person_far = self.walkaway > ABANDON_FRAMES

        self.abandoned = person_gone or person_far
        return self.abandoned


AT_CART = (620, 200, 720, 560)          # engaged: overlaps, same ground plane
FAR = (60, 200, 160, 560)               # visible, 440 px clear of the cart box
NEAR_BUT_LOOSE = (810, 200, 910, 560)   # visible, 10 px clear of the cart box

# Cart 1 of the 1764173272870 static-cart clip, to scale: a large box seen from
# above with its owner standing against the right edge. The two boxes are in
# contact, and their centroids are 232 px apart.
BIG_CART = (601, 377, 802, 664)             # Cart 1 at frame 240, verbatim
BESIDE_BIG_CART = (757, 188, 857, 439)      # Person 4 at frame 240, verbatim
AWAY_FROM_BIG_CART = (150, 188, 250, 439)   # the same person, actually away


# ---------------------------------------------------------------------------
section("the released cart whose owner then leaves — the case that was blind")
# ---------------------------------------------------------------------------
c = Cart()
f = 0
for _ in range(10):                      # owner with the cart
    f += 1
    c.step(f, linked_person=1, people={1: AT_CART})
check("not abandoned while the owner is at the cart", not c.abandoned)

f += 1
c.step(f, linked_person=None, people={1: FAR})   # link released, owner visible
check("not abandoned the instant the link is released", not c.abandoned)

for _ in range(ABANDON_FRAMES + 2):     # owner has left the frame entirely
    f += 1
    c.step(f, linked_person=None, people={}, gone={1: ABANDON_FRAMES + 5})
check("abandoned once the FORMER owner has been gone ABANDON_FRAMES",
      c.abandoned,
      "this is the case the old `linked`-gated test could never answer: "
      "a released cart is exactly the cart whose owner may have walked off")

# ---------------------------------------------------------------------------
section("...and it reaches PUSHOUT ALERT, which is the point")
# ---------------------------------------------------------------------------
score = compute_pops("UNKNOWN", "STATIC", True, "partial", bag_label="unbagged",
                     abandoned=True, linked=False, merch_removed=True)
event, _ = classify_event(score, False, "UNKNOWN", abandoned=True)
check(f"merchandise removed from an abandoned cart floors at {MERCH_REMOVED_FLOOR}",
      score == MERCH_REMOVED_FLOOR, f"score={score}")
check("and that is a PUSHOUT ALERT", event == "PUSHOUT ALERT", event)

score_no_abandon = compute_pops("UNKNOWN", "STATIC", True, "partial",
                                bag_label="unbagged", abandoned=False,
                                linked=False, merch_removed=True)
check("without abandonment the same cart cannot even reach HIGH",
      score_no_abandon < HIGH_SCORE,
      f"score={score_no_abandon} — this is why the gated test cost the clip its "
      f"pushout, not the scoring table")

# ---------------------------------------------------------------------------
section("a cart that never had an owner stays un-abandonable (the invariant)")
# ---------------------------------------------------------------------------
c = Cart()
f = 0
for _ in range(ABANDON_FRAMES * 4):
    f += 1
    c.step(f, linked_person=None, people={}, gone={1: 999, 2: 999})
check("a cart nobody ever owned is never abandoned, however long it sits",
      not c.abandoned,
      "classify_event() relies on this to keep a corral cart out of PUSHOUT ALERT")

# ---------------------------------------------------------------------------
section("a genuine handover is attended, not abandoned")
# ---------------------------------------------------------------------------
c = Cart()
f = 0
for _ in range(10):
    f += 1
    c.step(f, linked_person=1, people={1: AT_CART})
for _ in range(ABANDON_FRAMES * 2):
    # owner 1 gone from frame, but person 2 is engaged with the cart
    f += 1
    c.step(f, linked_person=None, people={2: AT_CART}, gone={1: 999})
check("someone else engaged with the cart keeps it attended",
      not c.abandoned,
      "the frames between one link ending and the next being confirmed must not "
      "read as abandonment")

for _ in range(ABANDON_FRAMES + 2):     # now person 2 leaves too
    f += 1
    c.step(f, linked_person=None, people={}, gone={1: 999, 2: 999})
check("once nobody is with it, the departure counts again", c.abandoned)

# ---------------------------------------------------------------------------
section("walkaway is measured against the owner, and reset on a change of hands")
# ---------------------------------------------------------------------------
c = Cart()
f = 0
for _ in range(10):
    f += 1
    c.step(f, linked_person=1, people={1: AT_CART})
for _ in range(ABANDON_FRAMES + 2):
    f += 1
    c.step(f, linked_person=None, people={1: FAR})
check("an owner who stays visible but walks far away abandons the cart",
      c.abandoned)

c = Cart()
f = 0
for _ in range(10):
    f += 1
    c.step(f, linked_person=1, people={1: AT_CART})
for _ in range(ABANDON_FRAMES - 1):     # owner 1 drifting off, nearly abandoned
    f += 1
    c.step(f, linked_person=None, people={1: FAR})
check("nearly abandoned", not c.abandoned)
f += 1
c.step(f, linked_person=2, people={1: FAR, 2: NEAR_BUT_LOOSE})
for _ in range(ABANDON_FRAMES - 1):
    f += 1
    c.step(f, linked_person=2, people={1: FAR, 2: NEAR_BUT_LOOSE})
check("a cart that changes hands does not inherit the old owner's walkaway count",
      not c.abandoned,
      "the counter measures ONE person's distance; before the `linked` gate came "
      "off it was only safe by accident")

# ---------------------------------------------------------------------------
section("the owner memory ages out with the cart")
# ---------------------------------------------------------------------------
c = Cart()
f = 0
for _ in range(10):
    f += 1
    c.step(f, linked_person=1, people={1: AT_CART})
f += STALE_CART_FRAMES + 5              # cart out of frame long enough to purge
for _ in range(ABANDON_FRAMES + 2):
    f += 1
    c.step(f, linked_person=None, people={}, gone={1: 999})
check("a cart gone longer than STALE_CART_FRAMES forgets its owner",
      not c.abandoned,
      "a display ID reused later must not inherit a stranger's departure")


# ---------------------------------------------------------------------------
section("standing beside a large cart is not walking away from it")
# ---------------------------------------------------------------------------
# The regression this section exists for. Cart 1 of the 1764173272870
# static-cart clip was scored 65 ABANDONED CART while Person 4 stood against it
# for the whole clip. Their boxes were in contact - gap 0.0 px, IoU 0.035, just
# under LINK_DRIFT_IOU, so `attended` was False - but the cart is large and seen
# from above, which put the centroids 232 px apart, past the flat 200 px bar the
# walkaway counter used to measure against. It therefore counted every frame the
# person was not overlapping the cart, crossed ABANDON_FRAMES at frame 239, and
# the finaliser latched the flag over the rest of the run.
_beside_iou = _iou(BIG_CART, BESIDE_BIG_CART)
_beside_gap = _gap(BIG_CART, BESIDE_BIG_CART)
check("the fixture reproduces the geometry: boxes touch, IoU below the bar",
      _beside_gap == 0.0 and _beside_iou < LINK_DRIFT_IOU,
      f"gap={_beside_gap} iou={_beside_iou:.4f}")

_ccx = (BIG_CART[0] + BIG_CART[2]) / 2
_ccy = (BIG_CART[1] + BIG_CART[3]) / 2
_pcx = (BESIDE_BIG_CART[0] + BESIDE_BIG_CART[2]) / 2
_pcy = (BESIDE_BIG_CART[1] + BESIDE_BIG_CART[3]) / 2
_centroid_dist = ((_pcx - _ccx) ** 2 + (_pcy - _ccy) ** 2) ** 0.5
check("...and that the old centroid measure called that person far away",
      _centroid_dist > 200, f"centroid distance={_centroid_dist:.1f}")

c = Cart(bbox=BIG_CART)
f = 0
for _ in range(ABANDON_FRAMES * 4):
    f += 1
    c.step(f, linked_person=1, people={1: BESIDE_BIG_CART})
check("a cart with its owner standing against it is never abandoned",
      not c.abandoned,
      "however long they stand there - this is the 1764173272870 false positive")

# ...and the fix must not have bought that off by making walkaway unreachable.
_diag = ((BIG_CART[2] - BIG_CART[0]) ** 2 + (BIG_CART[3] - BIG_CART[1]) ** 2) ** 0.5
_away_gap = _gap(BIG_CART, AWAY_FROM_BIG_CART)
check("the walking-away fixture really is past the bar",
      _away_gap > max(WALKAWAY_GAP_FRAC * _diag, WALKAWAY_MIN_GAP_PX),
      f"gap={_away_gap:.1f} diag={_diag:.1f}")
for _ in range(ABANDON_FRAMES + 2):
    f += 1
    c.step(f, linked_person=1, people={1: AWAY_FROM_BIG_CART})
check("...and the same owner actually leaving still abandons the cart",
      c.abandoned)


print(f"\n{_passed} passed, {_failed} failed")
if _failed:
    raise SystemExit(1)
print("All green.")
