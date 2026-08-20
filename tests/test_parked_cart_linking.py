# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""A cart that has never moved cannot be owned by someone walking past it.

The geometry is real: tests/fixtures/parked_cart_geometry.py holds the boxes from
the 1764200318790 INSIDE clip, where an empty cart parked inside the entrance was
given to a shopper entering the store on six frames of corner overlap at IoU
0.039-0.064, kept for the remaining 279 frames of the run, and then scored
ABANDONED CART (65) when she walked out of the doorway.

WHAT THIS PINS
--------------
1. "Parked" needs BOTH a small centroid sweep and LINK_STATIC_MIN_FRAMES of
   observation. Extent alone calls every newly appeared cart parked, including
   one a shopper is about to take out of the corral.
2. On a parked cart the co-movement exemption is granted only to a person at
   IoU >= LINK_STATIC_MIN_IOU. A parked cart can never fail the co-movement test
   on its own account, so at LINK_MIN_IOU that exemption was open to anyone whose
   box touched it.
3. A parked cart's confirmation bar is LINK_CONTESTED_FRAMES even with one
   candidate.
4. Neither gate applies to a cart that HAS moved — that is the golden OUTSIDE
   clip's own cart, which rolls in and comes to rest with its real handler
   working at it, and it is pinned end-to-end by
   tests/fixtures/golden/baseline_outside.json.
5. A link on a cart that has not moved since it was established is released when
   the owner is visible and no longer touching it, even though nobody is taking
   the cart over — and the release reports the cart as DISOWNED, which is what
   tells the tracker to forget the remembered owner too.
6. A person who leaves the FRAME still keeps the link, parked cart or not. That
   is abandonment, and it is the invariant tests/test_abandonment_after_release.py
   exists to protect.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import linker as L  # noqa: E402
from engine.config import (  # noqa: E402
    LINK_CONFIRM_FRAMES, LINK_CONTESTED_FRAMES, LINK_GRACE_FRAMES,
    LINK_DRIFT_FRAMES, LINK_GROUND_BAND, LINK_MIN_IOU,
    LINK_STATIC_MIN_FRAMES, LINK_STATIC_MIN_IOU, LINK_STATIC_SPREAD_PX,
)
from engine.motion import are_co_moving  # noqa: E402
from fixtures.parked_cart_geometry import (  # noqa: E402
    CART_BY_FRAME, CART_PARKED, CART_CENTROID_SPAN_PX, P4_BY_FRAME,
    P4_WALKING_AWAY, LINKED_ON_FRAME, OVERLAP_FRAMES,
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
# Harness
# ---------------------------------------------------------------------------
def _linker():
    """A linker whose display IDs are the raw IDs, so assertions read plainly."""
    return L.PersonCartLinker(lambda label, raw: raw)


class Scene:
    """Position histories the linker can read, advanced one frame at a time.

    The linker takes obj_positions as a plain dict of lists, which the tracker
    appends to before calling it. This does the same, so co-movement and the
    parked-cart measurement both see the histories they would see in the run.
    """

    def __init__(self, linker, first_frame=0):
        self.lk = linker
        self.positions = {}
        self.first_frame = {}
        self.gone = {}
        self._default_first = first_frame

    def step(self, frame, people, carts, centroids):
        """people/carts: {id: bbox}. centroids: {id: (cx, cy)} for this frame."""
        for oid, c in centroids.items():
            self.positions.setdefault(oid, []).append(c)
            self.first_frame.setdefault(oid, self._default_first)
        for oid in list(self.gone):
            self.gone[oid] = self.gone[oid] + 1 if oid not in people else 0
        for oid in people:
            self.gone[oid] = 0
        self.lk.update(
            person_bboxes=people, cart_bboxes=carts, frame_idx=frame,
            obj_disappeared=self.gone, obj_positions=self.positions,
            obj_first_frame=self.first_frame,
        )


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


#: The parked cart, held for as many frames as a test needs before the real
#: geometry starts. CART_PARKED is what the clip shows on every frame outside
#: CART_BY_FRAME.
def _hold_parked(scene, cart_id, first, last):
    box, centroid = CART_PARKED
    for f in range(first, last + 1):
        scene.step(f, {}, {cart_id: box}, {cart_id: centroid})


# ---------------------------------------------------------------------------
section("the clip's own numbers")
# ---------------------------------------------------------------------------
_cart_box = CART_BY_FRAME[LINKED_ON_FRAME][0]
_ious = [L._iou(CART_BY_FRAME[f][0], P4_BY_FRAME[f][0]) for f in OVERLAP_FRAMES]
_feet = [L.foot_ratio(P4_BY_FRAME[f][0], CART_BY_FRAME[f][0]) for f in OVERLAP_FRAMES]

check("she really did overlap the cart — the mislink was not an IoU failure",
      max(_ious) >= LINK_MIN_IOU, f"peak IoU {max(_ious):.3f}")
check("...but never at more than grazing contact",
      max(_ious) < LINK_STATIC_MIN_IOU, f"peak IoU {max(_ious):.3f}")
check("the ground-plane gate cannot see this one: both bodies are foreground",
      sum(_feet) / len(_feet) <= LINK_GROUND_BAND,
      f"mean foot ratio {sum(_feet) / len(_feet):+.3f} vs band {LINK_GROUND_BAND}")
check("she gave the accumulator exactly LINK_CONFIRM_FRAMES of overlap",
      len([i for i in _ious if i >= LINK_MIN_IOU]) >= LINK_CONFIRM_FRAMES,
      f"{len([i for i in _ious if i >= LINK_MIN_IOU])} qualifying frames")
check("the cart's whole-clip centroid sweep is under LINK_STATIC_SPREAD_PX",
      CART_CENTROID_SPAN_PX < LINK_STATIC_SPREAD_PX)

# The exemption, on its own, still passes her — which is why the IoU condition
# had to go where it went rather than into are_co_moving().
_cart_hist = [CART_PARKED[1]] * 8
_p4_hist = [P4_BY_FRAME[f][1] for f in sorted(P4_BY_FRAME)][:8]
check("a parked cart + a moving person is 'co-moving' whenever the exemption "
      "is granted",
      are_co_moving(_cart_hist, _p4_hist, static_a_ok=True))
check("...and is not, when it is withheld",
      not are_co_moving(_cart_hist, _p4_hist, static_a_ok=False))

# ---------------------------------------------------------------------------
section("what counts as parked")
# ---------------------------------------------------------------------------
lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 10, 1, LINK_STATIC_MIN_FRAMES - 1)
check("a cart watched for less than LINK_STATIC_MIN_FRAMES is not parked yet, "
      "however still it has been",
      not lk._is_parked(10, LINK_STATIC_MIN_FRAMES - 1))
_hold_parked(sc, 10, LINK_STATIC_MIN_FRAMES, LINK_STATIC_MIN_FRAMES + 1)
check("and is, once it has been watched that long",
      lk._is_parked(10, LINK_STATIC_MIN_FRAMES + 1))

# A cart that rolls: same observation length, moving centroid.
lk = _linker()
sc = Scene(lk)
for f in range(1, LINK_STATIC_MIN_FRAMES + 2):
    box = (400 + f * 3, 320, 570 + f * 3, 560)
    sc.step(f, {}, {10: box}, {10: _centre(box)})
check("a cart that has moved is never parked, however long it is watched",
      not lk._is_parked(10, LINK_STATIC_MIN_FRAMES + 1))

# ---------------------------------------------------------------------------
section("the real mislink no longer forms")
# ---------------------------------------------------------------------------
lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 1, 1, min(CART_BY_FRAME) - 1)
for f in sorted(CART_BY_FRAME):
    cart_box, cart_centroid = CART_BY_FRAME[f]
    people = {4: P4_BY_FRAME[f][0]} if f in P4_BY_FRAME else {}
    centroids = {1: cart_centroid}
    if 4 in people:
        centroids[4] = P4_BY_FRAME[f][1]
    sc.step(f, people, {1: cart_box}, centroids)
check("a shopper walking past the parked cart does not take ownership of it",
      lk.links.get(1) is None,
      f"links={lk.links} — she held it from frame {LINKED_ON_FRAME} to the end "
      f"of the run")

# Give her far longer than she actually had: the gate is not a timing accident.
lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 1, 1, 120)
frame = 121
for _ in range(LINK_CONTESTED_FRAMES * 3):
    frame += 1
    # Freeze her at her strongest overlap frame, so only the gate can stop her.
    best = max(OVERLAP_FRAMES, key=lambda f: L._iou(CART_BY_FRAME[f][0],
                                                    P4_BY_FRAME[f][0]))
    box, base = P4_BY_FRAME[best]
    # Walking, not standing — her real centroid track advances every frame.
    drift = (frame - 121) * 8
    sc.step(frame, {4: box}, {1: CART_PARKED[0]},
            {1: CART_PARKED[1], 4: (base[0] + drift, base[1])})
check("...not even over three times the contested window",
      lk.links.get(1) is None, f"links={lk.links}")

# ---------------------------------------------------------------------------
section("a person AT the parked cart still gets it, on the contested bar")
# ---------------------------------------------------------------------------
# Same parked cart, but a person overlapping it well above LINK_STATIC_MIN_IOU
# and sharing its ground plane — someone loading or unloading it.
_at_cart = (200, 300, 330, 640)
check("the control case really is over the static-cart IoU bar",
      L._iou(CART_PARKED[0], _at_cart) >= LINK_STATIC_MIN_IOU,
      f"IoU {L._iou(CART_PARKED[0], _at_cart):.3f}")
check("...and shares its ground plane",
      L._shares_ground_plane(L.foot_ratio(_at_cart, CART_PARKED[0])))

lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 1, 1, 60)
frame = 60
for _ in range(LINK_CONFIRM_FRAMES):
    frame += 1
    sc.step(frame, {7: _at_cart}, {1: CART_PARKED[0]},
            {1: CART_PARKED[1], 7: (_centre(_at_cart)[0] + (frame - 60) * 6,
                                    _centre(_at_cart)[1])})
check("LINK_CONFIRM_FRAMES is not enough on a parked cart",
      lk.links.get(1) is None, f"links={lk.links}")
for _ in range(LINK_CONTESTED_FRAMES - LINK_CONFIRM_FRAMES):
    frame += 1
    sc.step(frame, {7: _at_cart}, {1: CART_PARKED[0]},
            {1: CART_PARKED[1], 7: (_centre(_at_cart)[0] + (frame - 60) * 6,
                                    _centre(_at_cart)[1])})
check("LINK_CONTESTED_FRAMES is", lk.links.get(1) == 7, f"links={lk.links}")

# ---------------------------------------------------------------------------
section("a cart being taken out of the corral is only delayed, not blocked")
# ---------------------------------------------------------------------------
# Parked long enough to qualify, then someone takes it and it starts rolling.
# Once it has moved, the fast path is back.
lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 1, 1, LINK_STATIC_MIN_FRAMES + 10)
frame = LINK_STATIC_MIN_FRAMES + 10
cart_box, (ccx, ccy) = CART_PARKED
for i in range(LINK_CONFIRM_FRAMES + 2):
    frame += 1
    shift = (i + 1) * 6          # cart rolling away from the corral
    cb = (cart_box[0] + shift, cart_box[1], cart_box[2] + shift, cart_box[3])
    pb = (cb[0] + 30, cb[1] - 90, cb[2] + 30, cb[3])
    sc.step(frame, {8: pb}, {1: cb}, {1: (ccx + shift, ccy),
                                      8: (_centre(pb))})
check("a cart that starts moving is no longer parked",
      not lk._is_parked(1, frame))
check("and its taker gets it on the fast path",
      lk.links.get(1) == 8, f"links={lk.links}")

# ---------------------------------------------------------------------------
section("a link on a cart that never moved is released, and reported")
# ---------------------------------------------------------------------------
# A link CAN still form on a parked cart: a person at it, over the static IoU
# bar, on the contested threshold — someone loading it, or taking one out of the
# corral. What must not stand is such a link outliving the contact that made it
# while the cart still has not moved. Before this rule the no-taker case was
# unreleasable, and that is what turned a hollow link on the 1764200318790 clip
# into an ABANDONED CART at 65.
lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 10, 1, 60)
frame = 60
for _ in range(LINK_CONTESTED_FRAMES + 1):
    frame += 1
    sc.step(frame, {5: _at_cart}, {10: CART_PARKED[0]},
            {10: CART_PARKED[1], 5: (_centre(_at_cart)[0] + (frame - 60) * 6,
                                     _centre(_at_cart)[1])})
check("owner established on the parked cart", lk.links.get(10) == 5,
      f"links={lk.links}")
link_frame = frame

_away = (900, 300, 1000, 560)     # visible, same ground plane, no overlap
released_on = None
for _ in range(LINK_STATIC_MIN_FRAMES + LINK_DRIFT_FRAMES + 4):
    frame += 1
    sc.step(frame, {5: _away}, {10: CART_PARKED[0]},
            {10: CART_PARKED[1], 5: _centre(_away)})
    if released_on is None and lk.links.get(10) is None:
        released_on = frame
        check("the release reports the cart as disowned",
              10 in lk.disowned_carts, f"disowned={lk.disowned_carts}")
check("a link that never moved its cart is released once the contact stops",
      released_on is not None, f"links={lk.links}")
check("...and not before the cart has stood still LINK_STATIC_MIN_FRAMES "
      "since the link was made",
      released_on is None or released_on - link_frame >= LINK_STATIC_MIN_FRAMES,
      f"released {released_on} - linked {link_frame}")
check("disowned_carts is per-frame state, cleared by the next update",
      10 not in lk.disowned_carts, f"disowned={lk.disowned_carts}")

# The counter-case, and the more important one: a cart that MOVED under this
# owner and then came to rest. The same walk-away must keep the link, because
# that is a person parking their own cart and stepping away from it — the case
# POPS scores as abandonment, and releasing it here would close that route.
lk = _linker()
sc = Scene(lk)
frame = 0
cart_box = (400, 320, 570, 560)
beside_cart = (420, 230, 540, 570)
for i in range(LINK_GRACE_FRAMES + LINK_CONFIRM_FRAMES + 2):
    frame += 1
    shift = i * 4
    cb = (cart_box[0] + shift, cart_box[1], cart_box[2] + shift, cart_box[3])
    pb = (beside_cart[0] + shift, beside_cart[1],
          beside_cart[2] + shift, beside_cart[3])
    sc.step(frame, {5: pb}, {10: cb}, {10: _centre(cb), 5: _centre(pb)})
check("owner established on a moving cart", lk.links.get(10) == 5,
      f"links={lk.links}")

_stopped = (400 + (LINK_GRACE_FRAMES + LINK_CONFIRM_FRAMES + 1) * 4, 320,
            570 + (LINK_GRACE_FRAMES + LINK_CONFIRM_FRAMES + 1) * 4, 560)
for _ in range(LINK_STATIC_MIN_FRAMES + LINK_DRIFT_FRAMES + 4):
    frame += 1
    sc.step(frame, {5: _away}, {10: _stopped},
            {10: _centre(_stopped), 5: _centre(_away)})
check("a cart that DID move under its owner keeps him when he steps away",
      lk.links.get(10) == 5,
      f"links={lk.links} — this is the abandonment case, not a hollow link")
check("and is never reported disowned", not lk.disowned_carts,
      f"disowned={lk.disowned_carts}")

# ---------------------------------------------------------------------------
section("leaving the frame is still abandonment, not disownment")
# ---------------------------------------------------------------------------
lk = _linker()
sc = Scene(lk)
frame = 0
for i in range(LINK_GRACE_FRAMES + LINK_CONFIRM_FRAMES + 2):
    frame += 1
    shift = i * 4
    cb = (cart_box[0] + shift, cart_box[1], cart_box[2] + shift, cart_box[3])
    pb = (beside_cart[0] + shift, beside_cart[1],
          beside_cart[2] + shift, beside_cart[3])
    sc.step(frame, {5: pb}, {10: cb}, {10: _centre(cb), 5: _centre(pb)})
check("owner established", lk.links.get(10) == 5, f"links={lk.links}")
for _ in range(LINK_STATIC_MIN_FRAMES + LINK_DRIFT_FRAMES + 4):
    frame += 1
    sc.step(frame, {}, {10: _stopped}, {10: _centre(_stopped)})
check("an owner who has left the frame keeps the link, parked cart or not",
      lk.links.get(10) == 5,
      f"links={lk.links} — abandonment scoring needs this link to exist")
check("and the cart is not reported disowned", not lk.disowned_carts,
      f"disowned={lk.disowned_carts}")

# ---------------------------------------------------------------------------
section("the walk-past shopper, all the way to where the 65 came from")
# ---------------------------------------------------------------------------
# The full sequence: she grazes the cart, walks on across the frame in plain
# view, and then leaves it. With no link there is no owner, and with no owner
# the cart cannot be abandoned by anyone.
lk = _linker()
sc = Scene(lk)
_hold_parked(sc, 1, 1, min(CART_BY_FRAME) - 1)
for f in sorted(CART_BY_FRAME):
    cart_box_f, cart_centroid = CART_BY_FRAME[f]
    people = {4: P4_BY_FRAME[f][0]} if f in P4_BY_FRAME else {}
    centroids = {1: cart_centroid}
    if 4 in people:
        centroids[4] = P4_BY_FRAME[f][1]
    sc.step(f, people, {1: cart_box_f}, centroids)
frame = max(CART_BY_FRAME)
for f, (box, centroid) in sorted(P4_WALKING_AWAY.items()):
    while frame < f:
        frame += 1
        sc.step(frame, {4: box}, {1: CART_PARKED[0]},
                {1: CART_PARKED[1], 4: centroid})
check("she never owns the cart at any point in the crossing",
      lk.links.get(1) is None, f"links={lk.links}")
check("nor is anything disowned, because nothing was ever linked",
      not lk.disowned_carts, f"disowned={lk.disowned_carts}")
check("and the cart has no remembered owner for the tracker to inherit",
      not lk.person_raw_for_cart, f"{lk.person_raw_for_cart}")

# ---------------------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
