# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Which person the linker decides owns a cart, and why.

Every geometry case here comes from tests/fixtures/link_geometry.py — real boxes
from the 1764099569430 OUTSIDE clip, not drawn ones. A synthetic fixture with
tidy boxes cannot reproduce this bug: what breaks the association is a scene with
depth, where a person metres from a cart still overlaps it in 2D.

WHAT THIS PINS
--------------
1. The ground-plane test is judged on the MEAN of a candidate's window, never
   per frame. This is the whole reason the first design was wrong: per-frame it
   rejects the primary golden clip's own correct link on 39% of its frames.
2. A frame with no candidate does not discard the ones already accumulated.
3. A takeover release hands the cart to the taker instead of to nobody.
4. `claimed` is complete before the ID-swap search runs, so two carts cannot end
   up holding one person by dict-iteration luck.

See docs/missed_pushout_fix_plan.md.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import linker as L  # noqa: E402
from engine.config import (  # noqa: E402
    LINK_GROUND_BAND, LINK_BEHIND_BAND, LINK_MIN_IOU,
    LINK_CONFIRM_FRAMES, LINK_CONTESTED_FRAMES, LINK_GRACE_FRAMES,
    LINK_CANDIDATE_PATIENCE, LINK_DRIFT_FRAMES,
)
from fixtures.link_geometry import (  # noqa: E402
    LINK_CASES, foot_delta, cart_height, ends_inside_cart,
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


def _linker():
    """A linker whose display IDs are the raw IDs, so assertions read plainly."""
    return L.PersonCartLinker(lambda label, raw: raw)


def _step(lk, people, carts, frame, first_frame=None, gone=None, positions=None):
    """One update() call with the boilerplate filled in.

    Positions default to a short static history for every object: are_co_moving()
    returns True when both are static, which is the case these tests are about
    (a cart at a doorway and the people around it), and keeps the geometry the
    only variable.
    """
    ids = list(people) + list(carts)
    lk.update(
        person_bboxes=people,
        cart_bboxes=carts,
        frame_idx=frame,
        obj_disappeared=gone or {},
        obj_positions=positions or {
            i: [(0.0, 0.0)] * 6 for i in ids
        },
        obj_first_frame=first_frame or {i: 0 for i in ids},
    )


# ---------------------------------------------------------------------------
section("foot_ratio() on the real boxes")
# ---------------------------------------------------------------------------
for label, frame, pbox, cbox, should_link, why in LINK_CASES:
    ratio = L.foot_ratio(pbox, cbox)
    hand = foot_delta(pbox, cbox) / cart_height(cbox)
    check(f"{label}: ratio matches the fixture's own arithmetic",
          abs(ratio - hand) < 1e-9, f"{ratio} != {hand}")

_mislinks = [c for c in LINK_CASES if not c[4]]
_legit = [c for c in LINK_CASES if c[4]]
check("every mislink case is over the band",
      all(L.foot_ratio(c[2], c[3]) > LINK_GROUND_BAND for c in _mislinks),
      str([round(L.foot_ratio(c[2], c[3]), 2) for c in _mislinks]))
check("no legitimate case is over the band",
      all(L.foot_ratio(c[2], c[3]) <= LINK_GROUND_BAND for c in _legit),
      str([round(L.foot_ratio(c[2], c[3]), 2) for c in _legit]))

# The occluded handler is the case a symmetric |delta| <= band test rejects.
_occluded = next(c for c in LINK_CASES if "occluded by the basket" in c[0])
check("the occluded true handler would fail a SYMMETRIC band test",
      abs(L.foot_ratio(_occluded[2], _occluded[3])) > LINK_GROUND_BAND,
      "if this stops being true the test below no longer proves anything")
check("...but passes the one-sided test, because his box ends INSIDE the cart",
      L._shares_ground_plane(L.foot_ratio(_occluded[2], _occluded[3]))
      and ends_inside_cart(_occluded[2], _occluded[3]))

check("a person entirely above the cart's top is rejected as behind it",
      not L._shares_ground_plane(LINK_BEHIND_BAND - 0.01))
check("the band's own value is inclusive, not a strict bound",
      L._shares_ground_plane(LINK_GROUND_BAND)
      and L._shares_ground_plane(LINK_BEHIND_BAND))

# ---------------------------------------------------------------------------
section("the gate reads the MEAN of a window, not single frames")
# ---------------------------------------------------------------------------
# The primary golden clip's correct P1->C1 link measured mean +0.19 with 39% of
# its frames over the bar. Reproduce that shape: mostly plausible, a large
# minority implausible. A per-frame veto rejects it; the mean must not.
_plausible = (600, 200, 700, 560)      # feet at the cart's base
_implausible = (600, 200, 700, 760)    # feet a long way in front of it
_cart = (590, 320, 760, 560)

lk = _linker()
for f in range(1, LINK_CONFIRM_FRAMES + 1):
    # 1 frame in 3 implausible — over LINK_CONFIRM_FRAMES that is 33%
    box = _implausible if f % 3 == 0 else _plausible
    _step(lk, {1: box}, {10: _cart}, frame=LINK_GRACE_FRAMES + f)
check("a link whose frames are a third implausible still forms",
      lk.links.get(10) == 1,
      f"links={lk.links} — this is the primary clip's own pushout link")

lk = _linker()
for f in range(1, LINK_CONTESTED_FRAMES + 2):
    _step(lk, {1: _implausible}, {10: _cart}, frame=LINK_GRACE_FRAMES + f)
check("a link implausible on EVERY frame never forms",
      lk.links.get(10) is None, f"links={lk.links}")

# ---------------------------------------------------------------------------
section("the real mislink, against the real handler")
# ---------------------------------------------------------------------------
# P4 (foreground, truncated at the frame edge) and P2 (the actual handler) both
# overlap C1. On the clip P4 won and held the link for 41 frames. P2 must win.
_p4 = next(c for c in LINK_CASES if c[0].startswith("P4->C1"))[2]
_p2 = next(c for c in LINK_CASES if c[0].startswith("P2->C1 same pair"))[2]
_c1 = next(c for c in LINK_CASES if c[0].startswith("P4->C1"))[3]

check("both really do overlap the cart — the mislink was not an IoU failure",
      L._iou(_c1, _p4) >= LINK_MIN_IOU and L._iou(_c1, _p2) >= LINK_MIN_IOU,
      f"P4 IoU {L._iou(_c1, _p4):.3f}, P2 IoU {L._iou(_c1, _p2):.3f}")
check("the foreground body's overlap is the SMALLER of the two, so cumulative "
      "IoU alone cannot pick the right one reliably",
      L._iou(_c1, _p4) < L._iou(_c1, _p2))

lk = _linker()
for f in range(1, LINK_CONTESTED_FRAMES + 2):
    _step(lk, {4: _p4, 2: _p2}, {1: _c1}, frame=LINK_GRACE_FRAMES + f)
check("contested: the cart goes to the ground-sharing handler, not the "
      "foreground body", lk.links.get(1) == 2, f"links={lk.links}")

lk = _linker()
for f in range(1, LINK_CONTESTED_FRAMES + 2):
    _step(lk, {4: _p4}, {1: _c1}, frame=LINK_GRACE_FRAMES + f)
check("uncontested: the foreground body still does not get the cart",
      lk.links.get(1) is None, f"links={lk.links}")

# The same P4 owning the cart that is actually with him must survive.
_p4_own, _c4 = next((c[2], c[3]) for c in LINK_CASES if c[0].startswith("P4->C4"))
lk = _linker()
for f in range(1, LINK_CONFIRM_FRAMES + 1):
    _step(lk, {4: _p4_own}, {4: _c4}, frame=LINK_GRACE_FRAMES + f)
check("a foreground person still gets their OWN foreground cart",
      lk.links.get(4) == 4, f"links={lk.links} — 'distrust foreground people' "
      f"would break this")

# ---------------------------------------------------------------------------
section("a barren frame does not discard accumulated candidates")
# ---------------------------------------------------------------------------
lk = _linker()
frame = LINK_GRACE_FRAMES
for _ in range(LINK_CONFIRM_FRAMES - 1):
    frame += 1
    _step(lk, {1: _plausible}, {10: _cart}, frame=frame)
check("not linked yet — one frame short of confirmation",
      lk.links.get(10) is None)

frame += 1
_step(lk, {}, {10: _cart}, frame=frame)          # nobody in the frame at all
frame += 1
_step(lk, {1: _plausible}, {10: _cart}, frame=frame)
check("one barren frame in the middle does not reset the accumulator",
      lk.links.get(10) == 1, f"links={lk.links}")

lk = _linker()
frame = LINK_GRACE_FRAMES
for _ in range(LINK_CONFIRM_FRAMES - 1):
    frame += 1
    _step(lk, {1: _plausible}, {10: _cart}, frame=frame)
for _ in range(LINK_CANDIDATE_PATIENCE):
    frame += 1
    _step(lk, {}, {10: _cart}, frame=frame)
frame += 1
_step(lk, {1: _plausible}, {10: _cart}, frame=frame)
check(f"{LINK_CANDIDATE_PATIENCE} barren frames DO reset it",
      lk.links.get(10) is None, f"links={lk.links}")

# ---------------------------------------------------------------------------
section("a takeover release hands the cart to the taker")
# ---------------------------------------------------------------------------
# Establish P1 as owner, then move P1 away while P2 stands at the cart. The old
# code released the link and left the cart ownerless; it must now re-link to P2.
_far = (100, 200, 200, 560)       # same ground plane, no overlap with the cart
lk = _linker()
frame = LINK_GRACE_FRAMES
for _ in range(LINK_CONFIRM_FRAMES):
    frame += 1
    _step(lk, {1: _plausible}, {10: _cart}, frame=frame)
check("owner established", lk.links.get(10) == 1, f"links={lk.links}")

for _ in range(LINK_DRIFT_FRAMES):
    frame += 1
    _step(lk, {1: _far, 2: _plausible}, {10: _cart}, frame=frame)
check("the drifted owner lost the link", lk.links.get(10) != 1,
      f"links={lk.links}")
for _ in range(LINK_CONTESTED_FRAMES):
    frame += 1
    _step(lk, {1: _far, 2: _plausible}, {10: _cart}, frame=frame)
check("and the taker now owns the cart, rather than nobody owning it",
      lk.links.get(10) == 2, f"links={lk.links}")

# A foreground body is not a taker: it satisfies IoU but not the ground test.
lk = _linker()
frame = LINK_GRACE_FRAMES
for _ in range(LINK_CONFIRM_FRAMES):
    frame += 1
    _step(lk, {1: _plausible}, {10: _cart}, frame=frame)
for _ in range(LINK_DRIFT_FRAMES + 2):
    frame += 1
    _step(lk, {1: _far, 9: _implausible}, {10: _cart}, frame=frame)
check("a foreground body overlapping the cart does not trigger a takeover",
      lk.links.get(10) == 1,
      f"links={lk.links} — without the ground test in someone_else, the release "
      f"fires and then SEEDS the foreground body as the next owner")

# ---------------------------------------------------------------------------
section("claimed is complete before the ID-swap search (audited bug 5)")
# ---------------------------------------------------------------------------
# Two carts. Cart A's owner vanishes; cart B's owner is still visible and stands
# where cart A's owner was. The swap search must not hand B's person to A.
_cart_a = (590, 320, 760, 560)
_cart_b = (900, 320, 1070, 560)
_person_b = (910, 200, 1010, 560)

lk = _linker()
frame = LINK_GRACE_FRAMES
for _ in range(LINK_CONFIRM_FRAMES):
    frame += 1
    _step(lk, {1: _plausible, 2: _person_b},
          {10: _cart_a, 20: _cart_b}, frame=frame)
check("both carts linked to their own person",
      lk.links.get(10) == 1 and lk.links.get(20) == 2, f"links={lk.links}")

# Person 1 disappears; person 2 is the only body left, sitting near where
# person 1 was last seen so the swap search finds them.
frame += 1
positions = {i: [(0.0, 0.0)] * 6 for i in (2, 10, 20)}
positions[1] = [(650.0, 380.0)] * 6
positions[2] = [(650.0, 380.0)] * 6
lk.update(
    person_bboxes={2: _person_b},
    cart_bboxes={10: _cart_a, 20: _cart_b},
    frame_idx=frame,
    obj_disappeared={1: 3},
    obj_positions=positions,
    obj_first_frame={i: 0 for i in (1, 2, 10, 20)},
)
check("the still-linked person is not stolen by the cart whose owner left",
      lk.links.get(20) == 2 and lk.links.get(10) != 2,
      f"links={lk.links} — one person cannot own two carts, and cart 20's owner "
      f"is still standing right there")


print(f"\n{_passed} passed, {_failed} failed")
if _failed:
    raise SystemExit(1)
print("All green.")
