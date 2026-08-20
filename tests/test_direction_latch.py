# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Direction-latch tests — pure logic, no GPU and no video decode.

Run with:  python tests/test_direction_latch.py

The fixture is real. It is the per-frame direction_label of Cart 1 from the
1763942209000_B8A44F1EDC34-medium-OUTSIDE clip, read out of that run's tracking
JSON, plus the frames on which `abandoned` was true. That run is a confirmed
pushout — a cart wheeled out of the door and left standing outside holding
unbagged merchandise — and the annotated video showed PUSHOUT ALERT on 15 of
416 frames (153-167, 0.79s at 19 fps) while the POPS table reported PUSHOUT
ALERT for the cart overall.

The cause was not the classifier and not abandonment. The cart parked at y~290
from frame 105, so by frame 169 the trailing DIRECTION_WINDOW_S=4.0s delta had
fallen to 17px, under DIRECTION_MIN_DY, and the heading read UNKNOWN for the
remaining 13 seconds. compute_pops() caps an abandoned loaded cart at 65 in the
UNKNOWN branch, so every one of those frames scored ABANDONED CART, while the
finaliser — which reads direction off the peak event, logged at frame 104 while
the cart was still rolling — scored the same cart 75 / PUSHOUT ALERT.

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import DIRECTION_LATCH_FRAMES
from engine.motion import DirectionLatch
from engine.scoring import classify_event, compute_pops

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixture: frames 1..416 of that clip. Runs rather than a per-frame list,
# because the real timeline is exactly three of them.
# ---------------------------------------------------------------------------
TOTAL_FRAMES = 416
FPS = 19.0
OUTBOUND_RUN = (28, 170)        # the only OUTBOUND span in the whole clip
ABANDONED_RUNS = ((153, 167), (198, 416))


def c1_label(frame):
    if OUTBOUND_RUN[0] <= frame <= OUTBOUND_RUN[1]:
        return "OUTBOUND"
    return "UNKNOWN"


def c1_abandoned(frame):
    return any(lo <= frame <= hi for lo, hi in ABANDONED_RUNS)


def replay(latch, labels, key="C1"):
    """Feed a per-frame label sequence through the latch, return the heading it
    scored each frame with."""
    return [latch.resolve(key, lab) for lab in labels]


def score_frame(direction, abandoned):
    """What the live path scores this cart with. partial|unbagged is its voted
    fill and STATIC its speed for the whole parked stretch; merch_removed is
    False because the goods never left the cart — the cart left the store."""
    score = compute_pops(direction, "STATIC", True, "partial",
                         bag_label="unbagged", cart_detected=True,
                         abandoned=abandoned, linked=True, merch_removed=False)
    return score, classify_event(score, True, direction, abandoned=abandoned)[0]


# ---------------------------------------------------------------------------
section("the latch gate")
# ---------------------------------------------------------------------------
latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
out = replay(latch, ["OUTBOUND"] * (DIRECTION_LATCH_FRAMES - 1) + ["UNKNOWN"])
check("one frame short of the gate does NOT latch",
      out[-1] == "UNKNOWN", f"got {out[-1]!r}")

latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
out = replay(latch, ["OUTBOUND"] * DIRECTION_LATCH_FRAMES + ["UNKNOWN"] * 50)
check("a sustained outbound run holds through UNKNOWN",
      set(out) == {"OUTBOUND"}, f"got {sorted(set(out))}")

# A twitch of the trailing window on a cart parked INSIDE the store: one
# resolved OUTBOUND frame at a time, never consecutive. That is the shopper who
# parks at a shelf and steps away, and latching it would promote them from
# ABANDONED CART (65) to PUSHOUT ALERT (75).
latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
flicker = ["OUTBOUND", "UNKNOWN"] * 60
out = replay(latch, flicker)
check("outbound/unknown flicker never accumulates a latch",
      all(o == "UNKNOWN" for o, lab in zip(out, flicker) if lab == "UNKNOWN"),
      f"got {[o for o, lab in zip(out, flicker) if lab == 'UNKNOWN']}"[:60])

# ---------------------------------------------------------------------------
section("a resolved label always wins")
# ---------------------------------------------------------------------------
latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
replay(latch, ["OUTBOUND"] * DIRECTION_LATCH_FRAMES)
check("latched, then the cart is wheeled back inside: INBOUND is returned",
      latch.resolve("C1", "INBOUND") == "INBOUND")
check("...and the latch is CLEARED, so the UNKNOWN frames after it stay UNKNOWN",
      latch.resolve("C1", "UNKNOWN") == "UNKNOWN",
      "the latch must not outlive a resolved reversal")

# INBOUND is never itself latched: it is a kill switch worth INBOUND_SCORE
# before contents or abandonment are read, so holding it would silence the
# UNKNOWN abandonment tier for any cart that ever rolled inbound — including a
# cart wheeled in, loaded, and left.
latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
replay(latch, ["INBOUND"] * 200)
check("a long inbound run does NOT latch INBOUND",
      latch.resolve("C1", "UNKNOWN") == "UNKNOWN")

# Two carts must not share a latch.
latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
replay(latch, ["OUTBOUND"] * DIRECTION_LATCH_FRAMES)
check("the latch is per-key", latch.resolve("C2", "UNKNOWN") == "UNKNOWN")

# ---------------------------------------------------------------------------
section("the 1763942209000 OUTSIDE clip, replayed")
# ---------------------------------------------------------------------------
frames = list(range(1, TOTAL_FRAMES + 1))
raw_labels = [c1_label(f) for f in frames]

# Sanity: the fixture really does lose the heading at 171 and never get it back.
check("fixture: OUTBOUND ends at frame 170",
      raw_labels[169] == "OUTBOUND" and raw_labels[170] == "UNKNOWN")

latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
latched_labels = replay(latch, raw_labels)

before = [score_frame(raw_labels[f - 1], c1_abandoned(f))[1] for f in frames]
after = [score_frame(latched_labels[f - 1], c1_abandoned(f))[1] for f in frames]

n_before = before.count("PUSHOUT ALERT")
n_after = after.count("PUSHOUT ALERT")
# 15 frames is what the shipped run produced, and it is the exact overlap of
# `abandoned` (153-167, 198-416) with OUTBOUND (28-170).
check("without the latch the alert lasts 15 frames",
      n_before == 15, f"got {n_before}")
# 153-167 plus 198-416: every abandoned frame now clears HIGH_SCORE.
n_expected = sum(hi - lo + 1 for lo, hi in ABANDONED_RUNS)
check("with the latch it lasts every abandoned frame",
      n_after == n_expected == 234, f"got {n_after}, expected {n_expected}")
check("that is 0.79s -> 12.32s at 19 fps",
      (round(n_before / FPS, 2), round(n_after / FPS, 2)) == (0.79, 12.32),
      f"got {(round(n_before / FPS, 2), round(n_after / FPS, 2))}")

# The tier the cap produced on the frames it used to lose, pinned so the
# regression stays legible.
check("those frames used to read 65 / ABANDONED CART",
      score_frame("UNKNOWN", True) == (65, "ABANDONED CART"),
      f"got {score_frame('UNKNOWN', True)}")
check("latched, the same frames read 75 / PUSHOUT ALERT",
      score_frame("OUTBOUND", True) == (75, "PUSHOUT ALERT"),
      f"got {score_frame('OUTBOUND', True)}")

# The point of the fix: the live path and the finaliser now agree. The
# finaliser reads direction off the peak event (frame 104, OUTBOUND), so it
# scored 75 all along and the POPS table said PUSHOUT ALERT while the video
# said ABANDONED CART.
check("live path now matches the POPS table verdict on the final frame",
      after[-1] == "PUSHOUT ALERT", f"got {after[-1]!r}")

# Nothing before the cart was abandoned is promoted: a linked cart rolling out
# at walking pace is still scored on its contents alone.
check("frames before abandonment are unchanged by the latch",
      before[:152] == after[:152])

# ---------------------------------------------------------------------------
section("staff retrieval on that clip")
# ---------------------------------------------------------------------------
# Same clip, except at frame 250 a store employee takes the cart back inside:
# the heading resolves INBOUND, and `abandoned` goes false because the cart is
# attended again.
RETRIEVAL = 250
latch = DirectionLatch(DIRECTION_LATCH_FRAMES)
recovered_labels = [
    latch.resolve("C1", c1_label(f) if f < RETRIEVAL else "INBOUND")
    for f in frames
]
recovered = [
    score_frame(recovered_labels[f - 1],
                c1_abandoned(f) and f < RETRIEVAL)[1]
    for f in frames
]
check("the retrieval frames score the INBOUND kill switch",
      score_frame("INBOUND", False)[0] == 5,
      f"got {score_frame('INBOUND', False)[0]}")
check("no PUSHOUT frame after the cart is taken back in",
      "PUSHOUT ALERT" not in recovered[RETRIEVAL - 1:],
      f"{recovered[RETRIEVAL - 1:].count('PUSHOUT ALERT')} frames still alerting")
# ...but the alert raised BEFORE the retrieval is not retracted, and should not
# be by this change: an unattended loaded cart outside the door is a real event
# whoever collects it. Retraction is a separate decision.
check("the alert raised before the retrieval still stands",
      recovered.count("PUSHOUT ALERT") == 15 + (RETRIEVAL - 1 - 198 + 1),
      f"got {recovered.count('PUSHOUT ALERT')}")

print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
