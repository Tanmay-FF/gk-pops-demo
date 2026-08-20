# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Grab-and-run pushout escalation — pure scoring, no GPU and no video decode.

Run with:  python tests/test_grabrun_pushout.py

The policy under test: an abandoned cart whose classification history shows a
sustained run of LOADED observations followed by an EMPTY tail is a pushout
regardless of the direction label, because the merchandise left the cart while
the owner left the frame. Before this, that cart could only reach PUSHOUT ALERT
through the OUTBOUND branch's max(score, 75) floor, so the same incident
finished as "ABANDONED CART" at 65 whenever the direction window failed to
resolve OUTBOUND — see the 1764099569430 OUTSIDE clip, Cart 1, in
test_outside_clip_bag_label.py.

The negative cases matter more than the positive one. `abandoned` is not a
theft signal: tracker.py computes it from ABANDON_FRAMES (30) of lost person
track, or 30 frames with the owner further from the cart than
WALKAWAY_GAP_FRAC of its box diagonal, roughly a second at 30 fps. A
shopper who parks a loaded cart and steps to a shelf trips it constantly. What
those carts never produce is an empty tail — their fill stays loaded — and that
is the whole discriminator. Every "must NOT escalate" check below is guarding
against turning ordinary in-store shopping into pushout alerts.

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import GRABRUN_MIN_RUN_OBS
from engine.scoring import (
    HIGH_SCORE, LOGGABLE_EVENTS, MERCH_REMOVED_FLOOR, classify_event,
    compute_pops, merchandise_removed, prune_event_log,
    sync_events_with_snapshots,
)

_PASS: list[str] = []
_FAIL: list[str] = []


def check(label, ok, detail=""):
    (_PASS if ok else _FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))


def section(title):
    print(f"\n=== {title} ===")


def _score(fill, bag, direction="UNKNOWN", speed="STATIC",
           abandoned=True, linked=True, merch_removed=False):
    return compute_pops(direction, speed, True, fill, bag_label=bag,
                        cart_detected=True, abandoned=abandoned,
                        linked=linked, merch_removed=merch_removed)


def _event(score, direction="UNKNOWN", abandoned=True, linked=True):
    return classify_event(score, linked, direction, abandoned=abandoned)[0]


# ---------------------------------------------------------------------------
section("merchandise_removed — what counts as evidence")
# ---------------------------------------------------------------------------
_LOADED_RUN = ["partial"] * GRABRUN_MIN_RUN_OBS

check("a loaded run followed by an empty tail is removal",
      merchandise_removed(_LOADED_RUN + ["empty"] * 5, GRABRUN_MIN_RUN_OBS))

# The real door-side history: the cart is occluded and re-exposed, so a stray
# loaded observation lands inside the empty tail. The last observation is still
# empty, which is the question being asked.
check("a stray loaded frame inside the empty tail does not disqualify",
      merchandise_removed(["partial"] * 15 + ["empty"] * 3 + ["partial"]
                          + ["empty"] * 32, GRABRUN_MIN_RUN_OBS))

# Cart 1 of the 1764092528600 OUTSIDE clip, per-observation out of that run's
# tracking JSON. The goods plainly left this cart — 27 loaded observations, then
# two empty runs — and the FINAL observation re-reads "partial" at fill
# confidence 0.502. Tested because one 0.50 frame decided the tier: on an
# UNKNOWN heading the difference is 65 / ABANDONED CART against
# MERCH_REMOVED_FLOOR / PUSHOUT ALERT.
GOLDEN_1764092528600_C1 = (["partial"] * 27 + ["empty"] * 4
                           + ["partial"] * 4 + ["empty"] * 4 + ["partial"])
check("one trailing loaded observation does not undo an empty tail",
      merchandise_removed(GOLDEN_1764092528600_C1, GRABRUN_MIN_RUN_OBS))

# The other side of that tolerance, and why it is a RUN test and not a
# last-observation test with a fudge factor: the trailing loaded run in the
# parked-cart case below is exactly GRABRUN_MIN_RUN_OBS long, so it survives
# stripping and the cart still reads as ending loaded.
check("a cart that ends loaded is NOT removal (parked cart)",
      merchandise_removed(_LOADED_RUN + ["empty"] * 5 + ["partial"] * 4,
                          GRABRUN_MIN_RUN_OBS) is False)

check("a trailing loaded run one observation over the gate still ends loaded",
      merchandise_removed(_LOADED_RUN + ["empty"] * 5
                          + ["partial"] * (GRABRUN_MIN_RUN_OBS + 1),
                          GRABRUN_MIN_RUN_OBS) is False)

# Cart 1 of the 1763942209000 OUTSIDE clip: 40 observations, not one of them
# empty. The cart left the store WITH its merchandise, which is a pushout the
# direction latch scores as one — it is not merchandise being removed, and this
# field must not claim it is.
check("a cart that never read empty is NOT removal",
      merchandise_removed(["full"] * 3 + ["partial"] * 4 + ["full"]
                          + ["partial"] * 15 + ["full"] * 4 + ["partial"] * 6,
                          GRABRUN_MIN_RUN_OBS) is False)

check("a cart that was loaded the whole time is NOT removal",
      merchandise_removed(["full"] * 20, GRABRUN_MIN_RUN_OBS) is False)

# One noisy "partial" on a genuinely empty cart is the classifier being wrong,
# not merchandise. Without the run gate this alone would float an empty
# abandoned cart from 60 to a pushout.
check("a single noisy loaded frame cannot sustain a run",
      merchandise_removed(["empty"] * 10 + ["partial"] + ["empty"] * 10,
                          GRABRUN_MIN_RUN_OBS) is False)

check("a run one observation short of the gate does not qualify",
      merchandise_removed(["partial"] * (GRABRUN_MIN_RUN_OBS - 1) + ["empty"] * 8,
                          GRABRUN_MIN_RUN_OBS) is False)

check("an all-empty history is NOT removal",
      merchandise_removed(["empty"] * 12, GRABRUN_MIN_RUN_OBS) is False)

check("an empty history is NOT removal",
      merchandise_removed([], GRABRUN_MIN_RUN_OBS) is False)


# ---------------------------------------------------------------------------
section("UNKNOWN direction — the escalation")
# ---------------------------------------------------------------------------
# This is the cart the policy exists for: partial, unbagged, abandoned, the
# items gone. It scored 65 / ABANDONED CART purely because the direction window
# never resolved OUTBOUND.
_before = _score("partial", "unbagged")
_after = _score("partial", "unbagged", merch_removed=True)
check("without the evidence it is still capped at 65", _before == 65,
      f"got {_before}")
check("with the evidence it reaches MERCH_REMOVED_FLOOR",
      _after == MERCH_REMOVED_FLOOR, f"got {_after}")
check("MERCH_REMOVED_FLOOR clears HIGH_SCORE so the abandonment route opens",
      MERCH_REMOVED_FLOOR >= HIGH_SCORE)
check("event becomes PUSHOUT ALERT", _event(_after) == "PUSHOUT ALERT",
      f"got {_event(_after)!r}")
check("event without the evidence stays ABANDONED CART",
      _event(_before) == "ABANDONED CART", f"got {_event(_before)!r}")

_full = _score("full", "unbagged", merch_removed=True)
check("a full unbagged cart escalates the same way",
      _full >= MERCH_REMOVED_FLOOR and _event(_full) == "PUSHOUT ALERT",
      f"got {_full} / {_event(_full)!r}")

# Bagged means the items look paid for, which is the one reading that survives
# a cart going empty innocently: the customer transferred their own bags. The
# partial+bagged cap of 55 is spec (see test_grabrun_override.py) and the
# escalation must not lift it.
_bagged = _score("partial", "bagged", merch_removed=True)
check("partial + bagged is still capped at 55 even with the evidence",
      _bagged == 55, f"got {_bagged}")
check("and does not become a pushout", _event(_bagged) != "PUSHOUT ALERT",
      f"got {_event(_bagged)!r}")


# ---------------------------------------------------------------------------
section("what the escalation must NOT touch")
# ---------------------------------------------------------------------------
# The flag is only ever True alongside abandonment, but a stale True must not
# be able to invent a score on its own.
check("a not-abandoned UNKNOWN cart is unaffected",
      _score("partial", "unbagged", abandoned=False)
      == _score("partial", "unbagged", abandoned=False, merch_removed=True))

# An abandoned EMPTY cart takes the UNKNOWN branch's +25, and the escalation is
# gated on partial/full so it cannot reach it. (The 60 floor is OUTBOUND's.)
_empty_score = _score("empty", "not_applicable", merch_removed=True)
check("an abandoned EMPTY cart is untouched by the escalation",
      _empty_score == _score("empty", "not_applicable") == 25,
      f"got {_empty_score}")

_inbound = _score("partial", "unbagged", direction="INBOUND", merch_removed=True)
check("the INBOUND kill switch still returns 5 ahead of everything",
      _inbound == 5, f"got {_inbound}")
_invalid = compute_pops("UNKNOWN", "STATIC", False, "partial",
                        bag_label="unbagged", abandoned=True, linked=True,
                        merch_removed=True)
check("the not-valid kill switch still returns 5", _invalid == 5,
      f"got {_invalid}")

_out = _score("partial", "unbagged", direction="OUTBOUND")
check("OUTBOUND abandonment already floored at MERCH_REMOVED_FLOOR",
      _out >= MERCH_REMOVED_FLOOR, f"got {_out}")
check("and the flag changes nothing there",
      _score("partial", "unbagged", direction="OUTBOUND", merch_removed=True)
      == _out)

check("no cart detected still scores 0",
      compute_pops("UNKNOWN", "STATIC", True, "partial", bag_label="unbagged",
                   cart_detected=False, abandoned=True, merch_removed=True) == 0)


# ---------------------------------------------------------------------------
section("the escalation survives the event/snapshot sync in both directions")
# ---------------------------------------------------------------------------
# The live per-frame path and the end-of-run finaliser both score this cart, and
# sync_events_with_snapshots() keeps whichever read scored HIGHER as a unit. So
# the escalation only reaches the user if it survives regardless of which of the
# two paths produced it — a 75 that gets voted back to 65 by the other side is
# precisely the orig=75/recomp=60 defect the finaliser exists to prevent.
_ESCALATED = MERCH_REMOVED_FLOOR
_esc_event_name, _ = classify_event(_ESCALATED, True, "UNKNOWN", abandoned=True)

# Finaliser escalated, live row did not (e.g. logged before the empty tail).
_events = [{"cart_id": 1, "frame": 40, "event": "ABANDONED CART",
            "pops_score": 65, "fill": "partial", "bag": "unbagged"}]
_snapshots = {1: {"fill": "partial", "bag": "unbagged", "score": _ESCALATED,
                  "event": _esc_event_name}}
_max = {1: 65}
sync_events_with_snapshots(_events, _snapshots, _max)
check("a 65 live row is rewritten from the escalated snapshot",
      (_events[0]["pops_score"], _events[0]["event"])
      == (_ESCALATED, "PUSHOUT ALERT"),
      f"got {(_events[0]['pops_score'], _events[0]['event'])}")

# Live path escalated, finaliser recomputed lower (its `abandoned` came from a
# best-event context that did not carry the flag). The live reading is the floor.
_events = [{"cart_id": 2, "frame": 40, "event": _esc_event_name,
            "pops_score": _ESCALATED, "fill": "partial", "bag": "unbagged"}]
_snapshots = {2: {"fill": "partial", "bag": "unbagged", "score": 65,
                  "event": "ABANDONED CART"}}
_max = {2: 65}
sync_events_with_snapshots(_events, _snapshots, _max)
check("an escalated live row is not demoted by a 65 snapshot",
      (_events[0]["pops_score"], _events[0]["event"])
      == (_ESCALATED, "PUSHOUT ALERT"),
      f"got {(_events[0]['pops_score'], _events[0]['event'])}")
check("and the snapshot is lifted with it, so table and row agree",
      (_snapshots[2]["score"], _snapshots[2]["event"])
      == (_ESCALATED, "PUSHOUT ALERT"),
      f"got {(_snapshots[2]['score'], _snapshots[2]['event'])}")
check("max_pops follows the escalated reading", _max[2] == _ESCALATED,
      f"got {_max[2]}")

# The rewritten name has to still be loggable, or prune_event_log() drops the
# alert on the floor after the sync just promoted it.
check("PUSHOUT ALERT is a loggable event", "PUSHOUT ALERT" in LOGGABLE_EVENTS)
_kept, _dropped = prune_event_log(_events)
check("the promoted row survives pruning", len(_kept) == 1 and _dropped == 0,
      f"got kept={len(_kept)} dropped={_dropped}")


# ---------------------------------------------------------------------------
section("both tracker call sites are wired")
# ---------------------------------------------------------------------------
# The live per-frame path and the end-of-run finaliser both score the same cart,
# and a POPS table that disagrees with the printed live score is exactly the
# class of defect the finaliser was written to remove. Wiring only one of them
# reintroduces it, and no pure-scoring assertion can see that.
_tracker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "engine", "tracker.py")
with open(_tracker, encoding="utf-8") as fh:
    _src = fh.read()
_n_pass = _src.count("merch_removed=merch_removed")
_n_derive = _src.count("merchandise_removed(")
check("tracker.py passes merch_removed at both compute_pops call sites",
      _n_pass == 2, f"got {_n_pass}")
check("tracker.py derives it from merchandise_removed()", _n_derive == 2,
      f"got {_n_derive}")

print("\n" + "=" * 62)
print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
sys.exit(1 if _FAIL else 0)
