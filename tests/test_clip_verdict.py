# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""An empty cart that was never loaded is not a theft, and the clip says so.

THE RULE
--------
An abandoned cart whose fill is "empty" used to floor at 60 OUTBOUND
(ABANDONED CART) and gain 25 on an UNKNOWN heading, whatever its history. Now
both terms are skipped when `never_loaded` holds: no cart in the CLIP has had
GRABRUN_MIN_RUN_OBS consecutive reads saying it carried something.

WHAT MUST NOT BREAK
-------------------
  * A cart that held goods and then read empty (grab-and-run) keeps every
    floor: one sustained loaded run anywhere in the clip withdraws the
    exemption, including on a DIFFERENT track, which is how a tracker that
    re-acquires the cart after the goods are gone is covered.
  * A cart with no reading at all is not exempt and cannot make a clip clean.
  * never_loaded can never be true for a clip in which peak_sustained_fill()
    would call any cart loaded (property-tested below).

THE VERDICT
-----------
scoring.clip_verdict(): any surviving event -> its name; else a cart never
read -> INSUFFICIENT EVIDENCE; else a cart held goods -> LOW PRIORITY; else
NOTHING HAPPENED. _finalise_pops() is run on hand-built engine state, so the
whole end-of-run path is covered without a video.
"""
import copy
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import cart_state as CS  # noqa: E402
from engine.config import GRABRUN_MIN_RUN_OBS  # noqa: E402
from engine.scoring import (  # noqa: E402
    compute_pops, classify_event, clip_verdict, longest_loaded_run,
    peak_sustained_fill, LOGGABLE_EVENTS, MEDIUM_SCORE,
    CLIP_NOTHING_HAPPENED, CLIP_LOW_PRIORITY, CLIP_INSUFFICIENT_EVIDENCE,
)
from engine.tracker import TrackingEngine  # noqa: E402

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


def is_event(score, linked, direction, abandoned):
    name, _ = classify_event(score, linked, direction, abandoned=abandoned)
    return name in LOGGABLE_EVENTS and score >= MEDIUM_SCORE


# ---------------------------------------------------------------------------
section("compute_pops: never_loaded only touches an abandoned EMPTY cart")
for speed in ("STATIC", "SLOW", "MEDIUM", "FAST"):
    for linked in (False, True):
        before = compute_pops("OUTBOUND", speed, True, "empty",
                              abandoned=True, linked=linked)
        after = compute_pops("OUTBOUND", speed, True, "empty",
                             abandoned=True, linked=linked, never_loaded=True)
        check(f"OUTBOUND empty abandoned {speed} linked={linked}: 60 floor before",
              before == 60, f"got {before}")
        check(f"OUTBOUND empty abandoned {speed} linked={linked}: no event after",
              not is_event(after, linked, "OUTBOUND", True), f"scored {after}")
    before = compute_pops("UNKNOWN", speed, True, "empty", abandoned=True)
    after = compute_pops("UNKNOWN", speed, True, "empty", abandoned=True,
                         never_loaded=True)
    check(f"UNKNOWN empty abandoned {speed}: +25 skipped",
          before - after == 25, f"{before} -> {after}")
    check(f"UNKNOWN empty abandoned {speed}: no event after",
          not is_event(after, False, "UNKNOWN", True), f"scored {after}")

for direction in ("OUTBOUND", "UNKNOWN"):
    for fill, bag in (("partial", "unbagged"), ("full", "unbagged"),
                      ("full", "bagged"), ("partial", "bagged")):
        a = compute_pops(direction, "SLOW", True, fill, bag_label=bag, abandoned=True)
        b = compute_pops(direction, "SLOW", True, fill, bag_label=bag, abandoned=True,
                         never_loaded=True)
        check(f"{direction} {fill}|{bag} abandoned: never_loaded changes nothing",
              a == b, f"{a} vs {b}")
    a = compute_pops(direction, "SLOW", True, "unclassified", abandoned=True)
    b = compute_pops(direction, "SLOW", True, "unclassified", abandoned=True,
                     never_loaded=True)
    check(f"{direction} unclassified abandoned: no reading is not exempt",
          a == b, f"{a} vs {b}")

a = compute_pops("OUTBOUND", "FAST", True, "empty")
b = compute_pops("OUTBOUND", "FAST", True, "empty", never_loaded=True)
check("not abandoned: never_loaded changes nothing", a == b, f"{a} vs {b}")
check("INBOUND kill switch unchanged",
      compute_pops("INBOUND", "FAST", True, "empty", abandoned=True,
                   never_loaded=True) == 5)


# ---------------------------------------------------------------------------
section("longest_loaded_run is never weaker than peak_sustained_fill")
check("empty sequence", longest_loaded_run([]) == 0)
check("partial and full share one run",
      longest_loaded_run([f in ("partial", "full") for f in
                          ["partial", "full", "partial", "full", "partial"]]) == 5)
check("an empty read breaks the run",
      longest_loaded_run([True, True, False, True, True, True]) == 3)

rng = random.Random(0)
violations = 0
for _ in range(20000):
    n = rng.randint(0, 40)
    fills = [rng.choice(("empty", "empty", "partial", "full")) for _ in range(n)]
    loaded = [f in ("partial", "full") for f in fills]
    if (longest_loaded_run(loaded) < GRABRUN_MIN_RUN_OBS
            and peak_sustained_fill(fills, GRABRUN_MIN_RUN_OBS) is not None):
        violations += 1
check("20000 random histories: never_loaded implies no sustained fill",
      violations == 0, f"{violations} violations")


# ---------------------------------------------------------------------------
section("_record_loaded_read: what counts as a loaded read")


def bare_engine():
    """A TrackingEngine with only the state the end-of-run path touches."""
    eng = TrackingEngine.__new__(TrackingEngine)
    eng._cart_loaded_reads = defaultdict(list)
    eng._clip_loaded = False
    eng._clip_verdict = None
    eng._event_log = []
    eng._peak_pops_snapshot = {}
    eng._cart_cls_history = defaultdict(list)
    eng._max_pops_per_cart = {}
    eng._pops_cache = {}
    return eng


VALID_EMPTY = {"quality": "valid_cart", "fill": "empty", "bag": "not_applicable"}
VALID_PARTIAL = {"quality": "valid_cart", "fill": "partial", "bag": "unbagged"}
UNREADABLE = {"quality": "unclear", "fill": "partial", "bag": "unbagged"}
DET_EMPTY = CS.detector_empty_result(0.1)
DET_LOADED_BAD_CROP = {"quality": "unclear", "fill": "partial",
                       "detector_state": CS.NON_EMPTY, "p_non_empty": 0.9}

eng = bare_engine()
for r in (DET_EMPTY, VALID_PARTIAL, UNREADABLE, DET_LOADED_BAD_CROP, VALID_EMPTY):
    eng._record_loaded_read(1, r)
check("reads recorded in order, the unreadable one skipped",
      eng._cart_loaded_reads[1] == [False, True, True, False],
      str(eng._cart_loaded_reads[1]))

eng = bare_engine()
for _ in range(GRABRUN_MIN_RUN_OBS):
    eng._record_loaded_read(1, DET_LOADED_BAD_CROP)
check("NON-EMPTY reads with unreadable crops still make the clip loaded",
      eng._clip_loaded)

eng = bare_engine()
for i in range(GRABRUN_MIN_RUN_OBS - 1):
    eng._record_loaded_read(1, VALID_PARTIAL)
eng._record_loaded_read(1, VALID_EMPTY)
eng._record_loaded_read(1, VALID_PARTIAL)
check(f"a run of {GRABRUN_MIN_RUN_OBS - 1} is not sustained", not eng._clip_loaded)

eng = bare_engine()
for _ in range(GRABRUN_MIN_RUN_OBS):
    eng._record_loaded_read(1, VALID_PARTIAL)
eng._record_loaded_read(2, VALID_EMPTY)
check("one loaded cart makes the whole clip loaded", eng._clip_loaded)
for _ in range(20):
    eng._record_loaded_read(1, VALID_EMPTY)
check("and it stays latched when that cart reads empty later", eng._clip_loaded)


# ---------------------------------------------------------------------------
section("clip_verdict precedence")
ev = lambda cd, name, score: {"cart_id": cd, "event": name, "pops_score": score}  # noqa: E731

v = clip_verdict([], {}, False)
check("no carts -> NOTHING HAPPENED", v["verdict"] == CLIP_NOTHING_HAPPENED, v["verdict"])
v = clip_verdict([], {1: "empty", 2: "empty"}, False)
check("all empty, never loaded -> NOTHING HAPPENED",
      v["verdict"] == CLIP_NOTHING_HAPPENED and v["tier"] == "NONE", v["verdict"])
v = clip_verdict([], {1: "empty"}, True)
check("empty vote but the clip held a loaded run -> LOW PRIORITY",
      v["verdict"] == CLIP_LOW_PRIORITY, v["verdict"])
v = clip_verdict([], {1: "partial", 2: "empty"}, False)
check("a cart voted partial -> LOW PRIORITY", v["verdict"] == CLIP_LOW_PRIORITY, v["verdict"])
v = clip_verdict([], {1: "empty", 2: None}, False)
check("an unread cart -> INSUFFICIENT EVIDENCE",
      v["verdict"] == CLIP_INSUFFICIENT_EVIDENCE and v["carts_unassessed"] == [2],
      str(v))
v = clip_verdict([], {1: "partial", 2: "unclassified"}, True)
check("unread beats loaded: INSUFFICIENT EVIDENCE",
      v["verdict"] == CLIP_INSUFFICIENT_EVIDENCE, v["verdict"])
v = clip_verdict([ev(1, "MEDIUM PRIORITY", 40), ev(2, "PUSHOUT ALERT", 75),
                  ev(3, "HIGH PRIORITY", 78)], {1: "partial", 2: None, 3: "full"}, True)
check("events beat everything, most severe first",
      v["verdict"] == "PUSHOUT ALERT" and v["tier"] == "HIGH" and v["top_cart"] == 2,
      str(v))
v = clip_verdict([ev(4, "ABANDONED CART", 60)], {4: "empty"}, False)
check("an ABANDONED CART row is a MEDIUM verdict",
      v["verdict"] == "ABANDONED CART" and v["tier"] == "MEDIUM", str(v))


# ---------------------------------------------------------------------------
section("_finalise_pops on hand-built state")


def abandoned_empty_cart_state(clip_loaded):
    """Cart 1: read empty 20 times, owner walked off, live row ABANDONED CART 60."""
    eng = bare_engine()
    eng._clip_loaded = clip_loaded
    eng._cart_cls_history[1] = [("empty", "not_applicable", 0.9, 1.0)] * 20
    row = {"frame": 120, "timestamp": 6.0, "cart_id": 1, "event": "ABANDONED CART",
           "pops_score": 60, "fill": "empty", "bag": "not_applicable",
           "direction": "OUTBOUND", "linked": False, "speed_status": "SLOW",
           "abandoned": True, "never_loaded": False}
    eng._event_log = [row]
    eng._peak_pops_snapshot[1] = {
        "score": 60, "event": "ABANDONED CART", "color": (0, 0, 0),
        "fill": "empty", "bag": "not_applicable", "direction": "OUTBOUND",
        "quality": "valid_cart", "speed_status": "SLOW", "linked": False,
        "abandoned": True, "merch_removed": False, "never_loaded": False,
        "owner": 3, "timestamp": 6.0, "frame": 120}
    eng._max_pops_per_cart[1] = 60
    eng._pops_cache[1] = dict(eng._peak_pops_snapshot[1])
    return eng


eng = abandoned_empty_cart_state(clip_loaded=False)
eng._finalise_pops()
snap = eng._peak_pops_snapshot[1]
check("never loaded: the abandoned empty cart scores no event",
      snap["score"] < MEDIUM_SCORE and snap["never_loaded"], str(snap))
check("never loaded: its live ABANDONED CART row is pruned", eng._event_log == [],
      str(eng._event_log))
check("never loaded: verdict NOTHING HAPPENED",
      eng._clip_verdict["verdict"] == CLIP_NOTHING_HAPPENED, str(eng._clip_verdict))

eng = abandoned_empty_cart_state(clip_loaded=True)
eng._finalise_pops()
snap = eng._peak_pops_snapshot[1]
check("loaded elsewhere in the clip (split track): 60 floor kept",
      snap["score"] == 60 and snap["event"] == "ABANDONED CART"
      and not snap["never_loaded"], str(snap))
check("loaded elsewhere: verdict ABANDONED CART",
      eng._clip_verdict["verdict"] == "ABANDONED CART", str(eng._clip_verdict))
check("the surviving row carries never_loaded=False",
      eng._event_log and eng._event_log[0]["never_loaded"] is False, str(eng._event_log))

# Grab-and-run on ONE track: loaded run then an empty tail, owner gone. The
# engine latches _clip_loaded from those reads, so the floor must hold.
eng = abandoned_empty_cart_state(clip_loaded=False)
fills = ["partial"] * 9 + ["empty"] * 12
eng._cart_cls_history[1] = [(f, "unbagged" if f == "partial" else "not_applicable",
                             0.9, 0.9) for f in fills]
for f in fills:
    eng._record_loaded_read(1, {"quality": "valid_cart", "fill": f})
eng._finalise_pops()
snap = eng._peak_pops_snapshot[1]
check("grab-and-run on one track: still a PUSHOUT ALERT",
      snap["event"] == "PUSHOUT ALERT" and snap["score"] >= 75, str(snap))

# A second cart with no valid reading makes the clip unprovable.
eng = abandoned_empty_cart_state(clip_loaded=False)
eng._max_pops_per_cart[2] = 5
eng._pops_cache[2] = {"score": 5, "fill": "unclassified"}
eng._peak_pops_snapshot[2] = {"score": 5, "event": "LOW PRIORITY", "fill": "unclassified",
                              "direction": "UNKNOWN", "speed_status": "STATIC"}
eng._finalise_pops()
check("an unread second cart -> INSUFFICIENT EVIDENCE",
      eng._clip_verdict["verdict"] == CLIP_INSUFFICIENT_EVIDENCE
      and eng._clip_verdict["carts_unassessed"] == [2], str(eng._clip_verdict))

# Replaying the same captured state twice gives the same answer: the offline
# evaluation depends on _finalise_pops() being a pure function of its state.
a = abandoned_empty_cart_state(clip_loaded=False)
b = copy.deepcopy(a)
a._finalise_pops()
b._finalise_pops()
check("replay is deterministic",
      a._clip_verdict == b._clip_verdict and a._peak_pops_snapshot == b._peak_pops_snapshot)


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
