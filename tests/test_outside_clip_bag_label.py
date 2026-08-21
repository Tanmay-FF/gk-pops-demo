# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Bag-label regression tests for the 1764099569430 OUTSIDE clip.

Run with:  python tests/test_outside_clip_bag_label.py

The report: Cart 1 was a partially filled, UNBAGGED cart left outside the exit
gate; the person then lifted the items out and ran. The run log ended with

    [GRAB-RUN] Cart 1: empty vote overridden to 'partial' | sustained run >= 4
    [POPS] Cart 1: partial|unbagged UNKNOWN score=65 (orig=55 recomp=65)

and the UI showed PARTIAL / BAGGED / 55 / ABANDONED CART. Two separate defects
produced that, and each is pinned below.

  1. The live grab-and-run path in the frame loop restored the peak fill from
     history but copied the bag label from ONE frame - the first frame reaching
     the peak fill rank, because the running `fill_lbl` was reassigned inside
     the comparison loop. That frame read "bagged" while the whole history votes
     unbagged 6.59 to 5.62, and partial+bagged is capped at 55 by
     compute_pops(), so the event row was logged 10 points under the
     abandonment floor the cart had earned.

  2. The end-of-run sync then copied that event row BACK over the reconciled
     snapshot ("Events is truth for abandonment"), after the [POPS] line had
     already been printed. So the log described a verdict the UI never showed,
     and reconciliation could not raise any cart out of the 55 cap - a
     reconciled PUSHOUT ALERT would have been demoted to ABANDONED CART.

The fixture reproduces the logged aggregates for Cart 1 exactly: 51
observations, fills in the order the [DEBUG] line recorded, and bag counts
bagged 7 / unbagged 9 / not_applicable 35 with the confidence sums from the
[VOTE] line. The first loaded frame is "bagged" because that is the frame that
decided the label under the old code.

Deliberately stdlib only, no pytest - matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import GRABRUN_MIN_RUN_OBS
from engine.scoring import (
    MERCH_REMOVED_FLOOR, classify_event, compute_pops, merchandise_removed,
    peak_sustained_fill, sync_events_with_snapshots, vote_bag_for_loaded_cart,
)

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixture: (fill, bag, fill_conf, bag_conf) as _cart_cls_history holds it.
# 15 loaded observations, then the cart is emptied.
# ---------------------------------------------------------------------------
def _obs(fill, bag, fc, bc):
    return (fill, bag, fc, bc)


# bagged: 7 obs, conf sum 5.6223 | unbagged: 9 obs, conf sum 6.5865
_BAGGED_CONFS = [0.8113, 0.7935, 0.8402, 0.8010, 0.7684, 0.7920, 0.8159]
_UNBAGGED_CONFS = [0.7402, 0.7280, 0.7515, 0.7196, 0.7333, 0.7108, 0.7460,
                   0.7290, 0.7281]

# The bag reading is genuinely mixed at door distance. What matters for the
# regression is that observation 0 - the first frame to reach fill rank
# "partial" - is one of the bagged ones.
_BAG_SEQUENCE = (
    [("bagged", c) for c in _BAGGED_CONFS[:1]]
    + [("unbagged", c) for c in _UNBAGGED_CONFS[:2]]
    + [("bagged", c) for c in _BAGGED_CONFS[1:3]]
    + [("unbagged", c) for c in _UNBAGGED_CONFS[2:5]]
    + [("bagged", c) for c in _BAGGED_CONFS[3:6]]
    + [("unbagged", c) for c in _UNBAGGED_CONFS[5:8]]
    + [("bagged", c) for c in _BAGGED_CONFS[6:]]
    + [("unbagged", c) for c in _UNBAGGED_CONFS[8:]]
)

# Fill order exactly as the [DEBUG] line recorded it: 15 partial, then a mostly
# empty tail with one stray partial at index 18.
_FILL_ORDER = (
    ["partial"] * 15
    + ["empty"] * 3 + ["partial"] + ["empty"] * 32
)

OUTSIDE_C1 = []
_loaded_i = 0
for _fill in _FILL_ORDER:
    if _fill == "partial":
        _bag, _bc = _BAG_SEQUENCE[_loaded_i]
        _loaded_i += 1
        OUTSIDE_C1.append(_obs("partial", _bag, 0.8374, _bc))
    else:
        # An empty cart always reports not_applicable with confidence 1.0.
        OUTSIDE_C1.append(_obs("empty", "not_applicable", 0.8969, 1.0))


# Both helpers take the CACHED per-frame labels the frame loop starts from, so
# a non-firing override has to leave them untouched - that is the path every
# abandoned cart which is genuinely empty still takes.
def _old_live_bag(history, fill_lbl="empty", bag_lbl="not_applicable"):
    """The pre-fix live grab-and-run path, verbatim, for contrast."""
    rank = {"empty": 0, "partial": 1, "full": 2}
    for h_fill, h_bag, _, _ in history:
        if rank.get(h_fill, 0) > rank.get(fill_lbl, 0):
            fill_lbl = h_fill
            bag_lbl = h_bag
    return fill_lbl, bag_lbl


def _new_live_bag(history, fill_lbl="empty", bag_lbl="not_applicable"):
    """The fixed live grab-and-run path, mirroring tracker.py."""
    rank = {"empty": 0, "partial": 1, "full": 2}
    peak = max((f for f, _, _, _ in history), key=lambda f: rank.get(f, 0),
               default=fill_lbl)
    if rank.get(peak, 0) > rank.get(fill_lbl, 0):
        fill_lbl = peak
        bag_lbl = vote_bag_for_loaded_cart(history)
    return fill_lbl, bag_lbl


# ---------------------------------------------------------------------------
section("fixture matches the logged aggregates")
# ---------------------------------------------------------------------------
_counts: dict[str, int] = {}
_confs: dict[str, float] = {}
for _f, _b, _fc, _bc in OUTSIDE_C1:
    _counts[_b] = _counts.get(_b, 0) + 1
    _confs[_b] = _confs.get(_b, 0.0) + _bc
check("51 observations", len(OUTSIDE_C1) == 51, f"got {len(OUTSIDE_C1)}")
check("bag counts match the [VOTE] line",
      (_counts["bagged"], _counts["unbagged"], _counts["not_applicable"])
      == (7, 9, 35), f"got {_counts}")
check("bagged conf sum ~5.62", abs(_confs["bagged"] - 5.6223) < 0.01,
      f"got {_confs['bagged']:.4f}")
check("unbagged conf sum ~6.59", abs(_confs["unbagged"] - 6.5865) < 0.01,
      f"got {_confs['unbagged']:.4f}")

# ---------------------------------------------------------------------------
section("defect 1 — the live path read ONE frame's bag label")
# ---------------------------------------------------------------------------
_old_fill, _old_bag = _old_live_bag(OUTSIDE_C1)
check("old path restores fill to partial", _old_fill == "partial",
      f"got {_old_fill!r}")
check("old path lands on the first loaded frame's bag", _old_bag == "bagged",
      f"got {_old_bag!r}")

_new_fill, _new_bag = _new_live_bag(OUTSIDE_C1)
check("fixed path restores fill to partial", _new_fill == "partial",
      f"got {_new_fill!r}")
check("fixed path VOTES the bag across the history", _new_bag == "unbagged",
      f"got {_new_bag!r}")
check("not_applicable votes are excluded from the bag vote",
      vote_bag_for_loaded_cart(OUTSIDE_C1) == "unbagged")
check("a history with no loaded observation defaults to unbagged",
      vote_bag_for_loaded_cart(
          [_obs("empty", "not_applicable", 0.9, 1.0)] * 5) == "unbagged")
check("an empty history defaults to unbagged",
      vote_bag_for_loaded_cart(()) == "unbagged")

# A genuinely empty abandoned cart must come out of the block with the cached
# labels untouched - the guard must not reset the bag to anything.
_all_empty = [_obs("empty", "not_applicable", 0.9, 1.0)] * 6
check("override that does not fire leaves fill/bag alone",
      _new_live_bag(_all_empty) == ("empty", "not_applicable"),
      f"got {_new_live_bag(_all_empty)}")
check("a cart already reading partial keeps its own cached bag",
      _new_live_bag(OUTSIDE_C1, fill_lbl="partial", bag_lbl="bagged")
      == ("partial", "bagged"),
      f"got {_new_live_bag(OUTSIDE_C1, 'partial', 'bagged')}")

# ---------------------------------------------------------------------------
section("defect 1 — what the bag label costs in score")
# ---------------------------------------------------------------------------
# Person static outside the exit gate: direction UNKNOWN, speed STATIC.
def _score(fill, bag):
    return compute_pops("UNKNOWN", "STATIC", True, fill, bag_label=bag,
                        cart_detected=True, abandoned=True, linked=True)


_score_bagged = _score("partial", "bagged")
_score_unbagged = _score("partial", "unbagged")
check("one noisy frame's 'bagged' caps the cart at 55", _score_bagged == 55,
      f"got {_score_bagged}")
check("the voted 'unbagged' reaches the UNKNOWN abandonment floor of 65",
      _score_unbagged == 65, f"got {_score_unbagged}")
check("the sustained-run gate fires on this history",
      peak_sustained_fill([f for f, _, _, _ in OUTSIDE_C1],
                          GRABRUN_MIN_RUN_OBS) == "partial")

# ---------------------------------------------------------------------------
section("defect 2 — the sync must not clobber reconciliation")
# ---------------------------------------------------------------------------
_snap_score = _score("partial", "unbagged")
_snap_event, _ = classify_event(_snap_score, True, "UNKNOWN", abandoned=True)
snapshots = {1: {"fill": "partial", "bag": "unbagged", "score": _snap_score,
                 "event": _snap_event}}
events = [{"cart_id": 1, "frame": 90, "event": "ABANDONED CART",
           "pops_score": 55, "fill": "partial", "bag": "bagged"}]
max_pops = {1: _snap_score}
notes = sync_events_with_snapshots(events, snapshots, max_pops)

check("reconciled bag survives the sync", snapshots[1]["bag"] == "unbagged",
      f"got {snapshots[1]['bag']!r}")
check("reconciled score survives the sync", snapshots[1]["score"] == 65,
      f"got {snapshots[1]['score']}")
check("max_pops keeps the reconciled score", max_pops[1] == 65,
      f"got {max_pops[1]}")
check("the event row is rewritten from the snapshot",
      (events[0]["bag"], events[0]["pops_score"]) == ("unbagged", 65),
      f"got {(events[0]['bag'], events[0]['pops_score'])}")
check("no override note when reconciliation wins", notes == [],
      f"got {notes}")

# The reverse direction, and the price of making the vote the only authority: a
# live PUSHOUT ALERT at 75 against a reconciled 60 is rewritten DOWN. There is no
# score floor in either path any more — see sync_events_with_snapshots(). The
# escalation this file is about survives because the finaliser re-derives
# merch_removed from the same history the vote reads, not because the live frame
# is held against it.
snapshots = {2: {"fill": "empty", "bag": "not_applicable", "score": 60,
                 "event": "ABANDONED CART"}}
events = [{"cart_id": 2, "frame": 120, "event": "PUSHOUT ALERT",
           "pops_score": 75, "fill": "partial", "bag": "unbagged"}]
max_pops = {2: 60}
notes = sync_events_with_snapshots(events, snapshots, max_pops)
check("the reconciled reading stands, high live score or not",
      snapshots[2]["score"] == 60, f"got {snapshots[2]['score']}")
check("the snapshot is not blended with the live row",
      (snapshots[2]["fill"], snapshots[2]["bag"], snapshots[2]["event"])
      == ("empty", "not_applicable", "ABANDONED CART"),
      f"got {(snapshots[2]['fill'], snapshots[2]['bag'], snapshots[2]['event'])}")
check("and the row is brought down to it",
      (events[0]["pops_score"], events[0]["fill"], events[0]["event"])
      == (60, "empty", "ABANDONED CART"),
      f"got {(events[0]['pops_score'], events[0]['fill'], events[0]['event'])}")
check("max_pops stays reconciled", max_pops[2] == 60, f"got {max_pops[2]}")
check("the demotion is reported", len(notes) == 1, f"got {notes}")

# Every row for a cart is rewritten, not only the last one.
snapshots = {3: {"fill": "partial", "bag": "unbagged", "score": 65,
                 "event": "ABANDONED CART"}}
events = [
    {"cart_id": 3, "frame": 40, "event": "MEDIUM PRIORITY", "pops_score": 33,
     "fill": "partial", "bag": "bagged"},
    {"cart_id": 3, "frame": 90, "event": "ABANDONED CART", "pops_score": 55,
     "fill": "partial", "bag": "bagged"},
]
sync_events_with_snapshots(events, snapshots, {3: 65})
check("earlier rows are reconciled too",
      all(r["pops_score"] == 65 and r["bag"] == "unbagged" for r in events),
      f"got {[(r['frame'], r['pops_score'], r['bag']) for r in events]}")

# A cart with no snapshot must be left alone rather than crash.
events = [{"cart_id": 9, "frame": 10, "event": "ABANDONED CART",
           "pops_score": 55, "fill": "partial", "bag": "bagged"}]
sync_events_with_snapshots(events, {}, {})
check("a row with no snapshot is untouched", events[0]["pops_score"] == 55)

# ---------------------------------------------------------------------------
section("end-to-end — Cart 1 as it should finalise")
# ---------------------------------------------------------------------------
_fills = [f for f, _, _, _ in OUTSIDE_C1]
_final_fill = peak_sustained_fill(_fills, GRABRUN_MIN_RUN_OBS)
_final_bag = vote_bag_for_loaded_cart(OUTSIDE_C1)
_final_score = _score(_final_fill, _final_bag)
_final_event, _ = classify_event(_final_score, True, "UNKNOWN", abandoned=True)
check("finalised fill/bag", (_final_fill, _final_bag) == ("partial", "unbagged"),
      f"got {(_final_fill, _final_bag)}")
check("finalised score is 65 without merch_removed evidence",
      _final_score == 65, f"got {_final_score}")
# 65 is below HIGH_SCORE (71), so the abandonment route to PUSHOUT ALERT is not
# open on the score alone: direction UNKNOWN denies the cart OUTBOUND's +15
# base and its max(score, 75) abandonment floor.
check("65 alone classifies as ABANDONED CART, not PUSHOUT ALERT",
      _final_event == "ABANDONED CART", f"got {_final_event!r}")

# ...but this cart's history is not merely "abandoned while loaded". It holds a
# 15-observation partial run and then ends empty: the items were lifted out
# while the owner left. merchandise_removed() is that evidence, and it lifts
# the UNKNOWN floor to MERCH_REMOVED_FLOOR so the cart finishes as the pushout
# it is. See test_grabrun_pushout.py for the full policy including the parked
# cart it must NOT fire on.
_removed = merchandise_removed(_fills, GRABRUN_MIN_RUN_OBS)
check("merchandise_removed fires on Cart 1's history", _removed is True,
      f"got {_removed}")
_esc_score = compute_pops("UNKNOWN", "STATIC", True, _final_fill,
                          bag_label=_final_bag, abandoned=True, linked=True,
                          merch_removed=True)
_esc_event, _ = classify_event(_esc_score, True, "UNKNOWN", abandoned=True)
check("with merch_removed the cart reaches MERCH_REMOVED_FLOOR",
      _esc_score == MERCH_REMOVED_FLOOR, f"got {_esc_score}")
check("and finalises as PUSHOUT ALERT", _esc_event == "PUSHOUT ALERT",
      f"got {_esc_event!r}")

print("\n" + "=" * 62)
print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
sys.exit(1 if _FAIL else 0)
