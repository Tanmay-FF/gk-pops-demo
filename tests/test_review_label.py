# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The placement-free review label (engine/review_label.py).

What it pins:
  * every cart decision in cart_decision()'s documented order;
  * an empty cart is never a reason to review, abandoned or not, unless the
    split-track guard says it may be a loaded cart re-acquired;
  * bagged goods with the shopper still there are skipped, left behind are reviewed;
  * an unread cart is reviewed only past REVIEW_UNREAD_MIN_S; a blip is ignored;
  * the clip is REVIEW when any cart is, with that cart's reason;
  * the engine path builds the label from placement-free state only: changing
    every direction-derived input (event log, peak snapshots) leaves it unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import review_label as RL  # noqa: E402
from engine.config import (  # noqa: E402
    REVIEW_UNREAD_MIN_S, REVIEW_LOADED_MIN_RUN, CART_BLIP_S,
)

_passed = _failed = 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f" -- {extra}" if extra else ""))


def section(title):
    print(f"\n{title}")


N = REVIEW_LOADED_MIN_RUN
EMPTY = ("empty", "not_applicable", 0.95, 1.0)
PART_UNB = ("partial", "unbagged", 0.9, 0.9)
PART_BAG = ("partial", "bagged", 0.9, 0.9)


def cart(history=(), loaded=None, seen_s=10.0, never_loaded=True, abandoned=()):
    history = list(history)
    if loaded is None:
        loaded = [h[0] in ("partial", "full") for h in history]
    return RL.CartReviewFacts(seen_s=seen_s, history=history, loaded_reads=list(loaded),
                              never_loaded=never_loaded, abandoned_flags=list(abandoned))


def decision(c, **kw):
    return RL.cart_decision(c, **kw)[0]


# ---------------------------------------------------------------------------
section("cart decisions")
check("empty cart -> empty", decision(cart([EMPTY] * 20)) == RL.CART_EMPTY)
check("empty cart left behind, never loaded -> empty (not reviewed)",
      decision(cart([EMPTY] * 20, abandoned=[False] * 5 + [True] * 5)) == RL.CART_EMPTY)
check("empty cart left behind on a split loaded track -> split_track",
      decision(cart([EMPTY] * 20, never_loaded=False, abandoned=[True])) == RL.CART_SPLIT_TRACK)
check("loaded unbagged with shopper -> loaded",
      decision(cart([PART_UNB] * (N + 2))) == RL.CART_LOADED)
check("loaded bagged with shopper -> bagged_with_shopper",
      decision(cart([PART_BAG] * (N + 2), abandoned=[True, False])) == RL.CART_BAGGED_WITH_SHOPPER)
check("loaded bagged left at the end -> bagged_left",
      decision(cart([PART_BAG] * (N + 2), abandoned=[False, True])) == RL.CART_BAGGED_LEFT)
check("abandon_mode=ever: stepping away once counts",
      decision(cart([PART_BAG] * (N + 2), abandoned=[True, False]), abandon_mode="ever")
      == RL.CART_BAGGED_LEFT)
check("loaded run then empty, shopper gone -> goods_removed",
      decision(cart([PART_UNB] * (N + 2) + [EMPTY] * 15, abandoned=[True])) == RL.CART_GOODS_REMOVED)
check("same history, shopper still there -> not goods_removed",
      decision(cart([PART_UNB] * (N + 2) + [EMPTY] * 15, abandoned=[False])) != RL.CART_GOODS_REMOVED)

section("loaded basis")
short = cart([PART_UNB] * (N - 1))  # wins the vote, run too short
check("basis both: a short loaded run is not loaded", decision(short) == RL.CART_EMPTY)
check("basis vote: the same cart is loaded", decision(short, loaded_basis="vote") == RL.CART_LOADED)
noisy = cart([PART_UNB] * N + [EMPTY] * (3 * N))  # long run, loses the vote
check("basis both: a run that loses the vote is not loaded", decision(noisy) == RL.CART_EMPTY)
check("basis run: the same cart is loaded", decision(noisy, loaded_basis="run") == RL.CART_LOADED)
det_only = cart([], loaded=[True] * N, seen_s=5.0)  # NON-EMPTY reads, every crop failed quality
check("detector-only sustained loaded run -> loaded", decision(det_only) == RL.CART_LOADED)

section("unread carts")
check("unread blip -> blip", decision(cart(seen_s=CART_BLIP_S / 2)) == RL.CART_BLIP)
check("unread, brief -> unread_brief",
      decision(cart(seen_s=(CART_BLIP_S + REVIEW_UNREAD_MIN_S) / 2)) == RL.CART_UNREAD_BRIEF)
check("unread, long -> unread", decision(cart(seen_s=REVIEW_UNREAD_MIN_S + 1)) == RL.CART_UNREAD)

section("clip label")
lab = RL.review_label({})
check("no carts -> SKIP", lab["label"] == RL.SKIP and lab["reasons"] == ["No cart was tracked"])
lab = RL.review_label({1: cart([EMPTY] * 20), 2: cart([EMPTY] * 9, abandoned=[True])})
check("all empty -> SKIP 'Every cart read empty'",
      lab["label"] == RL.SKIP and lab["reasons"] == ["Every cart read empty"], lab["reasons"])
lab = RL.review_label({1: cart([EMPTY] * 20), 2: cart([PART_BAG] * (N + 2))})
check("empty + bagged with shopper -> SKIP naming the bagged cart",
      lab["label"] == RL.SKIP and any("Cart 2" in r for r in lab["reasons"]), lab["reasons"])
lab = RL.review_label({1: cart([EMPTY] * 20), 2: cart([PART_UNB] * (N + 2)),
                       3: cart(seen_s=CART_BLIP_S / 2)})
check("any loaded cart -> REVIEW with that cart's reason only",
      lab["label"] == RL.REVIEW and lab["reasons"] == ["Cart 2 held unbagged goods"], lab["reasons"])

# ---------------------------------------------------------------------------
section("engine: built from placement-free state only")
from collections import defaultdict  # noqa: E402
from engine.tracker import TrackingEngine  # noqa: E402
from engine.analytics_models import CartFactSample  # noqa: E402


def engine_label(event_log, snapshots):
    e = TrackingEngine.__new__(TrackingEngine)
    e._fps = 20.0
    e._cart_span = {1: [1, [0, 0, 10, 10], 400, [0, 0, 10, 10], 400],
                    2: [1, [50, 50, 90, 90], 300, [50, 50, 90, 90], 300]}
    e._cart_cls_history = {1: [EMPTY] * 30, 2: [PART_UNB] * (N + 4)}
    e._cart_loaded_reads = defaultdict(list, {1: [False] * 30, 2: [True] * (N + 4)})
    e._loaded_carts = {2}
    e._cart_facts = defaultdict(list)
    for cd, ab in ((1, True), (2, False)):
        e._cart_facts[cd] = [CartFactSample(t=0.0, frame=1, fill="empty", bag="not_applicable",
                                            fill_conf=0.9, quality="valid_cart", fill_stale_frames=0,
                                            linked=False, linked_person_display=None,
                                            abandoned_linker=ab)]
    e._event_log = event_log
    e._peak_pops_snapshot = snapshots
    return e._build_review_label()


a = engine_label([], {})
b = engine_label([{"cart_id": 2, "event": "PUSHOUT ALERT", "pops_score": 90, "direction": "OUTBOUND",
                   "abandoned": True}],
                 {1: {"direction": "INBOUND", "abandoned": False, "score": 5},
                  2: {"direction": "OUTBOUND", "abandoned": True, "score": 90}})
check("engine label: REVIEW for the loaded cart", a["label"] == RL.REVIEW, a)
check("engine label ignores event log and peak snapshots",
      (a["label"], a["reasons"]) == (b["label"], b["reasons"]), (a, b))
check("abandoned empty cart 1 is not a review reason",
      a["carts"][1]["decision"] == RL.CART_EMPTY, a["carts"][1])

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
