# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The placement-free review label (engine/review_label.py).

What it pins:
  * every cart decision in cart_decision()'s documented order;
  * an empty cart is never a reason to review, abandoned or not, unless the
    split-track guard says it may be a loaded cart re-acquired;
  * normal first: bagged goods whose shopper never left are skipped; ANY
    abandonment of a cart that held goods, bagged or unbagged, is reviewed;
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
section("cart decisions: normal first")
check("empty cart -> empty", decision(cart([EMPTY] * 20)) == RL.CART_EMPTY)
check("empty cart left behind, never loaded -> empty (not reviewed)",
      decision(cart([EMPTY] * 20, abandoned=[False] * 5 + [True] * 5)) == RL.CART_EMPTY)
check("empty cart left behind on a split loaded track -> split_track",
      decision(cart([EMPTY] * 20, never_loaded=False, abandoned=[True])) == RL.CART_SPLIT_TRACK)
check("loaded bagged, shopper never left -> bagged_with_shopper (normal)",
      decision(cart([PART_BAG] * (N + 2), abandoned=[False] * 5)) == RL.CART_BAGGED_WITH_SHOPPER)
check("loaded unbagged, shopper with it -> loaded (review)",
      decision(cart([PART_UNB] * (N + 2))) == RL.CART_LOADED)

section("any abandonment of a cart that held goods needs review")
check("bagged, shopper stepped away once and came back -> left_with_goods",
      decision(cart([PART_BAG] * (N + 2), abandoned=[False, True, False])) == RL.CART_LEFT_WITH_GOODS)
check("bagged, left at the end -> left_with_goods",
      decision(cart([PART_BAG] * (N + 2), abandoned=[False, True])) == RL.CART_LEFT_WITH_GOODS)
check("unbagged, left -> left_with_goods",
      decision(cart([PART_UNB] * (N + 2), abandoned=[True])) == RL.CART_LEFT_WITH_GOODS)
check("loaded run then empty, shopper gone -> goods_removed",
      decision(cart([PART_UNB] * (N + 2) + [EMPTY] * 15, abandoned=[True])) == RL.CART_GOODS_REMOVED)
check("a long loaded run the vote overrules, shopper left -> still reviewed",
      decision(cart([PART_UNB] * N + [EMPTY] * (3 * N) + [PART_UNB] * 2, abandoned=[True]))
      == RL.CART_GOODS_REMOVED)  # the 2 trailing loaded reads are noise; it ends empty
check("same history, shopper still there -> empty (normal)",
      decision(cart([PART_UNB] * N + [EMPTY] * (3 * N) + [PART_UNB] * 2, abandoned=[False]))
      == RL.CART_EMPTY)
check("abandon_mode=end: stepping away once does not count",
      decision(cart([PART_BAG] * (N + 2), abandoned=[True, False]), abandon_mode="end")
      == RL.CART_BAGGED_WITH_SHOPPER)

section("loaded basis")
short = cart([PART_UNB] * (N - 1))  # wins the vote, run too short
check("basis both: a short loaded run is not loaded", decision(short) == RL.CART_EMPTY)
check("basis vote: the same cart is loaded", decision(short, loaded_basis="vote") == RL.CART_LOADED)
noisy = cart([PART_UNB] * N + [EMPTY] * (3 * N))  # long run, loses the vote
check("basis both: a run that loses the vote is not loaded", decision(noisy) == RL.CART_EMPTY)
check("basis run: the same cart is loaded", decision(noisy, loaded_basis="run") == RL.CART_LOADED)
check("a short loaded run, shopper left -> not held, so empty (never a review reason)",
      decision(cart([PART_UNB] * (N - 1), abandoned=[True])) == RL.CART_EMPTY)
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
lab = RL.review_label({1: cart([EMPTY] * 20), 2: cart([PART_BAG] * (N + 2), abandoned=[False])})
check("empty + bagged with shopper -> SKIP naming the bagged cart",
      lab["label"] == RL.SKIP and any("Cart 2" in r for r in lab["reasons"]), lab["reasons"])
lab = RL.review_label({1: cart([EMPTY] * 20), 2: cart([PART_UNB] * (N + 2)),
                       3: cart(seen_s=CART_BLIP_S / 2)})
check("any loaded cart -> NEEDS_REVIEW with that cart's reason only",
      lab["label"] == RL.REVIEW and lab["reasons"] == ["Cart 2 held unbagged goods"], lab["reasons"])

section("Q1: the review's run is the engine's run")
from engine.config import GRABRUN_MIN_RUN_OBS  # noqa: E402
check("REVIEW_LOADED_MIN_RUN follows GRABRUN_MIN_RUN_OBS", N == GRABRUN_MIN_RUN_OBS)
check("a run of exactly GRABRUN_MIN_RUN_OBS held goods; shopper left -> reviewed",
      decision(cart([PART_UNB] * GRABRUN_MIN_RUN_OBS + [EMPTY] * 15, never_loaded=False,
                    abandoned=[True])) == RL.CART_GOODS_REMOVED)


def ev_cart(history=(), *, moved=200.0, linked=True, p=None, stacked=None, evidence=(), **kw):
    c = cart(history, **kw)
    c.moved_px, c.linked_ever, c.p_mean, c.p_stacked_mean = moved, linked, p, stacked
    c.read_evidence = list(evidence)
    return c


section("candidate rules are on by default (user, 2026-11-12: \"new + all\")")
rack = ev_cart([("full", "unbagged", 0.9, 0.9)] * (N + 4), moved=1.0, linked=False, p=1.0)
check("default: a static unlinked loaded box is a fixture", decision(rack) == RL.CART_FIXTURE)
check("static_unlinked off: the same box is reviewed", decision(rack, static_unlinked=False) == RL.CART_LOADED)
brief = ev_cart([PART_UNB] * 3, p=0.99, seen_s=2.0)
check("default: a brief all-loaded cart is promoted", decision(brief) == RL.CART_LOADED)
check("promote_brief off: the same cart is not promoted", decision(brief, promote_brief=False) == RL.CART_EMPTY)

section("R2: static, never-linked fixture")
check("static_unlinked: display rack -> fixture",
      decision(rack, static_unlinked=True) == RL.CART_FIXTURE)
check("static_unlinked: a linked static cart is still reviewed",
      decision(ev_cart(rack.history, moved=1.0, linked=True, p=1.0), static_unlinked=True)
      == RL.CART_LOADED)
check("static_unlinked: a moving unlinked cart is still reviewed",
      decision(ev_cart(rack.history, moved=300.0, linked=False, p=1.0), static_unlinked=True)
      == RL.CART_LOADED)
check("static_unlinked: missing evidence never fires",
      decision(cart(rack.history), static_unlinked=True) == RL.CART_LOADED)

section("R3: brief strongly-loaded cart")
check("promote_brief: 3 reads all loaded, P 0.99, moving -> loaded",
      decision(brief, promote_brief=True) == RL.CART_LOADED)
check("promote_brief: bagged with its shopper is still skipped",
      decision(ev_cart([PART_BAG] * 3, p=0.99), promote_brief=True) == RL.CART_BAGGED_WITH_SHOPPER)
check("promote_brief: shopper left -> left_with_goods",
      decision(ev_cart([PART_UNB] * 3, p=0.99, abandoned=[True]), promote_brief=True)
      == RL.CART_LEFT_WITH_GOODS)
check("promote_brief: detector-only reads (never classified) -> loaded, not unread_brief",
      decision(ev_cart([], loaded=[True] * 3, p=0.99, seen_s=1.8), promote_brief=True)
      == RL.CART_LOADED)
check("promote_brief: one empty read blocks it",
      decision(ev_cart([PART_UNB, EMPTY, PART_UNB], p=0.99), promote_brief=True) == RL.CART_EMPTY)
check("promote_brief: P 0.8 blocks it", decision(ev_cart([PART_UNB] * 3, p=0.8), promote_brief=True)
      == RL.CART_EMPTY)
check("promote_brief: static blocks it",
      decision(ev_cart([PART_UNB] * 3, p=0.99, moved=5.0), promote_brief=True) == RL.CART_EMPTY)
check("promote_brief: a single read is not enough",
      decision(ev_cart([PART_UNB], p=0.99), promote_brief=True) == RL.CART_EMPTY)

section("R4 / child: doubtful loaded reads dropped per read")
occl = ev_cart([PART_UNB] * N, evidence=[(0.7, 0.6, 0.0)] * N)
check("drop_occluded: every loaded read under a person at P 0.7 -> run 0, empty",
      decision(occl, drop_occluded=True) == RL.CART_EMPTY)
check("drop_occluded: off -> loaded", decision(occl, drop_occluded=False) == RL.CART_LOADED)
check("drop_occluded: on by default", decision(occl) == RL.CART_EMPTY)
check("drop_occluded: high-confidence reads under the pusher are kept",
      decision(ev_cart([PART_UNB] * N, evidence=[(0.95, 0.8, 0.0)] * N), drop_occluded=True)
      == RL.CART_LOADED)
check("drop_occluded: misaligned evidence never fires",
      decision(ev_cart([PART_UNB] * N, evidence=[(0.7, 0.6, 0.0)]), drop_occluded=True)
      == RL.CART_LOADED)
kid = ev_cart([PART_UNB] * N, evidence=[(0.9, 0.0, 0.9)] * N)
check("drop_child: loaded reads with a child in the cart -> empty",
      decision(kid, drop_child=True, child_source="detector") == RL.CART_EMPTY)
check("drop_child: off -> loaded", decision(kid, drop_child=False) == RL.CART_LOADED)

section("stacked carts")
stack = ev_cart([PART_UNB] * N, stacked=0.8)
check("stacked_not_held: a stack -> stacked", decision(stack, stacked_not_held=True) == RL.CART_STACKED)
check("stacked_not_held: off -> loaded", decision(stack, stacked_not_held=False) == RL.CART_LOADED)
check("stacked_not_held: on by default", decision(stack) == RL.CART_STACKED)

section("v2 classifier child / stack evidence (REVIEW_CHILD_SOURCE, REVIEW_STACKED_SOURCE)")
cls_kid = ev_cart([PART_UNB] * N, evidence=[(0.9, 0.0, 0.0, 0.99)] * N)
check("child either: classifier alone (P 0.99) drops the reads -> empty",
      decision(cls_kid, drop_child=True, child_source="either") == RL.CART_EMPTY)
check("child detector: classifier ignored -> loaded",
      decision(cls_kid, drop_child=True, child_source="detector") == RL.CART_LOADED)
check("child either: classifier below REVIEW_CLS_CHILD_MIN (P 0.3) -> loaded",
      decision(ev_cart([PART_UNB] * N, evidence=[(0.9, 0.0, 0.0, 0.3)] * N),
               drop_child=True, child_source="either") == RL.CART_LOADED)
check("child both: detector child but classifier says no -> loaded",
      decision(ev_cart([PART_UNB] * N, evidence=[(0.9, 0.0, 0.9, 0.1)] * N),
               drop_child=True, child_source="both") == RL.CART_LOADED)
check("child both: 3-tuple (legacy) evidence falls back to the detector -> empty",
      decision(kid, drop_child=True, child_source="both") == RL.CART_EMPTY)
check("child: master switch off -> classifier evidence never fires",
      decision(cls_kid, drop_child=False, child_source="either") == RL.CART_LOADED)
check("child default (2026-11-12): on, classifier head is the source -> empty",
      decision(cls_kid) == RL.CART_EMPTY)
check("child default: detector-only child (3-tuple, no classifier P) does not fire",
      decision(kid) == RL.CART_LOADED)
cls_stack = ev_cart([PART_UNB] * N, stacked=0.1)
cls_stack.cls_stacked_mean = 0.97
check("stacked either: classifier mean 0.97 alone -> stacked",
      decision(cls_stack, stacked_not_held=True, stacked_source="either") == RL.CART_STACKED)
check("stacked detector: classifier ignored -> loaded",
      decision(cls_stack, stacked_not_held=True, stacked_source="detector") == RL.CART_LOADED)
check("stacked both: detector says no -> loaded",
      decision(cls_stack, stacked_not_held=True, stacked_source="both") == RL.CART_LOADED)
check("stacked either: classifier mean 0.9 is under REVIEW_CLS_STACKED_MIN -> loaded",
      decision(ev_cart([PART_UNB] * N, stacked=0.1), stacked_not_held=True, stacked_source="either")
      == RL.CART_LOADED)
check("stacked both: legacy run (no classifier mean) falls back to the detector",
      decision(stack, stacked_not_held=True, stacked_source="both") == RL.CART_STACKED)
check("stacked: facts carry both means",
      RL.cart_decision(cls_stack)[1]["cls_stacked_mean"] == 0.97)

section("NOTHING_HAPPENED tier")
clean = lambda **kw: ev_cart([EMPTY] * 10, p=0.1, **kw)  # noqa: E731
lab = RL.review_label({1: clean(), 2: clean()}, confident_tier=True)
check("tier on: two clean empty carts -> NOTHING_HAPPENED", lab["label"] == RL.NOTHING_HAPPENED, lab)
lab = RL.review_label({1: clean(), 2: clean()})
check("tier on by default: same clip -> NOTHING_HAPPENED", lab["label"] == RL.NOTHING_HAPPENED, lab)
lab = RL.review_label({1: clean(), 2: clean()}, confident_tier=False)
check("tier off: same clip -> SKIP with confident evidence",
      lab["label"] == RL.SKIP and lab["confidence"]["confident"], lab["confidence"])
lab = RL.review_label({1: clean(), 2: clean(abandoned=[False, True])}, confident_tier=True)
check("Q3: an EMPTY cart left by its shopper blocks the tier (still SKIP)",
      lab["label"] == RL.SKIP and any("left by its shopper" in b for b in lab["confidence"]["blocked_by"]),
      lab["confidence"])
lab = RL.review_label({1: clean(), 2: ev_cart([EMPTY] * 3, p=0.1)}, confident_tier=True)
check("too few reads blocks the tier", lab["label"] == RL.SKIP)
lab = RL.review_label({1: clean(), 2: ev_cart([EMPTY] * 10, p=0.45)}, confident_tier=True)
check("mean P above the bar blocks the tier", lab["label"] == RL.SKIP)
lab = RL.review_label({1: clean(), 2: ev_cart([EMPTY] * 10)}, confident_tier=True)
check("unknown P (2-class backend) blocks the tier", lab["label"] == RL.SKIP)
lab = RL.review_label({1: clean(), 2: cart(seen_s=(CART_BLIP_S + REVIEW_UNREAD_MIN_S) / 2)},
                      confident_tier=True)
check("a brief unread cart blocks the tier", lab["label"] == RL.SKIP)
lab = RL.review_label({1: clean(), 2: cart(seen_s=CART_BLIP_S / 2)}, confident_tier=True)
check("a blip does not block the tier", lab["label"] == RL.NOTHING_HAPPENED, lab["confidence"])
lab = RL.review_label({1: clean(), 2: ev_cart([PART_BAG] * (N + 2), p=0.9)}, confident_tier=True)
check("bagged with shopper -> SKIP, never NOTHING_HAPPENED", lab["label"] == RL.SKIP)
check("no carts -> SKIP, never NOTHING_HAPPENED",
      RL.review_label({}, confident_tier=True)["label"] == RL.SKIP)
lab = RL.review_label({1: clean(), 2: ev_cart([PART_UNB] * (N + 2), p=0.9)}, confident_tier=True)
check("a review clip is never confident", lab["label"] == RL.REVIEW
      and not lab["confidence"]["confident"])

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


ev = TrackingEngine._read_evidence({"p_non_empty": 0.8}, (0, 0, 100, 100),
                                   [(50, 0, 150, 100), (0, 0, 10, 10)], [[10, 10, 30, 30, 0.9]])
check("read evidence: P, largest person cover, child-inside share", ev == (0.8, 0.5, 1.0, None), ev)
ev = TrackingEngine._read_evidence({"quality": "valid_cart"}, (0, 0, 100, 100), [], [])
check("read evidence: no detector P, no person, no child", ev == (None, 0.0, 0.0, None), ev)
ev = TrackingEngine._read_evidence({"quality": "valid_cart", "child_conf": 0.991234}, (0, 0, 100, 100), [], [])
check("read evidence: v2 classifier P(child) is the 4th element", ev == (None, 0.0, 0.0, 0.9912), ev)

a = engine_label([], {})
b = engine_label([{"cart_id": 2, "event": "PUSHOUT ALERT", "pops_score": 90, "direction": "OUTBOUND",
                   "abandoned": True}],
                 {1: {"direction": "INBOUND", "abandoned": False, "score": 5},
                  2: {"direction": "OUTBOUND", "abandoned": True, "score": 90}})
check("engine label: NEEDS_REVIEW for the loaded cart", a["label"] == RL.REVIEW, a)
check("engine label ignores event log and peak snapshots",
      (a["label"], a["reasons"]) == (b["label"], b["reasons"]), (a, b))
check("abandoned empty cart 1 is not a review reason",
      a["carts"][1]["decision"] == RL.CART_EMPTY, a["carts"][1])

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
