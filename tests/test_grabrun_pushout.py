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

from engine.config import GRABRUN_MIN_RUN_OBS, GRABRUN_TRAILING_NOISE_OBS
from engine.scoring import (
    HIGH_SCORE, LOGGABLE_EVENTS, MERCH_REMOVED_FLOOR, classify_event,
    compute_pops, merchandise_removed, peak_sustained_fill, prune_event_log,
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
section("a genuinely empty cart with a noisy loaded burst")
# ---------------------------------------------------------------------------
# Cart 1 of the 1763902526030 clip, from the Not_A_Pushout set. Ground truth:
# the cart is EMPTY. The fill vote agrees by a landslide — empty 1129.36 over
# 35 observations against partial 47.73 over 8, a 24x margin at mean partial
# confidence 0.746 — but those 8 partials include one burst of exactly 4 in a
# row, so a run gate of 4 promoted the cart to partial|unbagged, set
# merch_removed, and filed an OUTBOUND 75 PUSHOUT ALERT on an empty cart.
#
# Fill order exactly as the [DEBUG] line recorded it.
NOT_A_PUSHOUT_1763902526030_C1 = (
    ["partial", "empty", "partial", "partial", "empty"]
    + ["partial"] * 4
    + ["empty", "partial"]
    + ["empty"] * 32
)
check("43 observations", len(NOT_A_PUSHOUT_1763902526030_C1) == 43,
      f"got {len(NOT_A_PUSHOUT_1763902526030_C1)}")
check("8 partial / 35 empty, matching the logged vote counts",
      (sum(1 for f in NOT_A_PUSHOUT_1763902526030_C1 if f == "partial"),
       sum(1 for f in NOT_A_PUSHOUT_1763902526030_C1 if f == "empty")) == (8, 35))

# The discriminator, asserted rather than assumed: the longest loaded burst on
# this history is 4, and every confirmed removal in the corpus runs 9 or more.
_longest, _cur = 0, 0
for _f in NOT_A_PUSHOUT_1763902526030_C1:
    _cur = _cur + 1 if _f != "empty" else 0
    _longest = max(_longest, _cur)
check("longest loaded burst is 4 observations", _longest == 4, f"got {_longest}")
check("the run gate sits above it", GRABRUN_MIN_RUN_OBS > _longest,
      f"gate {GRABRUN_MIN_RUN_OBS}")

check("a noisy 4-observation burst is NOT removal",
      merchandise_removed(NOT_A_PUSHOUT_1763902526030_C1,
                          GRABRUN_MIN_RUN_OBS) is False)
check("and the empty verdict stands, so the fill is not promoted",
      peak_sustained_fill(NOT_A_PUSHOUT_1763902526030_C1,
                          GRABRUN_MIN_RUN_OBS) is None)

# The worst case once the override is off this cart. `abandoned` is computed
# from lost person track and is unrelated to fill, so even with the flag set
# OUTBOUND + empty takes the 60 floor and logs ABANDONED CART rather than the
# 75 / PUSHOUT ALERT the promoted fill produced. Pinned as the ceiling on what
# this history can now score.
#
# Re-running the clip itself lands lower still, at 0 with no event: the
# finaliser reads its abandonment context off the best logged event, and with
# the fill no longer promoted no abandoned row is logged at all.
_noisy_score = compute_pops("OUTBOUND", "STATIC", True, "empty",
                            bag_label="not_applicable", cart_detected=True,
                            abandoned=True, linked=True, merch_removed=False)
check("even with abandonment it is 60 / ABANDONED CART, not 75 / PUSHOUT ALERT",
      (_noisy_score, _event(_noisy_score, direction="OUTBOUND"))
      == (60, "ABANDONED CART"),
      f"got {(_noisy_score, _event(_noisy_score, direction='OUTBOUND'))}")


# ---------------------------------------------------------------------------
section("the two run thresholds are independent knobs")
# ---------------------------------------------------------------------------
# GRABRUN_MIN_RUN_OBS and GRABRUN_TRAILING_NOISE_OBS both count consecutive
# loaded observations, and wiring them to one number is a trap: raising the run
# gate to reject noise would simultaneously widen the trailing-noise window and
# convert the parked-cart negative above into a pushout. Pinned as arithmetic
# so nobody re-merges them.
_parked = (["partial"] * GRABRUN_MIN_RUN_OBS + ["empty"] * 5
           + ["partial"] * GRABRUN_TRAILING_NOISE_OBS)
check("a parked cart's trailing loaded run survives the noise strip",
      merchandise_removed(_parked, GRABRUN_MIN_RUN_OBS) is False)
# Stated as an invariant over run gates rather than as "passing the wrong
# value gives the wrong answer": a trailing loaded run at the noise threshold
# must survive for EVERY run gate, which is exactly what a shared constant
# cannot deliver.
check("a cart that ends loaded is not a removal at any run gate",
      all(merchandise_removed(["partial"] * _g + ["empty"] * 5
                              + ["partial"] * GRABRUN_TRAILING_NOISE_OBS, _g)
          is False for _g in range(2, 12)))


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
section("the escalation has to come from the finaliser, not just from a frame")
# ---------------------------------------------------------------------------
# The live per-frame path and the end-of-run finaliser both score this cart, and
# sync_events_with_snapshots() rewrites every row from the RECONCILED snapshot —
# there is no live-peak floor, in either direction. So a merch-removed escalation
# only reaches the user if the finaliser reaches it too: the finaliser reads
# `abandoned` from the cart's best event and re-derives merch_removed from the
# whole classification history, so an escalation the live path found is normally
# re-found there. What it cannot do any more is survive on the strength of the
# live frame alone.
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

# The other direction: live path escalated, finaliser recomputed lower (its
# `abandoned` came from a best-event context that did not carry the flag). The
# snapshot wins here too, and the row is rewritten DOWN — the price of making the
# reconciliation the single authority on a cart's contents. What keeps the real
# escalation safe is that the finaliser derives merch_removed from the same
# history the vote reads, so this shape means the evidence was not there at the
# end of the run either.
_events = [{"cart_id": 2, "frame": 40, "event": _esc_event_name,
            "pops_score": _ESCALATED, "fill": "partial", "bag": "unbagged"}]
_snapshots = {2: {"fill": "partial", "bag": "unbagged", "score": 65,
                  "event": "ABANDONED CART"}}
_max = {2: 65}
_notes = sync_events_with_snapshots(_events, _snapshots, _max)
check("an escalated live row IS rewritten from a 65 snapshot",
      (_events[0]["pops_score"], _events[0]["event"]) == (65, "ABANDONED CART"),
      f"got {(_events[0]['pops_score'], _events[0]['event'])}")
check("the snapshot is not lifted to meet it, so table and row still agree",
      (_snapshots[2]["score"], _snapshots[2]["event"]) == (65, "ABANDONED CART"),
      f"got {(_snapshots[2]['score'], _snapshots[2]['event'])}")
check("max_pops stays on the reconciled reading", _max[2] == 65,
      f"got {_max[2]}")
check("and the demotion is reported rather than silent", bool(_notes),
      f"got {_notes}")

# The rewritten name has to still be loggable, or prune_event_log() drops the
# row on the floor after the sync just rewrote it. Checked for both outcomes of
# the sync, since either can be what a row ends up carrying.
check("PUSHOUT ALERT is a loggable event", "PUSHOUT ALERT" in LOGGABLE_EVENTS)
check("so is ABANDONED CART", "ABANDONED CART" in LOGGABLE_EVENTS)
_kept, _dropped = prune_event_log(_events)
check("the rewritten row survives pruning", len(_kept) == 1 and _dropped == 0,
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
