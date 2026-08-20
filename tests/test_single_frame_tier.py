# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""One classifier observation must not decide a POPS tier.

Run with:  python tests/test_single_frame_tier.py

WHAT BROKE
----------
On the FF1763940475070 INSIDE clip Cart 3 finished the run as HIGH PRIORITY at
75 while the whole-history vote read it partial|bagged / 30 — a cart of bagged
groceries. Observation 18 of 30 read full|unbagged at bag confidence 0.6795,
against 28 bagged observations totalling 23.21. The live frame loop scored
whatever the LATEST classification said, so that one frame produced
OUTBOUND + full + unbagged + moving = 75, logged a HIGH PRIORITY event and
froze a 75 peak. Both peak-as-floor rules (tracker.py's reconciliation floor and
sync_events_with_snapshots) then kept the higher live reading, so the vote could
not undo it.

The fix is that the frame loop scores vote_classification() over the cart's
whole history — the same function the finaliser votes with — so the two paths
cannot disagree about labels and no single observation can carry a tier.

The fixtures below are REAL histories from that clip, lifted out of
`_cart_cls_history` per observation: INSIDE_C3 is the false positive, INSIDE_C2
is a confirmed pushout on the same clip and the same run. Both matter. A change
that suppresses the first by making loaded+unbagged evidence harder to reach
would take the second with it, and the second is the alert the demo exists for.

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.scoring import (
    HIGH_SCORE, MEDIUM_SCORE, PUSHOUT_SCORE, classify_event, compute_pops,
    vote_bag_for_loaded_cart, vote_classification,
)

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixtures: (fill, bag, fill_conf, bag_conf) exactly as _cart_cls_history holds
# them, sampled every CLASSIFY_EVERY_N_FRAMES=8 frames at 20 fps.
# ---------------------------------------------------------------------------
INSIDE_C3 = [
    ("full", "bagged", 0.52, 0.7727),
    ("partial", "bagged", 0.7082, 0.9021),
    ("partial", "bagged", 0.5703, 0.922),
    ("full", "bagged", 0.5407, 0.9653),
    ("partial", "bagged", 0.6218, 0.9459),
    ("full", "bagged", 0.6846, 0.9439),
    ("full", "bagged", 0.9278, 0.8796),
    ("full", "bagged", 0.627, 0.9069),
    ("full", "bagged", 0.9653, 0.9737),
    ("full", "bagged", 0.7581, 0.9714),
    ("partial", "bagged", 0.6207, 0.931),
    ("partial", "bagged", 0.6897, 0.836),
    ("partial", "bagged", 0.774, 0.9757),
    ("partial", "bagged", 0.8165, 0.981),
    ("partial", "bagged", 0.9301, 0.9581),
    ("partial", "bagged", 0.9755, 0.5845),
    ("partial", "bagged", 0.7362, 0.7199),
    # THE frame. One full|unbagged read at 0.6795 bag confidence — this is what
    # used to score 75 and take the whole run with it.
    ("full", "unbagged", 0.9129, 0.6795),
    ("empty", "not_applicable", 0.5036, 1.0),
    ("partial", "bagged", 0.7271, 0.502),
    ("partial", "bagged", 0.7385, 0.6052),
    ("partial", "bagged", 0.8708, 0.5994),
    ("partial", "bagged", 0.8311, 0.617),
    ("partial", "bagged", 0.6656, 0.7138),
    ("partial", "bagged", 0.8442, 0.8566),
    ("partial", "bagged", 0.8278, 0.7957),
    ("partial", "bagged", 0.9329, 0.813),
    ("partial", "bagged", 0.96, 0.8031),
    ("partial", "bagged", 0.9486, 0.9024),
    ("partial", "bagged", 0.9647, 0.8367),
]

#: The bad frame's index in INSIDE_C3, named so the assertions below read as
#: "this observation" rather than "17".
BAD_OBS = 17

INSIDE_C2 = [
    ("partial", "unbagged", 0.6454, 0.9668),
    ("full", "unbagged", 0.8193, 0.9901),
    ("full", "unbagged", 0.5815, 0.9634),
    ("full", "unbagged", 0.6688, 0.978),
    ("full", "unbagged", 0.3477, 0.9503),
    ("full", "unbagged", 0.4941, 0.9701),
    ("full", "unbagged", 0.5737, 0.9921),
    ("partial", "unbagged", 0.9474, 0.9681),
    ("partial", "unbagged", 0.9324, 0.9885),
    ("partial", "unbagged", 0.8468, 0.9766),
    ("partial", "unbagged", 0.4944, 0.9875),
    ("partial", "unbagged", 0.4601, 0.9888),
    ("partial", "unbagged", 0.4612, 0.9891),
    ("full", "unbagged", 0.4809, 0.9739),
    ("full", "unbagged", 0.4876, 0.9587),
    ("full", "unbagged", 0.4933, 0.959),
    ("full", "unbagged", 0.507, 0.9534),
    ("full", "unbagged", 0.5076, 0.9558),
    ("full", "unbagged", 0.4438, 0.9629),
    ("full", "unbagged", 0.4438, 0.9629),
    ("full", "unbagged", 0.3957, 0.9618),
    ("partial", "unbagged", 0.4464, 0.938),
    ("full", "unbagged", 0.3914, 0.9327),
    ("partial", "unbagged", 0.4983, 0.9536),
    ("full", "unbagged", 0.4433, 0.9638),
    ("partial", "unbagged", 0.3827, 0.9753),
    ("empty", "not_applicable", 0.3928, 1.0),
    ("partial", "unbagged", 0.492, 0.9713),
    ("partial", "unbagged", 0.4815, 0.9757),
    ("full", "unbagged", 0.5273, 0.9694),
    ("full", "unbagged", 0.795, 0.976),
]

_SPEEDS = ("STATIC", "SLOW", "MEDIUM", "FAST")


def live_labels(history, abandoned=False):
    """What the frame loop now scores: the vote, plus the loaded-bag constraint.

    Mirrors engine/tracker.py's live block. The grab-and-run branch is not
    reproduced here — tests/test_grabrun_override.py owns it — so this is only
    called with histories whose vote is not "empty".
    """
    fill, bag, _detail = vote_classification(history)
    if fill in ("partial", "full") and bag == "not_applicable":
        bag = vote_bag_for_loaded_cart(history)
    return fill, bag


def main():
    section("Cart 3: the vote overrules the single frame")
    fill, bag, detail = vote_classification(INSIDE_C3)
    check("vote reads the cart partial|bagged",
          (fill, bag) == ("partial", "bagged"), f"got {fill}|{bag}")
    check("bagged wins the bag vote by more than an order of magnitude",
          detail["bag_scores"]["bagged"] > 10 * detail["bag_scores"]["unbagged"],
          f"{detail['bag_scores']['bagged']:.1f} vs "
          f"{detail['bag_scores']['unbagged']:.1f}")
    check("exactly one observation reads unbagged",
          detail["bag_count"]["unbagged"] == 1)

    section("Cart 3: no speed reaches HIGH on the voted labels")
    for speed in _SPEEDS:
        vfill, vbag = live_labels(INSIDE_C3)
        score = compute_pops("OUTBOUND", speed, True, vfill, bag_label=vbag,
                             linked=True)
        event, _c = classify_event(score, True, "OUTBOUND")
        check(f"OUTBOUND/{speed} stays under HIGH_SCORE",
              score < HIGH_SCORE, f"score={score} {event}")

    section("Cart 3: the single frame WOULD have reached HIGH — the hazard")
    bad_fill, bad_bag, _fc, _bc = INSIDE_C3[BAD_OBS]
    bad_score = compute_pops("OUTBOUND", "SLOW", True, bad_fill,
                             bag_label=bad_bag, linked=True)
    bad_event, _c = classify_event(bad_score, True, "OUTBOUND")
    check("scoring that one observation alone clears HIGH_SCORE",
          bad_score >= HIGH_SCORE, f"score={bad_score} {bad_event}")
    check("...which is why the frame loop must not score a single observation",
          bad_event == "HIGH PRIORITY", bad_event)

    section("Cart 3: no PREFIX of the history reaches HIGH either")
    # The live path scores the vote as evidence accumulates, so the tier has to
    # hold at every point in the run, not just at the end. A check on the
    # complete history alone would pass even if observation 18 could still spike
    # the tier at the moment it arrived.
    worst = 0
    worst_at = None
    for n in range(1, len(INSIDE_C3) + 1):
        vfill, vbag = live_labels(INSIDE_C3[:n])
        if vfill == "empty":
            continue  # grab-and-run territory, not this test's subject
        for speed in _SPEEDS:
            score = compute_pops("OUTBOUND", speed, True, vfill,
                                 bag_label=vbag, linked=True)
            if score > worst:
                worst, worst_at = score, (n, speed, vfill, vbag)
    check("worst prefix score stays under HIGH_SCORE",
          worst < HIGH_SCORE, f"worst={worst} at {worst_at}")

    section("Cart 2 on the same clip: the real pushout still fires")
    fill2, bag2, _d2 = vote_classification(INSIDE_C2)
    check("vote reads the cart full|unbagged",
          (fill2, bag2) == ("full", "unbagged"), f"got {fill2}|{bag2}")
    vfill2, vbag2 = live_labels(INSIDE_C2)
    outbound = compute_pops("OUTBOUND", "SLOW", True, vfill2, bag_label=vbag2,
                            linked=True)
    check("walked calmly out the exit it is still HIGH at minimum",
          outbound >= HIGH_SCORE, f"score={outbound}")
    abandoned = compute_pops("UNKNOWN", "STATIC", True, vfill2, bag_label=vbag2,
                             linked=True, abandoned=True, merch_removed=True)
    ev2, _c = classify_event(abandoned, True, "UNKNOWN", abandoned=True)
    check("abandoned mid-store it is still a PUSHOUT ALERT",
          ev2 == "PUSHOUT ALERT", f"score={abandoned} {ev2}")

    section("A cart with no loaded evidence is not invented into one")
    empty_hist = [("empty", "not_applicable", 0.75, 1.0) for _ in range(10)]
    efill, ebag, _d = vote_classification(empty_hist)
    check("empty history votes empty|not_applicable",
          (efill, ebag) == ("empty", "not_applicable"), f"got {efill}|{ebag}")
    check("empty history cannot reach even MEDIUM outbound",
          compute_pops("OUTBOUND", "FAST", True, efill,
                       bag_label=ebag) < MEDIUM_SCORE)
    check("an empty history returns no verdict at all",
          vote_classification([]) == (None, None, {}))

    section("Loaded run with no bagging evidence still reads as the risky one")
    # The pre-vote live path fell through to vote_bag_for_loaded_cart(), whose
    # documented default is "unbagged" — a cart that carried merchandise with no
    # positive bagging evidence is the higher-risk read. Voting first must not
    # quietly replace that with "not_applicable", which scores as an empty cart.
    na_hist = ([("empty", "not_applicable", 0.75, 1.0) for _ in range(10)]
               + [("partial", "not_applicable", 0.8, 1.0) for _ in range(5)]
               + [("empty", "not_applicable", 0.75, 1.0)])
    nfill, nbag, _d = vote_classification(na_hist)
    check("bag vote alone lands on not_applicable", nbag == "not_applicable")
    from engine.config import GRABRUN_MIN_RUN_OBS
    from engine.scoring import peak_sustained_fill
    sustained = peak_sustained_fill([f for f, _b, _fc, _bc in na_hist],
                                    GRABRUN_MIN_RUN_OBS)
    check("the loaded run is still recognised", sustained == "partial", str(sustained))
    check("and the loaded-bag constraint restores the risky read",
          vote_bag_for_loaded_cart(na_hist) == "unbagged")
    grabrun = compute_pops("OUTBOUND", "SLOW", True, sustained,
                           bag_label=vote_bag_for_loaded_cart(na_hist),
                           abandoned=True, linked=True, merch_removed=True)
    check("so a grab-and-run on that history still floors at 75",
          grabrun >= HIGH_SCORE, f"score={grabrun}")

    section("Both paths call the same voter")
    # Structural, because the property is "these two cannot disagree" and the
    # only way they disagree again is by one of them growing its own vote.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "engine", "tracker.py"),
        encoding="utf-8").read()
    check("tracker.py votes in at least two places (frame loop + finaliser)",
          src.count("vote_classification(") >= 2,
          f"count={src.count('vote_classification(')}")
    check("no inline fill vote left in the finaliser",
          "fill_scores = {f: fill_conf[f]" not in src)

    section("PUSHOUT_SCORE is still out of reach for a bagged partial cart")
    for speed in _SPEEDS:
        score = compute_pops("OUTBOUND", speed, True, "partial",
                             bag_label="bagged", linked=True, abandoned=True)
        check(f"partial|bagged abandoned OUTBOUND/{speed} capped under PUSHOUT",
              score < PUSHOUT_SCORE, f"score={score}")

    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        for name in _FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
