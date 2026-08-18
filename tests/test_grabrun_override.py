# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Grab-and-run finalisation tests — pure scoring, no GPU and no video decode.

Run with:  python tests/test_grabrun_override.py

The fixture is not synthetic. HANNAFORD_C2 is the real classification history
of Cart 2 from the 2026-08-13 HANNAFORD GM DOOR clip, lifted per-frame out of
that run's tracking JSON. It is a CONFIRMED pushout: the cart held items, the
person lifted them out, and the live per-frame path scored it 75 / PUSHOUT
ALERT throughout. The end-of-run finaliser then re-voted it down to 60 /
ABANDONED CART, so the same incident finished with two different verdicts
(orig=75 recomp=60) depending on which side a coin-flip fill vote landed.

Each assertion pins one link in that chain. All of them fail silently in
production — the alert simply drops a severity tier, which looks identical to
"the model decided it wasn't a pushout".

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from engine.config import GRABRUN_MIN_RUN_OBS
from engine.scoring import classify_event, compute_pops, peak_sustained_fill

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixture: (fill, bag, fill_conf, bag_conf) exactly as _cart_cls_history holds
# it. Real values, sampled every CLASSIFY_EVERY_N_FRAMES=8 frames at 20 fps.
# ---------------------------------------------------------------------------
def _obs(fill, bag, fc, bc):
    return (fill, bag, fc, bc)


HANNAFORD_C2 = (
    # 3.1s-6.3s — cart loaded, bag reading genuinely mixed at close range
    [_obs("partial", "unbagged", 0.9041, 0.9939),
     _obs("partial", "unbagged", 0.9112, 0.6978),
     _obs("partial", "bagged",   0.9942, 0.5880),
     _obs("partial", "unbagged", 0.9390, 0.7269),
     _obs("partial", "bagged",   0.9866, 0.6899),
     _obs("partial", "unbagged", 0.8874, 0.5882),
     _obs("partial", "bagged",   0.8859, 0.9654),
     _obs("partial", "bagged",   0.9947, 0.9843),
     _obs("partial", "bagged",   0.9933, 0.9931)]
    # 6.7s-15.5s — items lifted out, cart reads empty
    + [_obs("empty", "not_applicable", 0.75, 1.0) for _ in range(24)]
    # 15.9s-21.1s — loaded again as the cart clears the door, all unbagged
    + [_obs("partial", "unbagged", 0.92, 0.90) for _ in range(12)]
    + [_obs("empty", "not_applicable", 0.75, 1.0)]
)

FILLS = [f for f, b, fc, bc in HANNAFORD_C2]


def _whole_history_bag(history):
    """The constraint at tracker.py:1195 — vote the bag across the FULL
    history, excluding not_applicable."""
    scores = defaultdict(float)
    for _f, bag, _fc, bc in history:
        if bag != "not_applicable":
            scores[bag] += bc
    return max(scores, key=scores.get) if scores else "unbagged"


def _run_local_bag(history, label):
    """The tempting alternative: vote the bag inside the run that triggered the
    override — which is what the replaced code did (it voted over `first_half`,
    and on this history that is exactly the early loaded run)."""
    fills = [f for f, b, fc, bc in history]
    start = fills.index(label)
    end = start
    while end + 1 < len(fills) and fills[end + 1] == label:
        end += 1
    scores = defaultdict(float)
    for _f, bag, _fc, bc in history[start:end + 1]:
        scores[bag] += bc
    return max(scores, key=scores.get) if scores else "unbagged"


# ---------------------------------------------------------------------------
section("peak_sustained_fill — the override gate")
# ---------------------------------------------------------------------------
sustained = peak_sustained_fill(FILLS, GRABRUN_MIN_RUN_OBS)
# If this returns None the finaliser keeps "empty", floors at 60, and a
# confirmed pushout is filed as ABANDONED CART.
check("real pushout history overrides empty -> partial",
      sustained == "partial", f"got {sustained!r}")

# Why the previous gate could not fire, kept executable so nobody reintroduces
# it: the history is items -> empty -> items, so NEITHER half is cleanly loaded
# nor cleanly empty.
mid = max(1, len(FILLS) // 2)
first, second = FILLS[:mid], FILLS[mid:]
n_items_first = sum(1 for f in first if f in ("partial", "full"))
n_empty_second = sum(1 for f in second if f == "empty")
old_gate = (n_items_first >= len(first) * 0.5
            and n_empty_second > len(second) * 0.7)
check("the replaced half-split gate would NOT have fired",
      old_gate is False,
      f"first_half items {n_items_first}/{len(first)}, "
      f"second_half empty {n_empty_second}/{len(second)}")

# Contiguity, not count, is the noise guard: 8 scattered single-observation
# partials must not be enough, while GRABRUN_MIN_RUN_OBS consecutive ones are.
noisy = ["empty", "partial"] * 8 + ["empty"] * 10
check("scattered single-frame noise does not override",
      peak_sustained_fill(noisy, GRABRUN_MIN_RUN_OBS) is None,
      f"{sum(1 for f in noisy if f == 'partial')} noisy partials, longest run 1")

exact = ["empty"] * 5 + ["partial"] * GRABRUN_MIN_RUN_OBS + ["empty"] * 5
check("a run of exactly GRABRUN_MIN_RUN_OBS qualifies",
      peak_sustained_fill(exact, GRABRUN_MIN_RUN_OBS) == "partial")

short = ["empty"] * 5 + ["partial"] * (GRABRUN_MIN_RUN_OBS - 1) + ["empty"] * 5
check("a run one observation short does not qualify",
      peak_sustained_fill(short, GRABRUN_MIN_RUN_OBS) is None)

both = (["partial"] * 6 + ["empty"] * 4 + ["full"] * 5 + ["empty"] * 6)
check("full outranks partial when both sustain a run",
      peak_sustained_fill(both, GRABRUN_MIN_RUN_OBS) == "full")

check("an all-empty history is left alone",
      peak_sustained_fill(["empty"] * 30, GRABRUN_MIN_RUN_OBS) is None)

# ---------------------------------------------------------------------------
section("bag resolution — the 75-vs-55 trap")
# ---------------------------------------------------------------------------
# Voting the bag inside the run that triggered the override lands on "bagged"
# here: that run is 4 unbagged / 5 bagged and bagged wins on confidence
# (4.22 vs 3.01). The late run is uniformly unbagged, which is why the
# whole-history vote gets it right and a run-local one does not.
local = _run_local_bag(HANNAFORD_C2, "partial")
whole = _whole_history_bag(HANNAFORD_C2)
check("run-local bag vote picks bagged on this history",
      local == "bagged", f"got {local!r}")
check("whole-history bag vote picks unbagged",
      whole == "unbagged", f"got {whole!r}")

score_local = compute_pops("OUTBOUND", "STATIC", True, "partial",
                           bag_label=local, cart_detected=True,
                           abandoned=True, linked=True)
score_whole = compute_pops("OUTBOUND", "STATIC", True, "partial",
                           bag_label=whole, cart_detected=True,
                           abandoned=True, linked=True)
# partial+bagged is capped at 55 AFTER the abandonment floor of 75, and 55 is
# under classify_event's 71 threshold — so the run-local vote silently demotes
# the alert even though the fill override fired correctly.
check("run-local bag caps the score under the PUSHOUT line",
      score_local == 55, f"got {score_local}")
check("whole-history bag reaches the abandonment floor",
      score_whole == 75, f"got {score_whole}")

# ---------------------------------------------------------------------------
section("end-to-end finalisation for Cart 2")
# ---------------------------------------------------------------------------
# Replays the finaliser's decision chain: empty vote -> sustained-run override
# -> partial/full constraint re-votes the bag -> recompute -> classify.
best_fill, best_bag = "empty", "not_applicable"
if best_fill == "empty":                      # abandoned=True for this cart
    override = peak_sustained_fill(FILLS, GRABRUN_MIN_RUN_OBS)
    if override:
        best_fill = override
if best_fill in ("partial", "full") and best_bag == "not_applicable":
    best_bag = _whole_history_bag(HANNAFORD_C2)

final = compute_pops("OUTBOUND", "STATIC", True, best_fill,
                     bag_label=best_bag, cart_detected=True,
                     abandoned=True, linked=True)
event, _color = classify_event(final, True, "OUTBOUND", abandoned=True)
check("finalised fill/bag", (best_fill, best_bag) == ("partial", "unbagged"),
      f"got {(best_fill, best_bag)}")
# The whole point: recomputed score must equal the live score (orig=75), not
# the 60 the old finaliser produced.
check("recomputed score matches the live score", final == 75, f"got {final}")
check("event is PUSHOUT ALERT", event == "PUSHOUT ALERT", f"got {event!r}")

# The 60 the old gate produced, pinned so the regression is legible.
old_score = compute_pops("OUTBOUND", "STATIC", True, "empty",
                         bag_label="not_applicable", cart_detected=True,
                         abandoned=True, linked=True)
old_event, _ = classify_event(old_score, True, "OUTBOUND", abandoned=True)
check("un-overridden empty verdict is the 60/ABANDONED CART regression",
      (old_score, old_event) == (60, "ABANDONED CART"),
      f"got {(old_score, old_event)}")

# ---------------------------------------------------------------------------
section("the override stays gated")
# ---------------------------------------------------------------------------
# A cart that was never loaded must not be promoted, however long it sits.
clean = ["empty"] * 46
promoted = peak_sustained_fill(clean, GRABRUN_MIN_RUN_OBS)
clean_score = compute_pops("OUTBOUND", "STATIC", True,
                           promoted or "empty", bag_label="not_applicable",
                           cart_detected=True, abandoned=True, linked=True)
check("genuinely empty abandoned cart stays at 60",
      clean_score == 60, f"got {clean_score}")

print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
