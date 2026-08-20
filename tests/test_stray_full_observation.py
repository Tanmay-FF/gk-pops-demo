# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
A single stray "full" observation must not become the finalised fill label.

Run with:  python tests/test_stray_full_observation.py

This pins the one place where gk-pops-enhanced deliberately disagrees with
gk-pops-code, so the disagreement does not get "fixed" back by whoever next
diffs the two POPS tables side by side.

The clip is sample_videos/1764120540680_B8A44F40EED7-medium-OUTSIDE.mp4, Cart 1.
Both codebases produce the SAME detections and the SAME per-frame
classifications on it — fill_count {'partial': 24, 'empty': 35, 'full': 1} and
bag_count {'unbagged': 23, 'bagged': 2, 'not_applicable': 35} are identical in
both runs, and only the confidence floats differ by the usual GPU
nondeterminism. Both vote "empty", and both then fire the grab-and-run
override. They pick a different label out of it:

  gk-pops-code  engine/tracker.py, first-half candidate loop:
      for candidate in ("full", "partial"):
          if first_fill_count.get(candidate, 0) > 0:
  finds full count == 1 and reports FULL.

  this repo, peak_sustained_fill(), requires a contiguous run of
  GRABRUN_MIN_RUN_OBS, and full's longest run is 1, so it reports PARTIAL.

The single "full" observation sits at index 22 at fill confidence 0.667,
against 24 partial observations at aggregate confidence 471.08. Every
selection rule that weighs evidence at all — contiguous run length,
confidence-weighted vote over the loaded observations, count share of the
loaded observations (1 of 25) — returns partial. The only rule that returns
full is "highest severity ever observed, count >= 1", which is precisely the
rule engine/scoring.py:peak_sustained_fill() documents replacing because
classifier noise satisfies it.

Reverting to it to match this one clip would trade a documented noise guard for
a coincidence of candidate ordering. If the cart really was full, the defect is
upstream: the classifier called a full cart partial 24 times out of 60.

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from engine.config import GRABRUN_MIN_RUN_OBS
from engine.scoring import (
    classify_event, compute_pops, merchandise_removed, peak_sustained_fill,
)

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixture: the fill sequence exactly as the run printed it, in order.
# ---------------------------------------------------------------------------
MEDIUM_OUTSIDE_C1_FILLS = (
    ["partial"] * 21
    + ["empty", "full", "partial", "partial", "empty", "partial"]
    + ["empty"] * 33
)

# Both codebases logged these counts, so the fixture has to reproduce them or
# it is not the same history.
_counts = defaultdict(int)
for f in MEDIUM_OUTSIDE_C1_FILLS:
    _counts[f] += 1

section("fixture matches both runs' logged counts")
check("60 observations", len(MEDIUM_OUTSIDE_C1_FILLS) == 60,
      f"got {len(MEDIUM_OUTSIDE_C1_FILLS)}")
check("fill_count partial=24 empty=35 full=1",
      (_counts["partial"], _counts["empty"], _counts["full"]) == (24, 35, 1),
      f"got {dict(_counts)}")

# ---------------------------------------------------------------------------
section("selection: the stray full loses")
# ---------------------------------------------------------------------------
sustained = peak_sustained_fill(MEDIUM_OUTSIDE_C1_FILLS, GRABRUN_MIN_RUN_OBS)
check("override fires at all (the empty vote does not stand)",
      sustained is not None, f"got {sustained!r}")
check("finalised fill is partial, not full",
      sustained == "partial", f"got {sustained!r}")

# The discriminator, stated as an assertion so a future edit that loosens
# selection to max-severity-present trips here rather than in production.
runs = defaultdict(int)
i, n = 0, len(MEDIUM_OUTSIDE_C1_FILLS)
while i < n:
    label, j = MEDIUM_OUTSIDE_C1_FILLS[i], i
    while j + 1 < n and MEDIUM_OUTSIDE_C1_FILLS[j + 1] == label:
        j += 1
    runs[label] = max(runs[label], j - i + 1)
    i = j + 1
check("full's longest run is 1, under GRABRUN_MIN_RUN_OBS",
      runs["full"] == 1 < GRABRUN_MIN_RUN_OBS, f"got {dict(runs)}")
check("partial's longest run is 21, well over it",
      runs["partial"] == 21, f"got {runs['partial']}")

# ---------------------------------------------------------------------------
section("the replaced gk-pops-code selection WOULD have said full")
# ---------------------------------------------------------------------------
# Kept executable so the divergence stays visible and nobody has to go read the
# other repo to understand why the two POPS tables disagree on this clip.
mid = max(1, len(MEDIUM_OUTSIDE_C1_FILLS) // 2)
first_half = MEDIUM_OUTSIDE_C1_FILLS[:mid]
second_half = MEDIUM_OUTSIDE_C1_FILLS[mid:]
first_count = defaultdict(int)
for f in first_half:
    first_count[f] += 1
second_count = defaultdict(int)
for f in second_half:
    second_count[f] += 1

old_gate = ((first_count["full"] + first_count["partial"]) >= len(first_half) * 0.5
            and second_count["empty"] > len(second_half) * 0.7)
check("the half-split gate happens to fire on this history too",
      old_gate is True,
      f"first-half loaded {first_count['full'] + first_count['partial']}/"
      f"{len(first_half)}, second-half empty "
      f"{second_count['empty']}/{len(second_half)}")

old_selection = next(
    (c for c in ("full", "partial") if first_count.get(c, 0) > 0), None)
check("its candidate loop selects full off a count of 1",
      old_selection == "full" and first_count["full"] == 1,
      f"got {old_selection!r} from full count {first_count['full']}")

# ---------------------------------------------------------------------------
section("the label difference is not a scoring regression")
# ---------------------------------------------------------------------------
# OUTBOUND + abandoned + fill in (partial, full) floors at MERCH_REMOVED_FLOOR,
# so on THIS clip both labels reach the same score through the same branch and
# the divergence is display-only. Asserting it both ways keeps the write-up
# honest: the labels are score-equivalent here, not in general.
scores = {
    fill: compute_pops("OUTBOUND", "STATIC", True, fill, bag_label="unbagged",
                       cart_detected=True, abandoned=True, linked=False,
                       merch_removed=True)
    for fill in ("full", "partial")
}
check("partial and full both score 75 on this context",
      scores == {"full": 75, "partial": 75}, f"got {scores}")
check("and both classify as PUSHOUT ALERT",
      all(classify_event(s, False, "OUTBOUND", abandoned=True)[0]
          == "PUSHOUT ALERT" for s in scores.values()))

# Not equivalent once the abandonment floor is out of the way: 65 vs 45 is a
# tier apart, which is why selection is worth a test instead of a shrug.
unfloored = {
    fill: compute_pops("OUTBOUND", "STATIC", True, fill, bag_label="unbagged",
                       cart_detected=True, abandoned=False, linked=False)
    for fill in ("full", "partial")
}
check("without the floor the two labels are a tier apart",
      unfloored == {"full": 65, "partial": 45}, f"got {unfloored}")

# The evidence that actually drove the alert on this clip.
check("merchandise_removed fires on this history",
      merchandise_removed(MEDIUM_OUTSIDE_C1_FILLS, GRABRUN_MIN_RUN_OBS) is True)

print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
