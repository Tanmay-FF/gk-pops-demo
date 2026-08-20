# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Every finalised event row must be reproducible from its own fields.

ONE ASSERTION CATCHES A WHOLE CLASS
-----------------------------------
An event row carries a score AND the inputs that score was computed from. So:

    compute_pops(row.direction, row.speed_status, True, row.fill, row.bag,
                 abandoned=row.abandoned, linked=row.linked) == row.pops_score

Any row that fails this describes a cart that never existed. Both golden clips
were shipping one before this test was written, and neither was visible in any
count or ID:

  * PUSHOUT ALERT | 75 | OUTBOUND | SLOW | abandoned=false — 75 came from a live
    frame where the cart WAS abandoned; sync_events_with_snapshots() rewrote the
    score from the reconciled snapshot and left the flag at the live row's value.
    Recomputing from the row's own fields gives 55.
  * MEDIUM PRIORITY | 45 | OUTBOUND | MEDIUM — 45 requires STATIC; MEDIUM gives 60.

Both are the same bug: a score taken from one moment beside a context taken from
another. The fix is that the sync rewrites every input it rewrites the score
from. This test is what keeps it fixed, and it needs no video to run.

The second half pins the tier rule: a loggable NAME is not enough, because
classify_event() returns "UNLINKED EXIT" from two different tiers.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.scoring import (  # noqa: E402
    compute_pops, classify_event, sync_events_with_snapshots, prune_event_log,
    LOGGABLE_EVENTS, MEDIUM_SCORE, HIGH_SCORE, PUSHOUT_SCORE,
)

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


def recompute(row):
    """The row's score, recomputed from the row's own fields."""
    score = compute_pops(
        row["direction"], row["speed_status"], True, row["fill"],
        bag_label=row["bag"], cart_detected=True,
        abandoned=row.get("abandoned", False), linked=row.get("linked", False),
        merch_removed=row.get("merch_removed", False),
    )
    # compute_pops does not know about the partial+bagged cap the finaliser
    # re-applies, so mirror it here or a legitimately capped row reads as broken.
    if row["fill"] == "partial" and row["bag"] == "bagged":
        score = min(score, 55)
    return score


def assert_coherent(label, rows):
    for row in rows:
        want = recompute(row)
        check(f"{label}: cart {row['cart_id']} {row['event']} is reproducible "
              f"from its own fields",
              row["pops_score"] == want,
              f"row says {row['pops_score']}, its own fields give {want} — "
              f"{row['fill']}|{row['bag']} {row['direction']} "
              f"{row['speed_status']} abandoned={row.get('abandoned')}")
        name, _ = classify_event(row["pops_score"], row.get("linked", False),
                                 row["direction"], abandoned=row.get("abandoned", False))
        check(f"{label}: cart {row['cart_id']} event name matches its score",
              row["event"] == name,
              f"row says {row['event']}, score {row['pops_score']} classifies "
              f"as {name}")


def _row(**kw):
    row = {
        "frame": 10, "timestamp": 0.5, "cart_id": 1, "event": "MEDIUM PRIORITY",
        "pops_score": 45, "fill": "partial", "bag": "unbagged",
        "direction": "OUTBOUND", "linked": False, "speed_status": "MEDIUM",
        "abandoned": False,
    }
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
section("the 1764092528600 row: score from the peak, abandonment from the frame")
# ---------------------------------------------------------------------------
# Live: an abandoned partial/unbagged cart, OUTBOUND SLOW -> 75, PUSHOUT ALERT.
# Reconciled snapshot agrees on fill/bag but was recorded on a frame where the
# cart was abandoned. The row logged at first crossing had abandoned=False.
snap = {
    1: {"fill": "partial", "bag": "unbagged", "score": 75,
        "event": "PUSHOUT ALERT", "direction": "OUTBOUND",
        "speed_status": "SLOW", "linked": False, "abandoned": True,
        "frame": 407, "timestamp": 20.35},
}
log = [_row(event="UNLINKED EXIT", pops_score=35, speed_status="SLOW",
            abandoned=False, bag="unbagged")]
sync_events_with_snapshots(log, snap, {1: 75})
assert_coherent("primary clip shape", log)
check("the row keeps the FIRST frame it reached the event",
      log[0]["frame"] == 10, str(log[0]["frame"]))
check("...and carries the peak's own frame separately, so it can still be seeked",
      log[0]["peak_frame"] == 407 and log[0]["peak_timestamp"] == 20.35,
      str((log[0].get("peak_frame"), log[0].get("peak_timestamp"))))

# ---------------------------------------------------------------------------
section("the 1764099569430 row: score from a STATIC peak, speed from a MEDIUM frame")
# ---------------------------------------------------------------------------
snap = {
    # linked=True is what made this MEDIUM PRIORITY rather than UNLINKED EXIT at
    # the same score: the peak frame had an owner. The row logged at first
    # crossing did not, and that mismatch is half of what made it unreadable.
    1: {"fill": "partial", "bag": "unbagged", "score": 45,
        "event": "MEDIUM PRIORITY", "direction": "OUTBOUND",
        "speed_status": "STATIC", "linked": True, "abandoned": False,
        "frame": 80, "timestamp": 3.93},
}
log = [_row(pops_score=35, speed_status="MEDIUM", bag="bagged")]
sync_events_with_snapshots(log, snap, {1: 45})
assert_coherent("outside clip shape", log)

# ---------------------------------------------------------------------------
section("the live-peak floor still wins as a UNIT, never blended")
# ---------------------------------------------------------------------------
# A row that scored higher live keeps its OWN reading, all fields together. If the
# floor took the score but left the snapshot's context, it would manufacture
# exactly the incoherence above.
snap = {
    1: {"fill": "empty", "bag": "not_applicable", "score": 5,
        "event": "INBOUND", "direction": "INBOUND", "speed_status": "STATIC",
        "linked": False, "abandoned": False, "frame": 200, "timestamp": 10.0},
}
log = [_row(event="PUSHOUT ALERT", pops_score=75, fill="partial", bag="unbagged",
            direction="OUTBOUND", speed_status="SLOW", abandoned=True)]
notes = sync_events_with_snapshots(log, snap, {1: 5})
check("the higher live reading is kept", log[0]["pops_score"] == 75,
      str(log[0]["pops_score"]))
check("and it is explained in the notes, not applied silently", bool(notes), str(notes))
assert_coherent("live floor", log)

# ---------------------------------------------------------------------------
section("an event needs the medium tier, not just a loggable name")
# ---------------------------------------------------------------------------
low_name, _ = classify_event(0, False, "OUTBOUND")
check("classify_event returns a LOGGABLE name below every tier",
      low_name in LOGGABLE_EVENTS,
      f"{low_name!r} — if this stops being true the guard below is dead code, "
      f"not a bug fix")

kept, dropped = prune_event_log([
    _row(cart_id=2, event=low_name, pops_score=0, fill="empty",
         bag="not_applicable", speed_status="STATIC"),
    _row(cart_id=3, event="MEDIUM PRIORITY", pops_score=45, speed_status="STATIC"),
])
check("a sub-threshold row is dropped", len(kept) == 1 and dropped == 1,
      f"kept={[r['event'] for r in kept]} dropped={dropped}")
check("the real event survives", kept and kept[0]["cart_id"] == 3)

for score in (MEDIUM_SCORE, HIGH_SCORE, PUSHOUT_SCORE):
    kept, _ = prune_event_log([_row(pops_score=score, event="MEDIUM PRIORITY")])
    check(f"a row at exactly {score} is kept — the tiers are inclusive",
          len(kept) == 1)


print(f"\n{_passed} passed, {_failed} failed")
if _failed:
    raise SystemExit(1)
print("All green.")
