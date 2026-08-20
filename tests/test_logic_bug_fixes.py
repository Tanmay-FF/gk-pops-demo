# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Regressions for eight logic defects found in the rule engine / POPS pipeline.

Run with:  python tests/test_logic_bug_fixes.py

Every one of these failed SILENTLY in production — no exception, no empty
output, just a wrong number or a missing alert that looked exactly like "the
model decided nothing happened". That is why they are pinned here.

The assertions deliberately go through the real entry points — evaluate_rules(),
compute_direction_label(), prune_event_log(), TrackingEngine._sanitize_timestamps()
— not through re-implementations of their internals. tests/test_grabrun_override.py
is the cautionary tale: it passes today while a later block in the same function
undoes what it pins, because it replayed the decision chain instead of calling it.

Deliberately stdlib + numpy, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import highlights, rules
from engine.analytics_models import (
    FACTS_SCHEMA_VERSION, TrackRecord, TrajectoryBundle, Zone,
)
from engine.config import (
    DIRECTION_WINDOW_S, RULE_MAX_SAMPLE_GAP_S, SPEED_MEDIUM,
)
from engine.motion import compute_direction_label
from engine.scoring import (
    LOGGABLE_EVENTS, classify_event, compute_pops, prune_event_log,
)

_PASS: list[str] = []
_FAIL: list[str] = []

CAM = "Outside (facing entrance)"


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def rec(raw, disp, label, ts, xy, speeds=None, bb=None):
    ts = np.asarray(ts, dtype=np.float32)
    pos = np.asarray(xy, dtype=np.float32)
    sp = (np.zeros(len(ts), np.float32) if speeds is None
          else np.asarray(speeds, np.float32))
    bbox = (np.empty((0, 4), np.float32) if bb is None
            else np.asarray(bb, np.float32))
    return TrackRecord(raw, label, disp, pos, ts,
                       np.arange(len(ts), dtype=np.int32), sp, bbox)


def bundle(tracks, w=1920, h=1080):
    return TrajectoryBundle("k", "v", w, h, 20.0, 4000,
                            {t.raw_id: t for t in tracks},
                            facts_schema_version=FACTS_SCHEMA_VERSION)


def door_zone():
    return Zone("d1", "Main Door",
                np.array([[400, 400], [600, 400], [600, 600], [400, 600]],
                         np.int32),
                applies_to="both", kind="door")


def analytics_zone():
    return Zone("z1", "Checkout",
                np.array([[10, 10], [100, 10], [100, 100], [10, 100]],
                         np.int32),
                applies_to="person", kind="analytics")


def parked_cart(dur_s=60.0, gap=0.5, at=(500.0, 500.0)):
    """A cart motionless in the doorway, sampled every `gap` seconds."""
    ts = list(np.arange(0.0, dur_s + 1e-9, gap))
    x, y = at
    return rec(1, 1, "cart", ts, [(x, y)] * len(ts), [0.0] * len(ts),
               [(x - 50, y - 50, x + 50, y + 50)] * len(ts))


# ---------------------------------------------------------------------------
section("1. direction is measured over a WINDOW, not the whole track")
# ---------------------------------------------------------------------------
# 40s at 20fps: enters (y down->up), mills about, leaves the way it came.
# _obj_positions is never trimmed, so a first-to-last delta cancels out.
enter = [(900.0, 1000.0 - 3.8 * i) for i in range(100)]
mill = [(900.0 + 0.5 * i, 620.0) for i in range(600)]
leave = [(1050.0, 620.0 + 3.8 * i) for i in range(100)]
whole = enter + mill + leave
wts = [i * 0.05 for i in range(len(whole))]

check("whole-track delta cancels to UNKNOWN (the bug)",
      compute_direction_label(whole, CAM) == "UNKNOWN")
check("windowed delta reports OUTBOUND while the cart leaves",
      compute_direction_label(whole, CAM, wts, DIRECTION_WINDOW_S) == "OUTBOUND")
check("windowed delta reports INBOUND while it arrives",
      compute_direction_label(enter, CAM, wts[:100],
                              DIRECTION_WINDOW_S) == "INBOUND")
check("milling in-store is still UNKNOWN, not a spurious heading",
      compute_direction_label(enter + mill, CAM, wts[:700],
                              DIRECTION_WINDOW_S) == "UNKNOWN")
# The window must be TIME-based rather than a sample count: positions are
# appended per DETECTION, so "the last 40 samples" is 2s for a cleanly tracked
# cart and minutes for a sparsely detected one. A cart detected only every 2s
# but genuinely covering ground must still resolve to a heading.
sparse_pos = ([(900.0, 200.0 + 100.0 * i) for i in range(10)]   # 100px/sample
              + [(900.0, 1200.0)] * 2)
sparse_ts = [i * 2.0 for i in range(len(sparse_pos))]           # 50 px/s
check("a sparsely-detected but genuinely moving track still resolves",
      compute_direction_label(sparse_pos, CAM, sparse_ts,
                              DIRECTION_WINDOW_S) == "OUTBOUND")
# And the converse: the same number of samples spread over a long period is a
# SLOW track, and a 4s window correctly declines to call it.
crawl_ts = [i * 20.0 for i in range(len(whole))]        # 800 samples over 4.4h
check("a genuinely crawling track is UNKNOWN, not a spurious heading",
      compute_direction_label(whole, CAM, crawl_ts,
                              DIRECTION_WINDOW_S) == "UNKNOWN")
# Callers that pre-slice their own window (rules._incoming_empty_rule) must be
# untouched by the new arguments.
check("omitting the window keeps the old whole-list behaviour",
      compute_direction_label(whole[-12:], CAM) == "OUTBOUND")

# The consequence the fix exists for.
out = compute_pops("OUTBOUND", "MEDIUM", True, "full", bag_label="unbagged")
unk = compute_pops("UNKNOWN", "MEDIUM", True, "full", bag_label="unbagged")
check("OUTBOUND scores an event, UNKNOWN does not",
      out >= 31 and unk < 31, f"OUTBOUND={out} UNKNOWN={unk}")


# ---------------------------------------------------------------------------
section("3. the synthesized clock derives from REAL frame numbers")
# ---------------------------------------------------------------------------
try:
    from engine.tracker import TrackingEngine
except Exception as e:                                   # pragma: no cover
    print(f"  SKIP  torch/ultralytics unavailable ({type(e).__name__})")
else:
    fps = 20.0
    # A cart parked for 4 minutes, detected 1 frame in 8 because shoppers keep
    # occluding it. CAP_PROP_POS_MSEC returns 0.0 on this container.
    frames = np.arange(100, 4900, 8, dtype=np.int32)
    broken = np.zeros(frames.size, dtype=np.float32)
    ts, synth = TrackingEngine._sanitize_timestamps(broken, frames, fps)
    true_span = float(frames[-1] - frames[0]) / fps
    got_span = float(ts[-1] - ts[0])
    check("a broken POS_MSEC is detected", synth is True)
    check("the rebuilt span matches real time (was 30s for a 240s block)",
          abs(got_span - true_span) < 0.05,
          f"true {true_span:.1f}s, rebuilt {got_span:.1f}s")
    check("thresholds are reachable again",
          got_span >= 180.0, f"{got_span:.1f}s vs RULE_ABANDONED_CART_S=180")
    # The old fallback counted SAMPLES, so it also lied to the density gate in
    # the opposite direction — claiming 8x denser sampling than reality.
    mean_gap = got_span / (ts.size - 1)
    check("the density gate sees the true sampling rate",
          abs(mean_gap - 8.0 / fps) < 0.01, f"{mean_gap:.3f}s")
    # A healthy clock must be passed through untouched.
    good = np.arange(frames.size, dtype=np.float32) * 0.4 + 5.0
    ts2, synth2 = TrackingEngine._sanitize_timestamps(good, frames, fps)
    check("a usable POS_MSEC is left alone",
          (not synth2) and np.allclose(ts2, good))
    # No frame record (state written before _obj_frames existed) must not crash.
    ts3, synth3 = TrackingEngine._sanitize_timestamps(
        broken, np.empty(0, dtype=np.int32), fps)
    check("a missing frame record degrades instead of raising",
          synth3 is True and ts3.size == broken.size)
    # A track too short to HAVE a span is not evidence of a broken clock. The
    # caller ORs this flag across every track, so counting a one-frame
    # detection as evidence marked entire healthy runs degraded.
    for n_short in (0, 1):
        short = np.zeros(n_short, dtype=np.float32)
        _t, s_short = TrackingEngine._sanitize_timestamps(
            short, np.arange(n_short, dtype=np.int32) + 1, fps)
        check(f"a {n_short}-sample track does not claim a broken clock",
              s_short is False)
    # Two real samples still detect a genuinely stuck clock.
    _t, s_stuck = TrackingEngine._sanitize_timestamps(
        np.zeros(2, dtype=np.float32), np.asarray([10, 11], dtype=np.int32), fps)
    check("two samples with an identical stamp still flag it", s_stuck is True)

check("FACTS_SCHEMA_VERSION was bumped so cached bundles are re-evaluated",
      FACTS_SCHEMA_VERSION >= 2, f"v{FACTS_SCHEMA_VERSION}")
stale = TrajectoryBundle("k", "v", 1920, 1080, 20.0, 100,
                         {1: parked_cart()}, facts_schema_version=1)
_f, reason = rules.evaluate_rules(stale, [door_zone()])
check("a v1 bundle reports unavailable rather than a silent all-clear",
      reason is not None and "re-run" in reason.lower(), repr(reason))


# ---------------------------------------------------------------------------
section("4. dedupe never collapses DIFFERENT carts")
# ---------------------------------------------------------------------------
t = np.arange(0.0, 20.0, 0.5)
c1 = rec(1, 1, "cart", t, [(300.0, 300.0)] * len(t))
c2 = rec(2, 2, "cart", t, [(1500.0, 900.0)] * len(t))    # other side of the store
f, _ = rules.evaluate_rules(bundle([c1, c2]), [],
                            thresholds={"abandoned_cart_s": 10.0})
ab = [x for x in f if x.rule_id == "abandoned_cart"]
check("two unattended carts produce two findings, not one",
      len(ab) == 2, f"got {len(ab)}")
check("each finding names its own cart",
      sorted(x.cart_display_id for x in ab) == [1, 2])
check("no cart is buried in evidence['also_carts']",
      all(not (x.evidence.get("also_carts") or []) for x in ab))

# Five carts in five aisles used to collapse into ONE finding.
many = [rec(i, i, "cart", t, [(200.0 * i, 300.0)] * len(t)) for i in range(1, 6)]
f5, _ = rules.evaluate_rules(bundle(many), [],
                             thresholds={"abandoned_cart_s": 10.0})
check("five unattended carts produce five findings",
      len([x for x in f5 if x.rule_id == "abandoned_cart"]) == 5)

# The merge this exists for must still happen: ONE cart whose interval was split.
split_ts = list(np.arange(0.0, 12.0, 0.5)) + list(np.arange(13.0, 25.0, 0.5))
one = rec(1, 1, "cart", split_ts, [(300.0, 300.0)] * len(split_ts))
f1, _ = rules.evaluate_rules(bundle([one]), [],
                             thresholds={"abandoned_cart_s": 10.0})
check("one cart's split interval still merges to a single finding",
      len([x for x in f1 if x.rule_id == "abandoned_cart"]) == 1)


# ---------------------------------------------------------------------------
section("6. suppressed sparse intervals are REPORTED, not swallowed")
# ---------------------------------------------------------------------------
# A cart in a busy doorway is occluded by the traffic it obstructs, so the
# intervals the density gate drops are disproportionately the SAFETY ones.
diag: list[str] = []
sparse = parked_cart(dur_s=60.0, gap=RULE_MAX_SAMPLE_GAP_S + 0.5)
f, reason = rules.evaluate_rules(bundle([sparse]), [door_zone()],
                                 diagnostics=diag)
bd = [x for x in f if x.rule_id == "blocked_door"]
check("the sparse interval is still (correctly) not asserted", len(bd) == 0)
check("but the run now SAYS it discarded something",
      any("sparsely observed" in d for d in diag), repr(diag))
check("the note carries a count", any("1 candidate" in d for d in diag))
check("it does not suppress findings via unavailable_reason",
      reason is None or "sparse" not in reason.lower(), repr(reason))

# Densely observed: no note, and the finding fires.
diag2: list[str] = []
dense = parked_cart(dur_s=60.0, gap=0.5)
f2, _ = rules.evaluate_rules(bundle([dense]), [door_zone()], diagnostics=diag2)
check("a well-observed 60s block still fires",
      len([x for x in f2 if x.rule_id == "blocked_door"]) == 1)
check("and reports no sparsity note",
      not any("sparsely observed" in d for d in diag2), repr(diag2))


# ---------------------------------------------------------------------------
section("7. 'clean' never hides a rule family that did not run")
# ---------------------------------------------------------------------------
# One analytics zone (the zone editor's DEFAULT kind) is enough to satisfy
# RULE_STATIC_KINDS, which used to make `reason` None and the whole run read as
# clean while the blocked-door rule had never been evaluated.
diag: list[str] = []
f, reason = rules.evaluate_rules(bundle([parked_cart()]), [analytics_zone()],
                                 diagnostics=diag)
state, _shown, _rem = highlights.ops_findings_state(reason, f)
check("state is still 'clean' (the zone-independent rules did run)",
      state == "clean", state)
check("but a note says the blocked-door rule did not run",
      any("Blocked-door rule did not run" in d for d in diag), repr(diag))

# With a door zone drawn, no such note.
diag2: list[str] = []
rules.evaluate_rules(bundle([parked_cart()]), [door_zone()], diagnostics=diag2)
check("drawing a door zone clears the note",
      not any("Blocked-door rule did not run" in d for d in diag2), repr(diag2))
check("but the missing AISLE family is reported instead",
      any("Static-cart rule did not run" in d for d in diag2), repr(diag2))

# Diagnostics must never be routed through unavailable_reason, which
# ops_findings_state() treats as "discard every finding".
diag3: list[str] = []
f3, reason3 = rules.evaluate_rules(bundle([parked_cart(at=(300.0, 300.0))]),
                                   [analytics_zone()],
                                   thresholds={"abandoned_cart_s": 10.0},
                                   diagnostics=diag3)
st3, shown3, _r3 = highlights.ops_findings_state(reason3, f3)
check("findings from the rules that DID run survive the notes",
      st3 == "findings" and len(shown3) >= 1, f"{st3} {len(shown3)}")
check("the notes are carried separately from the findings",
      len(highlights.ops_diagnostics(diag3)) >= 1)
check("ops_diagnostics de-duplicates",
      highlights.ops_diagnostics(["a", "a", " a ", "b"]) == ["a", "b"])

# The panel renders TWO different notices (the unavailable caveat and the
# coverage notes) and must not print either of them twice. A substring test
# cannot see duplication, which is how the first version of this fix shipped a
# doubled "did not run" heading on the empty+unavailable path.
from engine import ui_builder

_cases = {
    "empty+unavailable": ([], "No door zones drawn.", None),
    "empty+unavailable+notes": ([], "No door zones drawn.", ["Clock synthesised."]),
    "empty+clean+notes": ([], None, ["Clock synthesised."]),
    "findings+unavailable+notes": (list(f3), "No door zones drawn.",
                                   ["Clock synthesised."]),
}
for _name, _args in _cases.items():
    _html = ui_builder.build_operational_alerts(*_args)
    check(f"{_name}: 'did not run' appears at most once",
          _html.count("did not run") <= 1,
          f"x{_html.count('did not run')}")
    check(f"{_name}: 'Coverage notes' appears at most once",
          _html.count("Coverage notes") <= 1,
          f"x{_html.count('Coverage notes')}")
check("notes reach the panel when there are notes",
      "Coverage notes" in ui_builder.build_operational_alerts(
          [], None, ["Clock synthesised."]))


# ---------------------------------------------------------------------------
section("8. Zone.kind survives the API round trip")
# ---------------------------------------------------------------------------
# api.main instantiates TrackingEngine (and loads YOLO) at import, so the
# conversion helper cannot be imported here. Test the schema and the
# zone_editor helpers it now delegates to, plus the engine-side consequence.
from api.models import ZoneIn, ZoneOut
from engine import zone_editor

check("ZoneOut carries a kind", "kind" in ZoneOut.model_fields)
check("ZoneIn carries a kind", "kind" in ZoneIn.model_fields)
check("kind defaults to analytics for older clients",
      ZoneOut(zone_id="a", name="n", polygon=[[0, 0]], applies_to="person",
              color=[0, 0, 0]).kind == "analytics")
check("a door posted as a door stays a door",
      ZoneOut(zone_id="a", name="n", polygon=[[0, 0]], applies_to="person",
              color=[0, 0, 0], kind="door").kind == "door")
# The two derivations _zone_out_to_engine now performs.
check("a door's applies_to is coerced to 'both', not left at 'person'",
      zone_editor.coerce_applies_to("door", "person") == "both")
check("an analytics zone's applies_to is respected",
      zone_editor.coerce_applies_to("analytics", "cart") == "cart")
check("a door gets the door colour",
      zone_editor.zone_color_for("door", 0)
      == zone_editor._LAYOUT_COLORS_BGR["door"])
# The consequence: kind="analytics" cannot fire the blocked-door rule.
as_analytics = Zone("d1", "Main Door", door_zone().polygon,
                    applies_to="both", kind="analytics")
f_bad, _ = rules.evaluate_rules(bundle([parked_cart()]), [as_analytics])
f_good, _ = rules.evaluate_rules(bundle([parked_cart()]), [door_zone()])
check("kind='analytics' yields no blocked_door finding (the API bug)",
      not [x for x in f_bad if x.rule_id == "blocked_door"])
check("kind='door' yields one",
      len([x for x in f_good if x.rule_id == "blocked_door"]) == 1)


# ---------------------------------------------------------------------------
section("9. the event log contains only real, non-duplicated events")
# ---------------------------------------------------------------------------
# Reconciliation rewrites a cart's rows with classify_event(final_score, ...),
# which can return a name that is not an event at all.
demoted = compute_pops("UNKNOWN", "FAST", True, "partial", bag_label="bagged")
demoted_ev, _ = classify_event(demoted, False, "UNKNOWN")
check("a reconciled score really can produce a non-event",
      demoted < 31 and demoted_ev == "LOW PRIORITY",
      f"{demoted} -> {demoted_ev!r}")

# Scores are part of the contract now: prune_event_log() also drops a row below
# MEDIUM_SCORE, because "UNLINKED EXIT" is returned from two tiers and so the
# name alone stopped guaranteeing one. Each row here carries the score its own
# event name implies, so this section keeps testing what it was written to test —
# name filtering and duplicate collapse — rather than the new tier rule, which
# tests/test_event_row_coherence.py covers.
log = [
    {"cart_id": 2, "event": "MEDIUM PRIORITY", "frame": 10, "pops_score": 45},
    {"cart_id": 2, "event": "LOW PRIORITY", "frame": 50, "pops_score": 16},   # demoted
    {"cart_id": 3, "event": "UNLINKED EXIT", "frame": 20, "pops_score": 35},
    {"cart_id": 3, "event": "UNLINKED EXIT", "frame": 90, "pops_score": 35},  # duplicate
    {"cart_id": 4, "event": "MONITORING", "frame": 30, "pops_score": 8},      # never an event
]
kept, dropped = prune_event_log(log)
check("non-events are dropped", dropped == 3, f"dropped {dropped}")
check("every surviving row names a real event",
      all(r["event"] in {"PUSHOUT ALERT", "HIGH PRIORITY", "MEDIUM PRIORITY",
                         "UNLINKED EXIT", "ABANDONED CART"} for r in kept))
check("a cart never carries the same event twice",
      len({(r["cart_id"], r["event"]) for r in kept}) == len(kept))
check("the EARLIEST row wins a collapse",
      [r["frame"] for r in kept if r["cart_id"] == 3] == [20])
check("real events are preserved",
      sorted(r["cart_id"] for r in kept) == [2, 3])
check("an empty log does not raise", prune_event_log([]) == ([], 0))
check("None does not raise", prune_event_log(None) == ([], 0))


# ---------------------------------------------------------------------------
section("10. a walking pushout can reach HIGH PRIORITY")
# ---------------------------------------------------------------------------
walk = compute_pops("OUTBOUND", "MEDIUM", True, "full", bag_label="unbagged")
still = compute_pops("OUTBOUND", "STATIC", True, "full", bag_label="unbagged")
run = compute_pops("OUTBOUND", "FAST", True, "full", bag_label="unbagged")
check("full + unbagged at walking pace clears the 71 line (was 70)",
      walk >= 71, f"got {walk}")
# At MEDIUM pace this now reaches 80, which is PUSHOUT_SCORE — see section 12.
check("its event is at least HIGH PRIORITY",
      classify_event(walk, False, "OUTBOUND")[0]
      in ("HIGH PRIORITY", "PUSHOUT ALERT"),
      classify_event(walk, False, "OUTBOUND")[0])
slow = compute_pops("OUTBOUND", "SLOW", True, "full", bag_label="unbagged")
check("walking pace at the SLOW end clears it too",
      slow >= 71, f"got {slow}")
check(f"the high tier no longer hinges on SPEED_MEDIUM ({SPEED_MEDIUM} px/s), "
      f"which is resolution-dependent",
      slow >= 71 and walk >= 71 and run >= 71,
      f"SLOW={slow} MEDIUM={walk} FAST={run}")
# A loaded cart standing still near the exit is NOT leaving yet, and the
# operational rule engine owns that case with duration evidence POPS lacks.
check("a STATIC loaded cart is not promoted to HIGH",
      still < 71, f"got {still}")
check("it is still a MEDIUM-tier alert, not ignored",
      still >= 31, f"got {still}")

# The terms that are treated as SPEC must not have moved.
check("bagged merchandise is unaffected — items look paid for",
      compute_pops("OUTBOUND", "MEDIUM", True, "full", bag_label="bagged") == 40,
      str(compute_pops("OUTBOUND", "MEDIUM", True, "full", bag_label="bagged")))
check("partial + bagged is still capped at 55",
      compute_pops("OUTBOUND", "STATIC", True, "partial", bag_label="bagged",
                   abandoned=True, linked=True) == 55)
check("the abandonment floor for loaded carts is still 75",
      compute_pops("OUTBOUND", "STATIC", True, "partial", bag_label="unbagged",
                   abandoned=True, linked=True) == 75)
check("the abandonment floor for empty carts is still 60",
      compute_pops("OUTBOUND", "STATIC", True, "empty",
                   abandoned=True, linked=True) == 60)
check("an empty outbound cart is still not an event",
      compute_pops("OUTBOUND", "MEDIUM", True, "empty") < 31)
check("INBOUND is still floored to 5 by the kill switch",
      compute_pops("INBOUND", "FAST", True, "full", bag_label="unbagged") == 5)
check("the score is still clamped to 100",
      compute_pops("OUTBOUND", "FAST", True, "full", bag_label="unbagged",
                   abandoned=True) <= 100)


# ---------------------------------------------------------------------------
section("11. the INBOUND kill switch reports what it suppressed")
# ---------------------------------------------------------------------------
# compute_pops() returns INBOUND_SCORE for an inbound cart before looking at
# contents at all, and INBOUND_SCORE is under the 31 that logs an event. So a
# camera_placement that inverts the axis empties the Events tab, drops the
# banner and floors every POPS row while the detector keeps working and every
# box is still drawn -- which reads as "the detections stopped".
from engine.scoring import (
    INBOUND_SCORE, direction_suppressed_carts, inbound_suppression_note,
)

check("the kill switch and the detector's constant agree",
      compute_pops("INBOUND", "FAST", True, "full", bag_label="unbagged")
      == INBOUND_SCORE, f"got {compute_pops('INBOUND', 'FAST', True, 'full', bag_label='unbagged')}")
check("an inbound cart is under the event threshold by construction",
      INBOUND_SCORE < 31)

check("a clean run produces no note", inbound_suppression_note({}) is None)
check("an outbound scored cart produces no note",
      inbound_suppression_note(
          {2: {"direction": "OUTBOUND", "score": 75, "fill": "full"}}) is None)

_empty_only = {2: {"direction": "INBOUND", "score": INBOUND_SCORE, "fill": "empty"},
               4: {"direction": "INBOUND", "score": INBOUND_SCORE, "fill": "empty"}}
_n = inbound_suppression_note(_empty_only, "Outside (facing entrance)")
check("inbound empty carts are reported but not alarming",
      _n is not None and "read as empty" in _n)
check("the note names the carts", _n is not None and "C2, C4" in _n)
check("the note names the placement in force",
      "Outside (facing entrance)" in (_n or ""))

_loaded = {1: {"direction": "INBOUND", "score": INBOUND_SCORE, "fill": "partial"},
           2: {"direction": "INBOUND", "score": INBOUND_SCORE, "fill": "empty"},
           3: {"direction": "INBOUND", "score": INBOUND_SCORE, "fill": "full"}}
_n2 = inbound_suppression_note(_loaded, "Inside (facing exit)")
sup, loaded = direction_suppressed_carts(_loaded)
check("every inbound cart counts as suppressed", sup == [1, 2, 3])
check("only the LOADED ones are called out", loaded == [1, 3])
check("the loaded note points at the placement",
      "placement is inverted" in (_n2 or ""))
check("the loaded note names only the loaded carts",
      "(C1, C3)" in (_n2 or ""), _n2 or "")

# A cart that actually got scored must never be reported as suppressed.
check("a scored inbound cart is not counted",
      direction_suppressed_carts(
          {2: {"direction": "INBOUND", "score": 45, "fill": "full"}}) == ([], []))
# Long lists are capped so the notice stays readable.
_many = {i: {"direction": "INBOUND", "score": INBOUND_SCORE, "fill": "empty"}
         for i in range(1, 14)}
check("a long cart list is capped", "+5 more" in inbound_suppression_note(_many))
# Malformed snapshots must degrade, not raise.
for _bad in ({2: None}, {2: {}}, {2: {"direction": None, "score": None}},
             {2: {"direction": "INBOUND", "score": "n/a"}}, None):
    try:
        inbound_suppression_note(_bad)
        _ok = True
    except Exception as _e:
        _ok = False
    check(f"malformed snapshot {_bad!r} does not raise", _ok)

# It has to reach every surface, through the one shared projection.
_panel = ui_builder.build_operational_alerts([], None, [_n2])
_report_src = __import__("engine.case_report_builder",
                         fromlist=["_build_ops_findings_block"])


class _Res:
    rule_findings = []
    rules_unavailable_reason = None
    rule_diagnostics = [_n2]
    queue_spikes = []
    dwell_summary = []


check("the Operational Alerts panel carries it",
      "INBOUND kill switch" in _panel)
check("the panel still reports the all-clear alongside it",
      "No operational issues detected" in _panel)
check("the standalone case report carries it",
      "INBOUND kill switch"
      in _report_src._build_ops_findings_block(_Res()))
check("it de-duplicates like every other coverage note",
      highlights.ops_diagnostics([_n2, _n2]) == [_n2])


# ---------------------------------------------------------------------------
section("12. a high enough score IS a pushout, with no abandonment evidence")
# ---------------------------------------------------------------------------
from engine.scoring import HIGH_SCORE, MEDIUM_SCORE, PUSHOUT_SCORE

check("the pushout line sits above the high line",
      MEDIUM_SCORE < HIGH_SCORE < PUSHOUT_SCORE <= 100,
      f"{MEDIUM_SCORE}/{HIGH_SCORE}/{PUSHOUT_SCORE}")

# The new route: score alone.
check("at PUSHOUT_SCORE it is a pushout without abandonment",
      classify_event(PUSHOUT_SCORE, False, "OUTBOUND",
                     abandoned=False)[0] == "PUSHOUT ALERT")
check("one point below it is not",
      classify_event(PUSHOUT_SCORE - 1, False, "OUTBOUND",
                     abandoned=False)[0] == "HIGH PRIORITY")
check("a linked person does not soften it",
      classify_event(PUSHOUT_SCORE, True, "OUTBOUND",
                     abandoned=False)[0] == "PUSHOUT ALERT")
check("and it holds all the way to 100",
      classify_event(100, False, "OUTBOUND",
                     abandoned=False)[0] == "PUSHOUT ALERT")

# The OLD route must be untouched: 71..79 still needs the person to have left.
for _s in (HIGH_SCORE, PUSHOUT_SCORE - 1):
    check(f"score {_s} + abandoned is still a pushout",
          classify_event(_s, True, "OUTBOUND", abandoned=True)[0]
          == "PUSHOUT ALERT")
    check(f"score {_s} without abandonment is still HIGH PRIORITY",
          classify_event(_s, True, "OUTBOUND", abandoned=False)[0]
          == "HIGH PRIORITY")

# Lower tiers must not have moved.
check("the medium tier is unchanged",
      classify_event(MEDIUM_SCORE, False, "OUTBOUND")[0] == "UNLINKED EXIT")
check("abandonment below the high line is still ABANDONED CART",
      classify_event(60, True, "OUTBOUND", abandoned=True)[0] == "ABANDONED CART")
check("an abandoned EMPTY outbound cart stays one tier down",
      classify_event(
          compute_pops("OUTBOUND", "SLOW", True, "empty", abandoned=True,
                       linked=True), True, "OUTBOUND",
          abandoned=True)[0] == "ABANDONED CART")

# A pushout must remain unreachable for anything not leaving the store.
_bad = []
for _d in ("INBOUND", "UNKNOWN"):
    for _sp in ("STATIC", "SLOW", "MEDIUM", "FAST"):
        for _f in ("empty", "partial", "full", "unclassified"):
            for _b in ("bagged", "unbagged", "not_applicable"):
                for _lk in (True, False):
                    for _ab in (True, False):
                        _sc = compute_pops(_d, _sp, True, _f, bag_label=_b,
                                           abandoned=_ab, linked=_lk)
                        if classify_event(_sc, _lk, _d,
                                          abandoned=_ab)[0] == "PUSHOUT ALERT":
                            _bad.append((_d, _sp, _f, _b, _lk, _ab, _sc))
check("no INBOUND or UNKNOWN cart can be called a pushout",
      not _bad, f"{len(_bad)} combos: {_bad[:3]}")

# What it means in practice, end to end from the scorer.
_walk_full = compute_pops("OUTBOUND", "MEDIUM", True, "full", bag_label="unbagged")
check("walking a full unbagged cart out is now a pushout",
      classify_event(_walk_full, False, "OUTBOUND")[0] == "PUSHOUT ALERT",
      f"score={_walk_full}")
_bagged = compute_pops("OUTBOUND", "FAST", True, "full", bag_label="bagged")
check("a bagged full cart is not, however fast",
      classify_event(_bagged, False, "OUTBOUND")[0] != "PUSHOUT ALERT",
      f"score={_bagged}")

# It has to be a real event, and a high one, or nothing downstream reacts.
check("PUSHOUT ALERT is loggable", "PUSHOUT ALERT" in LOGGABLE_EVENTS)
from engine.scoring import HIGH_EVENTS
check("PUSHOUT ALERT counts as high", "PUSHOUT ALERT" in HIGH_EVENTS)


# ---------------------------------------------------------------------------
print(f"\n{'=' * 62}")
print(f"PASSED {len(_PASS)} / {len(_PASS) + len(_FAIL)}")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("All green.")
