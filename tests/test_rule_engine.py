# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Rule-engine tests — synthetic bundles only, no GPU and no video decode.

Run with:  python tests/test_rule_engine.py

Every assertion here pins a specific failure mode rather than just exercising
the happy path. The comments name what breaks if the assertion regresses,
because most of these are silent failures: the rule stops firing and the empty
alert list looks exactly like "nothing happened".

Deliberately stdlib + numpy only, no pytest — matches the rest of the repo,
which has no test infrastructure to hook into.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine.analytics_models import (
    CartFactSample, TrackRecord, TrajectoryBundle, Zone, FACTS_SCHEMA_VERSION,
)
from engine.analytics_builder import run_all
from engine import rules, ui_builder, zone_editor

W, H, FPS = 1280, 720, 30.0
_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_track(display_id, xy, n, *, t0=0.0, dt=1 / FPS, bbox_wh=(80, 60),
               jitter=0.0, label="cart", raw=1):
    """n samples parked at xy (plus optional jitter), evenly spaced in time."""
    rng = np.random.default_rng(display_id)
    pos = np.tile(np.asarray(xy, dtype=np.float32), (n, 1))
    if jitter:
        pos = pos + rng.normal(0, jitter, size=(n, 2)).astype(np.float32)
    ts = (t0 + np.arange(n) * dt).astype(np.float32)
    speeds = np.zeros(n, dtype=np.float32)
    if jitter:
        # Mimics how compute_motion() inflates speed from a jittering centroid.
        speeds[:] = jitter * 2 / max(dt * 4, 1e-6)
    bw, bh = bbox_wh
    bb = np.stack([pos[:, 0] - bw / 2, pos[:, 1] - bh / 2,
                   pos[:, 0] + bw / 2, pos[:, 1] + bh / 2], axis=1).astype(np.float32)
    return TrackRecord(raw_id=raw, label=label, display_id=display_id,
                       positions=pos, timestamps=ts,
                       frames=np.rint(ts * FPS).astype(np.int32),
                       speeds=speeds, bboxes=bb)


def make_facts(n, *, t0=0.0, dt=1 / FPS, fill="unclassified", every=8, linked=False):
    """Fact samples with a realistic staleness cycle (fill refreshes every 8th)."""
    return [
        CartFactSample(
            t=t0 + i * dt, frame=int((t0 + i * dt) * FPS), fill=fill,
            bag="not_applicable", fill_conf=0.9, quality="valid_cart",
            fill_stale_frames=i % every, linked=linked,
            linked_person_display=None, abandoned_linker=False)
        for i in range(n)
    ]


def make_bundle(tracks, facts=None, *, synth=False, version=FACTS_SCHEMA_VERSION,
                with_frame=False):
    return TrajectoryBundle(
        video_key="k", video_path="v.mp4", width=W, height=H, fps=FPS,
        total_frames=10000,
        tracks={r.raw_id: r for r in tracks},
        cart_facts=facts or {},
        facts_schema_version=version,
        timestamps_synthesized=synth,
        representative_frame=(np.zeros((H, W, 3), np.uint8) if with_frame else None),
    )


def rect_zone(zone_id, name, x, y, w, h, kind, applies_to="both"):
    poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
    return Zone(zone_id=zone_id, name=name, polygon=poly,
                applies_to=applies_to, kind=kind)


def door_zone():
    # applies_to defaults to "person" in the editor — kept here on purpose so
    # the test proves layout zones ignore it (see the applies_to case below).
    return rect_zone("d1", "Main Door", 600, 300, 120, 120, "door",
                     applies_to="person")


AISLE = rect_zone("a1", "Aisle 3", 200, 200, 300, 300, "aisle")
CORRAL = rect_zone("f1", "Corral", 900, 100, 200, 200, "fixture")


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def test_blocked_door():
    section("Blocked door: cart static in a door zone for 60s (threshold 45s)")
    n = int(60 * FPS)
    b = make_bundle([make_track(7, (660, 360), n)], {7: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [door_zone()])
    bd = [x for x in f if x.rule_id == "blocked_door"]
    check("fires", len(bd) == 1, f"got {[(x.rule_id, x.duration_s) for x in f]}")
    if bd:
        check("severity is SAFETY", bd[0].severity == "SAFETY")
        check("duration ~60s", 58 <= bd[0].duration_s <= 61, f"{bd[0].duration_s}")
        check("uses bbox-overlap membership",
              bd[0].evidence.get("membership") == "bbox_overlap")
        # Reusing the POPS name would collide with _EVENT_SEVERITY and
        # LOGGABLE_EVENTS, and would be consumed by POPS reconciliation.
        check("label is not a POPS event name",
              bd[0].label not in ("ABANDONED CART", "PUSHOUT ALERT",
                                  "HIGH PRIORITY", "MEDIUM PRIORITY"))


def test_applies_to_is_ignored_for_layout_zones():
    section("A door zone left at applies_to='person' must still match carts")
    # The editor's "Track type" dropdown defaults to "person". If rule
    # membership honoured that, every hand-drawn door would match zero cart
    # tracks and the rule would silently never fire.
    n = int(60 * FPS)
    b = make_bundle([make_track(8, (660, 360), n)], {8: make_facts(n)})
    check("zone under test really is applies_to='person'",
          door_zone().applies_to == "person")
    f, _ = rules.evaluate_rules(b, [door_zone()])
    check("still fires for a cart track",
          len([x for x in f if x.rule_id == "blocked_door"]) == 1)


def test_sparse_samples_rejected():
    section("Occlusion: 60s span from only 4 samples must NOT fire")
    # Positions are appended only when a track is DETECTED, so a cart occluded
    # for 20s at a time leaves samples far apart that an RLE reads as
    # continuous presence. Without the density gate this reports "static for
    # 60s" from four observations.
    b = make_bundle([make_track(9, (660, 360), 4, dt=20.0)],
                    {9: make_facts(4, dt=20.0)})
    f, _ = rules.evaluate_rules(b, [door_zone()])
    check("density gate rejects it",
          len([x for x in f if x.rule_id == "blocked_door"]) == 0,
          f"got {[(x.rule_id, x.n_samples) for x in f]}")


def test_jittering_cart_is_still_static():
    section("Stationary cart with a jittering bbox must count as static")
    # compute_motion() derives speed from a first-to-last delta over <=5
    # positions, so centroid jitter on a parked cart inflates the reading well
    # past SPEED_STATIC. A speed-only test never sees this cart as static.
    n = int(60 * FPS)
    jit = make_track(10, (660, 360), n, jitter=3.0)
    check("jitter really does inflate speed past SPEED_STATIC",
          jit.speeds.mean() > 10, f"mean {jit.speeds.mean():.0f} px/s")
    b = make_bundle([jit], {10: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [door_zone()])
    check("spread test rescues it",
          len([x for x in f if x.rule_id == "blocked_door"]) == 1)


def test_moving_cart_not_static():
    section("Cart driving through a door zone must NOT fire")
    n = int(60 * FPS)
    pos = np.stack([np.linspace(600, 900, n), np.full(n, 360.0)], 1).astype(np.float32)
    ts = (np.arange(n) / FPS).astype(np.float32)
    mov = TrackRecord(raw_id=1, label="cart", display_id=11, positions=pos,
                      timestamps=ts, frames=np.rint(ts * FPS).astype(np.int32),
                      speeds=np.full(n, 120.0, np.float32),
                      bboxes=np.stack([pos[:, 0] - 40, pos[:, 1] - 30,
                                       pos[:, 0] + 40, pos[:, 1] + 30], 1).astype(np.float32))
    b = make_bundle([mov], {11: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [door_zone()])
    check("does not fire", len([x for x in f if x.rule_id == "blocked_door"]) == 0)


def test_bbox_spans_door_but_centroid_outside():
    section("Cart body spans a thin doorway while its centroid sits outside")
    # This is the case blocked-door exists to catch, and it is exactly what
    # centroid-in-polygon membership misses.
    thin = rect_zone("d2", "Thin Door", 700, 300, 30, 120, "door")
    n = int(60 * FPS)
    spanning = make_track(12, (690, 360), n, bbox_wh=(120, 60))  # 630..750
    check("centroid is outside the polygon", not (700 <= 690 <= 730))
    b = make_bundle([spanning], {12: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [thin])
    check("overlap-fraction membership catches it",
          len([x for x in f if x.rule_id == "blocked_door"]) == 1)


def test_overlapping_zones():
    section("Overlapping aisle + door zones must both report")
    # analytics_builder.build_zone_membership is winner-takes-all: it collapses
    # to one zone index per sample, so the door would lose to a later-indexed
    # aisle and its rule would never fire. rules.py keeps a per-zone matrix.
    big_aisle = rect_zone("a9", "Big Aisle", 500, 200, 400, 300, "aisle")
    n = int(150 * FPS)
    b = make_bundle([make_track(13, (660, 360), n)], {13: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [big_aisle, door_zone()])
    check("door rule fires despite the overlap",
          len([x for x in f if x.rule_id == "blocked_door"]) == 1,
          f"got {[(x.rule_id, x.zone_name) for x in f]}")
    check("aisle rule fires too",
          len([x for x in f if x.rule_id == "static_cart"]) == 1)


def test_static_cart():
    section("Static cart in an aisle zone (threshold 120s)")
    n = int(150 * FPS)
    b = make_bundle([make_track(14, (350, 350), n)], {14: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [AISLE])
    sc = [x for x in f if x.rule_id == "static_cart"]
    check("fires", len(sc) == 1, f"got {[(x.rule_id, x.duration_s) for x in f]}")
    if sc:
        check("severity is WATCH", sc[0].severity == "WATCH")
        check("uses centroid membership",
              sc[0].evidence.get("membership") == "centroid")


def test_unattended_cart():
    section("Unattended cart: static 200s with nobody nearby (threshold 180s)")
    n = int(200 * FPS)
    b = make_bundle([make_track(15, (350, 350), n)], {15: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [])
    ab = [x for x in f if x.rule_id == "abandoned_cart"]
    check("fires", len(ab) == 1, f"got {[(x.rule_id, x.duration_s) for x in f]}")
    if ab:
        check("severity is ACTION", ab[0].severity == "ACTION")
        # POPS gates "ABANDONED CART" behind pops_score >= 31 and floors
        # INBOUND carts to 5, so sharing the name would make this unreachable.
        check("label is distinct from the POPS event",
              ab[0].label != "ABANDONED CART", ab[0].label)


def test_unattended_suppressed_by_nearby_person():
    section("Unattended must be suppressed while a person stands beside the cart")
    n = int(200 * FPS)
    cart = make_track(16, (350, 350), n, raw=1)
    person = make_track(1, (360, 360), n, label="person", raw=2)   # ~14px away
    b = make_bundle([cart, person], {16: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [])
    check("does not fire",
          len([x for x in f if x.rule_id == "abandoned_cart"]) == 0)


def test_unattended_suppressed_in_corral():
    section("Unattended must be suppressed inside a fixture zone (cart corral)")
    n = int(200 * FPS)
    b = make_bundle([make_track(17, (1000, 200), n)], {17: make_facts(n)})
    f, _ = rules.evaluate_rules(b, [CORRAL])
    check("corral carve-out applies",
          len([x for x in f if x.rule_id == "abandoned_cart"]) == 0)


def _inbound_cart(display_id, n):
    """Cart moving up-frame — INBOUND for 'Outside (facing entrance)'."""
    pos = np.stack([np.full(n, 640.0), np.linspace(600, 300, n)], 1).astype(np.float32)
    ts = (np.arange(n) / FPS).astype(np.float32)
    return TrackRecord(raw_id=1, label="cart", display_id=display_id, positions=pos,
                       timestamps=ts, frames=np.rint(ts * FPS).astype(np.int32),
                       speeds=np.full(n, 50.0, np.float32),
                       bboxes=np.stack([pos[:, 0] - 40, pos[:, 1] - 30,
                                        pos[:, 0] + 40, pos[:, 1] + 30], 1).astype(np.float32))


def test_incoming_empty():
    section("Incoming empty cart (INBOUND + empty)")
    n = int(6 * FPS)
    rec = _inbound_cart(18, n)
    b = make_bundle([rec], {18: make_facts(n, fill="empty")})
    f, _ = rules.evaluate_rules(b, [], camera_placement="Outside (facing entrance)")
    ie = [x for x in f if x.rule_id == "incoming_empty"]
    check("fires", len(ie) == 1, f"got {[x.rule_id for x in f]}")
    if ie:
        check("severity is INFO", ie[0].severity == "INFO")
        # It is whole-cart fill classification, not merchandise detection, and
        # must not be presented as the latter.
        check("declares itself a proxy", "proxy" in ie[0].evidence.get("basis", ""))


def test_incoming_full_does_not_fire():
    section("Incoming FULL cart must NOT fire the empty rule")
    n = int(6 * FPS)
    b = make_bundle([_inbound_cart(19, n)], {19: make_facts(n, fill="full")})
    f, _ = rules.evaluate_rules(b, [], camera_placement="Outside (facing entrance)")
    check("does not fire",
          len([x for x in f if x.rule_id == "incoming_empty"]) == 0)


# ---------------------------------------------------------------------------
# Degraded / unavailable states
# ---------------------------------------------------------------------------
def test_stale_bundle_is_unavailable_not_empty():
    section("Bundle cached before the rule engine → unavailable, not all-clear")
    n = int(60 * FPS)
    b = make_bundle([make_track(20, (660, 360), n)], {}, version=0)
    f, reason = rules.evaluate_rules(b, [door_zone()])
    check("no findings", len(f) == 0)
    check("reason explains the staleness",
          reason is not None and "re-run" in reason.lower(), repr(reason))


def test_no_zones_reports_a_reason():
    section("No door/aisle zones drawn → explicit reason")
    n = int(60 * FPS)
    b = make_bundle([make_track(21, (660, 360), n)], {21: make_facts(n)})
    _f, reason = rules.evaluate_rules(b, [])
    check("reason mentions zones",
          reason is not None and "zone" in reason.lower(), repr(reason))


def test_synthesized_timestamps_degrade_confidence():
    section("Synthesised timestamps must mark findings degraded")
    n = int(60 * FPS)
    b = make_bundle([make_track(22, (660, 360), n)], {22: make_facts(n)}, synth=True)
    f, _ = rules.evaluate_rules(b, [door_zone()])
    check("findings exist and are all degraded",
          len(f) > 0 and all(x.confidence == "degraded" for x in f),
          f"{[(x.rule_id, x.confidence) for x in f]}")


def test_empty_bundle_does_not_crash():
    section("Empty bundle / no tracks")
    f, reason = rules.evaluate_rules(make_bundle([], {}), [door_zone()])
    check("returns cleanly with a reason", f == [] and reason is not None, repr(reason))


def test_join_on_time():
    section("join_on_time staleness semantics")
    # The only sanctioned way to combine display_id-keyed facts with
    # raw_id-keyed track arrays. Parallel indexing misattributes silently.
    src = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    dst = np.array([-0.5, 0.0, 1.4, 5.0], dtype=np.float32)
    idx = rules.join_on_time(src, dst, 1.0)
    check("before the first sample -> -1", idx[0] == -1, str(idx))
    check("exact match -> 0", idx[1] == 0, str(idx))
    check("0.4s stale -> previous sample", idx[2] == 1, str(idx))
    check("3s stale -> -1 (not inherited)", idx[3] == -1, str(idx))


# ---------------------------------------------------------------------------
# Integration: run_all -> AnalyticsResult -> UI
# ---------------------------------------------------------------------------
def test_run_all_integration():
    section("run_all wires the rule engine without disturbing analytics")
    n = int(90 * FPS)
    b = make_bundle([make_track(23, (660, 360), n)], {23: make_facts(n)},
                    with_frame=True)
    res = run_all(b, [door_zone()], out_dir=None)
    check("rule_findings populated",
          any(x.rule_id == "blocked_door" for x in res.rule_findings),
          f"{[(x.rule_id, x.duration_s) for x in res.rule_findings]}")
    check("heatmap still produced", res.heatmap_composite is not None)
    return res


def test_alert_banner(findings):
    section("Alert banner surfaces the rule severity vocabulary")
    # The banner hard-filters on BACKED_UP/QUEUE_FORMING. Without its own
    # filter, every SAFETY/ACTION finding is dropped at that line.
    banner = ui_builder.build_alert_banner([], [], [], findings)
    check("banner rendered", bool(banner.strip()))
    check("names the issue", "BLOCKED DOOR" in banner)
    check("SAFETY selects the critical palette",
          "#dc2626" in banner or "7f1d1d" in banner,
          "amber would mean SAFETY was treated as merely congestion")
    check("headline is safety-specific", "Safety issue" in banner)
    check("old 3-arg call still works",
          ui_builder.build_alert_banner([], [], []) == "")


def test_operational_alerts_table(findings):
    section("Operational alerts table: findings, empty, and unavailable")
    html = ui_builder.build_operational_alerts(findings, None)
    check("renders findings", "BLOCKED DOOR" in html and "SAFETY" in html)
    check("shows a start time", "00:00" in html)

    empty = ui_builder.build_operational_alerts([], None)
    unavail = ui_builder.build_operational_alerts([], "No door zones drawn.")
    check("empty says no issues found", "No operational issues" in empty)
    check("unavailable says did not run", "did not run" in unavail)
    check("the two states are distinct", empty != unavail)
    # The important one: a run where rules could not execute must never be
    # presented as a clean bill of health.
    check("unavailable never claims all-clear",
          "No operational issues" not in unavail)


def test_rule_crash_is_isolated():
    section("A rule bug must not take analytics down with it")
    import engine.analytics_builder as ab
    n = int(90 * FPS)
    b = make_bundle([make_track(24, (660, 360), n)], {24: make_facts(n)},
                    with_frame=True)
    orig = ab.rule_engine.evaluate_rules

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic rule failure")

    ab.rule_engine.evaluate_rules = _boom
    try:
        print("  (a traceback below is expected — the crash is deliberate)")
        res = run_all(b, [door_zone()], out_dir=None)
        check("analytics survived", res.heatmap_composite is not None)
        check("reason reports the failure",
              "failed" in (res.rules_unavailable_reason or ""),
              repr(res.rules_unavailable_reason))
    finally:
        ab.rule_engine.evaluate_rules = orig


def test_validate_polygon():
    section("validate_polygon rejects unusable zone shapes")
    # fillPoly's even-odd rule punches holes where loops overlap, so membership
    # reports "outside" for pixels the user meant to include.
    ok, msg = zone_editor.validate_polygon([(0, 0), (100, 100), (100, 0), (0, 100)])
    check("symmetric bowtie rejected", not ok, msg)
    # A symmetric bowtie has ~zero net area, so the area check catches it and
    # the crossing test never runs. This one has area 10000 and must be caught
    # by the crossing test specifically.
    ok_x, msg_x = zone_editor.validate_polygon(
        [(0, 100), (400, 100), (400, 400), (200, 0)])
    check("self-intersecting shape with real area rejected", not ok_x, msg_x)
    check("...and rejected by the crossing test, not the area test",
          "cross" in msg_x.lower(), msg_x)
    check("plain square accepted",
          zone_editor.validate_polygon([(0, 0), (100, 0), (100, 100), (0, 100)])[0])
    check("triangle accepted",
          zone_editor.validate_polygon([(0, 0), (100, 0), (100, 100)])[0])
    check("degenerate tiny polygon still rejected",
          not zone_editor.validate_polygon([(0, 0), (1, 0), (1, 1)])[0])


# ---------------------------------------------------------------------------
def main():
    test_blocked_door()
    test_applies_to_is_ignored_for_layout_zones()
    test_sparse_samples_rejected()
    test_jittering_cart_is_still_static()
    test_moving_cart_not_static()
    test_bbox_spans_door_but_centroid_outside()
    test_overlapping_zones()
    test_static_cart()
    test_unattended_cart()
    test_unattended_suppressed_by_nearby_person()
    test_unattended_suppressed_in_corral()
    test_incoming_empty()
    test_incoming_full_does_not_fire()
    test_stale_bundle_is_unavailable_not_empty()
    test_no_zones_reports_a_reason()
    test_synthesized_timestamps_degrade_confidence()
    test_empty_bundle_does_not_crash()
    test_join_on_time()
    res = test_run_all_integration()
    test_alert_banner(res.rule_findings)
    test_operational_alerts_table(res.rule_findings)
    test_rule_crash_is_isolated()
    test_validate_polygon()

    print("\n" + "=" * 62)
    print(f"PASSED {len(_PASS)} / {len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print("FAILED:")
        for name in _FAIL:
            print("  -", name)
        return 1
    print("All green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
