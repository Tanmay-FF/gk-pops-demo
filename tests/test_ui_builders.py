# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
UI-builder tests — pure HTML generation, no GPU and no video decode.

Run with:  python tests/test_ui_builders.py

These pin the regressions that motivated the UI rework, because every one of
them fails silently in the browser rather than raising:

  * a builder emitting a hardcoded colour instead of a theme variable is
    invisible in one of the two themes;
  * a table missing its sort/export/seek hooks looks fine but does nothing;
  * an "empty" state that reads as "all clear" is a wrong answer, not a
    missing one.

Deliberately stdlib + numpy only, no pytest — matches tests/test_rule_engine.py.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine import theme as T
from engine import analytics_ui, ui_builder, zone_editor
from engine.analytics_models import Zone

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


# Colours that must never be emitted raw by an in-app builder: they are the
# old hardcoded dark palette and the old hardcoded light surfaces.
_BANNED_LITERALS = [
    "#1d2231", "#262c3e", "#323a52", "#161a25",   # old analytics dark surfaces
    "#dde0eb", "#a8acc0", "#7e8298",              # old analytics text ramp
]


def _has_banned(html: str) -> list[str]:
    return [c for c in _BANNED_LITERALS if c.lower() in html.lower()]


# ---------------------------------------------------------------------------
class _Finding:
    """Minimal RuleFinding stand-in — the builders only read attributes."""
    def __init__(self, rule_id, label, severity, cart, zone, t0, dur,
                 reasons=(), ongoing=False, confidence="high", evidence=None):
        self.rule_id = rule_id
        self.label = label
        self.severity = severity
        self.cart_display_id = cart
        self.zone_id = "z1"
        self.zone_name = zone
        self.start_t = t0
        self.end_t = t0 + dur
        self.duration_s = dur
        self.threshold_s = 45.0
        self.ongoing_at_eov = ongoing
        self.confidence = confidence
        self.n_samples = 30
        self.reasons = list(reasons)
        self.evidence = evidence or {}


class _Spike:
    def __init__(self, zid, name, sev):
        self.zone_id = zid
        self.zone_name = name
        self.severity = sev
        self.avg_dwell_s = 42.0
        self.max_dwell_s = 91.0
        self.n_visits = 7
        self.threshold_s = 30.0
        self.reasons = ["occupancy 6", "mean speed 8 px/s"]
        self.score = 78.0
        self.peak_occupancy = 6
        self.mean_occupancy = 4.1


class _DwellRow:
    def __init__(self, tid, label, zone, idx, t0, t1):
        self.display_id = tid
        self.track_label = label
        self.zone_name = zone
        self.zone_id = "z1"
        self.visit_index = idx
        self.enter_t = t0
        self.exit_t = t1
        self.dwell_seconds = t1 - t0


class _Result:
    def __init__(self, dwell_rows=(), spikes=(), edges=(), insight=""):
        self.dwell_rows = list(dwell_rows)
        self.queue_spikes = list(spikes)
        self.journey_edges = list(edges)
        self.insight_text = insight


EVENTS = [
    {"frame": 120, "timestamp": 4.0, "cart_id": 3, "event": "PUSHOUT ALERT",
     "pops_score": 88, "fill": "FULL", "bag": "unbagged", "speed_status": "FAST",
     "direction": "OUTBOUND", "linked": True},
    {"frame": 300, "timestamp": 10.0, "cart_id": 5, "event": "MEDIUM PRIORITY",
     "pops_score": 44, "fill": "PARTIAL", "bag": "not_applicable",
     "speed_status": "SLOW", "direction": "INBOUND", "linked": False},
]
MAX_POPS = {3: 88, 5: 44, 7: 12}
SNAPS = {
    3: {"score": 88, "event": "PUSHOUT ALERT", "fill": "FULL",
        "bag": "unbagged", "quality": "valid", "timestamp": 4.0},
    5: {"score": 44, "event": "MEDIUM PRIORITY", "fill": "PARTIAL",
        "bag": "not_applicable", "quality": "valid", "timestamp": 10.0},
    7: {"score": 12, "event": "MONITORING", "fill": "EMPTY",
        "bag": "not_applicable", "quality": "valid", "timestamp": 21.5},
}


def test_no_hardcoded_palette():
    section("Builders emit theme variables, not the old literal palette")
    panels = {
        "pops": ui_builder.build_pops_summary(MAX_POPS, SNAPS),
        "events": ui_builder.build_events_timeline(EVENTS),
        "ops": ui_builder.build_operational_alerts(
            [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 3, "Main Door",
                      12.0, 60.0, ["stationary 60s"])]),
        "ops_empty": ui_builder.build_operational_alerts([]),
        "video_info": ui_builder.build_video_info("/x/clip.mp4", 1280, 720, 30, 900, 900),
        "legend": ui_builder.build_legend(),
        "analytics_empty": analytics_ui.build_analytics_empty_state(False),
        "spikes": analytics_ui.build_queue_spikes_banner([_Spike("z1", "Checkout", "BACKED_UP")]),
        "journey": analytics_ui.build_journey_table(
            np.array([[0, 3], [2, 0]]), ["Checkout", "__OUTSIDE__"]),
    }
    for name, html in panels.items():
        bad = _has_banned(html)
        check(f"{name}: no legacy literal colours", not bad, str(bad))
    # The empty state used to be pure-white text on the light page.
    empty = analytics_ui.build_analytics_empty_state(False)
    check("analytics empty state uses themed ink", "var(--gk-ink" in empty)
    check("analytics empty state is not white-on-white",
          "#ffffff" not in empty.lower())


def test_theme_fallbacks():
    section("Every token carries a literal fallback for standalone export")
    for name, val in [("INK", T.INK), ("CARD", T.CARD), ("DANGER", T.DANGER),
                      ("ACCENT", T.ACCENT), ("GOOD", T.GOOD)]:
        ok = val.startswith("var(--gk-") and "," in val and val.endswith(")")
        check(f"{name} is var() with a fallback", ok, val)


def test_table_affordances():
    section("Tables ship sort / filter / export / sticky hooks")
    html = ui_builder.build_events_timeline(EVENTS)
    check("has a table id", "id='gk-events-table'" in html)
    check("columns are sortable", "data-sort=" in html)
    check("has a CSV export button", "gk-table-export" in html)
    check("has a filter input", "gk-table-filter" in html)
    check("uses the shared table class", "class='gk-table'" in html)
    check("scroll container present (page never scrolls sideways)",
          "gk-table-scroll" in html)


def test_row_seek():
    section("Rows carry the timestamp needed to seek the video")
    ev = ui_builder.build_events_timeline(EVENTS)
    check("event rows are seekable", "data-gk-seek=" in ev)
    check("seek value matches the event timestamp",
          "data-gk-seek='4.000'" in ev, "PUSHOUT at 4.0s")
    check("seekable rows are keyboard reachable", "tabindex='0'" in ev)

    pops = ui_builder.build_pops_summary(MAX_POPS, SNAPS)
    check("POPS rows seek to the peak", "data-gk-seek=" in pops)

    ops = ui_builder.build_operational_alerts(
        [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 3, "Main Door",
                  12.5, 60.0)])
    check("ops rows seek to the incident start",
          "data-gk-seek='12.500'" in ops)

    # A snapshot from a cached run predating the timestamp field must not
    # crash or emit a bogus seek target.
    legacy = {3: {"score": 88, "event": "PUSHOUT ALERT", "fill": "FULL",
                  "bag": "unbagged", "quality": "valid"}}
    old = ui_builder.build_pops_summary({3: 88}, legacy)
    check("legacy snapshot without a timestamp renders",
          "Cart 3" in old and "data-gk-seek" not in old)


def test_pops_sorted_by_risk():
    section("POPS table leads with the highest score, not arrival order")
    html = ui_builder.build_pops_summary(MAX_POPS, SNAPS)
    order = [int(m) for m in re.findall(r"Cart (\d+)</span>", html)]
    check("highest POPS first", order == [3, 5, 7], str(order))


def test_cart_flag_index():
    section("One cart can carry several operational categories")
    findings = [
        _Finding("abandoned_cart", "UNATTENDED CART (OPS)", "ACTION", 3, None,
                 30.0, 90.0),
        _Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 3, "Main Door",
                 40.0, 65.0, evidence={"also_carts": [9]}),
        _Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 3, "Main Door",
                 200.0, 20.0),
        _Finding("incoming_empty", "INCOMING EMPTY CART", "INFO", 5, None,
                 2.0, 4.0),
    ]
    idx = ui_builder.cart_flag_index(findings)
    check("a cart carries every category it fell into", len(idx[3]) == 2,
          str([f["label"] for f in idx[3]]))
    check("worst severity leads", idx[3][0]["severity"] == "SAFETY")
    check("repeats collapse into one chip with a count",
          idx[3][0]["count"] == 2 and idx[3][0]["longest_s"] == 65.0)
    check("earliest occurrence is kept", idx[3][0]["first_t"] == 40.0)
    check("deduped 'also_carts' still get the flag", 9 in idx and idx[9],
          str(sorted(idx)))
    check("no findings means no flags", ui_builder.cart_flag_index([]) == {})
    check("None is tolerated", ui_builder.cart_flag_index(None) == {})

    # Chip stack: the CSV exporter reads textContent, so adjacent chips need a
    # separator that is in the DOM but never painted.
    stack = ui_builder._flag_stack(idx[3])
    check("stack renders one chip per category", stack.count("gk-flag'") == 2,
          f"{stack.count('gk-flag')} chips")
    check("chips carry a separator for CSV export", "gk-sr-only" in stack)
    check("empty stack says n/a, not a blank cell",
          "n/a" in ui_builder._flag_stack([]))
    check("stack sorts by severity then breadth",
          ui_builder._flag_sort_value(idx[3])
          > ui_builder._flag_sort_value(idx[5]))


def test_pops_table_carries_categories():
    section("POPS table shows POPS events and operational categories together")
    findings = [
        _Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 7, "Main Door",
                 40.0, 65.0),
        _Finding("abandoned_cart", "UNATTENDED CART (OPS)", "ACTION", 7, None,
                 30.0, 90.0),
        # A cart the rule engine flagged that POPS never scored.
        _Finding("static_cart", "STATIC CART", "WATCH", 42, "Aisle 4",
                 12.0, 55.0),
    ]
    html = ui_builder.build_pops_summary(MAX_POPS, SNAPS, findings)
    check("the categories column exists", "Operational" in html)
    check("both of cart 7's categories render",
          "BLOCKED DOOR" in html and "UNATTENDED CART (OPS)" in html)
    check("category chips are outline, POPS events stay solid",
          "box-shadow:inset" in html and "color:#fff" in html)
    check("a flagged-but-unscored cart still gets a row",
          "Cart 42" in html and "not scored" in html)

    # cart_flag_index() decides the row set, so anything counting POPS rows has
    # to derive from it: a cart folded in via evidence["also_carts"] gets a row
    # while never appearing as a finding's own cart_display_id.
    folded = [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 7, "Main Door",
                       40.0, 65.0, evidence={"also_carts": [99]})]
    rows = ui_builder.build_pops_summary({7: 12}, {7: dict(SNAPS[7])}, folded)
    n_rows = len(re.findall(r"Cart (\d+)</span>", rows))
    check("a folded-in cart is a row AND is counted by the index",
          n_rows == len(set({7: 12}) | set(ui_builder.cart_flag_index(folded))),
          f"{n_rows} rows")
    check("the unscored row is still seekable to its first incident",
          "data-gk-seek='12.000'" in html)
    check("the strip discloses multi-category carts",
          "more than one" in html)
    check("no legacy literal colours", not _has_banned(html))

    # The pre-existing 2-argument call must keep working, and must not grow a
    # column of empty cells when there are no findings at all.
    plain = ui_builder.build_pops_summary(MAX_POPS, SNAPS)
    check("2-arg call still works", "Cart 3" in plain)
    check("no findings means no categories column",
          "Operational" not in plain)
    check("row order is unchanged without findings",
          [int(m) for m in re.findall(r"Cart (\d+)</span>", plain)] == [3, 5, 7])

    # A SAFETY category must not be tinted calm because POPS said MONITORING.
    # Assert on the ROW's style, not on any --gk-danger-bg in the html: the
    # outline chip carries that token too, so the loose check passes vacuously.
    quiet = ui_builder.build_pops_summary({7: 12}, {7: dict(SNAPS[7])})
    flagged = ui_builder.build_pops_summary(
        {7: 12}, {7: dict(SNAPS[7])},
        [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 7, "Main Door",
                  40.0, 65.0)])
    row_tint = "<tr class='gk-tr gk-seekable' style='background:var(--gk-danger-bg"
    check("worst label on the row wins the tint", row_tint in flagged)
    check("an unflagged MONITORING cart stays untinted", row_tint not in quiet)

    # INCOMING EMPTY CART fires on ordinary shopper entry. If an INFO category
    # tinted its row, most of the table would be coloured on a real clip and the
    # tint would stop meaning anything — the same reason _ROW_TONE leaves
    # MONITORING / INBOUND alone.
    info_only = ui_builder.build_pops_summary(
        {7: 12}, {7: dict(SNAPS[7])},
        [_Finding("incoming_empty", "INCOMING EMPTY CART", "INFO", 7, None,
                  1.5, 6.0)])
    check("an INFO-only category does not tint the row",
          "<tr class='gk-tr gk-seekable' style='background:" not in info_only)
    check("but the INFO category still shows as a chip",
          "INCOMING EMPTY CART" in info_only)
    watch_only = ui_builder.build_pops_summary(
        {7: 12}, {7: dict(SNAPS[7])},
        [_Finding("static_cart", "STATIC CART", "WATCH", 7, "Aisle 4",
                  12.0, 58.0)])
    check("WATCH is the tint floor",
          "<tr class='gk-tr gk-seekable' style='background:var(--gk-caution-bg"
          in watch_only)

    # Zone names reach this table for the first time via the chip tooltips.
    evil = '<img src=x onerror=alert(1)>"'
    xss = ui_builder.build_pops_summary(
        {7: 12}, {7: dict(SNAPS[7])},
        [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 7, evil,
                  40.0, 65.0)])
    check("zone names are escaped in chip tooltips", "<img src=x" not in xss)


def test_incoming_cart_without_items_label():
    section("An empty cart moving inward is named, not left as 'INBOUND'")
    LBL = ui_builder.INCOMING_NO_ITEMS

    inbound_empty = {"score": 5, "event": "INBOUND", "fill": "empty",
                     "bag": "not_applicable", "quality": "valid_cart",
                     "direction": "INBOUND", "timestamp": 4.0}
    html = ui_builder.build_pops_summary({5: 5}, {5: inbound_empty})
    check("the Event cell names the category", LBL in html)
    check("the bare direction no longer stands in for it",
          ">INBOUND<" not in html)
    check("it is rendered as an ordinary POPS badge",
          "gk-badge" in html and LBL in html)

    # Only inbound AND empty. Each half alone keeps the original event.
    inbound_full = dict(inbound_empty, fill="full")
    check("inbound but carrying items stays INBOUND",
          ui_builder.resolve_event_label(inbound_full, "INBOUND") == "INBOUND")
    outbound_empty = dict(inbound_empty, direction="OUTBOUND")
    check("empty but leaving is not an incoming cart",
          ui_builder.resolve_event_label(outbound_empty, "LOW PRIORITY")
          == "LOW PRIORITY")

    # The fill label's case comes from the classifier; the display uppercases.
    check("fill matching is case-insensitive",
          ui_builder.resolve_event_label(dict(inbound_empty, fill="EMPTY"),
                                         "INBOUND") == LBL)
    check("a snapshot with no direction (older cache) is left alone",
          ui_builder.resolve_event_label({"fill": "empty"}, "INBOUND")
          == "INBOUND")

    # A traffic category must never overwrite a theft-risk label.
    for risky in ("PUSHOUT ALERT", "HIGH PRIORITY", "MEDIUM PRIORITY",
                  "ABANDONED CART", "UNLINKED EXIT"):
        check(f"{risky} is never masked",
              ui_builder.resolve_event_label(inbound_empty, risky) == risky)

    # Naming is shared with the rule engine — one situation, one name.
    from engine.rules import RULE_LABELS
    check("the rule engine uses the same name",
          RULE_LABELS["incoming_empty"] == LBL, RULE_LABELS["incoming_empty"])
    check("the label has a standalone-report colour",
          LBL in ui_builder._EVENT_BADGE)
    check("the label carries a severity",
          ui_builder._event_severity(LBL) != "NEUTRAL")
    check("it does not tint the whole row", LBL not in ui_builder._ROW_TONE)


def test_category_reference_and_counts():
    section("The category vocabulary is documented where it is read")
    ref = ui_builder.build_category_reference()
    for name in ("BLOCKED DOOR", "UNATTENDED CART (OPS)", "STATIC CART",
                 ui_builder.INCOMING_NO_ITEMS):
        check(f"{name} is documented", name in ref)
    check("the reference is collapsed by default",
          "<details" in ref and "open" not in ref.split(">")[0])
    check("the fill-classifier proxy is disclosed", "Proxy" in ref)
    check("no legacy literal colours", not _has_banned(ref))

    ops = ui_builder.build_operational_alerts([
        _Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 3, "Main Door",
                 40.0, 65.0),
        _Finding("abandoned_cart", "UNATTENDED CART (OPS)", "ACTION", 3, None,
                 30.0, 90.0),
    ])
    check("ops table leads with per-category cart counts",
          "gk-inline-summary" in ops and "cart(s) flagged" in ops)
    check("an incident names the cart's other categories", "also:" in ops)


def test_empty_states_distinct():
    section("Empty states say which kind of empty they mean")
    all_clear = ui_builder.build_events_timeline([])
    no_carts = ui_builder.build_pops_summary({}, {})
    rules_ok = ui_builder.build_operational_alerts([])
    rules_na = ui_builder.build_operational_alerts([], "No door zones drawn.")
    check("no events reads as all clear", "All clear" in all_clear)
    check("no carts does NOT claim all clear",
          "All clear" not in no_carts and "No carts detected" in no_carts)
    check("rules-ran-clean says so", "No operational issues" in rules_ok)
    check("rules-did-not-run says so", "did not run" in rules_na)
    check("the two rule states stay distinct", rules_ok != rules_na)
    check("did-not-run never claims all clear",
          "No operational issues" not in rules_na)

    # The reason is PARTIAL: no door/aisle zones skips blocked-door and
    # static-cart, but unattended-cart and incoming-empty are zone-independent
    # and still run. Returning the notice INSTEAD of the table dropped their
    # findings — the tab badge counted them while the panel said nothing ran.
    partial = ui_builder.build_operational_alerts(
        [_Finding("incoming_empty", "INCOMING EMPTY CART", "INFO", 1, "-",
                  14.0, 6.0)],
        "No door zones drawn.")
    check("findings survive a partial did-not-run",
          "INCOMING EMPTY CART" in partial)
    check("...and are rendered as a real table", "gk-table" in partial)
    check("...with the caveat kept above them", "did not run" in partial)
    check("...and no false all-clear", "No operational issues" not in partial)
    check("panel row count matches what the tab badge counts",
          partial.count("INCOMING EMPTY CART") >= 1)

    # The invariant behind the report: the tab badge is len(rule_findings)
    # (see TrackingEngine._tab_counts), so the panel must render one row per
    # finding whether or not a partial "did not run" reason is present. It is
    # the mismatch, not either value alone, that the user sees as a lie.
    import re as _re
    trio = [_Finding("incoming_empty", "INCOMING EMPTY CART", "INFO", 1, "-", 4.0, 6.0),
            _Finding("abandoned_cart", "UNATTENDED CART", "ACTION", 2, "-", 9.0, 70.0),
            _Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 3, "Main Door", 12.0, 61.0)]
    for reason in (None, "No door zones drawn."):
        html = ui_builder.build_operational_alerts(trio, reason)
        body = html.split("<tbody")[-1] if "<tbody" in html else ""
        rows = len(_re.findall(r"<tr[ >]", body))
        check(f"badge {len(trio)} == rendered rows (reason={'yes' if reason else 'no'})",
              rows == len(trio), f"{rows} rows")


def test_alert_banner_consolidation():
    section("Alert banner caps its chips and links to the right tab")
    findings = [
        _Finding("blocked_door", "BLOCKED DOOR", "SAFETY", i, f"Door {i}",
                 10.0 * i, 60.0)
        for i in range(1, 4)
    ]
    spikes = [_Spike(f"z{i}", f"Zone {i}", "BACKED_UP") for i in range(4)]
    html = ui_builder.build_alert_banner(EVENTS, spikes, [], findings)
    n_chips = html.count("gk-alert-chip-t")
    check("chip count is capped", n_chips <= ui_builder._MAX_BANNER_CHIPS,
          f"{n_chips} chips")
    check("overflow is disclosed, not dropped silently",
          "more" in html.lower())
    check("chips link to a destination tab", "data-gk-tab=" in html)
    check("dismiss uses a class hook, not a stripped inline onclick",
          "gk-alert-dismiss" in html and "onclick=" not in html)
    check("safety drives the critical palette", "#dc2626" in html)

    quiet = ui_builder.build_alert_banner([], [], [], [])
    check("nothing to report renders nothing", quiet == "")

    # The pre-existing 3-argument call must keep working.
    legacy = ui_builder.build_alert_banner(EVENTS, spikes, [])
    check("old 3-arg signature still works", bool(legacy))


def test_alert_banner_headline_matches_evidence():
    section("Banner headline only names findings that actually exist")
    high = [{"frame": 90, "timestamp": 3.0, "cart_id": 4, "event": "HIGH PRIORITY",
             "pops_score": 71, "fill": "FULL", "bag": "unbagged",
             "speed_status": "FAST", "direction": "OUTBOUND", "linked": False}]

    # The reported bug: a lone HIGH PRIORITY cart, zero zone measurements, and
    # the banner announced "congestion forming in one or more zones".
    only_cart = ui_builder.build_alert_banner(high, [], [], [])
    check("a cart-only run never claims congestion",
          "congestion" not in only_cart.lower())
    check("a cart-only run names the cart finding",
          "1 high-priority cart" in only_cart)

    # ...and the mirror image: zones backed up, no cart events at all.
    only_zone = ui_builder.build_alert_banner(
        [], [_Spike("z1", "Checkout", "BACKED_UP")], [], [])
    check("a zone-only run does not claim flagged carts",
          "cart" not in only_zone.lower())
    check("a zone-only run says what backed up", "1 area backed up" in only_zone)

    # QUEUE_FORMING is the one case where the congestion wording is earned.
    queueing = ui_builder.build_alert_banner(
        [], [_Spike("z1", "Checkout", "QUEUE_FORMING")], [], [])
    check("measured queueing still reads as congestion",
          "congestion forming in 1 area" in queueing)

    # An ACTION rule is not necessarily the unattended-cart rule, and the
    # category name must pluralise without dragging "(OPS)" into the sentence.
    ops = ui_builder.build_alert_banner([], [], [], [
        _Finding("abandoned_cart", "UNATTENDED CART (OPS)", "ACTION", 3, None,
                 30.0, 90.0),
        _Finding("abandoned_cart", "UNATTENDED CART (OPS)", "ACTION", 8, None,
                 60.0, 75.0),
    ])
    check("rule headline is derived from the label",
          "2 unattended carts flagged" in ops, ops[ops.find("headline"):][:120])

    # Mixed run: both nouns present, both true.
    mixed = ui_builder.build_alert_banner(
        high, [_Spike("z1", "Checkout", "QUEUE_FORMING")], [], [])
    check("a mixed run names both", "high-priority cart" in mixed
          and "congestion forming" in mixed)
    check("safety wording is unchanged (case-report/VLM read it)",
          "Safety issue" in ui_builder.build_alert_banner(
              [], [], [], [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY",
                                    1, "Main Door", 5.0, 60.0)]))


def test_error_banner():
    section("Run failures surface at the top of the page")
    html = ui_builder.build_error_banner("Analysis failed", "RuntimeError: boom")
    check("names the failure", "Analysis failed" in html)
    check("carries the detail", "RuntimeError: boom" in html)
    check("is an aria-live alert", 'role="alert"' in html or "role='alert'" in html)
    check("is dismissible", "gk-alert-dismiss" in html)
    long_detail = "x" * 5000
    capped = ui_builder.build_error_banner("boom", long_detail)
    check("detail is length-capped", len(capped) < 2000, f"{len(capped)} chars")


def test_escaping():
    section("User-supplied text cannot break out of the markup")
    evil = '<img src=x onerror=alert(1)>"'
    zone = Zone(zone_id="z1", name=evil,
                polygon=np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32),
                applies_to="both", kind="door", color=(0, 200, 255))
    html = ui_builder.build_operational_alerts(
        [_Finding("blocked_door", "BLOCKED DOOR", "SAFETY", 1, evil, 1.0, 60.0)])
    check("zone name is escaped in the ops table", "<img src=x" not in html)
    info = ui_builder.build_video_info(f"/tmp/{evil}.mp4", 1, 1, 1, 1, 1)
    check("video name is escaped", "<img src=x" not in info)
    counts = ui_builder.build_tab_counts({"Events": {"n": 3, "tone": "DANGER"}})
    check("tab counts are attribute-safe",
          "&quot;" in counts or "'" not in counts.split("data-counts=")[1][:60])


def test_run_summary():
    section("Run summary surfaces the timings the pipeline already computes")
    html = ui_builder.build_run_summary(
        frames=1842, wall_s=63.0, encode_s=7.0, device="cuda",
        video_duration_s=61.4, n_people=12, n_carts=9, n_links=7,
        timings=[("YOLO", 41.0), ("classify", 12.0), ("pose", 0.0)])
    check("frame count present", "1,842" in html)
    check("wall time present", "63.0s" in html)
    check("realtime factor present", "×" in html)
    check("device present", "CUDA" in html)
    check("stage timings present", "YOLO 41.0s" in html)
    check("zero-cost stages are omitted", "pose" not in html)


def test_tab_counts():
    section("Tab badge payload")
    html = ui_builder.build_tab_counts(
        {"Events": {"n": 3, "tone": "DANGER"}, "POPS": {"n": 0, "tone": "INFO"}})
    check("has the host id", "id='gk-tab-counts'" in html)
    check("keeps non-zero counts", "Events" in html)
    check("drops zero counts", "POPS" not in html)
    check("is hidden from layout and AT", "hidden" in html and "aria-hidden" in html)


def test_table_row_cap():
    section("Large tables are capped, and say so")
    many = [dict(EVENTS[0], cart_id=i, timestamp=float(i)) for i in range(400)]
    html = ui_builder.build_events_timeline(many)
    check("renders a show-all control", "gk-table-showall" in html)
    check("discloses the visible/total split", "Showing" in html)
    hidden = html.count("gk-row-hidden")
    check("rows past the window start hidden", hidden > 0, f"{hidden} hidden")

    huge = [dict(EVENTS[0], cart_id=i, timestamp=float(i))
            for i in range(T.HARD_ROW_CAP + 50)]
    big = ui_builder.build_events_timeline(huge)
    check("hard cap is disclosed, never silent", "Capped at" in big)


def test_analytics_panels_render():
    section("Analytics panels render with real-shaped inputs")
    zones = [Zone(zone_id="z1", name="Checkout",
                  polygon=np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32),
                  applies_to="person", kind="analytics", color=(0, 200, 255))]
    summary = [{"zone_id": "z1", "zone_name": "Checkout", "applies_to": "person",
                "n_visits": 7, "n_unique": 5, "avg_dwell_s": 42.0, "p50_s": 30.0,
                "p95_s": 88.0, "max_s": 91.0, "total_s": 294.0}]
    rows = [_DwellRow(1, "person", "Checkout", 1, 5.0, 47.0),
            _DwellRow(2, "cart", "Checkout", 1, 9.0, 30.0)]
    dwell = analytics_ui.build_dwell_table(zones, summary, rows)
    check("dwell summary renders", "Per-zone dwell summary" in dwell)
    check("dwell visits are seekable", "data-gk-seek=" in dwell)
    check("dwell tables export", dwell.count("gk-table-export") >= 2)

    nozone = analytics_ui.build_dwell_table([], summary, rows)
    check("no zones gives guidance, not a blank", "Zone Editor" in nozone)

    res = _Result(dwell_rows=rows, spikes=[_Spike("z1", "Checkout", "BACKED_UP")],
                  insight="Checkout is the busiest zone.")
    head = analytics_ui.build_analytics_summary(zones, res)
    check("summary chips render", "gk-stat-chip" in head)
    check("insight renders", "Checkout is the busiest zone." in head)

    quiet = analytics_ui.build_queue_spikes_banner([])
    check("no spikes is a compact strip, not a full banner",
          "gk-inline-summary" in quiet)

    j = analytics_ui.build_journey_table(
        np.array([[0, 3], [2, 0]]), ["Checkout", "__OUTSIDE__"])
    check("journey renders", "Zone-to-zone transitions" in j)
    check("__OUTSIDE__ is humanised", "__OUTSIDE__" not in j)
    empty_j = analytics_ui.build_journey_table(None, [])
    check("journey without zones explains itself", "at least one zone" in empty_j)


def _mk_zones():
    """Two zones as the editor actually creates them, so the colour and
    applies_to defaults under test are the real ones."""
    tri = [(0, 0), (10, 0), (10, 10)]
    return [zone_editor.make_zone("Checkout", tri, "person", 0),
            zone_editor.make_zone("Aisle 3", tri, "person", 1)]


def test_zone_editing():
    section("Zone editing — rename / retype / applies-to")
    zones = _mk_zones()
    zid = zones[0].zone_id

    renamed, ch = zone_editor.rename_zone(zones, zid, "  Front door  ")
    check("rename trims and applies", renamed[0].name == "Front door" and ch)
    check("rename does not mutate the input list", zones[0].name == "Checkout")
    check("rename keeps list length", len(renamed) == 2)
    check("rename leaves other zones alone", renamed[1].name == "Aisle 3")

    _, ch = zone_editor.rename_zone(zones, zid, "Checkout")
    check("renaming to the same name is a no-op", not ch)
    blank, ch = zone_editor.rename_zone(zones, zid, "   ")
    check("blank rename is rejected", blank[0].name == "Checkout" and not ch)
    _, ch = zone_editor.rename_zone(zones, "nope", "X")
    check("rename of an unknown id is a no-op", not ch)

    # The two silent failures. A door left on applies_to="person" matches zero
    # cart tracks, so blocked-door and static-cart report nothing at all; a
    # door left in the analytics palette draws in the wrong colour.
    door, ch = zone_editor.retype_zone(zones, zid, "door")
    check("retype changes the kind", door[0].kind == "door" and ch)
    check("retype to a layout kind forces applies_to='both'",
          door[0].applies_to == "both", door[0].applies_to)
    check("retype recomputes the colour",
          door[0].color == zone_editor.zone_color_for("door", 0),
          str(door[0].color))
    check("retype leaves the polygon untouched",
          np.array_equal(door[0].polygon, zones[0].polygon))
    check("retype leaves the id stable", door[0].zone_id == zid)
    check("retype does not touch sibling zones",
          door[1].kind == "analytics" and door[1].color == zones[1].color)

    back, _ = zone_editor.retype_zone(door, zid, "analytics")
    check("retyping back restores an analytics colour",
          back[0].color == zone_editor.zone_color_for("analytics", 0),
          str(back[0].color))
    check("applies_to is not silently reverted with it",
          back[0].applies_to == "both")
    _, ch = zone_editor.retype_zone(door, zid, "door")
    check("retype to the current kind is a no-op (dropdown double-fires)",
          not ch)

    both, ch = zone_editor.set_zone_applies_to(zones, zid, "both")
    check("applies_to is editable on analytics zones",
          both[0].applies_to == "both" and ch)
    locked, ch = zone_editor.set_zone_applies_to(door, zid, "person")
    check("applies_to cannot be forced back to 'person' on a layout zone",
          locked[0].applies_to == "both" and not ch)
    check("coerce_applies_to covers every layout kind",
          all(zone_editor.coerce_applies_to(k, "person") == "both"
              for k in zone_editor.LAYOUT_KINDS))
    check("coerce_applies_to leaves analytics zones alone",
          zone_editor.coerce_applies_to("analytics", "cart") == "cart")


def test_zone_removal():
    section("Zone removal")
    zones = _mk_zones()
    zid = zones[0].zone_id
    left, removed = zone_editor.remove_zone(zones, zid)
    check("remove drops exactly one zone", len(left) == 1)
    check("remove drops the right one", left[0].name == "Aisle 3")
    check("remove hands back the removed zone", removed is not None
          and removed.zone_id == zid)
    check("remove does not mutate the input list", len(zones) == 2)
    check("survivors keep their colour", left[0].color == zones[1].color,
          str(left[0].color))

    same, removed = zone_editor.remove_zone(zones, "nope")
    check("removing an unknown id is a no-op",
          len(same) == 2 and removed is None)

    empty, removed = zone_editor.remove_zone([], "z1")
    check("removing from an empty list is safe",
          empty == [] and removed is None)

    check("find_zone locates by id", zone_editor.find_zone(zones, zid) == 0)
    check("find_zone reports -1 for a stale id",
          zone_editor.find_zone(zones, "gone") == -1)


def test_pose_toggle_is_recorded():
    section("Config panel records whether pose ran")
    args = (6, 12, "Outside (facing entrance)", "weights/q.pt", "weights/f.pt", 0.5)
    on  = ui_builder.build_config_info(*args, enable_pose=True)
    off = ui_builder.build_config_info(*args, enable_pose=False)
    check("pose row present when on", "Pose Estimation" in on)
    check("pose row present when off", "Pose Estimation" in off)
    check("on and off read differently", on != off)
    check("off says the overlay is absent", "no skeleton overlay" in off)
    # With pose off the annotated video simply has no skeletons. Unless the
    # run records the choice, that absence reads as a detection failure.
    check("off does not claim pose is on",
          "On - skeleton overlay" not in off)
    # Any caller that has not been updated must keep the previous behaviour.
    check("defaults to on for callers that omit it",
          "On - skeleton overlay" in ui_builder.build_config_info(*args))


def main():
    print("=" * 62)
    print("UI BUILDER TESTS")
    print("=" * 62)
    test_theme_fallbacks()
    test_no_hardcoded_palette()
    test_table_affordances()
    test_row_seek()
    test_pops_sorted_by_risk()
    test_cart_flag_index()
    test_pops_table_carries_categories()
    test_incoming_cart_without_items_label()
    test_category_reference_and_counts()
    test_empty_states_distinct()
    test_alert_banner_consolidation()
    test_alert_banner_headline_matches_evidence()
    test_error_banner()
    test_escaping()
    test_run_summary()
    test_tab_counts()
    test_table_row_cap()
    test_analytics_panels_render()
    test_zone_editing()
    test_zone_removal()
    test_pose_toggle_is_recorded()

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
