# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Case-report builder tests — pure HTML generation, no GPU and no video decode.

Run with:  python tests/test_case_report.py

The Operations Highlights section carries the same trap the operational alerts
tab does: "no issues found" and "the rules never ran" are DIFFERENT answers,
and rendering the second as the first is a clean bill of health for a clip
nothing was ever checked on. Those three states are pinned below, because none
of them raises when it is wrong — the report just reads confidently and lies.

Deliberately stdlib only, no pytest — matches tests/test_vlm_parse.py.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.analytics_models import AnalyticsResult, QueueSpike, RuleFinding
from engine.case_report_builder import build_case_report_html
from engine.vlm_analyzer import CaseReportData, FrameAnalysis

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


VIDEO_INFO = {"video_name": "clip.mp4", "width": 1280, "height": 720,
              "fps": 19.0, "total_frames": 420}
POPS_DATA = {"pops_summary": {"C1": {"max_score": 75, "peak_event": "OUTBOUND"}}}


def finding(**kw):
    base = dict(
        rule_id="unattended_cart", label="UNATTENDED CART (OPS)",
        severity="ACTION", cart_display_id=1, zone_id="z1", zone_name="Aisle 3",
        start_t=12.0, end_t=52.0, duration_s=40.0, threshold_s=30.0,
        reasons=["no linked person for 40s"],
    )
    base.update(kw)
    return RuleFinding(**base)


def spike(**kw):
    base = dict(zone_id="z2", zone_name="Checkout 1", avg_dwell_s=44.0,
                max_dwell_s=90.0, n_visits=12, threshold_s=30.0,
                severity="BACKED_UP", peak_occupancy=7,
                reasons=["occupancy above threshold"])
    base.update(kw)
    return QueueSpike(**base)


def build(analytics_result, report=None, event_log=None):
    gradio_html, standalone = build_case_report_html(
        report or CaseReportData(risk_level="HIGH", executive_summary="x"),
        [], POPS_DATA, event_log or [], {}, VIDEO_INFO,
        analytics_result=analytics_result,
    )
    return gradio_html, standalone


#: One timeline event, in the shape _build_timeline reads off the event log.
EVENT = {"timestamp": 14.2, "cart_id": 1, "event": "OUTBOUND", "pops_score": 75,
         "fill": "full", "bag": "no_bag", "direction": "OUTBOUND",
         "speed_status": "normal"}


# ---------------------------------------------------------------------------
section("the three states — findings / did-not-run / all-clear")

html_found, _ = build(AnalyticsResult(rule_findings=[finding()]))
check("section renders", "Operations Highlights" in html_found)
check("finding label shown", "UNATTENDED CART (OPS)" in html_found)
check("severity shown", "ACTION" in html_found)
check("zone shown", "Aisle 3" in html_found)
check("reason shown", "no linked person for 40s" in html_found)
check("does not claim all-clear",
      "No operational issues detected" not in html_found)

html_norun, _ = build(AnalyticsResult(
    rule_findings=[], rules_unavailable_reason="no monitored zones were drawn"))
check("did-not-run state renders", "did not run" in html_norun.lower())
check("did-not-run gives the reason",
      "no monitored zones were drawn" in html_norun)
check("did-not-run NEVER reads as all-clear",
      "No operational issues detected" not in html_norun,
      "<- a clean bill of health for an unchecked clip")

# Zone-independent rules (abandoned cart, incoming empty) can produce findings
# even while blocked-door/static-cart report "did not run" for lack of
# door/aisle zones. "did not run" must still win — promoting the
# zone-independent findings to a status they did not earn alone would silently
# report on rules that never actually ran over this clip's monitored zones.
html_norun_with_findings, _ = build(AnalyticsResult(
    rule_findings=[finding()],
    rules_unavailable_reason="no door/aisle zones drawn"))
check("did-not-run wins even when zone-independent rules found something",
      "did not run" in html_norun_with_findings.lower())
check("findings are not shown when rules did not run",
      "UNATTENDED CART (OPS)" not in html_norun_with_findings)

html_clear, _ = build(AnalyticsResult(rule_findings=[]))
check("all-clear state renders",
      "No operational issues detected" in html_clear)
check("all-clear is not the did-not-run state",
      "did not run" not in html_clear.lower())


# ---------------------------------------------------------------------------
section("highlights are curated, not the full ops table")

many = ([finding(severity="SAFETY", label="BLOCKED DOOR (OPS)", cart_display_id=2)]
        + [finding(severity="INFO", label=f"INFO FINDING {i}", cart_display_id=10 + i)
           for i in range(4)])
html_many, _ = build(AnalyticsResult(rule_findings=many))
check("severe finding shown", "BLOCKED DOOR (OPS)" in html_many)
check("low-severity findings rolled up, not listed",
      "INFO FINDING 0" not in html_many)
check("rollup states the count", "+4 lower-severity findings" in html_many)

only_info = [finding(severity="INFO", label=f"INFO FINDING {i}") for i in range(5)]
html_info, _ = build(AnalyticsResult(rule_findings=only_info))
check("with nothing severe, still shows evidence rows",
      "INFO FINDING 0" in html_info)
check("and rolls up the rest", "+2 lower-severity findings" in html_info)

sev_order = [finding(severity="ACTION", label="SECOND"),
             finding(severity="SAFETY", label="FIRST")]
html_ord, _ = build(AnalyticsResult(rule_findings=sev_order))
check("SAFETY sorts above ACTION",
      html_ord.index("FIRST") < html_ord.index("SECOND"))


# ---------------------------------------------------------------------------
section("precision markers are not dropped")

html_ong, _ = build(AnalyticsResult(
    rule_findings=[finding(ongoing_at_eov=True)]))
check("ongoing-at-end-of-video marked", "ongoing" in html_ong)

html_deg, _ = build(AnalyticsResult(
    rule_findings=[finding(confidence="degraded")]))
check("degraded confidence marked", "degraded" in html_deg)


# ---------------------------------------------------------------------------
section("congestion is independent of the rule engine")

# A clip too short for any rule threshold: rules unavailable, spikes still real.
html_ind, _ = build(AnalyticsResult(
    rules_unavailable_reason="video shorter than every rule threshold",
    queue_spikes=[spike()],
    dwell_summary=[{"zone_name": "Checkout 1", "avg_dwell_s": 44.0,
                    "p95_s": 88.0, "n_visits": 12}]))
check("spikes survive an unavailable rule engine", "BACKED UP" in html_ind)
check("dwell survives an unavailable rule engine", "Checkout 1" in html_ind)

html_nospike, _ = build(AnalyticsResult(rule_findings=[finding()]))
check("no congestion data gets its own empty state",
      "No congestion measured" in html_nospike)

html_dwell, _ = build(AnalyticsResult(dwell_summary=[
    {"zone_name": f"Zone {i}", "avg_dwell_s": float(i), "p95_s": 1.0,
     "n_visits": 3} for i in range(6)]))
check("dwell capped at top 3",
      all(f"Zone {i}" in html_dwell for i in (5, 4, 3))
      and not any(f"Zone {i}" in html_dwell for i in (2, 1, 0)),
      f'present: {[i for i in range(6) if f"Zone {i}" in html_dwell]}')
check("dwell ranked by avg descending",
      html_dwell.index("Zone 5") < html_dwell.index("Zone 4"))
check("zero-visit zones excluded",
      "Zone 9" not in build(AnalyticsResult(dwell_summary=[
          {"zone_name": "Zone 9", "avg_dwell_s": 99.0, "p95_s": 1.0,
           "n_visits": 0}]))[0])


# ---------------------------------------------------------------------------
section("safety and plumbing")

html_esc, _ = build(AnalyticsResult(
    rule_findings=[finding(zone_name='<script>alert(1)</script>')]))
check("user-typed zone name is escaped",
      "<script>" not in html_esc and "&lt;script&gt;" in html_esc)

html_none, _ = build(None)
check("absent analytics omits the section entirely",
      "Operations Highlights" not in html_none,
      "<- absent beats a fabricated all-clear")

_, standalone = build(AnalyticsResult(rule_findings=[finding()]))
check("section is in the downloadable report too",
      "Operations Highlights" in standalone and "UNATTENDED CART (OPS)" in standalone)

html_ins, _ = build(AnalyticsResult(
    rule_findings=[], insight_text="Checkout 1 saw the longest dwell."))
check("auto-insight surfaced",
      "Checkout 1 saw the longest dwell." in html_ins)

# Ordering: evidence before recommendations.
html_full, _ = build(AnalyticsResult(rule_findings=[finding()]))
check("ops sits after POPS analysis, before actionable insights",
      html_full.index("POPS Analysis") < html_full.index("Operations Highlights")
      < html_full.index("Actionable Insights"))


# ---------------------------------------------------------------------------
section("every text element carries its own colour")

# The Gradio tab renders this report inside the app's own stylesheet, whose
# dark-theme rules colour bare <li>, <strong>, <em>, <p>, <th> and <td>
# DIRECTLY. A tag that only inherits its colour from an ancestor loses to those
# rules and comes out near-white on this report's white and cream panels — the
# coverage notes under "Operations Highlights" were invisible for exactly that
# reason. Inline colours beat any non-!important rule, so this pins the shape
# rather than the palette: no emitted text tag may go out without its own.
#
# Deliberately BROADER than the defect: a <td> whose whole content is a
# fully-coloured <span> can never render invisibly, and two of those were given
# a colour to satisfy this rule rather than because they were broken. Holding
# "every text tag carries its own colour" is cheaper than parsing each tag's
# children to decide whether its colour matters.
_COLOURLESS = re.compile(r"<(?:li|strong|em|p|th|td)\b(?![^>]*\bcolor:)[^>]*>",
                         re.I)

# A POPULATED timeline and suspect profile: with empty inputs those sections
# render one-line placeholders and prove nothing. The timeline sits directly
# above Operations Highlights and emits the same shape of markup, so it is the
# likeliest place for this defect to recur.
REPORT_COLOUR = CaseReportData(
    risk_level="HIGH", executive_summary="Cart 1 left through the entrance.",
    risk_assessment="Push-out consistent with a grab-and-run.",
    suspect_profile="Male, dark jacket.",
    actionable_insights_lp="- Pull the till log for 14:02.",
    actionable_insights_leo="- Report as retail theft.",
    frame_analyses=[FrameAnalysis(frame_idx=270, timestamp=14.2,
                                  trigger="POPS_PEAK",
                                  description="Cart pushed past the lane.")])

html_colour, standalone_colour = build(
    AnalyticsResult(
        rule_findings=[finding()],
        rules_unavailable_reason="No door zones drawn.",
        rule_diagnostics=["Timing derived from frame rate.",
                          "4 intervals discarded as too sparse."],
        queue_spikes=[spike()],
        dwell_summary=[{"zone_name": "Checkout 1", "avg_dwell_s": 42.0,
                        "p95_s": 61.0, "n_visits": 12}],
        insight_text="Checkout 1 saw the longest dwell."),
    report=REPORT_COLOUR, event_log=[EVENT])

for _label, _doc in (("gradio tab", html_colour),
                     ("download", standalone_colour)):
    _offenders = _COLOURLESS.findall(_doc)
    check(f"no colourless text tag in the {_label} report",
          not _offenders, f"offenders: {_offenders[:3]}")

check("coverage notes are rendered, so the check above covered them",
      "Coverage notes for this run" in html_colour
      and "4 intervals discarded as too sparse." in html_colour)
check("the populated timeline is in that report too",
      "Incident Timeline" in html_colour
      and "Cart pushed past the lane." in html_colour
      and "No events recorded." not in html_colour)


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    print("FAILED: " + ", ".join(_FAIL))
sys.exit(1 if _FAIL else 0)
