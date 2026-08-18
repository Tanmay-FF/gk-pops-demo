# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Pure "what counts as a highlight" selection logic, shared between the
case-report HTML's Operations Highlights section and the tracking JSON's
`operational_highlights` key, so the two artifacts can never disagree on what
"severe" or "top" means.

Stdlib only, duck-typed via getattr() — same convention engine.ui_builder
uses for RuleFinding/QueueSpike consumers — so this module needs no import
from analytics_models/rules/ui_builder and can be imported from anywhere in
the engine package with zero cycle risk.
"""
from __future__ import annotations

#: Most urgent first. Anything unrecognised sorts last.
OPS_SEVERITY_ORDER = {"SAFETY": 0, "ACTION": 1, "WATCH": 2, "INFO": 3}
#: The severities that earn a row of their own in the highlights.
OPS_SEVERE = ("SAFETY", "ACTION")
#: Congestion severities worth reporting.
SPIKE_SEVERE = ("QUEUE_FORMING", "BACKED_UP")

_TOP_N_WHEN_NONE_SEVERE = 3
_TOP_N_DWELL = 3


def order_findings(rule_findings) -> list:
    """Findings sorted worst-severity-first, then earliest start_t."""
    return sorted(
        rule_findings or [],
        key=lambda f: (OPS_SEVERITY_ORDER.get(getattr(f, "severity", "INFO"), 9),
                       getattr(f, "start_t", 0.0)),
    )


def select_top_findings(rule_findings) -> tuple[list, int]:
    """(shown, remainder): every SAFETY/ACTION finding, or (if none) the top
    3 by severity/start_t.

    Callers that need the three-state distinction (rules never ran / ran
    clean / ran with findings) must go through ops_findings_state() instead —
    this function assumes `rule_findings` is already known to be non-empty
    and the rules are known to have run.
    """
    ordered = order_findings(rule_findings)
    severe = [f for f in ordered if getattr(f, "severity", "") in OPS_SEVERE]
    shown = severe if severe else ordered[:_TOP_N_WHEN_NONE_SEVERE]
    return shown, len(ordered) - len(shown)


def ops_findings_state(rules_unavailable_reason, rule_findings) -> tuple[str, list, int]:
    """The exact three-state decision the case report's Operations Highlights
    section renders. Returns (state, shown, remainder).

    state is one of "unavailable" / "clean" / "findings". shown/remainder are
    ([], 0) for the first two states.

    Precedence matters: `rules_unavailable_reason` is checked FIRST, even
    when `rule_findings` is non-empty. The abandoned-cart and incoming-empty
    rules are zone-independent and can produce findings while the
    blocked-door / static-cart rules report "did not run" for lack of
    door/aisle zones — that combination must still read as "did not run",
    not silently promote the zone-independent findings to a status they did
    not earn alone.
    """
    if rules_unavailable_reason:
        return "unavailable", [], 0
    findings = list(rule_findings or [])
    if not findings:
        return "clean", [], 0
    shown, remainder = select_top_findings(findings)
    return "findings", shown, remainder


def ops_diagnostics(rule_diagnostics) -> list[str]:
    """Non-suppressing rule-engine notes, de-duplicated, order preserved.

    Separate from ops_findings_state() by design: that function answers "did
    the rules run and what did they find", and returns ([], 0) for the
    "unavailable" state. These notes must survive all three of its states -
    "the clock was synthesised" and "4 intervals were too sparse to assert"
    are exactly the things a reader of a CLEAN report needs to see, and
    folding them into rules_unavailable_reason would blank every finding.
    """
    seen: set[str] = set()
    out: list[str] = []
    for note in (rule_diagnostics or []):
        text = str(note).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def select_congestion(queue_spikes, dwell_summary) -> tuple[list, list]:
    """(severe_spikes, top_dwell_zones).

    Deliberately independent of the rule engine: `rules_unavailable_reason`
    says nothing about whether congestion was measured, so this must never be
    gated behind it.
    """
    spikes = [s for s in (queue_spikes or [])
              if getattr(s, "severity", "NORMAL") in SPIKE_SEVERE]
    dwell = [d for d in (dwell_summary or []) if d.get("n_visits", 0) > 0]
    dwell = sorted(dwell, key=lambda d: d.get("avg_dwell_s", 0.0),
                   reverse=True)[:_TOP_N_DWELL]
    return spikes, dwell


def finding_to_dict(f) -> dict:
    """JSON-safe projection of a RuleFinding (or duck-typed stand-in).

    Single source of truth for wire field names (cart_id, not
    cart_display_id; ongoing_at_end_of_video, not ongoing_at_eov) — used for
    both the full rule_findings list and operational_highlights.top_findings,
    so the two can never drift apart in shape.
    """
    return {
        "rule_id": getattr(f, "rule_id", None),
        "label": getattr(f, "label", None),
        "severity": getattr(f, "severity", None),
        "cart_id": getattr(f, "cart_display_id", None),
        "zone_id": getattr(f, "zone_id", None),
        "zone_name": getattr(f, "zone_name", None),
        "start_t": getattr(f, "start_t", None),
        "end_t": getattr(f, "end_t", None),
        "duration_s": getattr(f, "duration_s", None),
        "threshold_s": getattr(f, "threshold_s", None),
        "ongoing_at_end_of_video": getattr(f, "ongoing_at_eov", False),
        "confidence": getattr(f, "confidence", "high"),
        "n_samples": getattr(f, "n_samples", 0),
        "reasons": list(getattr(f, "reasons", None) or []),
        "evidence": dict(getattr(f, "evidence", None) or {}),
    }


def spike_to_dict(s) -> dict:
    """JSON-safe projection of a QueueSpike.

    getattr-based rather than dataclasses.asdict() so duck-typed test
    stand-ins (plain classes, not dataclasses) serialize too.
    """
    return {
        "zone_id": getattr(s, "zone_id", None),
        "zone_name": getattr(s, "zone_name", None),
        "avg_dwell_s": getattr(s, "avg_dwell_s", 0.0),
        "max_dwell_s": getattr(s, "max_dwell_s", 0.0),
        "n_visits": getattr(s, "n_visits", 0),
        "threshold_s": getattr(s, "threshold_s", 0.0),
        "severity": getattr(s, "severity", "NORMAL"),
        "score": getattr(s, "score", 0.0),
        "peak_occupancy": getattr(s, "peak_occupancy", 0),
        "mean_occupancy": getattr(s, "mean_occupancy", 0.0),
        "static_fraction": getattr(s, "static_fraction", 0.0),
        "avg_speed_px_s": getattr(s, "avg_speed_px_s", 0.0),
        "reasons": list(getattr(s, "reasons", None) or []),
    }
