# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Case report HTML builder — generates professional LP / law-enforcement
incident reports from VLM analysis + POPS data.
"""
import base64
import html
import uuid
from datetime import datetime

from . import highlights
from .ui_builder import (
    _FONT, _EVENT_BADGE, _badge, _mmss, resolve_event_label,
)


# ---------------------------------------------------------------------------
# Risk-level styling
# ---------------------------------------------------------------------------
_RISK_COLORS = {
    "HIGH": ("#b71c1c", "#fce4ec"),
    "MEDIUM": ("#e65100", "#fff8e1"),
    "LOW": ("#2e7d32", "#e8f5e9"),
    "UNKNOWN": ("#546e7a", "#eceff1"),
}

#: (foreground, background) per operational rule severity — same palette shape
#: as _RISK_COLORS so the two panels read as one document.
_OPS_SEVERITY_COLORS = {
    "SAFETY": ("#b71c1c", "#fce4ec"),
    "ACTION": ("#e65100", "#fff8e1"),
    "WATCH":  ("#a16207", "#fefce8"),
    "INFO":   ("#546e7a", "#eceff1"),
}
#: Severity vocab + selection thresholds live in engine.highlights now — it is
#: the single source of truth shared with the tracking JSON's
#: operational_highlights key, so this report and that export can never
#: disagree on what counts as "severe" or "top".


def _esc(value) -> str:
    """Escape a value for HTML text content.

    Zone names come from the zone editor, i.e. they are user-typed, and this
    document is written to disk and opened in a browser. The older sections of
    this file interpolate engine-generated strings and escape nothing; anything
    added here goes through this.
    """
    return html.escape("" if value is None else str(value), quote=False)


def _section(title, body, icon=""):
    return (
        f'<div style="margin-bottom:18px;">'
        f'<h3 style="font-family:{_FONT};color:#1e3a5f;background:#eff6ff;'
        f'border-left:4px solid #2563eb;border-bottom:1px solid #bfdbfe;'
        f'padding:8px 12px;margin:0 0 10px;font-size:1rem;font-weight:800;'
        f'letter-spacing:0.02em;border-radius:4px 4px 0 0;">'
        f'<span style="color:#2563eb;margin-right:6px;">{icon}</span>{title}</h3>'
        f'{body}</div>'
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(video_info):
    incident_id = str(uuid.uuid4())[:8].upper()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vid_name = video_info.get("video_name", "Unknown")
    w = video_info.get("width", "?")
    h = video_info.get("height", "?")
    fps = video_info.get("fps", "?")
    total = video_info.get("total_frames", "?")

    return (
        f'<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);padding:14px 20px;'
        f'border-radius:8px;margin-bottom:14px;color:#fff;font-family:{_FONT};">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'<div style="font-size:0.65rem;letter-spacing:2px;color:#93c5fd;font-weight:700;">GATEKEEPER AI</div>'
        f'<h2 style="margin:2px 0 0;font-size:1.1rem;">POPS Incident Case Report</h2>'
        f'</div>'
        f'<div style="text-align:right;font-size:0.75rem;color:#bfdbfe;">'
        f'<div>#{incident_id}</div>'
        f'<div>{ts}</div>'
        f'</div></div>'
        f'<div style="margin-top:6px;font-size:0.75rem;color:#bfdbfe;">'
        f'{vid_name} &bull; {w}x{h} @ {fps} fps &bull; {total} frames'
        f'</div></div>'
    )


def _build_executive_summary(report_data):
    text = (report_data.executive_summary
            or '<em style="color:#64748b;">VLM analysis not available.</em>')
    return _section("Executive Summary", f'<p style="font-family:{_FONT};line-height:1.6;font-size:0.9rem;color:#1f2937;">{text}</p>', "&#128196;")


def _build_risk_assessment(report_data):
    level = report_data.risk_level or "UNKNOWN"
    fg, bg = _RISK_COLORS.get(level, _RISK_COLORS["UNKNOWN"])
    conf = report_data.confidence
    conf_pct = f"{conf * 100:.0f}%" if conf else "N/A"

    badge = (
        f'<span style="background:{fg};color:#fff;padding:4px 14px;border-radius:4px;'
        f'font-size:0.9rem;font-weight:800;font-family:{_FONT};letter-spacing:1px;">'
        f'{level}</span>'
    )
    body = (
        f'<div style="background:{bg};padding:12px 14px;border-radius:6px;'
        f'border-left:4px solid {fg};font-family:{_FONT};color:#1f2937;">'
        f'<div style="margin-bottom:8px;">{badge}'
        f'<span style="margin-left:10px;color:#475569;font-size:0.8rem;font-weight:600;">Confidence: {conf_pct}</span></div>'
        f'<p style="line-height:1.6;font-size:0.9rem;color:#1f2937;">{report_data.risk_assessment or "No assessment available."}</p>'
        f'</div>'
    )
    return _section("Risk Assessment", body, "&#9888;&#65039;")


def _build_suspect_profile(report_data):
    profile = report_data.suspect_profile
    if not profile:
        return ""

    # Group attributes into logical buckets so the panel reads at a glance.
    GROUPS = [
        ("Identity",     ["estimated age", "estimated gender", "build", "height",
                          "build & height"]),
        ("Hair & face",  ["hair", "facial hair"]),
        ("Clothing",     ["top", "bottom", "footwear", "outerwear", "headwear",
                          "clothing"]),
        ("Carry & gear", ["bags", "bags & carry", "accessories"]),
        ("Distinguishing", ["distinguishing", "marks", "tattoos"]),
        ("Behaviour",    ["cart", "cart at peak", "cart at peak event",
                          "behaviour", "behavior", "behaviour cues"]),
    ]

    parsed: list[tuple[str, str]] = []  # (label, value)
    free_text: list[str] = []
    for line in profile.strip().split("\n"):
        line = line.strip().lstrip("- ").lstrip("* ").strip()
        if not line:
            continue
        if ":" in line:
            label, _, value = line.partition(":")
            parsed.append((label.strip(), value.strip()))
        else:
            free_text.append(line)

    def _bucket_for(label: str):
        ll = label.lower()
        for name, keys in GROUPS:
            for k in keys:
                if k in ll:
                    return name
        return "Other"

    buckets: dict[str, list[tuple[str, str]]] = {}
    for label, value in parsed:
        buckets.setdefault(_bucket_for(label), []).append((label, value))

    # Render each non-empty bucket as a card.  Order matches GROUPS.
    bucket_order = [g[0] for g in GROUPS] + ["Other"]
    cards = ""
    for bname in bucket_order:
        items = buckets.get(bname)
        if not items:
            continue
        rows = ""
        for label, value in items:
            disp_value = value or "<span style='color:#94a3b8;'>not visible</span>"
            rows += (
                f'<tr>'
                f'<td style="padding:5px 10px 5px 0;font-weight:700;color:#1e3a5f;'
                f'font-size:0.78rem;white-space:nowrap;vertical-align:top;'
                f'text-transform:uppercase;letter-spacing:0.04em;">{label}</td>'
                f'<td style="padding:5px 0;font-size:0.88rem;color:#1f2937;'
                f'line-height:1.4;">{disp_value}</td>'
                f'</tr>'
            )
        cards += (
            f'<div style="flex:1 1 320px;min-width:280px;background:#f8fafc;'
            f'border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px;">'
            f'<div style="font-size:0.7rem;font-weight:800;color:#2563eb;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">'
            f'{bname}</div>'
            f'<table style="width:100%;border-collapse:collapse;font-family:{_FONT};">'
            f'{rows}'
            f'</table></div>'
        )

    free_html = ""
    if free_text:
        free_html = (
            f'<div style="margin-top:10px;padding:10px 14px;background:#f8fafc;'
            f'border:1px solid #e2e8f0;border-radius:6px;font-size:0.85rem;'
            f'line-height:1.5;color:#1f2937;">'
            + "<br/>".join(free_text) +
            f'</div>'
        )

    body = (
        f'<div style="background:linear-gradient(135deg,#fff,#f1f5f9);'
        f'padding:14px;border-radius:8px;border:1px solid #cbd5e1;'
        f'border-left:4px solid #b71c1c;">'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;">'
        f'{cards}'
        f'</div>'
        f'{free_html}'
        f'<div style="font-size:0.7rem;color:#64748b;margin-top:8px;'
        f'font-style:italic;">'
        f'Visual estimation only. Use as investigative lead, not legal evidence.'
        f'</div></div>'
    )
    return _section("Suspect Profile", body, "&#128100;")


def _build_timeline(event_log, frame_analyses):
    if not event_log and not frame_analyses:
        return _section("Incident Timeline",
                        f'<p style="color:#64748b;font-style:italic;">No events recorded.</p>', "&#128337;")

    # Merge events + frame analyses by timestamp
    entries = []
    for ev in event_log:
        entries.append({
            "ts": ev["timestamp"], "type": "event",
            "text": f'Cart {ev["cart_id"]}: {_badge(ev["event"])} - '
                    f'POPS {ev["pops_score"]} | {ev["fill"]}/{ev["bag"]} | '
                    f'{ev["direction"]} | Speed: {ev["speed_status"]}',
        })
    for fa in frame_analyses:
        if fa.description:
            entries.append({
                "ts": fa.timestamp, "type": "vlm",
                "text": f'<em style="color:#1565c0;">[VLM] {fa.trigger}:</em> {fa.description}',
            })
    entries.sort(key=lambda e: e["ts"])

    rows = ""
    for e in entries:
        icon = "&#128308;" if e["type"] == "event" else "&#128065;"
        rows += (
            f'<div style="display:flex;gap:10px;padding:6px 4px;border-bottom:1px solid #e2e8f0;'
            f'font-family:{_FONT};font-size:0.85rem;color:#1f2937;line-height:1.5;">'
            f'<div style="min-width:55px;color:#2563eb;font-weight:700;">{e["ts"]:.1f}s</div>'
            f'<div style="color:#1f2937;">{icon} {e["text"]}</div>'
            f'</div>'
        )
    return _section("Incident Timeline", rows, "&#128337;")


def _build_evidence_gallery(captures, frame_analyses):
    if not captures:
        return _section("Evidence Gallery",
                        f'<p style="color:#64748b;font-style:italic;">No frames captured.</p>', "&#128247;")

    # Build lookup: frame_idx -> description
    desc_map = {}
    for fa in frame_analyses:
        desc_map[fa.frame_idx] = fa.description

    cards = ""
    for cap in captures:
        b64 = base64.b64encode(cap["image_bytes"]).decode("utf-8")
        desc = desc_map.get(cap["frame_idx"], "")
        trigger = cap["trigger"]
        ts = cap["timestamp"]
        ctx = cap.get("pops_context", {})
        score = ctx.get("score", "")
        event = ctx.get("event", "")

        pops_badge = ""
        if score:
            ec = _EVENT_BADGE.get(event, "#546e7a")
            pops_badge = (
                f'<span style="background:{ec};color:#fff;padding:2px 8px;border-radius:4px;'
                f'font-size:0.75rem;font-weight:700;">POPS:{score}</span> '
            )

        cards += (
            f'<div style="display:inline-block;width:47%;vertical-align:top;margin:0.5% 1%;'
            f'background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;'
            f'font-family:{_FONT};color:#1f2937;">'
            f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;max-height:160px;object-fit:cover;">'
            f'<div style="padding:8px 12px;">'
            f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;font-weight:600;">'
            f'{ts:.1f}s &bull; {trigger}</div>'
            f'<div style="margin-bottom:4px;">{pops_badge}</div>'
            f'<div style="font-size:0.83rem;line-height:1.5;color:#1f2937;">{desc}</div>'
            f'</div></div>'
        )
    return _section("Evidence Gallery", f'<div>{cards}</div>', "&#128247;")


def _build_pops_analysis(peak_snapshots, pops_data):
    pops_summary = pops_data.get("pops_summary", {})
    if not pops_summary:
        return ""

    rows = ""
    for cd in sorted(pops_summary.keys()):
        info = pops_summary[cd]
        # pops_summary is keyed by "C{n}" (display id); peak_snapshots is
        # keyed by the raw integer cart id — translate before lookup.
        raw_cid = cd
        if isinstance(cd, str) and cd.startswith("C") and cd[1:].isdigit():
            raw_cid = int(cd[1:])
        snap = peak_snapshots.get(raw_cid, peak_snapshots.get(cd, {}))
        score = info.get("max_score", 0)
        # Same naming as the in-app POPS table — a report that called this row
        # "INBOUND" while the dashboard called it "INCOMING CART WITHOUT ITEMS"
        # would read as two different findings.
        event = resolve_event_label(snap, info.get("peak_event", "?"))
        fill = snap.get("fill", "?")
        bag = snap.get("bag", "?")
        direction = snap.get("direction", "?")

        ec = _EVENT_BADGE.get(event, "#546e7a")
        score_color = "#b71c1c" if score >= 71 else "#e65100" if score >= 31 else "#2e7d32"

        rows += (
            f'<tr style="border-bottom:1px solid #e2e8f0;background:#fff;color:#1f2937;">'
            f'<td style="padding:8px;font-weight:700;color:#1e3a5f;">{cd}</td>'
            f'<td style="padding:8px;color:#1f2937;"><span style="color:{score_color};font-weight:800;font-size:1.0rem;">{score}</span></td>'
            f'<td style="padding:8px;color:#1f2937;">{_badge(event)}</td>'
            f'<td style="padding:8px;color:#1f2937;">{fill}</td>'
            f'<td style="padding:8px;color:#1f2937;">{bag}</td>'
            f'<td style="padding:8px;color:#1f2937;">{direction}</td>'
            f'</tr>'
        )

    table = (
        f'<table style="width:100%;border-collapse:collapse;font-family:{_FONT};font-size:0.85rem;background:#fff;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">'
        f'<tr style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;">'
        f'<th style="padding:8px;text-align:left;color:#fff;">Cart</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Peak POPS</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Event</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Fill</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Bag</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Direction</th>'
        f'</tr>{rows}</table>'
    )
    return _section("POPS Analysis", table, "&#128202;")


def _ops_notice(title, detail, fg, bg):
    return (
        f'<div style="background:{bg};border-left:4px solid {fg};'
        f'padding:10px 14px;border-radius:6px;font-family:{_FONT};'
        f'font-size:0.85rem;color:#1f2937;">'
        f'<strong style="color:{fg};">{title}</strong>'
        + (f'<div style="margin-top:4px;color:#475569;">{detail}</div>'
           if detail else "")
        + '</div>'
    )


def _build_ops_findings_block(analytics_result):
    """Operational rule findings — three distinct states, never two.

    "No issues found" and "the rules could not run" are different answers and
    are rendered differently. Collapsing them would report a clean bill of
    health for a clip where nothing was ever checked, which is the failure
    build_operational_alerts exists to avoid.
    """
    reason = getattr(analytics_result, "rules_unavailable_reason", None)
    findings = getattr(analytics_result, "rule_findings", None)
    state, shown, remainder = highlights.ops_findings_state(reason, findings)

    # Coverage notes ride along with ALL THREE states, clean included, and come
    # from the same highlights helper the app panel and the JSON use. A report
    # that says "nothing crossed its threshold" without mentioning that the
    # clock was synthesised, or that four intervals were discarded as too
    # sparsely observed, is a clean bill of health nobody audited.
    notes = highlights.ops_diagnostics(
        getattr(analytics_result, "rule_diagnostics", None))
    notes_block = ""
    if notes:
        # Every colour in this document is inline ON THE ELEMENT, never
        # inherited: the Gradio tab renders this report inside the app's own
        # stylesheet, whose dark-theme rules colour bare <li>/<strong>/<th>
        # directly. A tag that only inherits its colour loses to those rules and
        # comes out near-white on this cream panel — which is exactly how these
        # coverage notes went invisible. Inline styles beat any non-!important
        # rule, so they hold in the tab and in the downloaded file alike.
        items = "".join(
            f'<li style="color:#475569;margin-bottom:4px;">{_esc(n)}</li>'
            for n in notes)
        notes_block = _ops_notice(
            "Coverage notes for this run",
            f'<ul style="margin:6px 0 0 18px;padding:0;color:#475569;">{items}</ul>',
            "#a16207", "#fefce8")

    if state == "unavailable":
        return notes_block + _ops_notice(
            "Operational rules did not run", _esc(reason),
            "#a16207", "#fefce8")
    if state == "clean":
        return notes_block + _ops_notice(
            "No operational issues detected",
            "Rules evaluated successfully and nothing crossed its threshold.",
            "#2e7d32", "#e8f5e9")

    # Highlights, not a second copy of the ops tab's 100-row table: `shown`
    # leads with everything severe, and when nothing is severe carries the
    # top few so the section still carries evidence rather than only a count.
    rows = ""
    for f in shown:
        sev = getattr(f, "severity", "INFO")
        fg, bg = _OPS_SEVERITY_COLORS.get(sev, _OPS_SEVERITY_COLORS["INFO"])
        dur = f"{getattr(f, 'duration_s', 0.0):.0f}s"
        if getattr(f, "ongoing_at_eov", False):
            # Still open when the video ended — the duration is a floor, not a
            # measurement, and saying so is the difference between "resolved in
            # 40s" and "was still happening when we stopped looking".
            dur = f"&ge; {dur} (ongoing)"
        why = "; ".join(_esc(r) for r in (getattr(f, "reasons", None) or []))
        if getattr(f, "confidence", "high") == "degraded":
            why += (" <span style='color:#a16207;font-weight:700;'>"
                    "[degraded: timing derived from frame rate]</span>")
        zone = _esc(getattr(f, "zone_name", None)) or "n/a"

        rows += (
            f'<tr style="border-bottom:1px solid #e2e8f0;background:#fff;">'
            f'<td style="padding:8px;color:#1f2937;"><span style="background:{fg};color:#fff;'
            f'padding:2px 8px;border-radius:4px;font-size:0.72rem;'
            f'font-weight:800;letter-spacing:0.04em;">{_esc(sev)}</span></td>'
            f'<td style="padding:8px;font-weight:700;color:#1e3a5f;">'
            f'{_esc(getattr(f, "label", ""))}</td>'
            f'<td style="padding:8px;color:#1f2937;">'
            f'C{_esc(getattr(f, "cart_display_id", "?"))}</td>'
            f'<td style="padding:8px;color:#1f2937;">{zone}</td>'
            f'<td style="padding:8px;color:#1f2937;white-space:nowrap;">'
            f'{_mmss(getattr(f, "start_t", 0.0))}</td>'
            f'<td style="padding:8px;color:#1f2937;white-space:nowrap;">{dur}</td>'
            f'<td style="padding:8px;color:#64748b;font-size:0.8rem;">{why}</td>'
            f'</tr>'
        )

    table = (
        f'<table style="width:100%;border-collapse:collapse;font-family:{_FONT};'
        f'font-size:0.85rem;background:#fff;border:1px solid #e2e8f0;'
        f'border-radius:6px;overflow:hidden;">'
        f'<tr style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;">'
        f'<th style="padding:8px;text-align:left;color:#fff;">Severity</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Issue</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Cart</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Zone</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Start</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Duration</th>'
        f'<th style="padding:8px;text-align:left;color:#fff;">Why</th>'
        f'</tr>{rows}</table>'
    )
    if remainder > 0:
        table += (
            f'<div style="font-size:0.78rem;color:#64748b;margin-top:6px;'
            f'font-family:{_FONT};">+{remainder} lower-severity finding'
            f'{"s" if remainder != 1 else ""} in the Operational Alerts tab.</div>'
        )
    return notes_block + table


def _build_congestion_block(analytics_result):
    """Queue spikes + top dwell zones.

    Independent of the rule engine on purpose: `rules_unavailable_reason` can
    be set because the clip is shorter than every rule threshold, which says
    nothing about whether congestion was measured. Gating this behind that
    field would hide data we actually have.
    """
    spikes, dwell = highlights.select_congestion(
        getattr(analytics_result, "queue_spikes", None),
        getattr(analytics_result, "dwell_summary", None))

    if not spikes and not dwell:
        return _ops_notice("No congestion measured",
                           "No zone occupancy or dwell data for this clip.",
                           "#546e7a", "#eceff1")

    blocks = ""
    if spikes:
        items = ""
        for s in spikes:
            sev = getattr(s, "severity", "")
            fg = "#b71c1c" if sev == "BACKED_UP" else "#e65100"
            reasons = ", ".join(_esc(r) for r in (getattr(s, "reasons", None) or [])[:3])
            items += (
                f'<li style="margin-bottom:6px;color:#1f2937;">'
                f'<span style="background:{fg};color:#fff;padding:1px 7px;'
                f'border-radius:4px;font-size:0.7rem;font-weight:800;">'
                f'{_esc(sev.replace("_", " "))}</span> '
                f'<strong style="color:#1f2937;">'
                f'{_esc(getattr(s, "zone_name", "?"))}</strong> '
                f'- avg dwell {getattr(s, "avg_dwell_s", 0.0):.1f}s, '
                f'peak occupancy {getattr(s, "peak_occupancy", 0)}'
                + (f' <span style="color:#64748b;">({reasons})</span>' if reasons else "")
                + '</li>'
            )
        blocks += (
            f'<h4 style="color:#1e3a5f;margin:0 0 8px;font-size:0.85rem;'
            f'font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">'
            f'Queue spikes</h4>'
            f'<ul style="margin:0 0 14px;padding-left:20px;line-height:1.6;'
            f'font-size:0.85rem;">{items}</ul>'
        )
    else:
        blocks += (
            f'<div style="font-size:0.85rem;color:#475569;margin-bottom:14px;">'
            f'No queue spikes - zone occupancy stayed within normal limits.'
            f'</div>'
        )

    if dwell:
        rows = ""
        for d in dwell:
            rows += (
                f'<tr style="border-bottom:1px solid #f1f5f9;">'
                f'<td style="padding:6px 10px;font-weight:700;color:#1e3a5f;">'
                f'{_esc(d.get("zone_name", "?"))}</td>'
                f'<td style="padding:6px 10px;color:#1f2937;">'
                f'{float(d.get("avg_dwell_s", 0.0)):.1f}s</td>'
                f'<td style="padding:6px 10px;color:#1f2937;">'
                f'{float(d.get("p95_s", 0.0)):.1f}s</td>'
                f'<td style="padding:6px 10px;color:#1f2937;">'
                f'{int(d.get("n_visits", 0))}</td>'
                f'</tr>'
            )
        blocks += (
            f'<h4 style="color:#1e3a5f;margin:0 0 8px;font-size:0.85rem;'
            f'font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">'
            f'Highest-dwell zones</h4>'
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-family:{_FONT};font-size:0.82rem;background:#fff;'
            f'border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">'
            f'<tr style="background:#f1f5f9;color:#1e3a5f;">'
            f'<th style="padding:6px 10px;text-align:left;color:#1e3a5f;">Zone</th>'
            f'<th style="padding:6px 10px;text-align:left;color:#1e3a5f;">Avg dwell</th>'
            f'<th style="padding:6px 10px;text-align:left;color:#1e3a5f;">p95</th>'
            f'<th style="padding:6px 10px;text-align:left;color:#1e3a5f;">Visits</th>'
            f'</tr>{rows}</table>'
        )
    return blocks


def _build_ops_highlights(analytics_result):
    """Operations Highlights — the operational read of the clip, alongside the
    theft-risk read the rest of the report carries.

    Rule findings are deliberately independent of POPS scoring (see
    RuleFinding's docstring), so this section is built on BOTH branches: a
    LOW-risk clip with a blocked fire exit still needs it.

    Returns "" when no analytics were supplied at all — an absent section is
    honest, whereas an "all clear" built from no data would not be.
    """
    if analytics_result is None:
        return ""

    insight = (getattr(analytics_result, "insight_text", "") or "").strip()
    insight_html = ""
    if insight:
        insight_html = (
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;'
            f'border-radius:6px;padding:10px 14px;margin-bottom:12px;'
            f'font-size:0.85rem;color:#1e3a5f;line-height:1.55;">'
            f'<strong style="color:#1e3a5f;">Auto-insight:</strong> '
            f'{_esc(insight)}</div>'
        )

    body = (
        f'<div style="font-family:{_FONT};color:#1f2937;">'
        f'{insight_html}'
        f'{_build_ops_findings_block(analytics_result)}'
        f'<div style="margin-top:14px;">'
        f'{_build_congestion_block(analytics_result)}'
        f'</div></div>'
    )
    return _section("Operations Highlights", body, "&#128737;")


def _build_insights(report_data):
    lp = report_data.actionable_insights_lp or "No LP recommendations available."
    leo = report_data.actionable_insights_leo or "No law enforcement summary available."

    # Convert bullet lines to HTML list
    def to_list(text):
        lines = [l.strip().lstrip("- ").lstrip("* ") for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return f'<p style="color:#1f2937;">{text}</p>'
        items = "".join(f"<li style='margin-bottom:6px;color:#1f2937;'>{l}</li>" for l in lines)
        return f"<ul style='margin:0;padding-left:20px;line-height:1.7;color:#1f2937;'>{items}</ul>"

    body = (
        f'<div style="font-family:{_FONT};font-size:0.9rem;color:#1f2937;'
        f'background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;">'
        f'<h4 style="color:#1e3a5f;margin:0 0 8px;font-size:0.9rem;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:0.04em;">Loss Prevention Team</h4>'
        f'{to_list(lp)}'
        f'<h4 style="color:#1e3a5f;margin:14px 0 8px;font-size:0.9rem;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:0.04em;">Law Enforcement / Authorities</h4>'
        f'{to_list(leo)}'
        f'</div>'
    )
    return _section("Actionable Insights", body, "&#128161;")


def _build_technical(report_data, video_info):
    items = [
        ("VLM Backend", report_data.vlm_backend),
        ("VLM Status", "Available" if report_data.vlm_available else f"Unavailable: {report_data.error_message}"),
        ("Analysis Time", f"{report_data.generation_time_seconds:.1f}s"),
        ("Video", video_info.get("video_name", "?")),
        ("Resolution", f'{video_info.get("width", "?")}x{video_info.get("height", "?")}'),
        ("FPS", str(video_info.get("fps", "?"))),
        ("Frames Analyzed", str(len(report_data.frame_analyses))),
    ]
    rows = "".join(
        f'<tr style="border-bottom:1px solid #f1f5f9;">'
        f'<td style="padding:6px 10px;font-weight:700;color:#475569;'
        f'text-transform:uppercase;letter-spacing:0.04em;font-size:0.72rem;">{k}</td>'
        f'<td style="padding:6px 10px;color:#1f2937;">{v}</td></tr>'
        for k, v in items
    )
    table = (
        f'<table style="width:100%;border-collapse:collapse;font-family:{_FONT};'
        f'font-size:0.82rem;background:#f8fafc;border:1px solid #e2e8f0;'
        f'border-radius:6px;overflow:hidden;">{rows}</table>'
    )
    return _section("Technical Details", table, "&#9881;")


def _vlm_unavailable_banner(report_data):
    if report_data.vlm_available:
        return ""
    return (
        f'<div style="background:#fff3e0;border-left:5px solid #e65100;padding:12px 16px;'
        f'border-radius:6px;margin-bottom:16px;font-family:{_FONT};font-size:0.85rem;">'
        f'&#9888; <strong style="color:#e65100;">VLM Analysis Unavailable</strong>'
        f' - '
        f'This is a data-only report. {report_data.error_message or ""}</div>'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_case_report_html(report_data, captures, pops_data,
                           event_log, peak_snapshots, video_info,
                           analytics_result=None):
    """Build the case report.

    analytics_result (optional AnalyticsResult): drives the Operations
    Highlights section — rule findings, queue spikes, dwell. Optional so the
    signature stays usable without it; the section is omitted when absent
    rather than rendered as an empty all-clear.

    Returns (gradio_html, standalone_html).
    """
    frame_analyses = report_data.frame_analyses if report_data else []

    suspect_section = _build_suspect_profile(report_data)
    body_parts = [
        _build_header(video_info),
        _vlm_unavailable_banner(report_data),
        _build_executive_summary(report_data),
        _build_risk_assessment(report_data),
    ]
    if suspect_section:
        body_parts.append(suspect_section)
    body_parts += [
        _build_timeline(event_log, frame_analyses),
        _build_evidence_gallery(captures, frame_analyses),
        _build_pops_analysis(peak_snapshots, pops_data),
        # Ops sits between the POPS table and the recommendations: both are
        # evidence, and the LP / law-enforcement actions should stay last.
        _build_ops_highlights(analytics_result),
        _build_insights(report_data),
        _build_technical(report_data, video_info),
        (f'<div style="text-align:center;color:#94a3b8;font-size:0.7rem;'
         f'font-family:{_FONT};padding:8px 0;border-top:1px solid #e2e8f0;margin-top:12px;">'
         f'<strong style="color:#94a3b8;">GATEKEEPER AI</strong> &bull; '
         f'POPS Case Report &bull; '
         f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>'),
    ]
    body_html = "\n".join(body_parts)

    # Gradio tab version — wrapped in an explicit "paper" container so the
    # report renders consistently against both light and dark Gradio themes.
    gradio_html = (
        f'<div style="font-family:{_FONT};max-width:900px;margin:8px auto;'
        f'background:#ffffff;color:#1f2937;border-radius:10px;padding:24px 28px;'
        f'box-shadow:0 2px 12px rgba(0,0,0,0.12);border:1px solid #e2e8f0;">'
        f'{body_html}'
        f'</div>'
    )

    # Standalone downloadable HTML
    standalone_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>POPS Incident Case Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: {_FONT}; background:#f1f5f9; padding:20px; color:#1e293b; }}
  .container {{ max-width:900px; margin:0 auto; background:#fff; border-radius:12px;
                padding:28px; box-shadow:0 4px 20px rgba(0,0,0,0.08); }}
  @media print {{
    body {{ background:#fff; padding:0; }}
    .container {{ box-shadow:none; padding:10px; }}
    img {{ max-height:300px; }}
  }}
</style>
</head>
<body>
<div class="container">
{body_html}
</div>
</body>
</html>'''

    return gradio_html, standalone_html
