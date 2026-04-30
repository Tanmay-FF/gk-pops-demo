# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Case report HTML builder — generates professional LP / law-enforcement
incident reports from VLM analysis + POPS data.
"""
import base64
import uuid
from datetime import datetime

from .ui_builder import _FONT, _EVENT_BADGE, _badge


# ---------------------------------------------------------------------------
# Risk-level styling
# ---------------------------------------------------------------------------
_RISK_COLORS = {
    "HIGH": ("#b71c1c", "#fce4ec"),
    "MEDIUM": ("#e65100", "#fff8e1"),
    "LOW": ("#2e7d32", "#e8f5e9"),
    "UNKNOWN": ("#546e7a", "#eceff1"),
}


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
    text = report_data.executive_summary or "<em>VLM analysis not available.</em>"
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
            "text": f'Cart {ev["cart_id"]}: {_badge(ev["event"])} &mdash; '
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
        event = info.get("peak_event", "?")
        fill = snap.get("fill", "?")
        bag = snap.get("bag", "?")
        direction = snap.get("direction", "?")

        ec = _EVENT_BADGE.get(event, "#546e7a")
        score_color = "#b71c1c" if score >= 71 else "#e65100" if score >= 31 else "#2e7d32"

        rows += (
            f'<tr style="border-bottom:1px solid #e2e8f0;background:#fff;color:#1f2937;">'
            f'<td style="padding:8px;font-weight:700;color:#1e3a5f;">{cd}</td>'
            f'<td style="padding:8px;"><span style="color:{score_color};font-weight:800;font-size:1.0rem;">{score}</span></td>'
            f'<td style="padding:8px;">{_badge(event)}</td>'
            f'<td style="padding:8px;color:#1f2937;">{fill}</td>'
            f'<td style="padding:8px;color:#1f2937;">{bag}</td>'
            f'<td style="padding:8px;color:#1f2937;">{direction}</td>'
            f'</tr>'
        )

    table = (
        f'<table style="width:100%;border-collapse:collapse;font-family:{_FONT};font-size:0.85rem;background:#fff;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">'
        f'<tr style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;">'
        f'<th style="padding:8px;text-align:left;">Cart</th>'
        f'<th style="padding:8px;text-align:left;">Peak POPS</th>'
        f'<th style="padding:8px;text-align:left;">Event</th>'
        f'<th style="padding:8px;text-align:left;">Fill</th>'
        f'<th style="padding:8px;text-align:left;">Bag</th>'
        f'<th style="padding:8px;text-align:left;">Direction</th>'
        f'</tr>{rows}</table>'
    )
    return _section("POPS Analysis", table, "&#128202;")


def _build_insights(report_data):
    lp = report_data.actionable_insights_lp or "No LP recommendations available."
    leo = report_data.actionable_insights_leo or "No law enforcement summary available."

    # Convert bullet lines to HTML list
    def to_list(text):
        lines = [l.strip().lstrip("- ").lstrip("* ") for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return f"<p>{text}</p>"
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
        f'&#9888; <strong>VLM Analysis Unavailable</strong> &mdash; '
        f'This is a data-only report. {report_data.error_message or ""}</div>'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_case_report_html(report_data, captures, pops_data,
                           event_log, peak_snapshots, video_info):
    """Build the case report.

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
        _build_insights(report_data),
        _build_technical(report_data, video_info),
        (f'<div style="text-align:center;color:#94a3b8;font-size:0.7rem;'
         f'font-family:{_FONT};padding:8px 0;border-top:1px solid #e2e8f0;margin-top:12px;">'
         f'<strong>GATEKEEPER AI</strong> &bull; POPS Case Report &bull; '
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
