# Built by Tanmay Thaker | MLE, Gatekeeper Systems <tthaker@gatekeepersystems.com>
"""
HTML generation for Gradio UI — POPS summary, events timeline, info tables.
"""
import os
from .scoring import HIGH_EVENTS, MEDIUM_EVENTS

_FONT = "'Nunito Sans', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"

# Badge / row colour maps (shared by POPS and events tables)
_EVENT_BADGE = {
    "PUSHOUT ALERT": "#b71c1c",
    "HIGH PRIORITY": "#c62828",
    "MEDIUM PRIORITY": "#e65100",
    "UNLINKED EXIT": "#ef6c00",
    "ABANDONED CART": "#b71c1c",
    "MONITORING": "#1565c0",
    "INBOUND": "#2e7d32",
    "LOW PRIORITY": "#2e7d32",
}
_EVENT_ROW_BG = {
    "PUSHOUT ALERT": "#fce4ec",
    "HIGH PRIORITY": "#fce4ec",
    "MEDIUM PRIORITY": "#fff8e1",
    "UNLINKED EXIT": "#fff3e0",
    "ABANDONED CART": "#fce4ec",
}


def _badge(event_name: str) -> str:
    ec = _EVENT_BADGE.get(event_name, "#546e7a")
    return (f"<span style='background:{ec};color:#fff;padding:5px 14px;"
            f"border-radius:5px;font-size:0.85rem;font-weight:800;"
            f"font-family:{_FONT};letter-spacing:0.3px;'>{event_name}</span>")


# ---------------------------------------------------------------------------
# Top-of-page alert banner (high-priority events + severe zone congestion)
# ---------------------------------------------------------------------------
def _alert_chip(title: str, detail: str) -> str:
    return (
        f"<div style='background:rgba(255,255,255,0.13);"
        f"border:1px solid rgba(255,255,255,0.30);border-radius:8px;"
        f"padding:6px 12px;line-height:1.25;'>"
        f"<div style='font-size:0.62rem;font-weight:800;letter-spacing:1.4px;"
        f"color:rgba(255,255,255,0.85);text-transform:uppercase;'>{title}</div>"
        f"<div style='font-size:0.86rem;font-weight:800;color:#fff;"
        f"margin-top:1px;'>{detail}</div></div>"
    )


def build_alert_banner(event_log, queue_spikes, spike_events=None) -> str:
    """Sticky top-of-page notification surfacing HIGH POPS events and severe
    zone congestion. Returns "" when nothing is worth flagging — caller is
    expected to hide the component in that case.

    Args:
        event_log:    list of per-frame logged events (HIGH_EVENTS get flagged).
        queue_spikes: list[QueueSpike] from analytics_builder — zone-bound.
        spike_events: list[dict] from analytics_result.spike_events — same data
                      but may also include `source: "crowd_cluster"` rows that
                      aren't tied to a user-defined zone.
    """
    high_events = [e for e in (event_log or []) if e.get("event") in HIGH_EVENTS]
    severe_spikes = [s for s in (queue_spikes or [])
                     if getattr(s, "severity", None) in ("BACKED_UP", "QUEUE_FORMING")]
    # Crowd-cluster spikes (or any spike_events row not already in queue_spikes)
    seen_zone_ids = {s.zone_id for s in severe_spikes}
    crowd_events = [e for e in (spike_events or [])
                    if e.get("severity") in ("BACKED_UP", "QUEUE_FORMING")
                    and e.get("zone_id") not in seen_zone_ids]

    if not high_events and not severe_spikes and not crowd_events:
        return ""

    # Keep the highest-severity event per cart so we don't double-count.
    _SEV = {"PUSHOUT ALERT": 2, "HIGH PRIORITY": 1}
    by_cart: dict = {}
    for e in high_events:
        cd = e["cart_id"]
        if cd not in by_cart or _SEV.get(e["event"], 0) > _SEV.get(by_cart[cd]["event"], 0):
            by_cart[cd] = e
    pushouts = [e for e in by_cart.values() if e["event"] == "PUSHOUT ALERT"]
    high_pri = [e for e in by_cart.values() if e["event"] == "HIGH PRIORITY"]
    backed_up = [s for s in severe_spikes if s.severity == "BACKED_UP"]
    queueing  = [s for s in severe_spikes if s.severity == "QUEUE_FORMING"]
    crowd_backed_up = [e for e in crowd_events if e.get("severity") == "BACKED_UP"]
    crowd_queueing  = [e for e in crowd_events if e.get("severity") == "QUEUE_FORMING"]

    # Hot palette when something truly critical happened, amber otherwise.
    critical = bool(pushouts or backed_up or crowd_backed_up)
    if critical:
        grad = "linear-gradient(135deg,#7f1d1d 0%,#dc2626 55%,#f97316 100%)"
        accent = "#fee2e2"
        shadow = "0 6px 22px rgba(220,38,38,0.32)"
    else:
        grad = "linear-gradient(135deg,#78350f 0%,#d97706 55%,#f59e0b 100%)"
        accent = "#fef3c7"
        shadow = "0 6px 22px rgba(217,119,6,0.30)"

    chips = []
    if pushouts:
        carts = ", ".join(f"C{e['cart_id']}" for e in pushouts[:5])
        if len(pushouts) > 5:
            carts += f" +{len(pushouts) - 5} more"
        chips.append(_alert_chip(f"Pushout × {len(pushouts)}", carts))
    if high_pri:
        carts = ", ".join(f"C{e['cart_id']}" for e in high_pri[:5])
        if len(high_pri) > 5:
            carts += f" +{len(high_pri) - 5} more"
        chips.append(_alert_chip(f"High Priority × {len(high_pri)}", carts))
    if backed_up:
        zones = ", ".join(s.zone_name for s in backed_up[:3])
        if len(backed_up) > 3:
            zones += f" +{len(backed_up) - 3} more"
        chips.append(_alert_chip(f"Zone Backed Up × {len(backed_up)}", zones))
    if queueing:
        zones = ", ".join(s.zone_name for s in queueing[:3])
        if len(queueing) > 3:
            zones += f" +{len(queueing) - 3} more"
        chips.append(_alert_chip(f"Queue Forming × {len(queueing)}", zones))
    if crowd_backed_up:
        labels = ", ".join(e.get("zone_name", "Crowd") for e in crowd_backed_up[:3])
        if len(crowd_backed_up) > 3:
            labels += f" +{len(crowd_backed_up) - 3} more"
        chips.append(_alert_chip(f"Crowd Backed Up × {len(crowd_backed_up)}", labels))
    if crowd_queueing:
        labels = ", ".join(e.get("zone_name", "Crowd") for e in crowd_queueing[:3])
        if len(crowd_queueing) > 3:
            labels += f" +{len(crowd_queueing) - 3} more"
        chips.append(_alert_chip(f"Crowd Spike × {len(crowd_queueing)}", labels))

    n_total = (len(pushouts) + len(high_pri) + len(severe_spikes)
               + len(crowd_events))
    headline = ("Action required — review the flagged carts and zones below"
                if critical else
                "Heads up — congestion forming in one or more zones")

    return (
        f"<div class='gk-alert-banner' role='alert' aria-live='assertive' "
        f"style='background:{grad};color:#fff;border-radius:10px;"
        f"padding:14px 18px;margin-bottom:12px;box-shadow:{shadow};"
        f"border:1px solid rgba(255,255,255,0.18);font-family:{_FONT};'>"
        f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap;'>"
        f"<div style='display:flex;align-items:center;gap:10px;flex-shrink:0;'>"
        f"<span class='gk-alert-pulse' style='width:12px;height:12px;border-radius:50%;"
        f"background:#fff;flex-shrink:0;'></span>"
        f"<div>"
        f"<div style='font-size:0.62rem;font-weight:800;letter-spacing:2px;"
        f"color:{accent};text-transform:uppercase;'>"
        f"Live Alerts &middot; {n_total} flagged"
        f"</div>"
        f"<div style='font-size:1.02rem;font-weight:800;color:#fff;line-height:1.2;"
        f"margin-top:2px;'>{headline}</div>"
        f"</div></div>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;flex:1;justify-content:flex-end;'>"
        f"{''.join(chips)}"
        f"</div>"
        f"<button onclick=\"this.closest('.gk-alert-banner').style.display='none'\" "
        f"style='background:rgba(255,255,255,0.16);color:#fff;"
        f"border:1px solid rgba(255,255,255,0.30);border-radius:6px;"
        f"width:30px;height:30px;font-size:1.1rem;font-weight:800;cursor:pointer;"
        f"flex-shrink:0;line-height:1;padding:0;' title='Dismiss'>&times;</button>"
        f"</div></div>"
    )


def styled_table(title, rows, header_gradient=("#1e3a5f", "#2563eb"), row_tint="#e8f0fe"):
    g1, g2 = header_gradient
    html = (
        f'<div style="padding:8px;background:#fff;border-radius:10px;font-family:{_FONT};">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.95rem;'
        f'font-family:{_FONT};border-radius:8px;overflow:hidden;'
        f'box-shadow:0 1px 6px rgba(0,0,0,0.08);line-height:1.5;">'
        f'<thead><tr style="background:linear-gradient(135deg,{g1},{g2});color:#fff;">'
        f'<th colspan="2" style="padding:14px 18px;text-align:left;font-size:1.05rem;'
        f'letter-spacing:0.5px;color:#fff;font-weight:700;">{title}</th>'
        f'</tr></thead><tbody>'
    )
    for i, (key, val) in enumerate(rows):
        bg = row_tint if i % 2 == 0 else "#ffffff"
        html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:12px 18px;font-weight:600;color:#111827;'
            f'border-bottom:1px solid #e2e8f0;width:45%;font-size:0.95rem;">{key}</td>'
            f'<td style="padding:12px 18px;color:#111827 !important;'
            f'border-bottom:1px solid #e2e8f0;font-size:0.95rem;">'
            f'<span style="color:#111827 !important;">{val}</span></td>'
            f'</tr>'
        )
    html += '</tbody></table></div>'
    return html


def build_video_info(source_path, w, h, fps, total_frames, processed):
    rows = [
        ("Video Name", os.path.basename(source_path)),
        ("Resolution", f"{w} x {h}"),
        ("Total Frames", str(total_frames)),
        ("Frames Processed", str(processed)),
    ]
    return styled_table("Video Information", rows,
                        header_gradient=("#0f4c75", "#3282b8"), row_tint="#e8f4fc")


def build_detection_info(n_people, n_carts, n_links):
    rows = [
        ("Unique Persons Detected", f"<b style='color:#1a1a2e !important;'>{n_people}</b>"),
        ("Unique Carts Detected",   f"<b style='color:#1a1a2e !important;'>{n_carts}</b>"),
        ("Person-Cart Links",       f"<b style='color:#1a1a2e !important;'>{n_links}</b>"),
    ]
    return styled_table("Detection Summary", rows,
                        header_gradient=("#1b5e20", "#43a047"), row_tint="#e8f5e9")


def build_config_info(link_confirm, link_grace, camera, quality_pt, fill_pt, threshold):
    rows = [
        ("Detection Model", "YOLOv26m (custom trained)"),
        ("Tracker", "BoTSORT (retail tuned)"),
        ("Link Confirmation", f"{link_confirm} frames co-movement + overlap"),
        ("Link Grace Period", f"{link_grace} frames before linking new cart"),
        ("Camera Placement", camera),
        ("Quality Model", os.path.basename(quality_pt) if quality_pt else "None"),
        ("Fill/Bag Model", os.path.basename(fill_pt) if fill_pt else "None"),
        ("Quality Threshold", f"{threshold:.2f}"),
    ]
    return styled_table("Model Configuration", rows,
                        header_gradient=("#4a148c", "#7b1fa2"), row_tint="#f3e5f5")


def build_legend():
    def sw(c):
        return f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:{c};vertical-align:middle;margin-right:6px;"></span>'
    rows = [
        (sw("#00e676") + "Green box", "Person"),
        (sw("#ffa500") + "Orange box", "Cart"),
        (sw("#ff32ff") + "Magenta line", "Confirmed link (person owns cart)"),
        (sw("#ff0000") + "Red text", "PUSHOUT ALERT / HIGH PRIORITY (POPS 71+)"),
        (sw("#ff8c00") + "Orange text", "MEDIUM PRIORITY / SUSPICIOUS (POPS 31-70)"),
        (sw("#00c853") + "Green text", "MONITORING / LOW PRIORITY (POPS 0-30)"),
        (sw("#34d399") + "Green label", "Valid Cart"),
        (sw("#ef4444") + "Red label", "Unclear Cart"),
    ]
    return styled_table("Annotation Legend", rows,
                        header_gradient=("#bf360c", "#e64a19"), row_tint="#fbe9e7")


def build_pops_summary(max_pops_per_cart, peak_snapshots):
    if not max_pops_per_cart:
        return "<p style='color:#94a3b8;text-align:center;padding:20px;'>No carts detected</p>"

    th_s = f"padding:14px 18px;text-align:left;color:#fff;font-weight:800;font-size:0.95rem;letter-spacing:0.3px;"
    td_s = f"padding:12px 18px;border-bottom:1px solid #e2e8f0;color:#111827;font-size:0.95rem;line-height:1.5;"

    rows_html = ""
    for i, cd in enumerate(sorted(max_pops_per_cart.keys())):
        ms = max_pops_per_cart[cd]
        peak = peak_snapshots.get(cd, {})
        evt  = peak.get("event", "CLEAR")
        qual = peak.get("quality", "unclassified")
        fill = peak.get("fill", "N/A")
        bag  = peak.get("bag", "N/A").replace("_", " ")

        if ms >= 71:
            sc = f"<b style='color:#b71c1c;'>{ms}</b>"
        elif ms >= 31:
            sc = f"<b style='color:#e65100;'>{ms}</b>"
        else:
            sc = f"<b style='color:#2e7d32;'>{ms}</b>"

        bg = "#e8f0fe" if i % 2 == 0 else "#ffffff"
        eb = _badge(evt)
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="{td_s}font-weight:800;">Cart {cd}</td>'
            f'<td style="{td_s}font-weight:700;">{sc}</td>'
            f'<td style="{td_s}">{eb}</td>'
            f'<td style="{td_s}font-weight:600;text-transform:uppercase;">{qual}</td>'
            f'<td style="{td_s}font-weight:600;text-transform:uppercase;">{fill}</td>'
            f'<td style="{td_s}font-weight:600;text-transform:uppercase;">{bag}</td>'
            f'</tr>'
        )

    return (
        f'<div style="padding:8px;font-family:{_FONT};">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.95rem;'
        f'font-family:{_FONT};border-radius:8px;overflow:hidden;'
        f'box-shadow:0 1px 6px rgba(0,0,0,0.08);line-height:1.5;">'
        f'<thead><tr style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;">'
        f'<th style="{th_s}">Cart</th>'
        f'<th style="{th_s}">POPS</th>'
        f'<th style="{th_s}">Event</th>'
        f'<th style="{th_s}">Quality</th>'
        f'<th style="{th_s}">Fill</th>'
        f'<th style="{th_s}">Bag</th>'
        f'</tr></thead><tbody>'
        + rows_html +
        f'</tbody></table></div>'
    )


def build_events_timeline(event_log):
    if not event_log:
        return (
            f'<div style="text-align:center;padding:30px;font-family:{_FONT};">'
            f'<span style="font-size:1.2rem;color:#2e7d32;font-weight:700;">ALL CLEAR</span>'
            f'<p style="color:#94a3b8;margin-top:8px;">No pushout events or suspicious activity detected</p>'
            f'</div>'
        )

    th_s = f"padding:14px 18px;text-align:left;color:#fff;font-weight:800;font-size:0.95rem;letter-spacing:0.3px;"
    td_s = f"padding:12px 18px;border-bottom:1px solid #e2e8f0;color:#111827;font-size:0.95rem;line-height:1.5;"
    rows_html = ""
    for evt in event_log:
        bg = _EVENT_ROW_BG.get(evt["event"], "#f5f5f5")
        b  = _badge(evt["event"])
        linked_str = "LINKED" if evt["linked"] else "NO LINK"
        bag_lbl = evt.get("bag", "N/A").replace("_", " ")
        spd = evt.get("speed_status", "N/A")
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="{td_s}color:#4b5563;">{evt["timestamp"]}s (F{evt["frame"]})</td>'
            f'<td style="{td_s}font-weight:800;">Cart {evt["cart_id"]}</td>'
            f'<td style="{td_s}">{b}</td>'
            f'<td style="{td_s}font-weight:700;">{evt["pops_score"]}</td>'
            f'<td style="{td_s}font-weight:600;text-transform:uppercase;">{evt["fill"]}</td>'
            f'<td style="{td_s}font-weight:600;text-transform:uppercase;">{bag_lbl}</td>'
            f'<td style="{td_s}font-weight:600;text-transform:uppercase;">{spd}</td>'
            f'<td style="{td_s}font-weight:500;">{evt["direction"]} | {linked_str}</td>'
            f'</tr>'
        )

    table = (
        f'<div style="padding:8px;font-family:{_FONT};">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.95rem;'
        f'font-family:{_FONT};border-radius:8px;overflow:hidden;'
        f'box-shadow:0 1px 6px rgba(0,0,0,0.08);line-height:1.5;">'
        f'<thead><tr style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;">'
        f'<th style="{th_s}">Time</th>'
        f'<th style="{th_s}">Cart</th>'
        f'<th style="{th_s}">Event</th>'
        f'<th style="{th_s}">POPS</th>'
        f'<th style="{th_s}">Fill</th>'
        f'<th style="{th_s}">Bag</th>'
        f'<th style="{th_s}">Speed</th>'
        f'<th style="{th_s}">Details</th>'
        f'</tr></thead><tbody>'
        + rows_html +
        f'</tbody></table></div>'
    )

    n_high = sum(1 for e in event_log if e["event"] in HIGH_EVENTS)
    n_med  = sum(1 for e in event_log if e["event"] in MEDIUM_EVENTS)
    if n_high > 0:
        bc, bt = "#c62828", f"ALERT: {n_high} HIGH-PRIORITY EVENT(s) DETECTED!"
    elif n_med > 0:
        bc, bt = "#e65100", f"WARNING: {n_med} MEDIUM-PRIORITY EVENT(S) DETECTED!"
    else:
        bc, bt = "#22c55e", "NO HIGH-RISK EVENTS"

    banner = (
        f'<div style="background:{bc};color:white;padding:16px;border-radius:8px;'
        f'text-align:center;font-weight:700;font-size:1.05rem;margin-bottom:12px;'
        f'font-family:{_FONT};letter-spacing:0.3px;">{bt}</div>'
    )
    return banner + table
