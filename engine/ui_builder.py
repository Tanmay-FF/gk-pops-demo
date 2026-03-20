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
