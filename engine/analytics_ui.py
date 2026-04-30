# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
HTML builders for the Analytics tab — dark-theme, high-contrast.

Color palette tuned for readability against the app's --bg-1 / --bg-2
surfaces. Font sizes bumped throughout so values pop on the dark
background. Inline-style only so the HTML is self-contained.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .analytics_models import (
    AnalyticsResult, DwellRow, JourneyEdge, QueueSpike, Zone,
)


_FONT = "'Inter', 'Nunito Sans', 'Segoe UI', Tahoma, sans-serif"

# ── High-contrast dark palette ───────────────────────────────────────────
_BG_1        = "#1d2231"   # surface (matches React --bg-1)
_BG_2        = "#262c3e"   # elevated card
_BG_3        = "#323a52"   # row header / hover
_BG_INSET    = "#161a25"

_BORDER      = "rgba(255,255,255,0.10)"
_BORDER_HARD = "rgba(255,255,255,0.18)"

# Text — bright enough to read at glance against _BG_1 / _BG_2
_TEXT_0      = "#ffffff"   # pure white for maximum contrast (titles, values)
_TEXT_1      = "#dde0eb"   # body text
_TEXT_2      = "#a8acc0"   # secondary labels (still very legible)
_TEXT_3      = "#7e8298"   # muted footnotes

# Brand colors — saturated so they pop against dark
_ACCENT      = "#6b9bff"
_ACCENT_HI   = "#8fb5ff"
_ACCENT_TINT = "rgba(107,155,255,0.18)"

_GOOD        = "#5fdc9c"
_GOOD_TINT   = "rgba(95,220,156,0.18)"
_WARN        = "#f0c870"
_WARN_TINT   = "rgba(240,200,112,0.18)"
_BAD         = "#ff8170"
_BAD_TINT    = "rgba(255,129,112,0.18)"

_PURPLE      = "#c4b5fd"
_PURPLE_HI   = "#ddd2ff"
_PURPLE_TINT = "rgba(196,181,253,0.18)"

_CYAN        = "#67e8f9"
_CYAN_TINT   = "rgba(103,232,249,0.18)"


_SEVERITY_COLOR = {
    "BACKED_UP":     (_BAD,    _BAD_TINT),
    "QUEUE_FORMING": (_WARN,   _WARN_TINT),
    "WATCH":         (_ACCENT, _ACCENT_TINT),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_dur(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{int(s):02d}s"


def _zone_swatch(color_bgr: tuple[int, int, int]) -> str:
    b, g, r = color_bgr
    css = f"rgb({r},{g},{b})"
    return (f"<span style='display:inline-block;width:12px;height:12px;"
            f"border-radius:3px;background:{css};vertical-align:middle;"
            f"margin-right:8px;box-shadow:0 0 0 1px rgba(255,255,255,0.18);'></span>")


def _empty_state(headline: str, sub: str = "") -> str:
    sub_html = ""
    if sub:
        sub_html = (f"<div style='color:{_TEXT_2};margin-top:8px;"
                    f"font-size:14px;'>{sub}</div>")
    return (
        f"<div style='text-align:center;padding:32px 16px;font-family:{_FONT};'>"
        f"<div style='font-size:15px;color:{_TEXT_0};font-weight:600;'>{headline}</div>"
        f"{sub_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Spike banner
# ---------------------------------------------------------------------------
def build_queue_spikes_banner(spikes: list[QueueSpike]) -> str:
    if not spikes:
        return (
            f"<div style='background:{_GOOD_TINT};color:{_GOOD};"
            f"border:1px solid {_GOOD};border-radius:10px;padding:14px 18px;"
            f"text-align:center;font-weight:700;font-size:14px;"
            f"font-family:{_FONT};margin-bottom:12px;letter-spacing:0.05em;"
            f"text-transform:uppercase;'>No queue spikes detected</div>"
        )

    by_sev: dict[str, list[QueueSpike]] = {}
    for s in spikes:
        by_sev.setdefault(s.severity, []).append(s)

    parts = []
    order = ["BACKED_UP", "QUEUE_FORMING", "WATCH"]
    headline_color, headline_tint = _SEVERITY_COLOR[
        order[next(i for i, k in enumerate(order) if k in by_sev)]
    ]
    n_zone_spikes = sum(1 for s in spikes if not str(s.zone_id).startswith("crowd_"))
    n_crowd_spikes = len(spikes) - n_zone_spikes
    headline_bits: list[str] = []
    if n_zone_spikes:
        headline_bits.append(f"{n_zone_spikes} zone(s)")
    if n_crowd_spikes:
        headline_bits.append(f"{n_crowd_spikes} crowd cluster(s)")
    headline_count = " + ".join(headline_bits) if headline_bits else f"{len(spikes)} item(s)"
    parts.append(
        f"<div style='background:{headline_tint};color:{headline_color};"
        f"border:1px solid {headline_color};padding:14px 18px;border-radius:10px;"
        f"text-align:center;font-weight:700;font-size:15px;font-family:{_FONT};"
        f"margin-bottom:12px;letter-spacing:0.05em;text-transform:uppercase;'>"
        f"Queue / dwell spikes — {headline_count} flagged</div>"
    )

    for sev in order:
        items = by_sev.get(sev, [])
        if not items:
            continue
        fg, bg = _SEVERITY_COLOR[sev]
        parts.append(
            f"<div style='background:{bg};border-left:4px solid {fg};"
            f"padding:12px 16px;margin:8px 0;border-radius:6px;font-family:{_FONT};'>"
        )
        parts.append(
            f"<div style='color:{fg};font-weight:700;font-size:13px;"
            f"letter-spacing:0.08em;text-transform:uppercase;'>"
            f"{sev.replace('_',' ')}</div>"
        )
        for s in items:
            reason_html = ""
            if getattr(s, "reasons", None):
                reason_html = (
                    f"<div style='color:{_TEXT_2};font-size:13px;margin-top:4px;'>"
                    f"Why: {' &middot; '.join(s.reasons)} "
                    f"<span style='color:{_TEXT_3};'>"
                    f"(score {getattr(s, 'score', 0):.0f}/100)</span></div>"
                )
            parts.append(
                f"<div style='color:{_TEXT_1};font-size:14px;margin-top:6px;'>"
                f"<b style='color:{_TEXT_0};font-size:15px;'>{s.zone_name}</b> — "
                f"avg {_fmt_dur(s.avg_dwell_s)}, "
                f"max {_fmt_dur(s.max_dwell_s)} across {s.n_visits} visit(s) "
                f"<span style='color:{_TEXT_2};'>(threshold {_fmt_dur(s.threshold_s)})</span>"
                f"{reason_html}</div>"
            )
        parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Dwell tables
# ---------------------------------------------------------------------------
def build_dwell_table(zones: list[Zone],
                      dwell_summary: list[dict],
                      dwell_rows: list[DwellRow],
                      *, max_visit_rows: int = 60) -> str:
    if not zones:
        return _empty_state(
            "Draw a zone in the Zone Editor tab to see store-flow analytics.",
            "Heatmap below works without zones.",
        )
    if not dwell_summary:
        return _empty_state("No dwell data computed.", "")

    th_s = (f"padding:12px 16px;text-align:left;color:{_TEXT_0};font-weight:700;"
            f"font-size:13px;letter-spacing:0.06em;vertical-align:top;"
            f"text-transform:uppercase;border-bottom:1px solid {_BORDER_HARD};"
            f"background:{_BG_3};")
    th_sub = (f"display:block;font-weight:500;font-size:11px;color:{_TEXT_2};"
              f"margin-top:3px;letter-spacing:0;text-transform:none;")
    td_s = (f"padding:12px 16px;border-bottom:1px solid {_BORDER};color:{_TEXT_1};"
            f"font-size:14px;line-height:1.5;")

    color_by_zone = {z.zone_id: z.color for z in zones}

    # ---- Per-zone summary ----
    summary_rows = ""
    for i, s in enumerate(dwell_summary):
        bg = _BG_1 if i % 2 == 0 else "transparent"
        sw = _zone_swatch(color_by_zone.get(s["zone_id"], (200, 200, 200)))
        summary_rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='{td_s}font-weight:700;color:{_TEXT_0};font-size:14.5px;'>"
            f"{sw}{s['zone_name']}"
            f"<span style='color:{_TEXT_2};font-size:11px;font-weight:700;"
            f"margin-left:10px;text-transform:uppercase;letter-spacing:0.06em;'>"
            f"{s['applies_to']}</span></td>"
            f"<td style='{td_s}font-weight:600;color:{_TEXT_0};'>{s['n_visits']}</td>"
            f"<td style='{td_s}font-weight:600;color:{_TEXT_0};'>{s['n_unique']}</td>"
            f"<td style='{td_s}font-weight:700;color:{_ACCENT_HI};'>{_fmt_dur(s['avg_dwell_s'])}</td>"
            f"<td style='{td_s}'>{_fmt_dur(s['p50_s'])}</td>"
            f"<td style='{td_s}'>{_fmt_dur(s['p95_s'])}</td>"
            f"<td style='{td_s}'>{_fmt_dur(s['max_s'])}</td>"
            f"<td style='{td_s}'>{_fmt_dur(s['total_s'])}</td>"
            f"</tr>"
        )

    summary_table = (
        f"<div style='font-family:{_FONT};'>"
        f"<div style='font-weight:700;color:{_TEXT_0};font-size:16px;"
        f"margin:4px 4px 4px;letter-spacing:0.01em;'>Per-zone dwell summary"
        f"<span style='font-weight:500;color:{_TEXT_2};font-size:12px;"
        f"margin-left:10px;'>visits &ge; 1 s only &middot; sub-second fly-throughs filtered out</span>"
        f"</div>"
        f"<div style='color:{_TEXT_2};font-size:11.5px;margin:0 4px 10px;line-height:1.5;'>"
        f"This table counts how long people <i>stay</i> in a zone &mdash; transient walk-throughs "
        f"are excluded so it reflects genuine engagement.</div>"
        f"<div style='border:1px solid {_BORDER};border-radius:10px;overflow:hidden;"
        f"background:{_BG_2};'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:{_FONT};'>"
        f"<thead><tr>"
        f"<th style='{th_s}'>Zone</th>"
        f"<th style='{th_s}'>Visits</th>"
        f"<th style='{th_s}'>Unique<span style='{th_sub}'>people / carts</span></th>"
        f"<th style='{th_s}'>Avg<span style='{th_sub}'>mean dwell</span></th>"
        f"<th style='{th_s}'>P50<span style='{th_sub}'>median — typical visit</span></th>"
        f"<th style='{th_s}'>P95<span style='{th_sub}'>95th pctile — long-tail</span></th>"
        f"<th style='{th_s}'>Max<span style='{th_sub}'>longest single visit</span></th>"
        f"<th style='{th_s}'>Total<span style='{th_sub}'>cumulative time</span></th>"
        f"</tr></thead><tbody>{summary_rows}</tbody></table></div>"
        f"<div style='color:{_TEXT_2};font-size:12.5px;margin-top:8px;padding:0 4px;line-height:1.6;'>"
        f"<b style='color:{_TEXT_1};'>P50</b> = median (half of visits were shorter). "
        f"<b style='color:{_TEXT_1};'>P95</b> = only 5% of visits lasted longer than this. "
        f"<b style='color:{_TEXT_1};'>Max</b> = the single longest individual visit recorded."
        f"</div></div>"
    )

    if not dwell_rows:
        return summary_table + _empty_state(
            "No individual visits met the dwell threshold yet.")

    # ---- Per-visit rows (truncated) ----
    sorted_rows = sorted(dwell_rows, key=lambda r: -r.dwell_seconds)[:max_visit_rows]
    visit_rows_html = ""
    for i, r in enumerate(sorted_rows):
        bg = _BG_1 if i % 2 == 0 else "transparent"
        prefix = "P" if r.track_label == "person" else "C"
        prefix_color = _ACCENT_HI if r.track_label == "person" else _WARN
        visit_rows_html += (
            f"<tr style='background:{bg};'>"
            f"<td style='{td_s}font-weight:700;color:{prefix_color};font-size:14.5px;'>"
            f"{prefix}{r.display_id}</td>"
            f"<td style='{td_s}color:{_TEXT_0};font-weight:600;'>{r.zone_name}</td>"
            f"<td style='{td_s}'>{r.visit_index}</td>"
            f"<td style='{td_s}'>{r.enter_t:.1f}s</td>"
            f"<td style='{td_s}'>{r.exit_t:.1f}s</td>"
            f"<td style='{td_s}font-weight:700;color:{_ACCENT_HI};font-size:14.5px;'>"
            f"{_fmt_dur(r.dwell_seconds)}</td>"
            f"</tr>"
        )

    visits_table = (
        f"<div style='font-family:{_FONT};margin-top:22px;'>"
        f"<div style='font-weight:700;color:{_TEXT_0};font-size:16px;"
        f"margin:4px 4px 12px;letter-spacing:0.01em;'>Top individual visits "
        f"<span style='color:{_TEXT_2};font-size:13px;font-weight:500;'>"
        f"longest first, up to {max_visit_rows}</span></div>"
        f"<div style='border:1px solid {_BORDER};border-radius:10px;overflow:hidden;"
        f"background:{_BG_2};'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:{_FONT};'>"
        f"<thead><tr>"
        f"<th style='{th_s}'>Track</th>"
        f"<th style='{th_s}'>Zone</th>"
        f"<th style='{th_s}'>Visit #</th>"
        f"<th style='{th_s}'>Entered</th>"
        f"<th style='{th_s}'>Exited</th>"
        f"<th style='{th_s}'>Dwell</th>"
        f"</tr></thead><tbody>{visit_rows_html}</tbody></table></div></div>"
    )

    return summary_table + visits_table


# ---------------------------------------------------------------------------
# Journey transition matrix
# ---------------------------------------------------------------------------
def _disp_label(label: str) -> str:
    return "Outside" if label == "__OUTSIDE__" else label


def build_journey_table(matrix: Optional[np.ndarray],
                        labels: list[str],
                        *, drop_outside_to_outside: bool = True) -> str:
    if matrix is None or matrix.size == 0 or not labels:
        return _empty_state(
            "Journey paths require at least one zone.",
            "Add a zone, then re-run analysis or click Recompute Analytics.",
        )

    n = matrix.shape[0]
    outside_idx = n - 1  # __OUTSIDE__ is always the last label

    # ---- Collect and rank all non-zero transitions ----
    transitions: list[tuple[int, str, str]] = []
    total_moves = 0
    for i in range(n):
        for j in range(n):
            if drop_outside_to_outside and i == outside_idx and j == outside_idx:
                continue
            v = int(matrix[i, j])
            if v > 0:
                transitions.append((v, _disp_label(labels[i]), _disp_label(labels[j])))
                total_moves += v
    transitions.sort(reverse=True)

    # ---- Ranked paths panel ----
    if not transitions:
        ranked_html = (
            f"<div style='color:{_TEXT_2};padding:16px;font-size:14px;"
            f"text-align:center;'>"
            f"No zone-to-zone transitions observed yet.</div>"
        )
    else:
        peak_count = transitions[0][0]
        rows_html = ""
        for idx, (count, src, dst) in enumerate(transitions):
            bar_pct = int(count / peak_count * 100)
            share_pct = count / total_moves * 100
            is_outside = src == "Outside" or dst == "Outside"
            bar_color = _CYAN if not is_outside else _TEXT_2
            arrow_color = _ACCENT_HI if not is_outside else _TEXT_2
            src_color = _TEXT_2 if src == "Outside" else _TEXT_0
            dst_color = _TEXT_2 if dst == "Outside" else _TEXT_0
            row_bg = _BG_1 if idx % 2 == 0 else "transparent"
            rows_html += (
                f"<div style='display:flex;align-items:center;gap:14px;"
                f"padding:11px 16px;border-bottom:1px solid {_BORDER};background:{row_bg};'>"
                f"<div style='min-width:120px;font-size:14.5px;color:{src_color};"
                f"font-weight:700;'>{src}</div>"
                f"<div style='font-size:18px;color:{arrow_color};font-weight:700;'>→</div>"
                f"<div style='min-width:120px;font-size:14.5px;color:{dst_color};"
                f"font-weight:700;'>{dst}</div>"
                # Bar
                f"<div style='flex:1;background:rgba(255,255,255,0.08);"
                f"border-radius:5px;height:9px;min-width:80px;overflow:hidden;'>"
                f"<div style='width:{bar_pct}%;background:{bar_color};height:100%;"
                f"border-radius:5px;box-shadow:0 0 8px {bar_color}55;'></div></div>"
                # Count + share
                f"<div style='min-width:48px;text-align:right;font-weight:700;"
                f"font-size:15px;color:{_TEXT_0};font-variant-numeric:tabular-nums;'>{count}×</div>"
                f"<div style='min-width:50px;text-align:right;font-size:13px;"
                f"color:{_TEXT_1};font-weight:600;font-variant-numeric:tabular-nums;'>"
                f"{share_pct:.0f}%</div>"
                f"</div>"
            )
        ranked_html = (
            f"<div style='border:1px solid {_BORDER};border-radius:10px;overflow:hidden;"
            f"background:{_BG_2};'>"
            f"<div style='display:flex;align-items:center;gap:14px;padding:11px 16px;"
            f"background:{_BG_3};border-bottom:1px solid {_BORDER_HARD};"
            f"font-size:12.5px;color:{_TEXT_0};font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;'>"
            f"<div style='min-width:120px;'>From</div>"
            f"<div style='width:18px;'></div>"
            f"<div style='min-width:120px;'>To</div>"
            f"<div style='flex:1;'>Frequency</div>"
            f"<div style='min-width:48px;text-align:right;'>Times</div>"
            f"<div style='min-width:50px;text-align:right;'>Share</div>"
            f"</div>"
            f"{rows_html}"
            f"</div>"
        )

    # ---- Compact matrix (secondary reference view) ----
    th_m = (f"padding:12px 14px;text-align:center;color:{_TEXT_0};font-weight:700;"
            f"font-size:12.5px;letter-spacing:0.08em;text-transform:uppercase;"
            f"background:{_BG_3};border-bottom:1px solid {_BORDER_HARD};")
    td_m = (f"padding:12px 14px;border-bottom:1px solid {_BORDER};color:{_TEXT_1};"
            f"font-size:15px;text-align:center;font-variant-numeric:tabular-nums;")
    lbl_m = (f"padding:12px 14px;border-bottom:1px solid {_BORDER};color:{_TEXT_0};"
             f"font-size:14px;font-weight:700;text-align:left;white-space:nowrap;"
             f"background:{_BG_1};")

    peak = max(int(matrix.max()), 1)

    def cell(v: int, is_diag_skip: bool) -> str:
        if is_diag_skip:
            return f"<td style='{td_m}color:{_TEXT_3};font-size:13px;'>—</td>"
        if v == 0:
            return f"<td style='{td_m}color:{_TEXT_3};font-size:13px;'>·</td>"
        intensity = v / peak
        # Heat ramp: bright cyan→accent-blue with strong alpha for visibility
        alpha = 0.30 + 0.55 * intensity
        # Mix cyan→blue based on intensity
        bg = f"rgba(107,155,255,{alpha:.2f})"
        return (f"<td style='{td_m}background:{bg};font-weight:700;"
                f"color:{_TEXT_0};font-size:16px;'>"
                f"{v}</td>")

    mat_header = "<tr>"
    mat_header += f"<th style='{th_m}text-align:left;'>↓ From / To →</th>"
    for lbl in labels:
        mat_header += f"<th style='{th_m}'>{_disp_label(lbl)}</th>"
    mat_header += "</tr>"

    mat_body = ""
    for i, row_lbl in enumerate(labels):
        mat_body += f"<tr><td style='{lbl_m}'>{_disp_label(row_lbl)}</td>"
        for j in range(n):
            skip = drop_outside_to_outside and i == outside_idx and j == outside_idx
            mat_body += cell(int(matrix[i, j]), skip)
        mat_body += "</tr>"

    matrix_html = (
        f"<div style='margin-top:22px;'>"
        f"<div style='font-weight:700;color:{_PURPLE_HI};font-size:13px;"
        f"margin-bottom:10px;letter-spacing:0.08em;text-transform:uppercase;'>"
        f"Full transition matrix</div>"
        f"<div style='overflow-x:auto;border:1px solid {_BORDER};border-radius:10px;"
        f"background:{_BG_2};'>"
        f"<table style='border-collapse:collapse;font-family:{_FONT};width:100%;'>"
        f"<thead>{mat_header}</thead><tbody>{mat_body}</tbody></table></div>"
        f"<div style='color:{_TEXT_2};font-size:12.5px;margin-top:8px;line-height:1.6;'>"
        f"Rows = origin &bull; Columns = destination &bull; "
        f"Brighter blue = more frequent &bull; Counts = number of transitions.</div>"
        f"</div>"
    )

    return (
        f"<div style='font-family:{_FONT};color:{_TEXT_1};'>"
        f"<div style='font-weight:700;color:{_TEXT_0};font-size:16px;"
        f"margin:4px 4px 4px;letter-spacing:0.01em;'>Zone-to-zone transitions"
        f"<span style='font-weight:600;color:{_ACCENT_HI};font-size:13px;"
        f"margin-left:12px;'>{total_moves} total moves</span>"
        f"<span style='font-weight:500;color:{_TEXT_2};font-size:12px;"
        f"margin-left:10px;'>every crossing counted, including fly-throughs</span></div>"
        f"<div style='color:{_TEXT_2};font-size:11.5px;margin:0 4px 14px;line-height:1.5;'>"
        f"This view counts <i>flow</i> &mdash; every time membership flips between zones "
        f"&mdash; so totals here can exceed the dwell-summary visits, which require "
        f"&ge; 1 s in-zone.</div>"
        f"{ranked_html}"
        f"{matrix_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Top-of-tab summary card
# ---------------------------------------------------------------------------
def build_analytics_summary(zones: list[Zone],
                            result: AnalyticsResult) -> str:
    n_zones  = len(zones)
    n_visits = len(result.dwell_rows)
    n_unique = len({(r.track_label, r.display_id) for r in result.dwell_rows})
    n_edges  = len(result.journey_edges)
    n_spikes = len(result.queue_spikes)

    chips = [
        (f"{n_zones}",  "Zones",       _ACCENT_HI, _ACCENT_TINT),
        (f"{n_visits}", "Visits",      _GOOD,      _GOOD_TINT),
        (f"{n_unique}", "Unique",      _CYAN,      _CYAN_TINT),
        (f"{n_edges}",  "Transitions", _PURPLE_HI, _PURPLE_TINT),
        (f"{n_spikes}", "Spikes",      _BAD,       _BAD_TINT),
    ]

    chips_html = (
        f"<div style='display:flex;flex-wrap:wrap;gap:10px;font-family:{_FONT};'>"
        + "".join(
            f"<div style='display:inline-flex;align-items:baseline;gap:8px;"
            f"background:{tint};border:1px solid {color};padding:8px 14px;"
            f"border-radius:999px;'>"
            f"<span style='color:{color};font-weight:700;font-size:18px;"
            f"font-variant-numeric:tabular-nums;'>{val}</span>"
            f"<span style='color:{_TEXT_1};font-size:11.5px;font-weight:700;"
            f"letter-spacing:0.08em;text-transform:uppercase;'>{label}</span>"
            f"</div>"
            for val, label, color, tint in chips
        )
        + f"</div>"
    )
    return chips_html + build_analytics_insight(result)


def build_analytics_insight(result: AnalyticsResult) -> str:
    """Narrative insight card — rendered below the chip strip.

    `result.insight_text` is a newline-separated bullet list (see
    `engine.analytics_builder.build_insight_text`); we render it as <ul>/<li>.
    """
    text = result.insight_text
    if not text:
        return ""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    items = "".join(
        f"<li style='margin:3px 0;'>{ln}</li>" for ln in lines
    )
    body = (
        f"<ul style='margin:8px 0 0 0;padding-left:1.25em;color:{_TEXT_0};"
        f"font-size:14.5px;'>{items}</ul>"
        if items
        else ""
    )
    return (
        f"<div style='background:{_ACCENT_TINT};border:1px solid {_ACCENT};"
        f"border-left-width:4px;border-radius:10px;padding:14px 18px;margin-top:12px;"
        f"font-family:{_FONT};font-size:14px;color:{_TEXT_0};line-height:1.65;'>"
        f"<span style='font-weight:700;letter-spacing:0.08em;color:{_ACCENT_HI};"
        f"font-size:11.5px;text-transform:uppercase;'>Insight</span>"
        f"{body}"
        f"</div>"
    )


def build_analytics_empty_state(has_video: bool) -> str:
    if not has_video:
        return _empty_state(
            "Upload a video to begin.",
            "Then draw zones in the Zone Editor tab and click Run Analysis.",
        )
    return _empty_state(
        "Run Analysis to populate this tab.",
        "Heatmap renders without zones; dwell, journeys and spikes need at least one zone.",
    )
