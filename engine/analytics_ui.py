# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
HTML builders for the Analytics tab.

Previously this module hardcoded its own dark palette (#1d2231 surfaces, pure
white text) while the page it renders into is light by default. Anything not
wrapped in one of its own dark cards — every section heading, the empty state,
the summary chip labels — came out white-on-white. Worst of all,
:func:`build_analytics_empty_state` is the tab's *default* value, so the app
shipped an invisible panel before a run had even happened.

Everything now goes through :mod:`engine.theme`, so these panels use the same
tokens as the rest of the app and follow the light/dark toggle.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import theme as T
from .analytics_models import (
    AnalyticsResult, DwellRow, JourneyEdge, QueueSpike, Zone,
)

_FONT = T.FONT

#: Spike severity -> shared severity vocabulary.
_SEVERITY_TONE = {
    "BACKED_UP":     "SAFETY",
    "QUEUE_FORMING": "ACTION",
    "WATCH":         "WATCH",
}


def _fmt_dur(seconds: float) -> str:
    return T.fmt_duration(seconds)


def _zone_swatch(color_bgr: tuple[int, int, int]) -> str:
    return T.swatch(color_bgr)


def _empty_state(headline: str, sub: str = "") -> str:
    return T.empty_state(headline, sub)


# ---------------------------------------------------------------------------
# Queue / dwell spikes
# ---------------------------------------------------------------------------
def build_queue_spikes_banner(spikes: list[QueueSpike]) -> str:
    """Compact congestion summary.

    This used to render a second full-width coloured banner announcing the
    same run the sticky top-of-page banner already announced. It is now an
    inline strip plus the per-zone detail, which is the part the sticky banner
    genuinely cannot show.
    """
    if not spikes:
        return (f"<div class='gk-inline-summary'>"
                f"{T.badge('No queue spikes detected', 'GOOD')}</div>")

    by_sev: dict[str, list[QueueSpike]] = {}
    for s in spikes:
        by_sev.setdefault(s.severity, []).append(s)

    n_zone = sum(1 for s in spikes if not str(s.zone_id).startswith("crowd_"))
    n_crowd = len(spikes) - n_zone
    bits = []
    if n_zone:
        bits.append(f"<b>{n_zone}</b> zone(s)")
    if n_crowd:
        bits.append(f"<b>{n_crowd}</b> crowd cluster(s)")
    order = ["BACKED_UP", "QUEUE_FORMING", "WATCH"]
    badges = [
        T.badge(f"{len(by_sev[sev])} {sev.replace('_', ' ').lower()}",
                _SEVERITY_TONE.get(sev, "NEUTRAL"))
        for sev in order if by_sev.get(sev)
    ]
    strip = (f"<div class='gk-inline-summary'>"
             f"<span>Queue / dwell spikes - {' + '.join(bits)}</span>"
             f"{''.join(badges)}</div>")

    blocks = []
    for sev in order:
        items = by_sev.get(sev, [])
        if not items:
            continue
        fg, bg = T.severity_colors(_SEVERITY_TONE.get(sev, "NEUTRAL"))
        lines = []
        for s in items:
            reason_html = ""
            if getattr(s, "reasons", None):
                reason_html = (
                    f"<div class='gk-dim' style='font-size:0.78rem;margin-top:4px;'>"
                    f"Why: {' &middot; '.join(T.esc(r) for r in s.reasons)} "
                    f"<span style='color:{T.INK_3};'>"
                    f"(score {getattr(s, 'score', 0):.0f}/100)</span></div>"
                )
            lines.append(
                f"<div style='font-size:0.85rem;margin-top:6px;color:{T.INK};'>"
                f"<b>{T.esc(s.zone_name)}</b> - avg {_fmt_dur(s.avg_dwell_s)}, "
                f"max {_fmt_dur(s.max_dwell_s)} across {s.n_visits} visit(s) "
                f"<span class='gk-dim'>(threshold {_fmt_dur(s.threshold_s)})"
                f"</span>{reason_html}</div>"
            )
        blocks.append(
            f"<div class='gk-notice' style='background:{bg};"
            f"border-left-color:{fg};'>"
            f"<div class='gk-notice-head' style='color:{fg};'>"
            f"{sev.replace('_', ' ')}</div>{''.join(lines)}</div>"
        )
    return strip + "".join(blocks)


# ---------------------------------------------------------------------------
# Dwell tables
# ---------------------------------------------------------------------------
def build_dwell_table(zones: list[Zone],
                      dwell_summary: list[dict],
                      dwell_rows: list[DwellRow],
                      *, max_visit_rows: int = 60) -> str:
    if not zones:
        return T.empty_state(
            "Draw a zone in the Zone Editor tab to see store-flow analytics.",
            "The traffic heatmap below works without zones.")
    if not dwell_summary:
        return T.empty_state("No dwell data computed.")

    color_by_zone = {z.zone_id: z.color for z in zones}

    summary_cols = [
        T.col("Zone", sort="text"),
        T.col("Visits", sort="num", align="right", width="90px"),
        T.col("Unique", sort="num", align="right", width="90px",
              sub="people / carts"),
        T.col("Avg", sort="num", align="right", width="90px", sub="mean dwell"),
        T.col("P50", sort="num", align="right", width="90px",
              sub="median - typical visit"),
        T.col("P95", sort="num", align="right", width="90px",
              sub="95th pctile - long tail"),
        T.col("Max", sort="num", align="right", width="90px",
              sub="longest single visit"),
        T.col("Total", sort="num", align="right", width="95px",
              sub="cumulative time"),
    ]
    summary_rows = []
    for s in dwell_summary:
        sw = _zone_swatch(color_by_zone.get(s["zone_id"], (200, 200, 200)))
        summary_rows.append(T.row([
            T.cell(f"{sw}<span class='gk-strong'>{T.esc(s['zone_name'])}</span>"
                   f"<span class='gk-dim' style='font-size:0.68rem;font-weight:700;"
                   f"margin-left:10px;text-transform:uppercase;letter-spacing:0.06em;'>"
                   f"{T.esc(s['applies_to'])}</span>",
                   sort_value=s["zone_name"]),
            T.cell(f"<b>{s['n_visits']}</b>", sort_value=s["n_visits"],
                   css_class="gk-num"),
            T.cell(f"<b>{s['n_unique']}</b>", sort_value=s["n_unique"],
                   css_class="gk-num"),
            T.cell(f"<b style='color:{T.ACCENT};'>{_fmt_dur(s['avg_dwell_s'])}</b>",
                   sort_value=s["avg_dwell_s"], css_class="gk-num"),
            T.cell(_fmt_dur(s["p50_s"]), sort_value=s["p50_s"], css_class="gk-num"),
            T.cell(_fmt_dur(s["p95_s"]), sort_value=s["p95_s"], css_class="gk-num"),
            T.cell(_fmt_dur(s["max_s"]), sort_value=s["max_s"], css_class="gk-num"),
            T.cell(_fmt_dur(s["total_s"]), sort_value=s["total_s"], css_class="gk-num"),
        ]))

    summary_note = (
        "Counts how long people <i>stay</i> in a zone - visits under 1 s are "
        "filtered out so this reflects genuine engagement rather than "
        "walk-throughs. <b>P50</b> = median (half of visits were shorter). "
        "<b>P95</b> = only 5% lasted longer. <b>Max</b> = the single longest "
        "visit recorded."
    )
    summary_table = T.render_table(
        summary_cols, summary_rows,
        title="Per-zone dwell summary",
        subtitle="visits ≥ 1 s only",
        note=summary_note,
        table_id="gk-dwell-summary",
        csv_name="dwell_summary.csv",
        filterable=False,
        visible_rows=None,
    )

    if not dwell_rows:
        return summary_table + T.empty_state(
            "No individual visits met the dwell threshold yet.")

    visit_cols = [
        T.col("Track", sort="text", width="90px"),
        T.col("Zone", sort="text"),
        T.col("Visit #", sort="num", align="right", width="80px"),
        T.col("Entered", sort="num", align="right", width="95px"),
        T.col("Exited", sort="num", align="right", width="95px"),
        T.col("Dwell", sort="num", align="right", width="100px"),
    ]
    sorted_rows = sorted(dwell_rows, key=lambda r: -r.dwell_seconds)[:max_visit_rows]
    visit_rows = []
    for r in sorted_rows:
        prefix = "P" if r.track_label == "person" else "C"
        prefix_color = T.ACCENT if r.track_label == "person" else T.WARN
        visit_rows.append(T.row([
            T.cell(f"<b style='color:{prefix_color};'>{prefix}{r.display_id}</b>",
                   sort_value=f"{prefix}{r.display_id:05d}"),
            T.cell(T.esc(r.zone_name), sort_value=r.zone_name),
            T.cell(str(r.visit_index), sort_value=r.visit_index, css_class="gk-num"),
            T.cell(f"{r.enter_t:.1f}s", sort_value=r.enter_t, css_class="gk-num"),
            T.cell(f"{r.exit_t:.1f}s", sort_value=r.exit_t, css_class="gk-num"),
            T.cell(f"<b style='color:{T.ACCENT};'>{_fmt_dur(r.dwell_seconds)}</b>",
                   sort_value=r.dwell_seconds, css_class="gk-num"),
        ], seek_t=r.enter_t))

    visits_table = T.render_table(
        visit_cols, visit_rows,
        title="Top individual visits",
        subtitle=f"longest first, up to {max_visit_rows} · "
                 f"click a row to jump the video there",
        table_id="gk-dwell-visits",
        csv_name="dwell_visits.csv",
        visible_rows=None,
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
        return T.empty_state(
            "Journey paths require at least one zone.",
            "Add a zone, then re-run analysis or click Recompute Analytics.")

    n = matrix.shape[0]
    outside_idx = n - 1                      # __OUTSIDE__ is always last

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

    header = T.section_title(
        "Zone-to-zone transitions",
        f"{total_moves} total moves · every crossing counted",
        "Counts <i>flow</i> - every time membership flips between zones - so "
        "totals here can exceed the dwell-summary visits, which require ≥ 1 s "
        "in-zone.",
    )

    # ---- Ranked paths ----
    if not transitions:
        ranked_html = T.empty_state("No zone-to-zone transitions observed yet.")
    else:
        peak_count = transitions[0][0]
        rows = []
        for count, src, dst in transitions:
            is_outside = src == "Outside" or dst == "Outside"
            bar_color = T.CYAN if not is_outside else T.INK_3
            rows.append(T.row([
                T.cell(f"<b>{T.esc(src)}</b>", sort_value=src),
                T.cell(f"<span style='color:{T.ACCENT};font-weight:700;'>→</span>",
                       sort_value=None),
                T.cell(f"<b>{T.esc(dst)}</b>", sort_value=dst),
                T.cell(T.bar_meter(count / peak_count * 100, bar_color),
                       sort_value=count),
                T.cell(f"<b>{count}×</b>", sort_value=count, css_class="gk-num"),
                T.cell(f"{count / total_moves * 100:.0f}%",
                       sort_value=count / total_moves, css_class="gk-num"),
            ]))
        ranked_html = T.render_table(
            [T.col("From", sort="text", width="150px"),
             T.col("", sort=None, width="30px", align="center"),
             T.col("To", sort="text", width="150px"),
             T.col("Frequency", sort="num"),
             T.col("Times", sort="num", align="right", width="80px"),
             T.col("Share", sort="num", align="right", width="80px")],
            rows,
            table_id="gk-journey-ranked",
            csv_name="zone_transitions.csv",
            visible_rows=40,
        )

    # ---- Full matrix (secondary reference view) ----
    peak = max(int(matrix.max()), 1)

    def cell(v: int, is_diag_skip: bool) -> str:
        base = ("padding:10px 13px;border-bottom:1px solid " + T.BORDER
                + ";text-align:center;font-variant-numeric:tabular-nums;")
        if is_diag_skip:
            return f"<td style='{base}color:{T.INK_3};'>n/a</td>"
        if v == 0:
            return f"<td style='{base}color:{T.INK_3};'>·</td>"
        alpha = 0.18 + 0.55 * (v / peak)
        return (f"<td style='{base}background:rgba(37,99,235,{alpha:.2f});"
                f"font-weight:700;color:{T.INK};'>{v}</td>")

    th = ("padding:10px 13px;text-align:center;font-weight:700;font-size:0.7rem;"
          f"letter-spacing:0.06em;text-transform:uppercase;background:{T.CARD_2};"
          f"color:{T.INK};border-bottom:1px solid {T.BORDER_HARD};")
    lbl = (f"padding:10px 13px;border-bottom:1px solid {T.BORDER};color:{T.INK};"
           f"font-size:0.83rem;font-weight:700;text-align:left;white-space:nowrap;"
           f"background:{T.CARD_2};")

    mat_header = f"<tr><th style='{th}text-align:left;'>↓ From / To →</th>"
    for l in labels:
        mat_header += f"<th style='{th}'>{T.esc(_disp_label(l))}</th>"
    mat_header += "</tr>"

    mat_body = ""
    for i, row_lbl in enumerate(labels):
        mat_body += f"<tr><td style='{lbl}'>{T.esc(_disp_label(row_lbl))}</td>"
        for j in range(n):
            skip = drop_outside_to_outside and i == outside_idx and j == outside_idx
            mat_body += cell(int(matrix[i, j]), skip)
        mat_body += "</tr>"

    matrix_html = (
        f"<div class='gk-table-wrap'>"
        f"<div class='gk-table-toolbar'>"
        f"<div class='gk-section-title'>Full transition matrix"
        f"<span class='gk-section-sub'>rows = origin · columns = destination · "
        f"brighter = more frequent</span></div></div>"
        f"<div class='gk-table-scroll'>"
        f"<table class='gk-table' style='width:100%;'>"
        f"<thead>{mat_header}</thead><tbody>{mat_body}</tbody></table>"
        f"</div></div>"
    )

    return f"<div>{header}{ranked_html}{matrix_html}</div>"


# ---------------------------------------------------------------------------
# Top-of-tab summary
# ---------------------------------------------------------------------------
def build_analytics_summary(zones: list[Zone],
                            result: AnalyticsResult) -> str:
    n_zones  = len(zones)
    n_visits = len(result.dwell_rows)
    n_unique = len({(r.track_label, r.display_id) for r in result.dwell_rows})
    n_edges  = len(result.journey_edges)
    n_spikes = len(result.queue_spikes)

    chips = "".join([
        T.stat_chip(n_zones,  "Zones",       "INFO"),
        T.stat_chip(n_visits, "Visits",      "GOOD"),
        T.stat_chip(n_unique, "Unique",      "INFO"),
        T.stat_chip(n_edges,  "Transitions", "NEUTRAL"),
        T.stat_chip(n_spikes, "Spikes",      "SAFETY" if n_spikes else "GOOD"),
    ])
    return (f"<div class='gk-stat-row'>{chips}</div>"
            + build_analytics_insight(result))


def build_analytics_insight(result: AnalyticsResult) -> str:
    """Narrative insight card rendered below the chip strip.

    ``result.insight_text`` is a newline-separated bullet list (see
    ``engine.analytics_builder.build_insight_text``).
    """
    text = result.insight_text
    if not text:
        return ""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    items = "".join(f"<li style='margin:3px 0;'>{ln}</li>" for ln in lines)
    body = (f"<ul style='margin:8px 0 0 0;padding-left:1.25em;color:{T.INK};"
            f"font-size:0.88rem;line-height:1.6;'>{items}</ul>")
    return T.notice("Insight", body, tone="INFO")


def build_analytics_empty_state(has_video: bool) -> str:
    if not has_video:
        return T.empty_state(
            "Upload a video to begin.",
            "Then draw zones in the Zone Editor tab and click Run Analysis.")
    return T.empty_state(
        "Run Analysis to populate this tab.",
        "The heatmap renders without zones; dwell, journeys and spikes need "
        "at least one zone.")
