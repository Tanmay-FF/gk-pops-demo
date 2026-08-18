# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Shared UI design tokens and HTML primitives.

Every HTML panel rendered into the Gradio page (POPS, Events, Operational
Alerts, Analytics, Case Report) goes through this module so there is exactly
ONE colour system instead of three.

Why CSS variables rather than literals
--------------------------------------
The page defines ``--gk-*`` custom properties in two palettes (light + dark)
and a toggle swaps between them. A builder that emits ``#ffffff`` cannot
follow that toggle, which is how the Analytics tab ended up rendering
white-on-white in light mode. Every token below is emitted as
``var(--gk-name, <literal>)`` so:

  * inside the app the live theme wins, and the toggle reaches the panels;
  * in a standalone export (the downloadable case report has no Gradio page
    around it) the literal fallback keeps the document readable.

Anything rendered inside an ``<iframe srcdoc>`` — the 3D / 2D BEV and Floor
Map documents — is deliberately NOT converted: an iframe does not inherit the
parent page's custom properties, so those builders keep their own literals.
"""
from __future__ import annotations

import html as _html
import itertools
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
FONT = "'Inter', 'Nunito Sans', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"


def _v(name: str, fallback: str) -> str:
    """A themed colour: live variable, with a literal fallback for exports."""
    return f"var(--gk-{name}, {fallback})"


# Surfaces
BG          = _v("bg",            "#f1f5f9")
CARD        = _v("card",          "#ffffff")
CARD_2      = _v("card-2",        "#f8fafc")
ROW_ALT     = _v("row-alt",       "#f8fafc")
BORDER      = _v("border",        "#e2e8f0")
BORDER_HARD = _v("border-strong", "#cbd5e1")
SHADOW      = _v("shadow",        "0 1px 2px rgba(15,23,42,0.04), 0 4px 16px rgba(15,23,42,0.05)")

# Text
INK   = _v("ink",   "#0f172a")
INK_2 = _v("ink-2", "#475569")
INK_3 = _v("ink-3", "#94a3b8")

# Semantic pairs — foreground + matching tinted background
ACCENT      = _v("accent",     "#2563eb")
ACCENT_HI   = _v("accent-hi",  "#1d4ed8")
ACCENT_BG   = _v("accent-bg",  "#eff6ff")

DANGER      = _v("danger",     "#dc2626")
DANGER_BG   = _v("danger-bg",  "#fef2f2")

WARN        = _v("warn",       "#ea580c")
WARN_BG     = _v("warn-bg",    "#fff7ed")

CAUTION     = _v("caution",    "#a16207")
CAUTION_BG  = _v("caution-bg", "#fefce8")

GOOD        = _v("good",       "#059669")
GOOD_BG     = _v("good-bg",    "#ecfdf5")

INFO        = _v("info",       "#2563eb")
INFO_BG     = _v("info-bg",    "#eff6ff")

PURPLE      = _v("purple",     "#7c3aed")
PURPLE_BG   = _v("purple-bg",  "#f5f3ff")

CYAN        = _v("cyan",       "#0891b2")
CYAN_BG     = _v("cyan-bg",    "#ecfeff")

NEUTRAL     = _v("neutral",    "#475569")
NEUTRAL_BG  = _v("neutral-bg", "#f1f5f9")

RADIUS = "var(--gk-radius, 10px)"

#: severity name -> (foreground token, tint token). Shared by the operational
#: rule table, the alert banner and the POPS event badges so one vocabulary
#: renders identically everywhere.
SEVERITY: dict[str, tuple[str, str]] = {
    "SAFETY":  (DANGER,  DANGER_BG),
    "ACTION":  (WARN,    WARN_BG),
    "WATCH":   (CAUTION, CAUTION_BG),
    "INFO":    (INFO,    INFO_BG),
    "DANGER":  (DANGER,  DANGER_BG),
    "WARN":    (WARN,    WARN_BG),
    "GOOD":    (GOOD,    GOOD_BG),
    "NEUTRAL": (NEUTRAL, NEUTRAL_BG),
}


def severity_colors(name: str) -> tuple[str, str]:
    return SEVERITY.get(str(name).upper(), (NEUTRAL, NEUTRAL_BG))


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------
def esc(value: Any) -> str:
    """Escape a value for safe interpolation into HTML text/attributes."""
    return _html.escape("" if value is None else str(value), quote=True)


def badge(text: str, severity: str = "NEUTRAL", *, solid: bool = True) -> str:
    fg, bg = severity_colors(severity)
    if solid:
        return (f"<span class='gk-badge' style='background:{fg};color:#fff;'>"
                f"{esc(text)}</span>")
    return (f"<span class='gk-badge' style='background:{bg};color:{fg};"
            f"box-shadow:inset 0 0 0 1px {fg};'>{esc(text)}</span>")


def swatch(color_bgr: Sequence[int]) -> str:
    """Colour chip from an OpenCV BGR triple."""
    b, g, r = color_bgr
    return (f"<span class='gk-swatch' style='background:rgb({int(r)},{int(g)},"
            f"{int(b)});'></span>")


def empty_state(headline: str, sub: str = "", *, tone: str = "NEUTRAL",
                icon: str = "") -> str:
    """The single empty/placeholder idiom for every panel.

    Previously each tab invented its own ("ALL CLEAR" text, a green card, a
    bare grey <p>, a centred div), so "nothing here" looked like four
    different application states.
    """
    fg, bg = severity_colors(tone)
    sub_html = f"<div class='gk-empty-sub'>{sub}</div>" if sub else ""
    icon_html = (f"<div class='gk-empty-icon' style='color:{fg};'>{icon}</div>"
                 if icon else "")
    toned = tone.upper() != "NEUTRAL"
    style = f"background:{bg};border-color:{fg};" if toned else ""
    head_color = fg if toned else INK
    return (
        f"<div class='gk-empty' style='{style}'>{icon_html}"
        f"<div class='gk-empty-head' style='color:{head_color};'>"
        f"{headline}</div>{sub_html}</div>"
    )


def notice(headline: str, body: str = "", *, tone: str = "INFO") -> str:
    """Inline callout — the 'rules did not run' / 'all clear' style card."""
    fg, bg = severity_colors(tone)
    body_html = f"<div class='gk-notice-body'>{body}</div>" if body else ""
    return (
        f"<div class='gk-notice' style='background:{bg};border-left-color:{fg};'>"
        f"<div class='gk-notice-head' style='color:{fg};'>{headline}</div>"
        f"{body_html}</div>"
    )


def section_title(title: str, sub: str = "", note: str = "") -> str:
    sub_html = f"<span class='gk-section-sub'>{sub}</span>" if sub else ""
    note_html = f"<div class='gk-section-note'>{note}</div>" if note else ""
    return (f"<div class='gk-section-title'>{title}{sub_html}</div>{note_html}")


def stat_chip(value: Any, label: str, tone: str = "ACCENT") -> str:
    fg, bg = severity_colors(tone) if tone.upper() in SEVERITY else (ACCENT, ACCENT_BG)
    return (f"<div class='gk-stat-chip' style='background:{bg};border-color:{fg};'>"
            f"<span class='gk-stat-val' style='color:{fg};'>{esc(value)}</span>"
            f"<span class='gk-stat-lbl'>{esc(label)}</span></div>")


def fmt_duration(seconds: float) -> str:
    if seconds is None or seconds <= 0:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}m{int(s):02d}s"


def fmt_mmss(t: float) -> str:
    t = max(0.0, float(t or 0.0))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------
#: rows past this are not emitted into the DOM at all. A 10-minute clip can
#: log thousands of events; pasting all of them into a gr.HTML string is what
#: made the Events tab crawl. The footer says so explicitly rather than
#: silently truncating.
HARD_ROW_CAP = 2000

_table_seq = itertools.count(1)


def col(label: str, *, sort: Optional[str] = "text", sub: str = "",
        align: str = "left", width: str = "") -> dict:
    """Column spec. ``sort`` is "text" | "num" | None (not sortable)."""
    return {"label": label, "sort": sort, "sub": sub, "align": align,
            "width": width}


def cell(html: str, *, sort_value: Any = None, style: str = "",
         css_class: str = "") -> dict:
    """One table cell. ``sort_value`` drives client-side sorting when the
    displayed text is not directly comparable (e.g. "1m20s", "≥ 45s")."""
    return {"html": html, "sort_value": sort_value, "style": style,
            "class": css_class}


def row(cells: Sequence[dict], *, tone: Optional[str] = None,
        seek_t: Optional[float] = None, css_class: str = "") -> dict:
    """One table row.

    ``seek_t`` makes the row clickable and seeks the tracked-output video to
    that timestamp — the reason an operator can read "PUSHOUT at 47s" and land
    on the frame instead of scrubbing for it.
    """
    return {"cells": list(cells), "tone": tone, "seek_t": seek_t,
            "class": css_class}


def render_table(columns: Sequence[dict],
                 rows: Sequence[dict],
                 *,
                 title: str = "",
                 subtitle: str = "",
                 note: str = "",
                 table_id: Optional[str] = None,
                 csv_name: str = "export.csv",
                 sortable: bool = True,
                 filterable: bool = True,
                 exportable: bool = True,
                 visible_rows: Optional[int] = 250,
                 empty: Optional[str] = None) -> str:
    """Render the one table style used across the app.

    Behaviour comes from classes + delegated JS in ``static/app.js``: column
    sort, live text filter, CSV export, sticky header, and "show all" for
    capped row sets. Every table gets them by construction rather than each
    builder hand-rolling a <table> string.
    """
    if not rows:
        return empty if empty is not None else empty_state("No rows.")

    tid = table_id or f"gk-tbl-{next(_table_seq)}"
    total = len(rows)
    rendered = list(rows[:HARD_ROW_CAP])
    hard_capped = total > HARD_ROW_CAP
    n_visible = len(rendered) if not visible_rows else min(visible_rows, len(rendered))

    # ---- head ----
    ths = []
    for i, c in enumerate(columns):
        sort_attr = (f" data-sort='{c.get('sort')}' tabindex='0' role='button'"
                     if sortable and c.get("sort") else "")
        sub = (f"<span class='gk-th-sub'>{c['sub']}</span>" if c.get("sub") else "")
        width = f"width:{c['width']};" if c.get("width") else ""
        ind = ("<span class='gk-sort-ind' aria-hidden='true'></span>"
               if sortable and c.get("sort") else "")
        ths.append(
            f"<th data-col='{i}'{sort_attr} style='text-align:{c.get('align','left')};"
            f"{width}'>{c['label']}{ind}{sub}</th>"
        )
    thead = f"<thead><tr>{''.join(ths)}</tr></thead>"

    # ---- body ----
    trs = []
    for ri, r in enumerate(rendered):
        classes = ["gk-tr"]
        if r.get("class"):
            classes.append(r["class"])
        if r.get("seek_t") is not None:
            classes.append("gk-seekable")
        if visible_rows and ri >= visible_rows:
            classes.append("gk-row-hidden")
        style = ""
        if r.get("tone"):
            _fg, bg = severity_colors(r["tone"])
            style = f"background:{bg};"
        seek = ""
        if r.get("seek_t") is not None:
            seek = (f" data-gk-seek='{float(r['seek_t']):.3f}'"
                    f" tabindex='0' role='button'"
                    f" title='Jump the video to this moment'")
        tds = []
        for ci, cl in enumerate(r["cells"]):
            align = columns[ci].get("align", "left") if ci < len(columns) else "left"
            dv = ("" if cl.get("sort_value") is None
                  else f" data-v=\"{esc(cl['sort_value'])}\"")
            cls = f" class='{cl['class']}'" if cl.get("class") else ""
            tds.append(f"<td{cls}{dv} style='text-align:{align};{cl.get('style','')}'>"
                       f"{cl['html']}</td>")
        trs.append(f"<tr class='{' '.join(classes)}' style='{style}'{seek}>"
                   f"{''.join(tds)}</tr>")
    tbody = f"<tbody>{''.join(trs)}</tbody>"

    # ---- toolbar ----
    actions = []
    if filterable:
        actions.append(
            f"<input type='search' class='gk-table-filter' data-target='{tid}' "
            f"placeholder='Filter rows…' aria-label='Filter table rows'>")
    if exportable:
        actions.append(
            f"<button type='button' class='gk-btn-mini gk-table-export' "
            f"data-target='{tid}' data-name='{esc(csv_name)}'>Export CSV</button>")
    toolbar = ""
    if title or actions:
        title_html = ""
        if title:
            sub = f"<span class='gk-section-sub'>{subtitle}</span>" if subtitle else ""
            title_html = f"<div class='gk-section-title'>{title}{sub}</div>"
        toolbar = (f"<div class='gk-table-toolbar'>{title_html}"
                   f"<div class='gk-table-actions'>{''.join(actions)}</div></div>")

    # ---- footer ----
    foot_bits = []
    if n_visible < len(rendered):
        foot_bits.append(
            f"<span class='gk-table-count' data-total='{len(rendered)}'>"
            f"Showing {n_visible:,} of {len(rendered):,} rows</span>"
            f"<button type='button' class='gk-btn-mini gk-table-showall' "
            f"data-target='{tid}'>Show all</button>")
    elif total > 1:
        foot_bits.append(f"<span class='gk-table-count'>{total:,} rows</span>")
    if hard_capped:
        foot_bits.append(
            f"<span class='gk-table-warn'>Capped at {HARD_ROW_CAP:,} of "
            f"{total:,} - the full set is in the tracking JSON download.</span>")
    if note:
        foot_bits.append(f"<span class='gk-table-note'>{note}</span>")
    footer = (f"<div class='gk-table-foot'>{''.join(foot_bits)}</div>"
              if foot_bits else "")

    return (
        f"<div class='gk-table-wrap'>{toolbar}"
        f"<div class='gk-table-scroll'>"
        f"<table class='gk-table' id='{tid}'>{thead}{tbody}</table>"
        f"</div>{footer}</div>"
    )


def bar_meter(pct: float, color: str = ACCENT) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return (f"<div class='gk-meter'><div class='gk-meter-fill' "
            f"style='width:{pct:.1f}%;background:{color};'></div></div>")
