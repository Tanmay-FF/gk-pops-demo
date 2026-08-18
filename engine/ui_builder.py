# Built by Tanmay Thaker | MLE, Gatekeeper Systems <tthaker@gatekeepersystems.com>
"""
HTML generation for the Gradio UI — POPS summary, events timeline,
operational alerts, info tables, the sticky alert banner and the run summary.

All colours come from :mod:`engine.theme` (CSS variables with literal
fallbacks) rather than hardcoded hex, so the panels follow the light/dark
toggle instead of fighting it. Tables are built by :func:`theme.render_table`,
which gives every one of them sorting, filtering, CSV export, a sticky header
and honest row caps without each builder re-implementing them.
"""
import os
import re

from . import theme as T
# highlights is stdlib-only and imports nothing from this package (see its
# docstring), so this cannot create a cycle. Shared so the panel, the case
# report and the JSON render one identical set of coverage notes.
from . import highlights
from .scoring import HIGH_EVENTS, MEDIUM_EVENTS

_FONT = T.FONT

# Literal hex, deliberately: engine/case_report_builder.py imports this map to
# style the STANDALONE downloadable report, which is opened outside the app
# where no --gk-* variables exist.
_EVENT_BADGE = {
    "PUSHOUT ALERT": "#b71c1c",
    "HIGH PRIORITY": "#c62828",
    "MEDIUM PRIORITY": "#e65100",
    "UNLINKED EXIT": "#ef6c00",
    "ABANDONED CART": "#b71c1c",
    "MONITORING": "#1565c0",
    "INBOUND": "#2e7d32",
    "INCOMING CART WITHOUT ITEMS": "#1565c0",
    "LOW PRIORITY": "#2e7d32",
}

#: An empty cart moving inward. Named as an operational category rather than a
#: direction, because "INBOUND" answers which way the cart went and leaves the
#: useful part — that it came in with no merchandise — for the reader to infer
#: from a separate column.
#:
#: DISPLAY ONLY. This is not a POPS event and is deliberately not one: event
#: names drive FrameCapturer, which JPEG-encodes a frame for every new event
#: name and hands it to the VLM, so minting one here would push every ordinary
#: shopper entry into the case report. compute_pops() is untouched, the score
#: is untouched, the event log is untouched — only the word in the cell changes.
#:
#: Same string as rules.RULE_LABELS["incoming_empty"] on purpose: the rule
#: engine reaches this conclusion from its own evidence, and one situation
#: should not have two names in one app.
INCOMING_NO_ITEMS = "INCOMING CART WITHOUT ITEMS"

#: Events that must never be masked by the label above. An inbound cart is
#: score-floored to 5 so it normally reads INBOUND, but a peak captured while
#: the cart was flagged for risk keeps that name — theft severity outranks a
#: traffic category.
_NEVER_MASK = frozenset({
    "PUSHOUT ALERT", "HIGH PRIORITY", "MEDIUM PRIORITY",
    "ABANDONED CART", "UNLINKED EXIT",
})

#: POPS event -> shared severity vocabulary. One mapping drives the badge
#: colour, the row tint and the alert banner, so an event cannot read as
#: "high" in one panel and "medium" in another.
_EVENT_SEVERITY = {
    "PUSHOUT ALERT":   "SAFETY",
    "HIGH PRIORITY":   "SAFETY",
    "ABANDONED CART":  "SAFETY",
    "MEDIUM PRIORITY": "ACTION",
    "UNLINKED EXIT":   "ACTION",
    "MONITORING":      "INFO",
    "INBOUND":         "GOOD",
    # INFO, matching RULE_SEVERITIES["incoming_empty"] — the same colour the
    # operational chip for this category uses, so one situation reads the same
    # whichever column it appears in.
    "INCOMING CART WITHOUT ITEMS": "INFO",
    "LOW PRIORITY":    "GOOD",
}
#: Events that tint their whole row rather than just carrying a badge.
_ROW_TONE = {
    "PUSHOUT ALERT": "SAFETY", "HIGH PRIORITY": "SAFETY",
    "ABANDONED CART": "SAFETY", "MEDIUM PRIORITY": "WATCH",
    "UNLINKED EXIT": "ACTION",
}


#: Shared severity ordering. Used to pick the worst label on a row and to sort
#: a cart's flags worst-first — the same ranking rules.py sorts findings by.
_SEV_RANK = {"SAFETY": 4, "ACTION": 3, "WATCH": 2, "INFO": 1, "GOOD": 0,
             "NEUTRAL": 0}


def _event_severity(event_name: str) -> str:
    return _EVENT_SEVERITY.get(event_name, "NEUTRAL")


def resolve_event_label(peak: dict, event_name: str) -> str:
    """The POPS event refined by what the peak snapshot already recorded.

    Reads `direction` and `fill` off the snapshot rather than inferring from
    the event name. classify_event() never sees the fill — it takes score,
    linked, direction and abandoned — so "INBOUND" is the most it can say, and
    an empty cart entering is indistinguishable there from a full one.

    Both fields are frozen at the cart's peak-score frame by the same writer,
    so they describe one instant and cannot be mismatched.
    """
    if event_name in _NEVER_MASK:
        return event_name
    if str((peak or {}).get("direction", "")).strip().upper() != "INBOUND":
        return event_name
    if str((peak or {}).get("fill", "")).strip().lower() != "empty":
        return event_name
    return INCOMING_NO_ITEMS


def _worst_tone(*tones):
    """The highest-severity tone given, or None when none of them tint a row."""
    real = [t for t in tones if t]
    if not real:
        return None
    return max(real, key=lambda t: _SEV_RANK.get(str(t).upper(), 0))


def _badge(event_name: str) -> str:
    """POPS event badge. Kept for engine.case_report_builder, which renders
    into a standalone document and needs literal colours."""
    ec = _EVENT_BADGE.get(event_name, "#546e7a")
    return (f"<span style='background:{ec};color:#fff;padding:4px 12px;"
            f"border-radius:5px;font-size:0.8rem;font-weight:800;"
            f"font-family:{_FONT};letter-spacing:0.3px;'>{event_name}</span>")


def _badge_themed(event_name: str) -> str:
    """In-app badge — follows the theme."""
    return T.badge(event_name, _event_severity(event_name))


def _mmss(t: float) -> str:
    return T.fmt_mmss(t)


# ---------------------------------------------------------------------------
# Sticky top-of-page alert banner
# ---------------------------------------------------------------------------
#: The banner is the ONE canonical alert surface. Beyond this many chips it
#: rolls up into "+N more" — an unbounded chip list pushed the whole dashboard
#: below the fold on a busy clip.
_MAX_BANNER_CHIPS = 4


def _alert_chip(title: str, detail: str, *, tab: str = "") -> str:
    """One banner chip. When a destination tab is given the chip becomes a
    button that navigates there — the banner used to say "review the flagged
    carts below" while nothing below was actually a target."""
    inner = (f"<div class='gk-alert-chip-t'>{T.esc(title)}</div>"
             f"<div class='gk-alert-chip-d'>{T.esc(detail)}</div>")
    if tab:
        return (f"<button type='button' class='gk-alert-chip' "
                f"data-gk-tab='{T.esc(tab)}' "
                f"title='Open the {T.esc(tab)} tab'>{inner}</button>")
    return f"<div class='gk-alert-chip'>{inner}</div>"


def _names(items, fmt, limit=3):
    txt = ", ".join(fmt(i) for i in items[:limit])
    if len(items) > limit:
        txt += f" +{len(items) - limit} more"
    return txt


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _headline_noun(label: str) -> str:
    """Rule label -> prose noun. "UNATTENDED CART (OPS)" is a category name
    built to be unmistakable in a table; inside a sentence the parenthetical
    reads as noise and blocks pluralisation."""
    return re.sub(r"\s*\([^)]*\)", "", label).strip().lower() or label.lower()


def build_alert_banner(event_log, queue_spikes, spike_events=None,
                       rule_findings=None) -> str:
    """Sticky notification surfacing HIGH POPS events, severe zone congestion
    and operational rule findings. Returns "" when nothing is worth flagging —
    the caller hides the component in that case.

    Args:
        event_log:    per-frame logged events (HIGH_EVENTS get flagged).
        queue_spikes: list[QueueSpike] from analytics_builder — zone-bound.
        spike_events: list[dict] — same data, but may include
                      ``source: "crowd_cluster"`` rows not tied to a zone.
        rule_findings: list[RuleFinding]. These carry their OWN severity
                      vocabulary (INFO/WATCH/ACTION/SAFETY) rather than
                      SpikeSeverity, so they need their own filter — reusing
                      the congestion filter would drop every one silently.
                      Only SAFETY and ACTION reach the banner; WATCH/INFO live
                      in the operational-alerts table.
    """
    high_events = [e for e in (event_log or []) if e.get("event") in HIGH_EVENTS]
    severe_spikes = [s for s in (queue_spikes or [])
                     if getattr(s, "severity", None) in ("BACKED_UP", "QUEUE_FORMING")]
    seen_zone_ids = {s.zone_id for s in severe_spikes}
    crowd_events = [e for e in (spike_events or [])
                    if e.get("severity") in ("BACKED_UP", "QUEUE_FORMING")
                    and e.get("zone_id") not in seen_zone_ids]

    safety_rules = [f for f in (rule_findings or [])
                    if getattr(f, "severity", None) == "SAFETY"]
    action_rules = [f for f in (rule_findings or [])
                    if getattr(f, "severity", None) == "ACTION"]

    if (not high_events and not severe_spikes and not crowd_events
            and not safety_rules and not action_rules):
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

    # A blocked fire exit outranks a queue, so SAFETY joins the critical test.
    critical = bool(pushouts or backed_up or crowd_backed_up or safety_rules)
    if critical:
        grad = "linear-gradient(135deg,#7f1d1d 0%,#dc2626 55%,#f97316 100%)"
        accent = "#fee2e2"
        shadow = "0 6px 22px rgba(220,38,38,0.32)"
    else:
        grad = "linear-gradient(135deg,#78350f 0%,#d97706 55%,#f59e0b 100%)"
        accent = "#fef3c7"
        shadow = "0 6px 22px rgba(217,119,6,0.30)"

    # (sort_key, chip_html) — highest severity first so the roll-up drops the
    # least important chips rather than an arbitrary tail.
    scored: list[tuple[int, str]] = []
    if pushouts:
        scored.append((100, _alert_chip(
            f"Pushout × {len(pushouts)}",
            _names(pushouts, lambda e: f"C{e['cart_id']}", 5),
            tab="POPS")))
    if safety_rules:
        for label, group in _group_rules(safety_rules).items():
            scored.append((95, _alert_chip(
                f"{label} × {len(group)}", _rule_detail(group),
                tab="Operational Alerts")))
    if high_pri:
        scored.append((80, _alert_chip(
            f"High Priority × {len(high_pri)}",
            _names(high_pri, lambda e: f"C{e['cart_id']}", 5),
            tab="POPS")))
    if backed_up:
        scored.append((70, _alert_chip(
            f"Zone Backed Up × {len(backed_up)}",
            _names(backed_up, lambda s: s.zone_name),
            tab="Analytics")))
    if crowd_backed_up:
        scored.append((65, _alert_chip(
            f"Crowd Backed Up × {len(crowd_backed_up)}",
            _names(crowd_backed_up, lambda e: e.get("zone_name", "Crowd")),
            tab="Analytics")))
    if action_rules:
        for label, group in _group_rules(action_rules).items():
            scored.append((60, _alert_chip(
                f"{label} × {len(group)}", _rule_detail(group),
                tab="Operational Alerts")))
    if queueing:
        scored.append((40, _alert_chip(
            f"Queue Forming × {len(queueing)}",
            _names(queueing, lambda s: s.zone_name),
            tab="Analytics")))
    if crowd_queueing:
        scored.append((35, _alert_chip(
            f"Crowd Spike × {len(crowd_queueing)}",
            _names(crowd_queueing, lambda e: e.get("zone_name", "Crowd")),
            tab="Analytics")))

    scored.sort(key=lambda x: -x[0])
    chips = [c for _, c in scored[:_MAX_BANNER_CHIPS]]
    overflow = len(scored) - len(chips)
    if overflow > 0:
        chips.append(f"<span class='gk-alert-more'>+{overflow} more "
                     f"in the tabs below</span>")

    n_total = (len(pushouts) + len(high_pri) + len(severe_spikes)
               + len(crowd_events) + len(safety_rules) + len(action_rules))

    # Every noun in the headline has to name a group that actually produced a
    # chip. The chain used to hardcode one per branch, so a run whose only
    # finding was a HIGH PRIORITY cart fell through to the congestion wording
    # and announced queues that were never measured — likewise "carts and
    # zones" on a zone-only run, and "unattended carts" for any ACTION rule.
    if safety_rules:
        headline = "Safety issue - egress or pathway obstructed"
    else:
        bits = []
        if pushouts:
            bits.append(_plural(len(pushouts), "pushout"))
        if high_pri:
            bits.append(_plural(len(high_pri), "high-priority cart"))
        n_backed = len(backed_up) + len(crowd_backed_up)
        if n_backed:
            bits.append(f"{_plural(n_backed, 'area')} backed up")
        if action_rules:
            for label, group in _group_rules(action_rules).items():
                bits.append(
                    f"{_plural(len(group), _headline_noun(label))} flagged")
        n_queue = len(queueing) + len(crowd_queueing)
        if n_queue:
            bits.append(f"congestion forming in {_plural(n_queue, 'area')}")
        lead = "Action required" if critical else "Heads up"
        headline = f"{lead} - {', '.join(bits)}"

    return (
        f"<div class='gk-alert-banner' role='alert' aria-live='assertive' "
        f"style='background:{grad};box-shadow:{shadow};'>"
        f"<div class='gk-alert-row'>"
        f"<div class='gk-alert-lead'>"
        f"<span class='gk-alert-pulse'></span>"
        f"<div>"
        f"<div class='gk-alert-kicker' style='color:{accent};'>"
        f"Live Alerts &middot; {n_total} flagged</div>"
        # Escaped because the headline is no longer a set of literals — it now
        # carries text derived from RuleFinding.label, like every other chip.
        f"<div class='gk-alert-headline'>{T.esc(headline)}</div>"
        f"</div></div>"
        f"<div class='gk-alert-chips'>{''.join(chips)}</div>"
        f"<button type='button' class='gk-alert-dismiss' "
        f"title='Dismiss' aria-label='Dismiss alerts'>&times;</button>"
        f"</div></div>"
    )


def _group_rules(findings) -> dict:
    """One doorway blocked by two carts is one incident, not two."""
    by_rule: dict = {}
    for f in findings:
        by_rule.setdefault(f.label, []).append(f)
    return by_rule


def _rule_detail(group) -> str:
    places = _names(group, lambda g: (g.zone_name or f"C{g.cart_display_id}"))
    longest = max(g.duration_s for g in group)
    return f"{places} · up to {longest:.0f}s"


def build_error_banner(message: str, detail: str = "") -> str:
    """Failure notice rendered into the sticky banner host.

    Errors used to land in the Video Info tab — eleventh in the tab bar — so a
    failed run showed a transient toast and parked the reason somewhere nobody
    looks. This puts it at the top of the page instead.
    """
    detail_html = ""
    if detail:
        detail_html = (f"<div class='gk-alert-chip' style='max-width:100%;'>"
                       f"<div class='gk-alert-chip-t'>Detail</div>"
                       f"<div class='gk-alert-chip-d' style='font-weight:600;"
                       f"font-size:0.78rem;'>{T.esc(detail)[:400]}</div></div>")
    return (
        f"<div class='gk-alert-banner' role='alert' aria-live='assertive' "
        f"style='background:linear-gradient(135deg,#7f1d1d 0%,#b91c1c 60%,"
        f"#dc2626 100%);box-shadow:0 6px 22px rgba(220,38,38,0.32);'>"
        f"<div class='gk-alert-row'>"
        f"<div class='gk-alert-lead'>"
        f"<span class='gk-alert-pulse'></span>"
        f"<div>"
        f"<div class='gk-alert-kicker' style='color:#fecaca;'>Run failed</div>"
        f"<div class='gk-alert-headline'>{T.esc(message)}</div>"
        f"</div></div>"
        f"<div class='gk-alert-chips'>{detail_html}</div>"
        f"<button type='button' class='gk-alert-dismiss' "
        f"title='Dismiss' aria-label='Dismiss'>&times;</button>"
        f"</div></div>"
    )


# ---------------------------------------------------------------------------
# Operational categories — per-cart multi-label view
# ---------------------------------------------------------------------------
# One cart can be in several operational categories at once: a cart can be
# unattended AND standing across the exit AND motionless in a monitored zone.
# The findings table answers "what happened, when" one incident per row, which
# structurally cannot say "these three describe the same cart". This index is
# the other axis — cart -> every category it fell into — and it is what lets
# the per-cart POPS table carry a stack of labels in one cell.
#
# Chip text is the finding's OWN label, never a second short vocabulary
# maintained here: rules.py deliberately names the operational one "UNATTENDED
# CART (OPS)" so it cannot be read as the theft-scored POPS "ABANDONED CART",
# and a private abbreviation in this file would quietly undo that. It also
# means a rule added later renders here with no edit.
def cart_flag_index(rule_findings) -> dict:
    """cart display id -> list of flag dicts, worst severity first.

    Flags are collapsed per rule_id, so a cart that blocked the door twice
    carries one BLOCKED DOOR chip reading "x2" rather than two identical chips.

    ``evidence["also_carts"]`` is folded in deliberately. rules._dedupe()
    merges the second cart in one doorway into a single finding and records it
    there; without this, that cart's row would show no flag at all while the
    findings table said the doorway was blocked.
    """
    idx: dict[int, dict] = {}
    for f in rule_findings or []:
        label = getattr(f, "label", "") or ""
        key = getattr(f, "rule_id", "") or label
        if not key:
            continue
        sev = getattr(f, "severity", "INFO")
        dur = float(getattr(f, "duration_s", 0.0) or 0.0)
        start = getattr(f, "start_t", None)
        zone = getattr(f, "zone_name", None)
        also = (getattr(f, "evidence", {}) or {}).get("also_carts") or []
        carts = [getattr(f, "cart_display_id", None)] + list(also)
        for cd in carts:
            if cd is None:
                continue
            slot = idx.setdefault(int(cd), {})
            e = slot.get(key)
            if e is None:
                e = {"rule_id": key, "label": label, "severity": sev,
                     "count": 0, "longest_s": 0.0, "first_t": None,
                     "zones": [], "ongoing": False, "degraded": False}
                slot[key] = e
            e["count"] += 1
            e["longest_s"] = max(e["longest_s"], dur)
            if start is not None:
                e["first_t"] = (float(start) if e["first_t"] is None
                                else min(e["first_t"], float(start)))
            if zone and zone not in e["zones"]:
                e["zones"].append(zone)
            e["ongoing"] = e["ongoing"] or bool(getattr(f, "ongoing_at_eov", False))
            e["degraded"] = (e["degraded"]
                             or getattr(f, "confidence", "high") == "degraded")

    out: dict[int, list] = {}
    for cd, slot in idx.items():
        out[cd] = sorted(
            slot.values(),
            key=lambda e: (-_SEV_RANK.get(str(e["severity"]).upper(), 0),
                           -e["longest_s"]))
    return out


def _flag_chip(flag: dict) -> str:
    """One category chip.

    OUTLINE styling (``solid=False``) against the POPS event badge's solid
    fill, so the two vocabularies stay visually separable inside one row —
    an operational category is not a theft score.
    """
    text = flag["label"]
    if flag["count"] > 1:
        text = f"{text} ×{flag['count']}"

    tip = [flag["label"], flag["severity"]]
    if flag["zones"]:
        tip.append(", ".join(flag["zones"][:3]))
    if flag["longest_s"] > 0:
        longest = f"longest {flag['longest_s']:.0f}s"
        tip.append(f"≥ {longest} (ongoing at end of clip)"
                   if flag["ongoing"] else longest)
    if flag["first_t"] is not None:
        tip.append(f"from {_mmss(flag['first_t'])}")
    if flag["degraded"]:
        tip.append("timing derived from frame rate")

    return (f"<span class='gk-flag' title='{T.esc(' · '.join(tip))}'>"
            f"{T.badge(text, flag['severity'], solid=False)}</span>")


def _flag_stack(flags) -> str:
    """Every category a cart fell into, as wrapping chips in one cell.

    Chips are joined by a visually-hidden comma: the CSV export reads
    ``td.textContent``, so without it two adjacent chips export as one run-on
    string ("BLOCKED DOOR STATIC CART").
    """
    if not flags:
        return "<span class='gk-dim'>n/a</span>"
    sep = "<span class='gk-sr-only'>, </span>"
    return (f"<span class='gk-flags'>"
            f"{sep.join(_flag_chip(f) for f in flags)}</span>")


def _flag_sort_value(flags) -> int:
    """Worst severity first, then how many categories — so sorting the column
    surfaces the cart in the most trouble, not the alphabetically first."""
    if not flags:
        return 0
    worst = max(_SEV_RANK.get(str(f["severity"]).upper(), 0) for f in flags)
    return worst * 1000 + min(len(flags), 999)


def category_counts_from_index(idx: dict) -> list:
    """rule_id -> cart count, worst-severity first. `idx` is cart_flag_index()'s
    output.

    Counts CARTS, not findings: the point is "how much of this is there", and
    one cart blocking a door for two spells is one problem. Pure data, no
    HTML — shared by _category_counts (chip strip) and the tracking JSON's
    operational_highlights.category_counts key.
    """
    per_cat: dict[str, dict] = {}
    for flags in idx.values():
        for f in flags:
            e = per_cat.setdefault(f["rule_id"], {"rule_id": f["rule_id"],
                                                  "label": f["label"],
                                                  "severity": f["severity"],
                                                  "carts": 0})
            e["carts"] += 1
    return sorted(per_cat.values(),
                  key=lambda e: -_SEV_RANK.get(str(e["severity"]).upper(), 0))


def _category_counts(rule_findings) -> str:
    """Compact strip of how many carts fell into each category."""
    idx = cart_flag_index(rule_findings)
    ordered = category_counts_from_index(idx)
    if not ordered:
        return ""
    chips = ""
    for e in ordered:
        text = "{} {}".format(e["label"], e["carts"])
        chips += (f"<span class='gk-flag'>"
                  f"{T.badge(text, e['severity'], solid=False)}</span>")
    return (f"<div class='gk-inline-summary'>"
            f"<b>{len(idx)}</b> cart(s) flagged{chips}"
            f"<span style='margin-left:auto;color:{T.INK_3};'>"
            f"A cart can fall into more than one category.</span></div>")


#: Category reference. Deliberately states the SIGNAL each rule actually uses,
#: including where it is a proxy — "no merchandise" comes from the whole-cart
#: fill classifier, not from merchandise detection, and a table that implied
#: otherwise would oversell the model.
_CATEGORY_DOCS = [
    ("BLOCKED DOOR", "SAFETY",
     "A cart or other object detected across a doorway or exit region and "
     "persisting over frames. Primarily a safety and egress-compliance "
     "signal, with secondary operational value in keeping entrances clear.",
     "Cart bbox overlapping a <b>door</b> zone while motionless, held past "
     "the blocked-door threshold."),
    ("UNATTENDED CART (OPS)", "ACTION",
     "A cart left standing with nobody near it, outside any designated cart "
     "area. Drives the retrieval workflow - deliberately named apart from the "
     "POPS theft event <i>ABANDONED CART</i>, which is a risk score, not an "
     "operational task.",
     "Cart motionless with no person within the attended radius, excluding "
     "carts parked inside a <b>fixture</b> (corral / bay) zone."),
    ("STATIC CART", "WATCH",
     "A cart that remains motionless in a monitored zone - for example "
     "stalled, stuck, or blocking a path - determined from an unchanging cart "
     "position across frames, distinct from an abandoned cart by location and "
     "intent. Supports monitoring, housekeeping alerts, and dwell-based "
     "analytics.",
     "Positional spread inside a sliding window staying under a few pixels "
     "while the cart sits in an <b>aisle</b> or analytics zone."),
    (INCOMING_NO_ITEMS, "INFO",
     "An empty cart entering the store or a monitored zone - a cart detected "
     "with no merchandise, moving inward. Distinguishes normal shopper entry "
     "from other movements, feeds cart-count and traffic analytics, and helps "
     "suppress unnecessary triggers on empty carts.",
     "Two independent routes, and they can disagree. The <b>POPS</b> table "
     "labels a cart from its peak-frame direction and voted fill, so it says "
     "this as soon as a cart's peak reads inbound + empty. The <b>rule "
     "engine</b> is stricter: inbound over the entry window AND several fresh "
     "&ldquo;empty&rdquo; classifications, which a cart in view for a second "
     "or two will not reach. <b>Proxy:</b> emptiness comes from the whole-cart "
     "fill classifier, so a single small item may not register."),
]


def build_category_reference() -> str:
    """Collapsed reference for the operational categories.

    Static — it documents the vocabulary, so it renders before any run and
    answers "what does this label mean" where the label is being read.
    """
    rows = "".join(
        f"<tr class='gk-tr'>"
        f"<td style='width:190px;'>{T.badge(name, sev, solid=False)}</td>"
        f"<td>{what}</td>"
        f"<td style='width:34%;' class='gk-dim'>{how}</td>"
        f"</tr>"
        for name, sev, what, how in _CATEGORY_DOCS
    )
    return (
        f"<details class='gk-details'>"
        f"<summary>What these categories mean "
        f"<span class='gk-section-sub'>{len(_CATEGORY_DOCS)} categories &middot; "
        f"a cart can be in several at once</span></summary>"
        f"<div class='gk-table-wrap' style='margin-top:8px;'>"
        f"<div class='gk-table-scroll'>"
        f"<table class='gk-table'>"
        f"<thead><tr><th>Category</th><th>What it means</th>"
        f"<th>Detected from</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"</div></div></details>"
    )


# ---------------------------------------------------------------------------
# Operational alerts (rule engine)
# ---------------------------------------------------------------------------
def build_operational_alerts(rule_findings, unavailable_reason=None,
                            diagnostics=None) -> str:
    """Table of operational rule findings — one row per incident.

    An empty findings list is NOT rendered as "no issues found" when the rules
    could not actually run: ``unavailable_reason`` is shown instead. Silently
    reporting "all clear" for a run where no zones were drawn would be a wrong
    answer dressed up as a clean bill of health.

    ``unavailable_reason`` is PARTIAL, though, and that distinction is the
    whole of the bug this shape replaced. rules.py sets it when there are no
    door/aisle zones, which skips exactly two of the four rules — blocked-door
    and static-cart. The other two, unattended cart and incoming empty cart,
    are zone-independent, still run, and still produce findings. Returning the
    notice INSTEAD of the table discarded every one of them: the tab badge
    counted them (it reads ``rule_findings`` directly) while this panel said
    nothing had run. So the caveat now sits ABOVE the findings rather than in
    place of them.

    ``diagnostics`` is the finer-grained version of the same idea and is shown
    in ALL THREE states, clean included. ``unavailable_reason`` only fires when
    NEITHER zone-dependent rule could run; these notes name the individual
    family that was skipped, flag a synthesised clock, and report intervals the
    density gate threw away. Every one of those turns an apparent "all clear"
    into "we did not actually look", which is the difference a reader needs.
    """
    caveat = (T.notice("Some operational rules did not run", unavailable_reason,
                       tone="WATCH")
              if unavailable_reason else "")
    # Deliberately its own block rather than appended to `caveat`: the
    # empty-findings path below renders its OWN version of the caveat (with a
    # trailing "the rules that did run found nothing"), so folding the two
    # together printed that heading twice on exactly the run that most needs to
    # be read carefully.
    notes = highlights.ops_diagnostics(diagnostics)
    notes_block = ""
    if notes:
        items = "".join(f"<li>{T.esc(n)}</li>" for n in notes)
        notes_block = T.notice(
            "Coverage notes for this run",
            f"<ul style='margin:6px 0 0 18px;padding:0;'>{items}</ul>",
            tone="WATCH")

    if not rule_findings:
        if unavailable_reason:
            # Say both halves: which rules were skipped, AND that the ones
            # that did run came back clean. Without the second sentence this
            # reads as "nothing was evaluated", which is not what happened.
            return notes_block + T.notice(
                "Some operational rules did not run",
                unavailable_reason + " The rules that did run found nothing.",
                tone="WATCH")
        # Still prefixed with the notes: "nothing crossed its threshold" is only
        # the whole truth when nothing was skipped or degraded either.
        return notes_block + T.notice(
            "No operational issues detected",
            "Rules evaluated successfully and nothing crossed its threshold.",
            tone="GOOD")

    columns = [
        T.col("Severity", sort="text", width="110px"),
        T.col("Issue", sort="text"),
        T.col("Cart", sort="text", width="110px"),
        T.col("Zone", sort="text"),
        T.col("Start", sort="num", width="82px", sub="mm:ss"),
        T.col("Duration", sort="num", width="110px"),
        T.col("Why", sort=None),
    ]

    # Which OTHER categories each cart is in, so an incident row can say "this
    # cart is also blocking the door" — the cross-category link the one-row-per-
    # incident shape otherwise hides.
    flag_idx = cart_flag_index(rule_findings)

    rows = []
    for f in rule_findings:
        sev = getattr(f, "severity", "INFO")
        dur = f"{f.duration_s:.0f}s"
        if getattr(f, "ongoing_at_eov", False):
            dur = f"&ge; {dur} (ongoing)"
        why = "; ".join(getattr(f, "reasons", []) or [])
        if getattr(f, "confidence", "high") == "degraded":
            why += (f" <span style='color:{T.CAUTION};font-weight:700;'>"
                    f"[degraded: timing derived from frame rate]</span>")
        also = (getattr(f, "evidence", {}) or {}).get("also_carts") or []
        cart_txt = f"C{f.cart_display_id}"
        if also:
            cart_txt += f" (+{', '.join('C' + str(c) for c in also)})"

        this_rule = getattr(f, "rule_id", "") or f.label
        others = [e["label"] for e in flag_idx.get(int(f.cart_display_id), [])
                  if e["rule_id"] != this_rule]
        also_line = ""
        if others:
            also_line = (f"<div class='gk-dim' style='font-size:0.74rem;"
                         f"margin-top:3px;'>also: {T.esc(', '.join(others))}"
                         f"</div>")

        rows.append(T.row([
            T.cell(T.badge(sev, sev), sort_value=sev),
            T.cell(f"<span class='gk-strong'>{T.esc(f.label)}</span>{also_line}",
                   sort_value=f.label),
            T.cell(T.esc(cart_txt), sort_value=f.cart_display_id),
            T.cell(T.esc(f.zone_name) if f.zone_name else "n/a",
                   sort_value=f.zone_name or ""),
            T.cell(_mmss(f.start_t), sort_value=f.start_t, css_class="gk-num"),
            T.cell(dur, sort_value=f.duration_s, css_class="gk-num"),
            T.cell(why, css_class="gk-dim"),
        ], tone=sev, seek_t=f.start_t))

    return caveat + notes_block + _category_counts(rule_findings) + T.render_table(
        columns, rows,
        title="Operational findings",
        subtitle="one row per incident · click a row to jump the video there",
        table_id="gk-ops-table",
        csv_name="operational_alerts.csv",
        visible_rows=100,
    )


# ---------------------------------------------------------------------------
# Key/value info tables
# ---------------------------------------------------------------------------
def styled_table(title, rows, header_gradient=None, row_tint=None,
                 *, tone: str = "INFO"):
    """Two-column key/value card.

    ``header_gradient`` / ``row_tint`` are accepted for backwards
    compatibility and ignored — colours now come from the theme so these
    cards follow the light/dark toggle.
    """
    body = "".join(
        f"<tr class='gk-tr'><td style='width:45%;font-weight:600;'>{k}</td>"
        f"<td>{v}</td></tr>"
        for k, v in rows
    )
    fg, _bg = T.severity_colors(tone)
    return (
        f"<div class='gk-table-wrap'>"
        f"<div class='gk-table-toolbar'>"
        f"<div class='gk-section-title' style='color:{fg};'>{title}</div>"
        f"</div>"
        f"<div class='gk-table-scroll'>"
        f"<table class='gk-table'><tbody>{body}</tbody></table>"
        f"</div></div>"
    )


def build_video_info(source_path, w, h, fps, total_frames, processed):
    rows = [
        ("Video Name", T.esc(os.path.basename(source_path))),
        ("Resolution", f"{w} x {h}"),
        ("Total Frames", str(total_frames)),
        ("Frames Processed", str(processed)),
    ]
    return styled_table("Video Information", rows, tone="INFO")


def build_detection_info(n_people, n_carts, n_links):
    rows = [
        ("Unique Persons Detected", f"<b>{n_people}</b>"),
        ("Unique Carts Detected",   f"<b>{n_carts}</b>"),
        ("Person-Cart Links",       f"<b>{n_links}</b>"),
    ]
    return styled_table("Detection Summary", rows, tone="GOOD")


def build_config_info(link_confirm, link_grace, camera, quality_pt, fill_pt,
                      threshold, enable_pose=True):
    rows = [
        ("Detection Model", "YOLOv26m (custom trained)"),
        ("Tracker", "BoTSORT (retail tuned)"),
        # Stated explicitly: with pose off the annotated video has no
        # skeletons, and that absence should read as a setting, not a fault.
        ("Pose Estimation", "On - skeleton overlay" if enable_pose
                            else "Off - no skeleton overlay"),
        ("Link Confirmation", f"{link_confirm} frames co-movement + overlap"),
        ("Link Grace Period", f"{link_grace} frames before linking new cart"),
        ("Camera Placement", T.esc(camera)),
        ("Quality Model", T.esc(os.path.basename(quality_pt)) if quality_pt else "None"),
        ("Fill/Bag Model", T.esc(os.path.basename(fill_pt)) if fill_pt else "None"),
        ("Quality Threshold", f"{threshold:.2f}"),
    ]
    return styled_table("Model Configuration", rows, tone="INFO")


def build_legend():
    def sw(c):
        return (f'<span class="gk-swatch" style="background:{c};'
                f'box-shadow:none;"></span>')
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
    return styled_table("Annotation Legend", rows, tone="WARN")


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------
def build_run_summary(*, frames, wall_s, encode_s, device, video_duration_s,
                      n_people, n_carts, n_links, timings=None) -> str:
    """Compact footer of what the run actually cost.

    The pipeline already accumulates every one of these numbers and then
    printed them to the server console only — surfacing them tells you at a
    glance whether a slow run was the model or the video writer.
    """
    pills = []
    pills.append(f"<span class='gk-run-pill'><b>{frames:,}</b> frames</span>")
    pills.append(f"<span class='gk-run-pill'><b>{wall_s:.1f}s</b> total</span>")
    if video_duration_s > 0 and wall_s > 0:
        pills.append(f"<span class='gk-run-pill'>"
                     f"<b>{video_duration_s / wall_s:.2f}×</b> realtime</span>")
    pills.append(f"<span class='gk-run-pill'>{T.esc(str(device)).upper()}</span>")
    pills.append(f"<span class='gk-run-pill'><b>{n_people}</b> people · "
                 f"<b>{n_carts}</b> carts · <b>{n_links}</b> links</span>")
    for name, secs in (timings or []):
        if secs and secs > 0.05:
            pills.append(f"<span class='gk-run-pill'>{T.esc(name)} "
                         f"{secs:.1f}s</span>")
    if encode_s > 0.05:
        pills.append(f"<span class='gk-run-pill'>encode {encode_s:.1f}s</span>")
    return (f"<div class='gk-run-summary'>"
            f"<span style='font-weight:800;letter-spacing:0.06em;"
            f"text-transform:uppercase;font-size:0.64rem;'>Run</span>"
            f"{''.join(pills)}</div>")


def build_tab_counts(counts: dict) -> str:
    """Hidden host element carrying per-tab badge counts.

    ``counts`` maps a tab label to ``{"n": int, "tone": "DANGER"|...}``.
    app.js reads the data attribute and injects the badges — Gradio's layout
    API has no supported way to relabel a gr.Tab after construction, and a
    failed injection simply means no badge rather than a broken page.
    """
    import json
    payload = json.dumps({k: v for k, v in (counts or {}).items() if v.get("n")})
    return (f"<div id='gk-tab-counts' data-counts='{T.esc(payload)}' "
            f"hidden aria-hidden='true'></div>")


# ---------------------------------------------------------------------------
# POPS summary
# ---------------------------------------------------------------------------
def build_pops_summary(max_pops_per_cart, peak_snapshots, rule_findings=None):
    """Per-cart table: the peak POPS event AND every operational category.

    ``rule_findings`` is optional so every existing 2-argument caller keeps
    working. When supplied, each cart's row carries a stack of category chips —
    a cart that is unattended AND blocking the exit shows both, which is the
    one thing the per-incident findings table cannot express.

    Ordering stays POPS-first: cart id is arrival order, and re-ranking by
    operational severity would move a pushout below a static cart. The
    Operational column sorts by worst severity for the other reading.
    """
    flag_idx = cart_flag_index(rule_findings)
    show_flags = bool(flag_idx)
    # A cart only enters max_pops_per_cart via the POPS scoring path, so a cart
    # the rule engine flagged can be missing from it entirely. Dropping those
    # rows would hide a blocked door because the cart was never scored.
    flag_only = sorted(cd for cd in flag_idx if cd not in (max_pops_per_cart or {}))

    if not max_pops_per_cart and not flag_only:
        return T.empty_state(
            "No carts detected in this clip.",
            "POPS scores appear here once a cart is tracked for long enough "
            "to classify.")

    columns = [
        T.col("Cart", sort="num", width="90px"),
        T.col("POPS", sort="num", width="80px", sub="peak score"),
        # Wider than the old 170px: "INCOMING CART WITHOUT ITEMS" is a real
        # label now, and a badge that wraps mid-phrase is harder to scan.
        T.col("Event", sort="text", width="210px"),
    ]
    if show_flags:
        columns.append(T.col("Operational", sort="num", width="215px",
                             sub="rule-engine categories"))
    columns += [
        T.col("Quality", sort="text"),
        T.col("Fill", sort="text"),
        T.col("Bag", sort="text"),
        T.col("Peak at", sort="num", width="90px", sub="mm:ss"),
    ]

    def _row(cd, *, scored: bool):
        flags = flag_idx.get(cd, [])
        if scored:
            ms = max_pops_per_cart[cd]
            peak = peak_snapshots.get(cd, {})
            # An empty cart moving inward is named as such rather than left as
            # a bare "INBOUND" with the reader joining it to the Fill column.
            evt = resolve_event_label(peak, peak.get("event", "CLEAR"))
            if ms >= 71:
                score_color = T.DANGER
            elif ms >= 31:
                score_color = T.WARN
            else:
                score_color = T.GOOD
            score_cell = T.cell(
                f"<b style='color:{score_color};font-size:0.98rem;'>{ms}</b>",
                sort_value=ms, css_class="gk-num")
            event_cell = T.cell(_badge_themed(evt), sort_value=evt)
            pops_tone = _ROW_TONE.get(evt)
        else:
            # Flagged by the rule engine but never POPS-scored. Say that
            # plainly rather than printing a 0 that reads as "assessed, clear".
            peak = {}
            evt = ""
            score_cell = T.cell("<span class='gk-dim'>n/a</span>",
                                sort_value=-1, css_class="gk-num")
            event_cell = T.cell("<span class='gk-dim'>not scored</span>",
                                sort_value="")
            pops_tone = None

        qual = peak.get("quality", "unclassified")
        fill = peak.get("fill", "N/A")
        bag = str(peak.get("bag", "N/A")).replace("_", " ")
        t_peak = peak.get("timestamp")
        # A flag-only cart has no peak to seek to; its earliest incident is the
        # next best target so the row stays clickable.
        seek_t = t_peak
        if seek_t is None and flags:
            firsts = [f["first_t"] for f in flags if f["first_t"] is not None]
            seek_t = min(firsts) if firsts else None

        cells = [
            T.cell(f"<span class='gk-strong'>Cart {cd}</span>", sort_value=cd),
            score_cell,
            event_cell,
        ]
        if show_flags:
            cells.append(T.cell(_flag_stack(flags),
                                sort_value=_flag_sort_value(flags)))
        cells += [
            T.cell(T.esc(qual), sort_value=qual, css_class="gk-upper"),
            T.cell(T.esc(fill), sort_value=fill, css_class="gk-upper"),
            T.cell(T.esc(bag), sort_value=bag, css_class="gk-upper"),
            T.cell(_mmss(t_peak) if t_peak is not None else "n/a",
                   sort_value=t_peak if t_peak is not None else -1,
                   css_class="gk-num"),
        ]
        # Worst label on the row wins the tint: a SAFETY category must not be
        # rendered calm because the cart's POPS event was MONITORING.
        #
        # WATCH is the floor, matching _ROW_TONE's policy of letting
        # MONITORING / INBOUND / LOW PRIORITY carry a badge and no tint. This
        # table lists EVERY cart, and INCOMING EMPTY CART fires on ordinary
        # shopper entry — tinting those rows would colour most of the table and
        # cost the tint its meaning.
        flag_tone = None
        if flags and _SEV_RANK.get(str(flags[0]["severity"]).upper(), 0) >= \
                _SEV_RANK["WATCH"]:
            flag_tone = flags[0]["severity"]
        return T.row(cells, tone=_worst_tone(pops_tone, flag_tone),
                     seek_t=seek_t)

    rows = [_row(cd, scored=True)
            for cd in sorted(max_pops_per_cart or {},
                             key=lambda c: -max_pops_per_cart[c])]
    # Unscored carts last — they have no risk score to rank them by.
    rows += [_row(cd, scored=False) for cd in flag_only]

    sub = "sorted by risk · click a row to jump the video to the peak"
    if show_flags:
        sub = ("sorted by risk · POPS event plus every operational category "
               "· click a row to jump the video")

    head = ""
    if show_flags:
        n_flagged = sum(1 for cd in flag_idx)
        n_multi = sum(1 for f in flag_idx.values() if len(f) > 1)
        bits = [f"<b>{n_flagged}</b> cart(s) carry an operational category"]
        if n_multi:
            bits.append(T.badge(f"{n_multi} in more than one", "WATCH"))
        if flag_only:
            bits.append(T.badge(f"{len(flag_only)} never POPS-scored", "INFO"))
        head = (f"<div class='gk-inline-summary'>{' '.join(bits)}"
                f"<span style='margin-left:auto;color:{T.INK_3};'>"
                f"Categories are rule-engine outcomes, not theft scores - "
                f"see the Operational Alerts tab for times and evidence."
                f"</span></div>")

    return head + T.render_table(
        columns, rows,
        title="Peak POPS by cart",
        subtitle=sub,
        table_id="gk-pops-table",
        csv_name="pops_summary.csv",
        visible_rows=100,
    )


# ---------------------------------------------------------------------------
# Events timeline
# ---------------------------------------------------------------------------
def build_events_timeline(event_log):
    if not event_log:
        return T.empty_state(
            "All clear",
            "No pushout events or suspicious activity detected in this clip.",
            tone="GOOD", icon="✓")

    n_high = sum(1 for e in event_log if e["event"] in HIGH_EVENTS)
    n_med  = sum(1 for e in event_log if e["event"] in MEDIUM_EVENTS)

    # Compact strip, not a full-width banner. The sticky banner at the top of
    # the page is the alert surface; this only says what is in THIS table.
    bits = [f"<b>{len(event_log)}</b> event(s) logged"]
    if n_high:
        bits.append(T.badge(f"{n_high} high", "SAFETY"))
    if n_med:
        bits.append(T.badge(f"{n_med} medium", "WATCH"))
    if not n_high and not n_med:
        bits.append(T.badge("no high-risk events", "GOOD"))
    summary = (f"<div class='gk-inline-summary'>{' '.join(bits)}"
               f"<span style='margin-left:auto;color:{T.INK_3};'>"
               f"Click any row to jump the video to that moment.</span></div>")

    columns = [
        T.col("Time", sort="num", width="110px", sub="mm:ss · frame"),
        T.col("Cart", sort="num", width="90px"),
        T.col("Event", sort="text", width="170px"),
        T.col("POPS", sort="num", width="76px"),
        T.col("Fill", sort="text"),
        T.col("Bag", sort="text"),
        T.col("Speed", sort="text"),
        T.col("Details", sort="text"),
    ]

    rows = []
    for evt in event_log:
        ts = evt.get("timestamp", 0.0)
        linked_str = "LINKED" if evt.get("linked") else "NO LINK"
        bag_lbl = str(evt.get("bag", "N/A")).replace("_", " ")
        spd = evt.get("speed_status", "N/A")
        rows.append(T.row([
            T.cell(f"{_mmss(ts)} <span class='gk-dim' style='font-size:0.76rem;'>"
                   f"F{evt.get('frame', 0)}</span>",
                   sort_value=ts, css_class="gk-num"),
            T.cell(f"<span class='gk-strong'>Cart {evt['cart_id']}</span>",
                   sort_value=evt["cart_id"]),
            T.cell(_badge_themed(evt["event"]), sort_value=evt["event"]),
            T.cell(f"<b>{evt.get('pops_score', 0)}</b>",
                   sort_value=evt.get("pops_score", 0), css_class="gk-num"),
            T.cell(T.esc(evt.get("fill", "N/A")), css_class="gk-upper"),
            T.cell(T.esc(bag_lbl), css_class="gk-upper"),
            T.cell(T.esc(spd), css_class="gk-upper"),
            T.cell(f"{T.esc(evt.get('direction', ''))} | {linked_str}",
                   css_class="gk-dim"),
        ], tone=_ROW_TONE.get(evt["event"]), seek_t=ts))

    table = T.render_table(
        columns, rows,
        title="Event timeline",
        table_id="gk-events-table",
        csv_name="pops_events.csv",
        visible_rows=150,
    )
    return summary + table
