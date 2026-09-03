# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Operational rule engine — turns recorded facts into the four operational
outcomes (blocked door, static cart, unattended cart, incoming empty cart).

Runs POST-HOC over a cached TrajectoryBundle rather than inside the frame loop.
That is deliberate: TrackingEngine.recompute_analytics() exists so that editing
a zone re-runs analytics without re-running YOLO, and evaluating rules in the
live loop would throw that away — every threshold tweak would cost a full GPU
pass. Post-hoc also means whole-timeline visibility, which is what makes
correct interval closing, gap merging, and real duration reporting possible.

Deliberately separate from POPS scoring. `scoring.py` already emits
"ABANDONED CART" as a theft-risk label gated behind pops_score >= 31, and an
INBOUND cart is floored to 5 by the kill switch — so an operational
abandonment routed through compute_pops() would be structurally unreachable.

Pure numpy + stdlib (plus cv2.fillPoly for rasterisation) and STATELESS, so it
stays unit-testable without torch/ultralytics and safe to call from the shared
engine singleton.
"""
from __future__ import annotations

import numpy as np

from .analytics_models import (
    CartFactSample, RuleFinding, TrajectoryBundle, Zone, FACTS_SCHEMA_VERSION,
)
from .config import (
    SPEED_STATIC,
    RULE_ENGINE_ENABLED,
    RULE_BLOCKED_DOOR_S, RULE_STATIC_CART_S, RULE_ABANDONED_CART_S,
    RULE_STATIC_POS_SPREAD_PX, RULE_STATIC_WINDOW_S,
    RULE_DOOR_OVERLAP_FRAC,
    RULE_ATTENDED_GAP_FRAC, RULE_ATTENDED_MIN_PX, RULE_ATTENDED_RADIUS_PX,
    RULE_TIME_GRID_HZ, RULE_GRID_STALENESS_S,
    RULE_INTERVAL_MERGE_S, RULE_MAX_SAMPLE_GAP_S, RULE_MIN_SAMPLES,
    RULE_EMPTY_CONFIRM_OBS, RULE_ENTRY_WINDOW_S,
    RULE_DOOR_KINDS, RULE_STATIC_KINDS, RULE_DESIGNATED_AREA_KINDS,
)

try:                                    # cv2 is present in the app, but keep
    import cv2                          # rules importable for pure unit tests
except ImportError:                      # pragma: no cover
    cv2 = None


# Human-facing labels. "UNATTENDED CART (OPS)" is deliberately NOT the string
# "ABANDONED CART" — that name already carries theft severity 3 in the POPS
# reconciliation table and membership in LOGGABLE_EVENTS/MEDIUM_EVENTS.
RULE_LABELS = {
    "blocked_door":   "BLOCKED DOOR",
    "static_cart":    "STATIC CART",
    "abandoned_cart": "UNATTENDED CART (OPS)",
    # Named for what an operator reads, and matched by the POPS table's own
    # label for the same situation (ui_builder.INCOMING_NO_ITEMS) so the app
    # has ONE name for it. rule_id stays "incoming_empty" — that is the stable
    # key in the JSON; only the human-facing string moved.
    "incoming_empty": "INCOMING CART WITHOUT ITEMS",
}
RULE_SEVERITIES = {
    "blocked_door":   "SAFETY",
    "static_cart":    "WATCH",
    "abandoned_cart": "ACTION",
    "incoming_empty": "INFO",
}

#: Duration thresholds the UI is allowed to override at recompute time.
#: config.py supplies the defaults; the sidebar sliders pass a dict of the same
#: shape into evaluate_rules(). Re-tuning is a post-hoc, zero-GPU operation
#: (see the module docstring), which is what makes it a live control rather
#: than a source edit.
DEFAULT_THRESHOLDS = {
    "blocked_door_s":   RULE_BLOCKED_DOOR_S,
    "static_cart_s":    RULE_STATIC_CART_S,
    "abandoned_cart_s": RULE_ABANDONED_CART_S,
}


def resolve_thresholds(overrides: dict | None = None) -> dict:
    """Merge user overrides over the config defaults, dropping empties.

    A None or missing key falls back to config so a partially-populated dict
    from the UI can never silently zero a threshold (which would make every
    rule fire on every cart).
    """
    out = dict(DEFAULT_THRESHOLDS)
    for k, v in (overrides or {}).items():
        if k in out and v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv > 0:
                out[k] = fv
    return out


# ---------------------------------------------------------------------------
# Time joins
# ---------------------------------------------------------------------------
def join_on_time(src_t: np.ndarray, dst_t: np.ndarray,
                 max_staleness_s: float) -> np.ndarray:
    """For each dst_t, the index of the nearest PRECEDING src_t sample,
    or -1 when the nearest one is staler than max_staleness_s.

    This is the only sanctioned way to combine cart_facts (display_id-keyed,
    its own sample count) with TrackRecord arrays (raw_id-keyed, a different
    sample count, remapped by re-identification). Indexing the two in parallel
    silently attributes fill/zone/speed to the wrong instant.
    """
    if src_t.size == 0 or dst_t.size == 0:
        return np.full(dst_t.shape, -1, dtype=np.int64)
    idx = np.searchsorted(src_t, dst_t, side="right") - 1
    idx = np.clip(idx, -1, src_t.size - 1)
    out = idx.astype(np.int64)
    valid = out >= 0
    # Reject matches that are too old to describe dst_t.
    stale = np.zeros(dst_t.shape, dtype=bool)
    stale[valid] = (dst_t[valid] - src_t[out[valid]]) > max_staleness_s
    out[stale] = -1
    return out


# ---------------------------------------------------------------------------
# Zone membership (rule-specific)
# ---------------------------------------------------------------------------
def _rasterize(zones: list[Zone], h: int, w: int) -> list[np.ndarray]:
    masks = []
    for z in zones:
        m = np.zeros((h, w), dtype=np.uint8)
        if z.polygon.size and cv2 is not None:
            cv2.fillPoly(m, [z.polygon.astype(np.int32)], 1)
        masks.append(m)
    return masks


def rule_zone_membership(bundle: TrajectoryBundle,
                         zones: list[Zone]) -> dict[int, np.ndarray]:
    """raw_id -> shape (n_samples, n_zones) bool.

    Differs from analytics_builder.build_zone_membership() in two ways that
    matter here, which is why this is a parallel implementation rather than a
    call into that one:

    1. NOT winner-takes-all. That function collapses to one zone index per
       sample, so a cart standing where a door zone overlaps an aisle zone
       registers only the higher-indexed zone and the door rule would never
       fire.
    2. Ignores Zone.applies_to. A door is a PLACE, not a track-type filter,
       and the zone editor defaults "Track type" to "person" — so honouring it
       would make every hand-drawn door zone silently match zero carts.
    """
    h, w = int(bundle.height), int(bundle.width)
    n_zones = len(zones)
    masks = _rasterize(zones, h, w)

    out: dict[int, np.ndarray] = {}
    for raw_id, rec in bundle.tracks.items():
        n = rec.n_samples
        arr = np.zeros((n, n_zones), dtype=bool)
        if n == 0 or n_zones == 0:
            out[raw_id] = arr
            continue
        xs = np.clip(rec.positions[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(rec.positions[:, 1].astype(np.int32), 0, h - 1)
        for zi, m in enumerate(masks):
            arr[:, zi] = m[ys, xs] == 1
        out[raw_id] = arr
    return out


def door_overlap_membership(bundle: TrajectoryBundle,
                            zones: list[Zone]) -> dict[int, np.ndarray]:
    """raw_id -> (n_samples, n_zones) bool using BBOX-OVERLAP FRACTION.

    A cart can block a doorway while its centroid sits outside a thin door
    polygon, so centroid membership under-detects exactly the case the
    blocked-door rule exists to catch. Falls back to centroid membership for
    tracks with no bbox history.

    Overlap is measured as (cart bbox ∩ zone mask) / bbox area, which is
    scale-relative and therefore degrades far more gracefully with perspective
    than a fixed pixel radius would.
    """
    h, w = int(bundle.height), int(bundle.width)
    n_zones = len(zones)
    masks = _rasterize(zones, h, w)
    # Integral images make per-sample rectangle sums O(1).
    integrals = [m.cumsum(axis=0).cumsum(axis=1).astype(np.int64) for m in masks]

    centroid_fallback = rule_zone_membership(bundle, zones)
    out: dict[int, np.ndarray] = {}
    for raw_id, rec in bundle.tracks.items():
        n = rec.n_samples
        if n == 0 or n_zones == 0:
            out[raw_id] = np.zeros((n, n_zones), dtype=bool)
            continue
        if not rec.has_bboxes:
            out[raw_id] = centroid_fallback[raw_id]
            continue

        bb = rec.bboxes
        x1 = np.clip(bb[:, 0].astype(np.int32), 0, w - 1)
        y1 = np.clip(bb[:, 1].astype(np.int32), 0, h - 1)
        x2 = np.clip(bb[:, 2].astype(np.int32), 0, w - 1)
        y2 = np.clip(bb[:, 3].astype(np.int32), 0, h - 1)
        area = np.maximum((x2 - x1) * (y2 - y1), 1).astype(np.float64)

        arr = np.zeros((n, n_zones), dtype=bool)
        for zi, integ in enumerate(integrals):
            # Inclusive-exclusive rectangle sum on the integral image.
            total = (integ[y2, x2] - integ[y1, x2]
                     - integ[y2, x1] + integ[y1, x1])
            arr[:, zi] = (total / area) >= RULE_DOOR_OVERLAP_FRAC
        out[raw_id] = arr
    return out


# ---------------------------------------------------------------------------
# Static test — two signals, see config for why speed alone is not enough
# ---------------------------------------------------------------------------
def static_mask(rec_positions: np.ndarray, rec_speeds: np.ndarray,
                timestamps: np.ndarray) -> np.ndarray:
    """Per-sample bool: is this track effectively motionless here?

    POSITIONAL SPREAD is authoritative wherever it is measurable; the speed
    reading is only a fallback for samples too early in a track to have a
    window behind them.

    That ordering is the whole point. compute_motion() derives speed from a
    first-to-last delta over the last <=5 positions, so bbox jitter on a
    PHYSICALLY STATIONARY cart inflates the reading — a cart sitting in a
    doorway can easily register 45 px/s against a SPEED_STATIC of 10 and never
    be seen as static at all. Max-deviation-from-window-centroid does not care
    about jitter: a cart that stays put has a small spread no matter how noisy
    its per-frame centroid is.
    """
    n = rec_positions.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    n_sp = min(n, rec_speeds.shape[0])
    slow = np.zeros(n, dtype=bool)
    slow[:n_sp] = rec_speeds[:n_sp] < SPEED_STATIC

    # A window must actually span some time before its spread means anything —
    # otherwise the first few samples of every track look "static" simply
    # because they were captured milliseconds apart.
    min_span = 0.5 * RULE_STATIC_WINDOW_S
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        t_end = float(timestamps[i])
        lo = int(np.searchsorted(timestamps, t_end - RULE_STATIC_WINDOW_S, side="left"))
        win = rec_positions[lo:i + 1]
        span = t_end - float(timestamps[lo])
        if win.shape[0] < 3 or span < min_span:
            out[i] = slow[i]                     # not enough window yet
            continue
        centre = win.mean(axis=0)
        spread = float(np.sqrt(((win - centre) ** 2).sum(axis=1)).max())
        out[i] = spread <= RULE_STATIC_POS_SPREAD_PX
    return out


# ---------------------------------------------------------------------------
# Interval machinery
# ---------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start_idx, end_idx_exclusive)."""
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_and_qualify(mask: np.ndarray, timestamps: np.ndarray,
                       threshold_s: float, stats: dict | None = None,
                       ) -> list[tuple[int, int, float, float, list[str]]]:
    """Turn a per-sample condition into qualified intervals.

    Returns (start_idx, end_idx_exclusive, t0, t1, reasons) for every interval
    that survives gap-merging, the duration threshold, and the SAMPLE DENSITY
    gate.

    The density gate is not optional. Positions are only appended when a track
    is DETECTED, so a cart occluded for 30s leaves two samples 30s apart that
    an RLE reads as continuous presence — "static for 30 seconds" inferred from
    two observations. Rejecting sparse intervals is what stops that.

    `stats` is an optional counter dict. The density gate discards intervals
    that were over threshold — a cart in a busy doorway is occluded by the very
    traffic it is obstructing, so the intervals it drops are disproportionately
    the SAFETY-severity ones. Dropping them is right; dropping them silently is
    not, so the count is reported as a diagnostic rather than swallowed.
    """
    runs = _runs(mask)
    if not runs:
        return []

    # Bridge short gaps so a one-frame detection dropout doesn't split a block.
    merged: list[list[int]] = []
    for s, e in runs:
        if merged:
            prev_end = merged[-1][1]
            gap = float(timestamps[min(s, timestamps.size - 1)]
                        - timestamps[min(prev_end - 1, timestamps.size - 1)])
            if gap <= RULE_INTERVAL_MERGE_S:
                merged[-1][1] = e
                continue
        merged.append([s, e])

    out = []
    for s, e in merged:
        n = e - s
        t0 = float(timestamps[s])
        t1 = float(timestamps[min(e - 1, timestamps.size - 1)])
        dur = max(0.0, t1 - t0)
        if dur < threshold_s:
            continue
        reasons: list[str] = []
        if n < RULE_MIN_SAMPLES:
            if stats is not None:           # too few observations to assert
                stats["suppressed_sparse"] = stats.get("suppressed_sparse", 0) + 1
            continue
        mean_gap = dur / max(1, n - 1)
        if mean_gap > RULE_MAX_SAMPLE_GAP_S:
            if stats is not None:           # observed too sparsely to assert
                stats["suppressed_sparse"] = stats.get("suppressed_sparse", 0) + 1
            continue
        reasons.append(f"{dur:.0f}s over {n} observations")
        out.append((s, e, t0, t1, reasons))
    return out


# ---------------------------------------------------------------------------
# Person-presence grid (for the unattended-cart rule)
# ---------------------------------------------------------------------------
def _person_presence(bundle: TrajectoryBundle) -> tuple[np.ndarray, np.ndarray]:
    """(grid_t, positions) where positions is (n_grid, n_persons, 2), NaN when
    that person has no sample within RULE_GRID_STALENESS_S of the grid instant.

    Each track only has samples for frames where it was DETECTED, and every
    track has its own timestamps, so "was anyone near this cart at time t"
    cannot be answered by parallel indexing — everything is resampled onto one
    shared grid first.
    """
    persons = [r for r in bundle.tracks.values()
               if r.label == "person" and r.n_samples > 0]
    all_t = [r.timestamps for r in bundle.tracks.values() if r.n_samples > 0]
    if not all_t:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 0, 2), dtype=np.float32)

    t_min = float(min(float(t[0]) for t in all_t))
    t_max = float(max(float(t[-1]) for t in all_t))
    step = 1.0 / max(RULE_TIME_GRID_HZ, 1e-6)
    if t_max <= t_min:
        grid = np.asarray([t_min], dtype=np.float32)
    else:
        grid = np.arange(t_min, t_max + step, step, dtype=np.float32)

    if not persons:
        return grid, np.zeros((grid.size, 0, 2), dtype=np.float32)

    pos = np.full((grid.size, len(persons), 2), np.nan, dtype=np.float32)
    for pi, rec in enumerate(persons):
        idx = join_on_time(rec.timestamps, grid, RULE_GRID_STALENESS_S)
        ok = idx >= 0
        pos[ok, pi, :] = rec.positions[idx[ok]]
    return grid, pos


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def _cart_records(bundle: TrajectoryBundle):
    return [(raw, rec) for raw, rec in bundle.tracks.items()
            if rec.label == "cart" and rec.n_samples > 0]


def _facts_arrays(samples: list[CartFactSample]):
    """(t, fill, stale, linked, abandoned_linker) as parallel arrays."""
    if not samples:
        return (np.zeros(0, dtype=np.float32), [], np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
    t = np.asarray([s.t for s in samples], dtype=np.float32)
    order = np.argsort(t, kind="stable")     # facts must be monotonic for searchsorted
    t = t[order]
    fill = [samples[i].fill for i in order]
    stale = np.asarray([samples[i].fill_stale_frames for i in order], dtype=np.int32)
    linked = np.asarray([samples[i].linked for i in order], dtype=bool)
    aband = np.asarray([samples[i].abandoned_linker for i in order], dtype=bool)
    return t, fill, stale, linked, aband


def _zone_subset(zones: list[Zone], kinds: tuple[str, ...]) -> list[Zone]:
    return [z for z in zones if getattr(z, "kind", "analytics") in kinds]


def _finding(rule_id: str, rec, zone: Zone | None, t0: float, t1: float,
             threshold_s: float, n_samples: int, reasons: list[str],
             *, ongoing: bool, degraded: bool, evidence: dict) -> RuleFinding:
    return RuleFinding(
        rule_id=rule_id,
        label=RULE_LABELS[rule_id],
        severity=RULE_SEVERITIES[rule_id],
        cart_display_id=int(rec.display_id),
        zone_id=(zone.zone_id if zone else None),
        zone_name=(zone.name if zone else None),
        start_t=round(t0, 2), end_t=round(t1, 2),
        duration_s=round(max(0.0, t1 - t0), 2),
        threshold_s=threshold_s,
        ongoing_at_eov=ongoing,
        confidence=("degraded" if degraded else "high"),
        n_samples=int(n_samples),
        reasons=reasons,
        evidence=evidence,
    )


def _static_in_zone_rule(bundle, zones, kinds, threshold_s, rule_id,
                         membership_fn, eov_t, degraded, stats=None):
    """Shared implementation for blocked-door and static-cart.

    The two categories are the same question — "is a cart sitting still in a
    place it shouldn't?" — differing only in which zone kind is monitored, how
    membership is measured, and how long is too long.
    """
    zs = _zone_subset(zones, kinds)
    if not zs:
        return []
    membership = membership_fn(bundle, zs)
    findings: list[RuleFinding] = []

    for raw, rec in _cart_records(bundle):
        still = static_mask(rec.positions, rec.speeds, rec.timestamps)
        if not still.any():
            continue
        inside = membership.get(raw)
        if inside is None or inside.size == 0:
            continue
        for zi, zone in enumerate(zs):
            cond = still & inside[:, zi]
            for s, e, t0, t1, reasons in _merge_and_qualify(
                    cond, rec.timestamps, threshold_s, stats):
                findings.append(_finding(
                    rule_id, rec, zone, t0, t1, threshold_s, e - s, reasons,
                    ongoing=bool(e >= rec.n_samples and abs(t1 - eov_t) < 1.0),
                    degraded=degraded,
                    evidence={
                        "mean_speed_px_s": round(float(
                            rec.speeds[s:min(e, rec.speeds.size)].mean()
                        ), 2) if rec.speeds.size else 0.0,
                        "membership": ("bbox_overlap"
                                       if membership_fn is door_overlap_membership
                                       else "centroid"),
                    },
                ))
    return findings


def _attendance_bar(rec) -> np.ndarray:
    """Per-sample distance at which a person counts as attending this cart.

    `RULE_ATTENDED_GAP_FRAC * cart_box_diagonal`, floored at
    RULE_ATTENDED_MIN_PX. Scaling by the cart's own apparent size is the
    cheapest available proxy for depth: the same pixel count is arm's reach at
    the front of frame and half the room at the far door, so a flat radius
    answers "is anyone near this cart" differently depending only on where the
    cart happens to be standing.

    Falls back to the flat RULE_ATTENDED_RADIUS_PX for a track with no recorded
    boxes. Defensive only — every cart track in the trajectory cache carries
    them — but a silent bar of zero would mark every such cart unattended.
    """
    n = rec.n_samples
    if not rec.has_bboxes:
        return np.full(n, float(RULE_ATTENDED_RADIUS_PX), dtype=np.float32)
    bb = rec.bboxes
    diag = np.hypot(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
    return np.maximum(RULE_ATTENDED_GAP_FRAC * diag,
                      RULE_ATTENDED_MIN_PX).astype(np.float32)


def _abandoned_cart_rule(bundle, zones, eov_t, degraded,
                         threshold_s=RULE_ABANDONED_CART_S, stats=None):
    """Cart stationary with nobody nearby, outside any designated cart area.

    "Unattended" is answered by PERSON PROXIMITY, not by the live linker's
    person_gone/person_far. They are different questions: the linker asks "did
    the person who was with this cart leave?" (and can only ever fire for a
    cart that was linked in the first place), while proximity asks "is anyone
    near this cart now?" — which is the operationally correct question for a
    retrieval workflow. The linker's verdict is carried as corroboration.
    """
    grid_t, ppos = _person_presence(bundle)
    designated = _zone_subset(zones, RULE_DESIGNATED_AREA_KINDS)
    exempt = rule_zone_membership(bundle, designated) if designated else {}

    findings: list[RuleFinding] = []
    for raw, rec in _cart_records(bundle):
        still = static_mask(rec.positions, rec.speeds, rec.timestamps)
        if not still.any():
            continue

        # Nobody within the attendance bar, evaluated on the shared grid then
        # mapped back onto this cart's own samples. The bar scales with the
        # CART'S OWN apparent size rather than being a flat pixel count, because
        # a pixel count means a different floor distance at every depth — see
        # RULE_ATTENDED_GAP_FRAC for the clip that documents it.
        near = np.full(rec.n_samples, np.inf, dtype=np.float32)
        bar = _attendance_bar(rec)
        if grid_t.size and ppos.shape[1]:
            gi = join_on_time(grid_t, rec.timestamps, 1.0 / max(RULE_TIME_GRID_HZ, 1e-6) * 2)
            unattended = np.ones(rec.n_samples, dtype=bool)
            ok = gi >= 0
            if ok.any():
                d = np.linalg.norm(
                    ppos[gi[ok]] - rec.positions[ok][:, None, :], axis=2)
                with np.errstate(invalid="ignore"):
                    near[ok] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
                unattended[ok] = near[ok] > bar[ok]
        else:
            unattended = np.ones(rec.n_samples, dtype=bool)

        # Carts parked in a corral / bay are where carts belong.
        outside_corral = np.ones(rec.n_samples, dtype=bool)
        ex = exempt.get(raw)
        if ex is not None and ex.size:
            outside_corral = ~ex.any(axis=1)
        cond = still & unattended & outside_corral

        # Diagnostic, not a rule: a cart that was still long enough on its own
        # but lost the interval to the ATTENDANCE test specifically. That
        # distinction is invisible in the output — every reason for not firing
        # looks the same from outside — and it is the question asked every time
        # someone re-checks why a parked cart was not flagged, so it is counted
        # rather than re-derived by hand. The corral carve-out is held in the
        # baseline so a cart sitting where carts belong is not reported as an
        # attendance suppression, which would be the wrong explanation.
        qualified = _merge_and_qualify(cond, rec.timestamps, threshold_s, stats)
        if not qualified and stats is not None:
            if _merge_and_qualify(still & outside_corral, rec.timestamps,
                                  threshold_s, None):
                stats.setdefault("suppressed_attended", []).append(int(rec.display_id))

        for s, e, t0, t1, reasons in qualified:
            corroborated = False
            samples = bundle.cart_facts.get(int(rec.display_id)) or []
            if samples:
                ft, _fill, _stale, _linked, aband = _facts_arrays(samples)
                fi = join_on_time(ft, rec.timestamps[s:e], RULE_GRID_STALENESS_S)
                valid = fi[fi >= 0]
                corroborated = bool(valid.size and aband[valid].any())
            if corroborated:
                reasons = reasons + ["linker also reports the attendant left"]
            findings.append(_finding(
                "abandoned_cart", rec, None, t0, t1, threshold_s,
                e - s, reasons,
                ongoing=bool(e >= rec.n_samples and abs(t1 - eov_t) < 1.0),
                degraded=degraded,
                evidence={"linker_corroborated": corroborated,
                          # The bar this cart was actually judged against, and
                          # how close anyone got. Reported per finding because
                          # the bar is now per-cart, so quoting the constant
                          # would no longer describe the decision.
                          "attended_bar_px": round(float(np.median(bar[s:e])), 1),
                          "nearest_person_px": (
                              round(float(np.min(near[s:e])), 1)
                              if np.isfinite(near[s:e]).any() else None),
                          "attended_basis": ("cart_diagonal_fraction"
                                             if rec.has_bboxes else "flat_radius")},
            ))
    return findings


def _incoming_empty_rule(bundle, zones, camera_placement, eov_t, degraded):
    """Empty cart moving inward shortly after it first appears.

    PROXY, and labelled as one: "no merchandise" is inferred from the
    whole-cart fill classifier, not from merchandise detection. It cannot
    distinguish an empty cart from one holding a single small item.

    Direction is recomputed here over the ENTRY WINDOW rather than reused from
    POPS, for two reasons: compute_direction_label() uses a whole-track
    first-to-last delta, which is meaningless for a cart that enters and then
    wanders; and the live loop overwrites a linked cart's direction with its
    person's, state that never reaches the bundle.
    """
    from .motion import compute_direction_label

    findings: list[RuleFinding] = []
    for raw, rec in _cart_records(bundle):
        samples = bundle.cart_facts.get(int(rec.display_id)) or []
        if not samples:
            continue
        t0 = float(rec.timestamps[0])
        win_end = t0 + RULE_ENTRY_WINDOW_S
        win = rec.timestamps <= win_end
        if win.sum() < 2:
            continue

        direction = compute_direction_label(
            [tuple(p) for p in rec.positions[win]], camera_placement)
        if direction != "INBOUND":
            continue

        # Count CLASSIFIED observations of "empty" — with
        # CLASSIFY_EVERY_N_FRAMES, consecutive frames repeat one reading.
        ft, fill, stale, _linked, _ab = _facts_arrays(samples)
        in_win = (ft >= t0) & (ft <= win_end)
        fresh_empty = [
            fill[i] for i in np.flatnonzero(in_win)
            if stale[i] == 0
        ]
        n_empty = sum(1 for f in fresh_empty if f == "empty")
        if n_empty < RULE_EMPTY_CONFIRM_OBS:
            continue

        t_end = float(rec.timestamps[win][-1])
        findings.append(_finding(
            "incoming_empty", rec, None, t0, t_end, 0.0,
            int(win.sum()),
            [f"inbound with {n_empty} classified 'empty' observations"],
            ongoing=False, degraded=degraded,
            evidence={"direction": direction,
                      "classified_observations": len(fresh_empty),
                      "basis": "whole-cart fill classifier (proxy for merchandise)"},
        ))
    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def evaluate_rules(bundle: TrajectoryBundle,
                   zones: list[Zone],
                   *, camera_placement: str = "Outside (facing entrance)",
                   thresholds: dict | None = None,
                   diagnostics: list[str] | None = None,
                   ) -> tuple[list[RuleFinding], str | None]:
    """Evaluate all operational rules. Returns (findings, unavailable_reason).

    `unavailable_reason` is a string whenever the rules could not meaningfully
    run. Callers must render that as an explicit "did not run" state — an empty
    findings list would otherwise read as "no issues found", which is a wrong
    answer rather than a missing one.

    `thresholds` optionally overrides the config durations (see
    DEFAULT_THRESHOLDS). Omit it and config.py wins, which keeps every existing
    caller and the test suite on the previous behaviour.

    `diagnostics`, when a list is passed, is EXTENDED with non-suppressing
    notes: which rule families did not run, whether the clock was synthesised,
    and how many candidate intervals the density gate discarded. An
    out-parameter rather than a third return value on purpose - this function
    has ~20 two-value unpacking call sites, and the alternative to arity
    stability is a ValueError in every one of them.

    These notes deliberately do NOT go into `unavailable_reason`.
    highlights.ops_findings_state() checks that field FIRST and discards every
    finding when it is set, so routing "the blocked-door rule had no door zone"
    through it would blank the unattended-cart findings that did run - the
    common case, since most runs have only analytics zones drawn.
    """
    notes: list[str] = []
    th = resolve_thresholds(thresholds)
    if not RULE_ENGINE_ENABLED:
        return [], "Rule engine disabled in config."
    if bundle is None or not bundle.tracks:
        return [], "No tracks in this run."
    if cv2 is None:                                        # pragma: no cover
        return [], "OpenCV unavailable - zone rasterisation not possible."

    # A bundle cached by an older build has no fact timeline. Report that
    # rather than silently returning zero findings.
    if getattr(bundle, "facts_schema_version", 0) < FACTS_SCHEMA_VERSION:
        return [], ("This result was cached by an earlier build whose recorded "
                    "facts the rules can no longer trust - re-run the analysis "
                    "to evaluate operational rules.")

    zones = list(zones or [])
    door_zones = _zone_subset(zones, RULE_DOOR_KINDS)
    static_zones = _zone_subset(zones, RULE_STATIC_KINDS)

    degraded = bool(getattr(bundle, "timestamps_synthesized", False))

    eov_t = 0.0
    sov_t = None
    for rec in bundle.tracks.values():
        if rec.n_samples:
            eov_t = max(eov_t, float(rec.timestamps[-1]))
            first = float(rec.timestamps[0])
            sov_t = first if sov_t is None else min(sov_t, first)
    observed_s = max(0.0, eov_t - (sov_t or 0.0))

    findings: list[RuleFinding] = []
    stats: dict = {}

    # Blocked door — bbox overlap, shortest fuse, safety severity.
    findings += _static_in_zone_rule(
        bundle, zones, RULE_DOOR_KINDS, th["blocked_door_s"], "blocked_door",
        door_overlap_membership, eov_t, degraded, stats)

    # Static cart in a monitored zone — same primitive, centroid membership.
    findings += _static_in_zone_rule(
        bundle, zones, RULE_STATIC_KINDS, th["static_cart_s"], "static_cart",
        rule_zone_membership, eov_t, degraded, stats)

    # Unattended cart — zone-independent (uses fixture zones only as a carve-out).
    findings += _abandoned_cart_rule(bundle, zones, eov_t, degraded,
                                     th["abandoned_cart_s"], stats)

    # Incoming empty cart — zone-independent proxy.
    findings += _incoming_empty_rule(
        bundle, zones, camera_placement, eov_t, degraded)

    findings = _dedupe(findings)
    findings.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 0), f.duration_s),
                  reverse=True)

    # Per-FAMILY reporting. `static_zones` includes kind "analytics", which is
    # the zone editor's DEFAULT - so one dwell zone drawn anywhere used to make
    # `reason` None and the whole run read as "rules ran clean" while the
    # blocked-door rule had never been evaluated for want of a door zone.
    if not door_zones:
        notes.append("Blocked-door rule did not run: no zones of kind 'door' "
                     "are drawn. Draw one in the Zone Editor.")
    if not static_zones:
        notes.append("Static-cart rule did not run: no 'aisle' or analytics "
                     "zones are drawn.")
    if degraded:
        notes.append("CAP_PROP_POS_MSEC was unusable for this video, so timing "
                     "was derived from the frame index and frame rate. "
                     "Durations are approximate.")
    # Durations are compared against the clip, not against an assumed length:
    # the clips this runs on vary, and a threshold longer than the footage can
    # never be met by any cart. _merge_and_qualify drops those intervals on
    # `dur < threshold_s` and ongoing_at_eov is only set AFTER that gate, so a
    # clip that ends mid-interval gets no partial credit either. Silence from a
    # rule that could not possibly fire reads as "nothing happened", which is
    # the wrong answer rather than a missing one.
    unreachable = [(RULE_LABELS[rid], th[key]) for rid, key in
                   (("blocked_door", "blocked_door_s"),
                    ("static_cart", "static_cart_s"),
                    ("abandoned_cart", "abandoned_cart_s"))
                   if th[key] > observed_s]
    if unreachable and observed_s > 0:
        detail = ", ".join(f"{name} ({thr:g}s)" for name, thr in unreachable)
        notes.append(
            f"This clip is only {observed_s:.1f}s of observed time, shorter "
            f"than the threshold for: {detail}. Those rules cannot fire at any "
            f"cart's behaviour - lower the thresholds in the sidebar and "
            f"Recompute analytics (no GPU work) to evaluate them on a clip "
            f"this short.")

    attended = stats.get("suppressed_attended") or []
    if attended:
        ids = ", ".join(f"Cart {c}" for c in sorted(set(attended)))
        notes.append(
            f"{ids} stood still long enough to cross the unattended-cart "
            f"threshold but read as ATTENDED - somebody was within "
            f"{RULE_ATTENDED_GAP_FRAC:g}x the cart's own box diagonal for "
            f"enough of the time to break the interval up. Not a duration "
            f"problem, so lowering the threshold will not surface it.")

    n_sup = int(stats.get("suppressed_sparse", 0))
    if n_sup:
        notes.append(
            f"{n_sup} candidate interval(s) crossed a duration threshold but "
            f"were discarded as too sparsely observed to assert (fewer than "
            f"{RULE_MIN_SAMPLES} observations, or averaging worse than one "
            f"every {RULE_MAX_SAMPLE_GAP_S:g}s). A cart in a busy doorway is "
            f"occluded by the traffic it obstructs, so re-check the video "
            f"before reading this run as clear.")

    reason = None
    if not door_zones and not static_zones:
        reason = ("No door / aisle zones drawn - the blocked-door and "
                  "static-cart rules did not run. Draw them in the Zone Editor.")
    if diagnostics is not None:
        diagnostics.extend(notes)
    return findings, reason


_SEV_ORDER = {"SAFETY": 4, "ACTION": 3, "WATCH": 2, "INFO": 1}


def _dedupe(findings: list[RuleFinding]) -> list[RuleFinding]:
    """Collapse findings describing one physical situation.

    Keyed on (rule_id, zone_id, CART) — one cart whose interval was split by
    track-id churn or a membership flicker collapses back into a single
    incident, which is what this exists for.

    Different carts are NEVER merged, even in the same zone at the same time.
    The key used to omit the cart, and abandoned_cart / incoming_empty always
    carry zone_id=None, so every zone-independent finding in a run shared one
    key: five carts abandoned in five different aisles collapsed into ONE
    finding attributed to whichever cart sorted first, with the rest buried in
    evidence["also_carts"]. These findings drive a retrieval work list and the
    tab badges, so the count IS the deliverable - merging distinct carts makes
    it wrong rather than tidier.
    """
    kept: list[RuleFinding] = []
    for f in sorted(findings, key=lambda x: (x.rule_id, x.zone_id or "",
                                             x.cart_display_id, x.start_t)):
        dup = None
        for k in kept:
            if (k.rule_id != f.rule_id or k.zone_id != f.zone_id
                    or k.cart_display_id != f.cart_display_id):
                continue
            # Overlapping in time, same rule, same place.
            if f.start_t <= k.end_t and k.start_t <= f.end_t:
                dup = k
                break
        if dup is None:
            kept.append(f)
            continue
        dup.start_t = min(dup.start_t, f.start_t)
        dup.end_t = max(dup.end_t, f.end_t)
        dup.duration_s = round(dup.end_t - dup.start_t, 2)
        dup.ongoing_at_eov = dup.ongoing_at_eov or f.ongoing_at_eov
        others = dup.evidence.setdefault("also_carts", [])
        if f.cart_display_id != dup.cart_display_id and f.cart_display_id not in others:
            others.append(f.cart_display_id)
    return kept


# ---------------------------------------------------------------------------
# Video overlay index
# ---------------------------------------------------------------------------
def overlay_index(bundle: TrajectoryBundle, findings) -> dict[int, list[dict]]:
    """Video frame number -> the rule badges to draw on that frame.

    Findings are post-hoc: they are only known once the whole track exists, so
    they cannot be drawn during the frame loop that writes the video. This maps
    each finding back onto the frames it covers, for a second pass over the
    already-written AVI (see video_io.reencode_to_mp4's frame_hook).

    Deliberately re-derived from the SAME finding objects the panel renders,
    rather than from a live approximation computed in the frame loop. A
    forward-only static/attendance test cannot reproduce `_attendance_bar` or
    the density gate, both of which read the whole track, so a live overlay
    would contradict the operational-alerts table on the same run.

    ``evidence["also_carts"]`` is folded in for parity with
    ui_builder.cart_flag_index(), which reads the same key. _dedupe() no longer
    merges distinct carts, so it is empty on a live run — but the two surfaces
    must not be able to disagree about a cart if that ever changes.
    """
    by_display: dict[int, list[TrackRecord]] = {}
    for rec in bundle.tracks.values():
        if rec.label != "cart":
            continue
        by_display.setdefault(int(rec.display_id), []).append(rec)

    out: dict[int, list[dict]] = {}
    for f in findings or []:
        rule_id = getattr(f, "rule_id", "") or ""
        text = RULE_LABELS.get(rule_id, getattr(f, "label", "") or rule_id)
        zone_name = getattr(f, "zone_name", None)
        severity = getattr(f, "severity", "INFO")
        degraded = getattr(f, "confidence", "high") != "high"
        start_t = float(getattr(f, "start_t", 0.0))
        end_t = float(getattr(f, "end_t", 0.0))
        also = (getattr(f, "evidence", {}) or {}).get("also_carts") or []
        carts = [getattr(f, "cart_display_id", None)] + list(also)
        for cd in carts:
            if cd is None:
                continue
            for rec in by_display.get(int(cd), []):
                # A re-identified track can carry positions with no boxes; the
                # badge is anchored to a box, so there is nothing to draw.
                if rec.bboxes.shape[0] < rec.timestamps.shape[0]:
                    continue
                sel = np.flatnonzero((rec.timestamps >= start_t)
                                     & (rec.timestamps <= end_t))
                for i in sel:
                    out.setdefault(int(rec.frames[i]), []).append({
                        "bbox": rec.bboxes[i],
                        # Which cart this is about. Two carts side by side in
                        # one doorway both get badged, and once a badge has
                        # been nudged clear of its neighbour it is no longer
                        # obvious from position alone which one it names.
                        "cart_display_id": int(cd),
                        "text": text,
                        "zone_name": zone_name,
                        "severity": severity,
                        "degraded": degraded,
                        # Counts up while the badge is on screen, so a viewer
                        # can see the threshold being crossed rather than being
                        # told after the fact that it was.
                        "elapsed_s": max(0.0, float(rec.timestamps[i]) - start_t),
                    })
    return out
