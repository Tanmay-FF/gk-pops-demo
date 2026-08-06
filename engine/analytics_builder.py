# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Pure-function analytics layer.  Consumes a TrajectoryBundle (already detected
+ tracked) plus user-defined Zones, returns an AnalyticsResult.

Imports only numpy + cv2 — no torch, no ultralytics — so this module is fast
to import and easy to unit-test against synthetic bundles.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from .analytics_models import (
    AnalyticsResult, DwellRow, JourneyEdge, QueueSpike,
    TrajectoryBundle, Zone,
)
from . import rules as rule_engine
from .config import (
    SPEED_STATIC,
    ZONE_CONGESTION_MIN_OCCUPANCY,
    ZONE_CONGESTION_HIGH_OCCUPANCY,
    ZONE_CONGESTION_STATIC_FRAC,
    ZONE_CONGESTION_LOW_AVG_SPEED,
    ZONE_CONGESTION_DWELL_ANCHOR_S,
    ZONE_CONGESTION_WATCH_SCORE,
    ZONE_CONGESTION_QUEUE_FORMING_SCORE,
    ZONE_CONGESTION_BACKED_UP_SCORE,
    CROWD_CLUSTER_RADIUS_PX,
    CROWD_CLUSTER_MIN_SIZE,
    CROWD_CLUSTER_MIN_DURATION_S,
    CROWD_CLUSTER_GAP_TOLERANCE_S,
    CROWD_CLUSTER_QUEUE_FORMING,
    CROWD_CLUSTER_BACKED_UP,
)


OUTSIDE_LABEL = "__OUTSIDE__"


# ---------------------------------------------------------------------------
# Zone membership
# ---------------------------------------------------------------------------
def build_zone_membership(bundle: TrajectoryBundle,
                          zones: list[Zone]) -> dict[int, np.ndarray]:
    """raw_id -> shape (N,) int8: zone index for each sample (-1 = outside).
    Each zone is rasterized once to a boolean mask; per-track lookup is O(N).
    Respects Zone.applies_to (person/cart/both) per track label.
    Last-zone-wins on overlap."""
    H, W = bundle.height, bundle.width

    rasterized: list[tuple[Zone, np.ndarray]] = []
    for z in zones:
        mask = np.zeros((H, W), dtype=np.uint8)
        if z.polygon.size:
            cv2.fillPoly(mask, [z.polygon.astype(np.int32)], 1)
        rasterized.append((z, mask))

    membership: dict[int, np.ndarray] = {}
    for raw_id, rec in bundle.tracks.items():
        n = rec.n_samples
        arr = np.full(n, -1, dtype=np.int8)
        if n == 0 or not zones:
            membership[raw_id] = arr
            continue
        xs = np.clip(rec.positions[:, 0].astype(np.int32), 0, W - 1)
        ys = np.clip(rec.positions[:, 1].astype(np.int32), 0, H - 1)
        for zi, (z, mask) in enumerate(rasterized):
            if z.applies_to != "both" and z.applies_to != rec.label:
                continue
            inside = mask[ys, xs] == 1
            arr = np.where(inside, zi, arr)
        membership[raw_id] = arr.astype(np.int8)
    return membership


# ---------------------------------------------------------------------------
# Dwell
# ---------------------------------------------------------------------------
def compute_dwell(bundle: TrajectoryBundle,
                  zones: list[Zone],
                  membership: dict[int, np.ndarray],
                  *, min_dwell_s: float = 1.0,
                  ) -> tuple[list[DwellRow], list[dict]]:
    """RLE the membership array → per-visit DwellRow; aggregate per zone."""
    rows: list[DwellRow] = []
    visit_counters: dict[tuple[int, str], int] = defaultdict(int)

    for raw_id, rec in bundle.tracks.items():
        arr = membership.get(raw_id)
        if arr is None or arr.size == 0:
            continue
        change_idxs = np.flatnonzero(np.diff(arr)) + 1
        run_starts = np.concatenate([[0], change_idxs])
        run_ends   = np.concatenate([change_idxs, [arr.size]])
        for s, e in zip(run_starts, run_ends):
            zi = int(arr[s])
            if zi == -1:
                continue
            z = zones[zi]
            if z.applies_to != "both" and z.applies_to != rec.label:
                continue                                # safety: shouldn't happen post-membership
            t0 = float(rec.timestamps[s])
            t1 = float(rec.timestamps[e - 1])
            dwell = max(0.0, t1 - t0)
            if dwell < min_dwell_s:
                continue
            key = (rec.display_id, z.zone_id)
            visit_counters[key] += 1
            rows.append(DwellRow(
                track_label=rec.label,
                display_id=rec.display_id,
                zone_id=z.zone_id, zone_name=z.name,
                enter_t=t0, exit_t=t1, dwell_seconds=dwell,
                visit_index=visit_counters[key],
            ))

    summary: list[dict] = []
    for z in zones:
        zone_rows = [r for r in rows if r.zone_id == z.zone_id]
        if not zone_rows:
            summary.append({
                "zone_id": z.zone_id, "zone_name": z.name,
                "applies_to": z.applies_to,
                "n_visits": 0, "n_unique": 0,
                "avg_dwell_s": 0.0, "p50_s": 0.0, "p95_s": 0.0,
                "max_s": 0.0, "total_s": 0.0,
            })
            continue
        dwells = np.array([r.dwell_seconds for r in zone_rows], dtype=np.float64)
        summary.append({
            "zone_id": z.zone_id, "zone_name": z.name,
            "applies_to": z.applies_to,
            "n_visits": len(zone_rows),
            "n_unique": len({(r.track_label, r.display_id) for r in zone_rows}),
            "avg_dwell_s": float(dwells.mean()),
            "p50_s": float(np.median(dwells)),
            "p95_s": float(np.percentile(dwells, 95)),
            "max_s": float(dwells.max()),
            "total_s": float(dwells.sum()),
        })
    return rows, summary


# ---------------------------------------------------------------------------
# Impressions (per-person fixture engagement, used by the 2D BEV)
# ---------------------------------------------------------------------------
def compute_impressions(bundle: TrajectoryBundle,
                        fixtures: list[dict],
                        *, min_dwell_s: float = 1.5,
                        slow_speed_px_s: float = 80.0,
                        ) -> dict[int, list[dict]]:
    """Per-person fixture engagement.

    A person *impresses* a fixture when their centroid is inside the
    fixture's pixel rectangle for >= min_dwell_s while moving below
    slow_speed_px_s. Walking-past visits are filtered out by the speed gate.

    fixtures: list of {"id": str, "label": str,
                       "x1": int, "y1": int, "x2": int, "y2": int}.
    Returns: display_id (int) -> ordered list of impression dicts.
    """
    if not fixtures:
        return {}

    H, W = bundle.height, bundle.width
    rasters: list[tuple[dict, np.ndarray]] = []
    for fx in fixtures:
        x1 = max(0, int(fx["x1"]))
        y1 = max(0, int(fx["y1"]))
        x2 = min(W, int(fx["x2"]))
        y2 = min(H, int(fx["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue
        m = np.zeros((H, W), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        rasters.append((fx, m))

    impressions: dict[int, list[dict]] = {}
    for raw_id, rec in bundle.tracks.items():
        if rec.label != "person" or rec.n_samples == 0:
            continue
        xs = np.clip(rec.positions[:, 0].astype(np.int32), 0, W - 1)
        ys = np.clip(rec.positions[:, 1].astype(np.int32), 0, H - 1)
        slow = rec.speeds <= slow_speed_px_s

        person_imps: list[dict] = []
        for fx, mask in rasters:
            inside = (mask[ys, xs] == 1) & slow
            if not inside.any():
                continue
            # Run-length encode the boolean array; keep runs >= min_dwell_s.
            change_idxs = np.flatnonzero(np.diff(inside.astype(np.int8))) + 1
            run_starts = np.concatenate([[0], change_idxs])
            run_ends   = np.concatenate([change_idxs, [inside.size]])
            for s, e in zip(run_starts, run_ends):
                if not bool(inside[s]):
                    continue
                t0 = float(rec.timestamps[s])
                t1 = float(rec.timestamps[e - 1])
                dwell = max(0.0, t1 - t0)
                if dwell < min_dwell_s:
                    continue
                person_imps.append({
                    "fixture_id": fx["id"],
                    "fixture_label": fx["label"],
                    "enter_t": round(t0, 2),
                    "dwell_s": round(dwell, 2),
                })
        if person_imps:
            person_imps.sort(key=lambda d: d["enter_t"])
            impressions[rec.display_id] = person_imps

    return impressions


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def compute_heatmap(bundle: TrajectoryBundle,
                    *, label_filter: Optional[str] = "person",
                    sigma_px: int = 24,
                    out_png_path: Optional[str] = None,
                    background: Optional[np.ndarray] = None,
                    ) -> tuple[Optional[str], np.ndarray, np.ndarray]:
    """Scatter centroids → blur → log-scale → JET colormap → alpha blend.
    Returns (saved_png_path_or_None, normalized_density(H,W), composite_bgr(H,W,3))."""
    H, W = bundle.height, bundle.width
    acc = np.zeros((H, W), dtype=np.float32)

    for rec in bundle.tracks.values():
        if label_filter and rec.label != label_filter:
            continue
        if rec.positions.size == 0:
            continue
        xs = np.clip(rec.positions[:, 0].astype(np.int32), 0, W - 1)
        ys = np.clip(rec.positions[:, 1].astype(np.int32), 0, H - 1)
        np.add.at(acc, (ys, xs), 1.0)

    if acc.max() > 0:
        # Downsample → blur at proportionally smaller sigma → upscale.
        # GaussianBlur is the dominant cost (separable, O(H*W*ksize*2)); on a
        # 720p+ frame with sigma=24, the ksize-145 kernel runs ~4× faster on
        # half-res input. Skipped on small clips where the kernel is cheap.
        scale = 2 if min(H, W) >= 256 else 1
        if scale > 1:
            acc_small = cv2.resize(acc, (W // scale, H // scale),
                                   interpolation=cv2.INTER_AREA)
            sigma_eff = sigma_px / scale
        else:
            acc_small = acc
            sigma_eff = sigma_px
        ksize = int(max(3, 6 * sigma_eff + 1))
        if ksize % 2 == 0:
            ksize += 1
        blurred_small = cv2.GaussianBlur(acc_small, (ksize, ksize), sigma_eff)
        blurred = (cv2.resize(blurred_small, (W, H), interpolation=cv2.INTER_LINEAR)
                   if scale > 1 else blurred_small)
        normalized = np.log1p(blurred)
        peak = float(normalized.max())
        if peak > 0:
            normalized = normalized / peak
        else:
            normalized = np.zeros_like(normalized)
    else:
        normalized = acc

    heat_u8 = (normalized * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    if background is None:
        bg = np.full((H, W, 3), 30, dtype=np.uint8)
    else:
        bg = (cv2.resize(background, (W, H))
              if background.shape[:2] != (H, W) else background.copy())

    # alpha-blend only where there's measurable density (avoids tinting cold zones)
    mask = (heat_u8 > 5).astype(np.float32)[:, :, None]
    composite = (color.astype(np.float32) * mask * 0.7
                 + bg.astype(np.float32) * (1.0 - mask * 0.7)).astype(np.uint8)

    saved_path: Optional[str] = None
    if out_png_path:
        os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
        cv2.imwrite(out_png_path, composite)
        saved_path = out_png_path

    return saved_path, normalized, composite


# ---------------------------------------------------------------------------
# Journeys
# ---------------------------------------------------------------------------
def compute_journeys(bundle: TrajectoryBundle,
                     zones: list[Zone],
                     membership: dict[int, np.ndarray],
                     *, label_filter: Optional[str] = "person",
                     ) -> tuple[list[JourneyEdge], np.ndarray, list[str]]:
    """Walk each track's RLE'd membership, emit zone→zone transitions.
    The OUTSIDE bucket sits at index len(zones)."""
    n = len(zones) + 1
    labels = [z.name for z in zones] + [OUTSIDE_LABEL]
    matrix = np.zeros((n, n), dtype=np.int64)
    edges: list[JourneyEdge] = []

    def name_of(zi: int) -> str:
        return zones[zi].name if zi != -1 else OUTSIDE_LABEL

    def idx_of(zi: int) -> int:
        return zi if zi != -1 else len(zones)

    for raw_id, rec in bundle.tracks.items():
        if label_filter and rec.label != label_filter:
            continue
        arr = membership.get(raw_id)
        if arr is None or arr.size < 2:
            continue
        change_idxs = np.flatnonzero(np.diff(arr)) + 1
        if change_idxs.size == 0:
            continue
        run_starts = np.concatenate([[0], change_idxs])
        run_ends   = np.concatenate([change_idxs, [arr.size]])
        seq = [(int(arr[s]), float(rec.timestamps[s])) for s, _ in zip(run_starts, run_ends)]
        for i in range(len(seq) - 1):
            src_zi, _ = seq[i]
            dst_zi, t_at = seq[i + 1]
            matrix[idx_of(src_zi), idx_of(dst_zi)] += 1
            edges.append(JourneyEdge(
                track_label=rec.label, display_id=rec.display_id,
                src_zone=name_of(src_zi),
                dst_zone=name_of(dst_zi),
                transition_t=t_at,
            ))
    return edges, matrix, labels


# ---------------------------------------------------------------------------
# Zone congestion — multi-signal scoring
# ---------------------------------------------------------------------------
# A zone is congested when several signals agree. Each signal contributes
# weighted points to a 0..100 score; severity is bucketed off the score.
#
# Signals & weights (max contribution):
#   peak_occupancy   →  40 pts   (most direct measure)
#   static_fraction  →  25 pts   (fraction of in-zone samples standing still)
#   avg_in_zone_speed → 20 pts   (lower = more stalled)
#   avg_dwell_s      →  15 pts   (anchored at the existing dwell threshold)


def _zone_intervals(bundle: TrajectoryBundle,
                    membership: dict[int, np.ndarray],
                    zi: int) -> list[tuple[float, float]]:
    """Per-track (enter_t, exit_t) intervals for visits to zone index `zi`."""
    intervals: list[tuple[float, float]] = []
    for raw_id, rec in bundle.tracks.items():
        arr = membership.get(raw_id)
        if arr is None or arr.size == 0:
            continue
        in_zone = (arr == zi).astype(np.int8)
        if in_zone.sum() == 0:
            continue
        # RLE: pad with zeros, find +1 (enter) and -1 (exit) edges.
        diff = np.diff(np.concatenate([[0], in_zone, [0]]))
        starts = np.flatnonzero(diff == 1)
        ends   = np.flatnonzero(diff == -1)              # exclusive end index
        ts = rec.timestamps
        for s, e in zip(starts, ends):
            t0 = float(ts[s])
            t1 = float(ts[min(e - 1, ts.size - 1)])
            intervals.append((t0, max(t1, t0)))
    return intervals


def _peak_concurrent(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    """Sweep-line peak concurrent occupancy + total dwell time."""
    if not intervals:
        return 0, 0.0
    events: list[tuple[float, int]] = []
    for t0, t1 in intervals:
        events.append((t0, +1))
        # Tie-break: process exits before enters at the same instant so
        # back-to-back visits don't inflate the count.
        events.append((t1, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    cur, peak = 0, 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    total_dwell = sum(t1 - t0 for t0, t1 in intervals)
    return peak, total_dwell


def _zone_motion_stats(bundle: TrajectoryBundle,
                       membership: dict[int, np.ndarray],
                       zi: int,
                       static_speed_px_s: float,
                       ) -> tuple[float, float, int]:
    """(static_fraction, mean_speed_px_s, n_in_zone_samples) for zone index `zi`."""
    n_total = 0
    n_static = 0
    speed_sum = 0.0
    for raw_id, rec in bundle.tracks.items():
        arr = membership.get(raw_id)
        if arr is None or arr.size == 0:
            continue
        n = min(arr.size, rec.speeds.size)
        if n == 0:
            continue
        mask = arr[:n] == zi
        if not mask.any():
            continue
        speeds = rec.speeds[:n][mask]
        n_total += int(speeds.size)
        n_static += int(np.count_nonzero(speeds < static_speed_px_s))
        speed_sum += float(speeds.sum())
    if n_total == 0:
        return 0.0, 0.0, 0
    return n_static / n_total, speed_sum / n_total, n_total


def _score_congestion(*, peak_occ: int, mean_occ: float,
                      static_frac: float, avg_speed: float,
                      avg_dwell_s: float,
                      n_in_zone_samples: int,
                      ) -> tuple[float, list[str]]:
    """Combine the four signals into a 0..100 score + human-readable reasons."""
    score = 0.0
    reasons: list[str] = []

    # 1) Peak concurrent occupancy (max 40 pts).
    if peak_occ >= ZONE_CONGESTION_HIGH_OCCUPANCY:
        score += 40.0
        reasons.append(f"peak {peak_occ} concurrent")
    elif peak_occ >= ZONE_CONGESTION_MIN_OCCUPANCY:
        span = max(1, ZONE_CONGESTION_HIGH_OCCUPANCY - ZONE_CONGESTION_MIN_OCCUPANCY)
        pts = 15.0 + 25.0 * (peak_occ - ZONE_CONGESTION_MIN_OCCUPANCY) / span
        score += pts
        reasons.append(f"peak {peak_occ} concurrent")

    # 2) Static fraction (max 25 pts) — only meaningful with enough samples.
    if n_in_zone_samples >= 30 and static_frac >= ZONE_CONGESTION_STATIC_FRAC:
        score += 25.0 * min(1.0, static_frac / 0.70)
        reasons.append(f"{int(round(static_frac * 100))}% stationary")

    # 3) Mean in-zone speed (max 20 pts).
    if n_in_zone_samples >= 30 and 0.0 < avg_speed < ZONE_CONGESTION_LOW_AVG_SPEED:
        score += 20.0 * (1.0 - avg_speed / ZONE_CONGESTION_LOW_AVG_SPEED)
        reasons.append(f"avg in-zone speed {avg_speed:.0f} px/s")

    # 4) Avg dwell, anchored at the existing threshold (max 15 pts).
    if avg_dwell_s >= ZONE_CONGESTION_DWELL_ANCHOR_S:
        excess = avg_dwell_s - ZONE_CONGESTION_DWELL_ANCHOR_S
        score += min(15.0, 15.0 * excess / 90.0)
        reasons.append(f"avg dwell {avg_dwell_s:.0f}s")

    # Mean occupancy is informational only (kept on the QueueSpike record),
    # not a scoring contributor — it's already implied by peak + dwell.
    _ = mean_occ
    return min(100.0, score), reasons


def _bucket_severity(score: float) -> str:
    if score >= ZONE_CONGESTION_BACKED_UP_SCORE:
        return "BACKED_UP"
    if score >= ZONE_CONGESTION_QUEUE_FORMING_SCORE:
        return "QUEUE_FORMING"
    if score >= ZONE_CONGESTION_WATCH_SCORE:
        return "WATCH"
    return "NORMAL"


def compute_zone_congestion(bundle: TrajectoryBundle,
                            zones: list[Zone],
                            membership: dict[int, np.ndarray],
                            dwell_summary: list[dict],
                            ) -> list[dict]:
    """Per-zone multi-signal congestion metrics. One dict per zone."""
    by_zone_id = {s["zone_id"]: s for s in dwell_summary}
    clip_dur = (bundle.total_frames / bundle.fps) if bundle.fps > 0 else 0.0
    out: list[dict] = []

    for zi, z in enumerate(zones):
        ds = by_zone_id.get(z.zone_id, {})
        intervals = _zone_intervals(bundle, membership, zi)
        peak_occ, total_dwell = _peak_concurrent(intervals)
        mean_occ = (total_dwell / clip_dur) if clip_dur > 0 else 0.0
        static_frac, avg_speed, n_samples = _zone_motion_stats(
            bundle, membership, zi, SPEED_STATIC,
        )
        score, reasons = _score_congestion(
            peak_occ=peak_occ, mean_occ=mean_occ,
            static_frac=static_frac, avg_speed=avg_speed,
            avg_dwell_s=ds.get("avg_dwell_s", 0.0),
            n_in_zone_samples=n_samples,
        )
        out.append({
            "zone_id": z.zone_id,
            "zone_name": z.name,
            "score": score,
            "severity": _bucket_severity(score),
            "peak_occupancy": peak_occ,
            "mean_occupancy": mean_occ,
            "static_fraction": static_frac,
            "avg_speed_px_s": avg_speed,
            "avg_dwell_s": ds.get("avg_dwell_s", 0.0),
            "max_dwell_s": ds.get("max_s", 0.0),
            "n_visits": ds.get("n_visits", 0),
            "reasons": reasons,
        })
    return out


def detect_queue_spikes(dwell_summary: list[dict],
                        *, threshold_s: float = 30.0,
                        congestion_metrics: list[dict] | None = None,
                        ) -> tuple[list[QueueSpike], list[dict]]:
    """Build QueueSpike records from the multi-signal congestion metrics.

    Falls back to legacy avg-dwell-only behaviour when `congestion_metrics`
    is None (so external callers / tests that hit the old signature still
    work). Severity NORMAL is filtered out so only zones worth surfacing
    appear in the result."""
    spikes: list[QueueSpike] = []
    events: list[dict] = []

    if congestion_metrics is None:
        # Legacy single-signal fallback: avg dwell vs threshold.
        for s in dwell_summary:
            avg = s["avg_dwell_s"]
            if avg < threshold_s:
                continue
            sev = ("BACKED_UP" if avg >= 120.0 else
                   "QUEUE_FORMING" if avg >= 60.0 else "WATCH")
            spikes.append(QueueSpike(
                zone_id=s["zone_id"], zone_name=s["zone_name"],
                avg_dwell_s=avg, max_dwell_s=s["max_s"],
                n_visits=s["n_visits"], threshold_s=threshold_s, severity=sev,
            ))
            events.append({
                "zone_id": s["zone_id"], "zone_name": s["zone_name"],
                "severity": sev, "avg_dwell_s": avg,
                "max_dwell_s": s["max_s"], "n_visits": s["n_visits"],
                "threshold_s": threshold_s,
            })
        return spikes, events

    for m in congestion_metrics:
        if m["severity"] == "NORMAL":
            continue
        spike = QueueSpike(
            zone_id=m["zone_id"], zone_name=m["zone_name"],
            avg_dwell_s=m["avg_dwell_s"], max_dwell_s=m["max_dwell_s"],
            n_visits=m["n_visits"], threshold_s=threshold_s,
            severity=m["severity"],
            score=m["score"],
            peak_occupancy=m["peak_occupancy"],
            mean_occupancy=m["mean_occupancy"],
            static_fraction=m["static_fraction"],
            avg_speed_px_s=m["avg_speed_px_s"],
            reasons=list(m["reasons"]),
        )
        spikes.append(spike)
        events.append({
            "zone_id": m["zone_id"], "zone_name": m["zone_name"],
            "severity": m["severity"], "score": m["score"],
            "avg_dwell_s": m["avg_dwell_s"], "max_dwell_s": m["max_dwell_s"],
            "n_visits": m["n_visits"], "threshold_s": threshold_s,
            "peak_occupancy": m["peak_occupancy"],
            "static_fraction": m["static_fraction"],
            "avg_speed_px_s": m["avg_speed_px_s"],
            "reasons": list(m["reasons"]),
        })
    return spikes, events


# ---------------------------------------------------------------------------
# Insight narrative
# ---------------------------------------------------------------------------
def build_insight_text(bundle: TrajectoryBundle,
                       zones: list[Zone],
                       dwell_summary: list[dict],
                       queue_spikes,
                       journey_edges,
                       ) -> str:
    """Generate a bullet-point narrative from the computed analytics data.

    Returns a newline-separated string; each line is one bullet. Renderers
    split on ``\\n`` and decorate (HTML <ul>/<li> for the Analytics card,
    `white-space: pre-line` + `• ` prefix for the BEV2D download page).
    """
    n_people = sum(1 for r in bundle.tracks.values() if r.label == "person")
    n_carts  = sum(1 for r in bundle.tracks.values() if r.label == "cart")
    dur_s    = bundle.total_frames / bundle.fps if bundle.fps > 0 else 0

    def _fmt(s: float) -> str:
        if s < 60:
            return f"{s:.0f}s"
        return f"{int(s // 60)}m{int(s % 60):02d}s"

    bullets: list[str] = []

    # Clip overview — split into two bullets so the user scans them faster
    bullets.append(f"Clip duration: {_fmt(dur_s)}.")
    bullets.append(f"Tracked {n_people} person(s) and {n_carts} cart(s).")

    if not zones or not dwell_summary:
        bullets.append(
            "No zones defined — draw zones in the Zone Editor tab for dwell and journey insights."
        )
        return "\n".join(bullets)

    # Zone dwell breakdown — one bullet per zone
    for s in dwell_summary:
        if s["n_visits"] == 0:
            bullets.append(f"{s['zone_name']}: no visits.")
        else:
            bullets.append(
                f"{s['zone_name']}: {s['n_visits']} visit(s), avg {_fmt(s['avg_dwell_s'])}"
                f" (peak {_fmt(s['max_s'])})"
                f" from {s['n_unique']} unique track(s)."
            )

    # Most/least engaged zone — only compute "average dwell" winners over
    # zones with enough visits to be statistically meaningful, otherwise the
    # average is just one or two samples and a single outlier wins trivially.
    # Always also surface the longest single visit, which is robust to sample
    # size and a better proxy for "deepest engagement".
    visited = [s for s in dwell_summary if s["n_visits"] > 0]
    if visited:
        MIN_SAMPLES_FOR_AVG = 3
        confident = [s for s in visited if s["n_visits"] >= MIN_SAMPLES_FOR_AVG]
        if confident:
            hottest = max(confident, key=lambda s: s["avg_dwell_s"])
            coldest = min(confident, key=lambda s: s["avg_dwell_s"])
            bullets.append(
                f"Longest average dwell: {hottest['zone_name']} "
                f"({_fmt(hottest['avg_dwell_s'])} avg over {hottest['n_visits']} visits)."
            )
            if hottest["zone_id"] != coldest["zone_id"]:
                bullets.append(
                    f"Shortest average dwell: {coldest['zone_name']} "
                    f"({_fmt(coldest['avg_dwell_s'])} avg over {coldest['n_visits']} visits)."
                )
        else:
            bullets.append(
                f"Average-dwell ranking skipped — no zone yet has "
                f"{MIN_SAMPLES_FOR_AVG}+ visits (sample too small)."
            )

        longest_peak = max(visited, key=lambda s: s["max_s"])
        bullets.append(
            f"Longest single visit: {longest_peak['zone_name']} "
            f"({_fmt(longest_peak['max_s'])})."
        )

    # Queue spikes — one bullet per spike, plus a fallback when none
    if queue_spikes:
        for s in queue_spikes:
            bullets.append(f"Queue/dwell alert: {s.zone_name} ({s.severity}).")
    else:
        bullets.append("No queue spikes detected.")

    # Journey summary
    if journey_edges:
        bullets.append(f"{len(journey_edges)} zone transition(s) recorded across all tracks.")

    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Zone-free crowd-cluster detection
# ---------------------------------------------------------------------------
def _crowd_severity(peak: int, duration: float) -> str:
    bp_size, bp_dur = CROWD_CLUSTER_BACKED_UP
    qf_size, qf_dur = CROWD_CLUSTER_QUEUE_FORMING
    if peak >= bp_size and duration >= bp_dur:
        return "BACKED_UP"
    if peak >= qf_size and duration >= qf_dur:
        return "QUEUE_FORMING"
    return "WATCH"


def _per_frame_persons(bundle: TrajectoryBundle
                       ) -> dict[int, list[tuple[int, float, float, float]]]:
    """frame_idx -> list of (raw_id, x, y, t) for every person sample."""
    out: dict[int, list[tuple[int, float, float, float]]] = {}
    for raw_id, rec in bundle.tracks.items():
        if rec.label != "person" or rec.n_samples == 0:
            continue
        for i in range(rec.n_samples):
            f = int(rec.frames[i])
            out.setdefault(f, []).append((
                int(raw_id),
                float(rec.positions[i, 0]),
                float(rec.positions[i, 1]),
                float(rec.timestamps[i]),
            ))
    return out


def _connected_components(persons: list, radius_px: float) -> list[list[int]]:
    """Union-find clustering on pairwise centroid distance. Returns groups
    as lists of indices into `persons`."""
    n = len(persons)
    if n < 2:
        return [[0]] if n == 1 else []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    r2 = radius_px * radius_px
    for i in range(n):
        _, xi, yi, _ = persons[i]
        for j in range(i + 1, n):
            _, xj, yj, _ = persons[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= r2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def detect_crowd_clusters(bundle: TrajectoryBundle,
                          *, radius_px: float = CROWD_CLUSTER_RADIUS_PX,
                          min_cluster_size: int = CROWD_CLUSTER_MIN_SIZE,
                          min_duration_s: float = CROWD_CLUSTER_MIN_DURATION_S,
                          gap_tolerance_s: float = CROWD_CLUSTER_GAP_TOLERANCE_S,
                          ) -> list[QueueSpike]:
    """Zone-free queue detection.

    Walks the trajectory bundle frame-by-frame, finds connected components of
    person centroids within `radius_px` of each other, and emits a synthetic
    QueueSpike whenever a cluster of >= `min_cluster_size` people persists for
    at least `min_duration_s`. Severity is bucketed off (peak_size, duration).

    The returned spikes flow into `AnalyticsResult.queue_spikes` alongside the
    zone-based spikes; the existing `build_alert_banner` surfaces the
    QUEUE_FORMING / BACKED_UP ones in the sticky UI banner with no extra
    plumbing required.
    """
    per_frame = _per_frame_persons(bundle)
    if not per_frame:
        return []

    # Per-frame cluster samples: (t, cx, cy, size, members frozenset)
    samples: list[tuple[float, float, float, int, frozenset]] = []
    for f in sorted(per_frame.keys()):
        persons = per_frame[f]
        for grp in _connected_components(persons, radius_px):
            if len(grp) < min_cluster_size:
                continue
            cx = sum(persons[i][1] for i in grp) / len(grp)
            cy = sum(persons[i][2] for i in grp) / len(grp)
            t  = persons[grp[0]][3]
            members = frozenset(persons[i][0] for i in grp)
            samples.append((t, cx, cy, len(grp), members))

    # Stitch samples into events: same event if within gap_tolerance_s,
    # within radius_px, and sharing at least one member id.
    events: list[list[tuple[float, float, float, int, frozenset]]] = []
    r2 = radius_px * radius_px
    for s in samples:
        t, cx, cy, _, mem = s
        merged = False
        for ev in reversed(events):
            lt, lcx, lcy, _, lmem = ev[-1]
            if t - lt > gap_tolerance_s:
                continue
            if (cx - lcx) ** 2 + (cy - lcy) ** 2 > r2:
                continue
            if not (mem & lmem):
                continue
            ev.append(s)
            merged = True
            break
        if not merged:
            events.append([s])

    spikes: list[QueueSpike] = []
    for idx, ev in enumerate(events):
        duration = ev[-1][0] - ev[0][0]
        if duration < min_duration_s:
            continue
        peak = max(s[3] for s in ev)
        mean = sum(s[3] for s in ev) / len(ev)
        severity = _crowd_severity(peak, duration)
        spikes.append(QueueSpike(
            zone_id=f"crowd_{idx}",
            zone_name=f"Crowd cluster ({peak} people, {duration:.1f}s)",
            avg_dwell_s=duration,
            max_dwell_s=duration,
            n_visits=peak,
            threshold_s=min_duration_s,
            severity=severity,
            score=min(100.0, peak * duration),
            peak_occupancy=peak,
            mean_occupancy=mean,
            static_fraction=0.0,
            avg_speed_px_s=0.0,
            reasons=[f"{peak} people clustered for {duration:.1f}s"],
        ))
    return spikes


def run_all(bundle: TrajectoryBundle,
            zones: list[Zone],
            *, dwell_threshold_s: float = 30.0,
            heatmap_background: Optional[np.ndarray] = None,
            out_dir: Optional[str] = None,
            min_dwell_s: float = 1.0,
            camera_placement: str = "Outside (facing entrance)",
            ) -> AnalyticsResult:
    """Single entry point used by the UI.  Empty zones → only the heatmap is computed.

    Layout zones (kind != "analytics") are filtered out here — they're for the
    Floor Map BEV's structural overlay and never participate in dwell, journey,
    or congestion analytics.

    The operational rule engine is the exception: it reads door / aisle /
    fixture zones, which is why it gets the FULL zone list rather than
    analytics_zones.
    """
    analytics_zones = [z for z in zones if getattr(z, "kind", "analytics") == "analytics"]
    membership = build_zone_membership(bundle, analytics_zones)

    if analytics_zones:
        dwell_rows, dwell_summary = compute_dwell(
            bundle, analytics_zones, membership, min_dwell_s=min_dwell_s)
        edges, matrix, labels = compute_journeys(
            bundle, analytics_zones, membership, label_filter="person")
        congestion_metrics = compute_zone_congestion(
            bundle, analytics_zones, membership, dwell_summary,
        )
        spikes, spike_events = detect_queue_spikes(
            dwell_summary, threshold_s=dwell_threshold_s,
            congestion_metrics=congestion_metrics,
        )
    else:
        dwell_rows, dwell_summary = [], []
        edges, matrix, labels = [], None, []
        spikes, spike_events = [], []

    # Zone-free crowd-cluster spikes — fire even when the user has not drawn
    # any zones. These flow into the same alert banner as the zone-based ones.
    crowd_spikes = detect_crowd_clusters(bundle)
    if crowd_spikes:
        spikes = list(spikes) + crowd_spikes
        spike_events.extend([{
            "zone_id": s.zone_id,
            "zone_name": s.zone_name,
            "severity": s.severity,
            "peak_occupancy": s.peak_occupancy,
            "mean_occupancy": s.mean_occupancy,
            "avg_dwell_s": s.avg_dwell_s,
            "max_dwell_s": s.max_dwell_s,
            "n_visits": s.n_visits,
            "threshold_s": s.threshold_s,
            "reasons": list(s.reasons),
            "source": "crowd_cluster",
        } for s in crowd_spikes])

    heatmap_png_path = None
    if out_dir is not None:
        heatmap_png_path = os.path.join(out_dir, f"heatmap_{bundle.video_key}.png")
    saved_path, density, composite = compute_heatmap(
        bundle, sigma_px=24, out_png_path=heatmap_png_path,
        background=heatmap_background, label_filter="person")

    insight = build_insight_text(bundle, analytics_zones, dwell_summary, spikes, edges)

    # --- Operational rule engine ------------------------------------------
    # Gets the full zone list (door / aisle / fixture), not analytics_zones.
    # Isolated behind try/except: a rule bug must never take down dwell,
    # heatmap, or the POPS dashboard alongside it.
    try:
        rule_findings, rules_reason = rule_engine.evaluate_rules(
            bundle, zones, camera_placement=camera_placement)
    except Exception as e:                                   # pragma: no cover
        import traceback
        traceback.print_exc()
        rule_findings, rules_reason = [], f"Rule evaluation failed: {e}"

    return AnalyticsResult(
        dwell_rows=dwell_rows,
        dwell_summary=dwell_summary,
        heatmap_png_path=saved_path,
        heatmap_array=density,
        heatmap_composite=composite,
        journey_edges=edges,
        journey_matrix=matrix,
        journey_labels=labels,
        queue_spikes=spikes,
        spike_events=spike_events,
        insight_text=insight,
        rule_findings=rule_findings,
        rules_unavailable_reason=rules_reason,
    )
