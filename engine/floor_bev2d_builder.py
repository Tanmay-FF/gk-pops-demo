# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Animated 2D floor-plan BEV builder.

Generates a self-contained HTML/Canvas2D document showing person/cart
tracks animated over a calibrated floor plan with zone overlays,
heatmap, journey details, and analytics sidebar.

Public surface:
    build_floor_2d_html(bundle, zones, analytics, H=None) -> str
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import numpy as np

from .analytics_models import AnalyticsResult, TrackRecord, TrajectoryBundle, Zone

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _apply_homography(positions: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography H to pixel-space positions (N,2) -> floor (N,2)."""
    if positions.shape[0] == 0:
        return positions.copy()
    ones = np.ones((positions.shape[0], 1), dtype=np.float64)
    hom = np.hstack([positions.astype(np.float64), ones])   # (N, 3)
    warped = (H @ hom.T).T                                   # (N, 3)
    w = warped[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return (warped[:, :2] / w).astype(np.float32)


def _to_floor(positions: np.ndarray, H: Optional[np.ndarray]) -> np.ndarray:
    """Convert pixel positions to floor coords. H=None → divide by 100."""
    if H is not None:
        return _apply_homography(positions, H)
    return (positions / 100.0).astype(np.float32)


def _track_to_floor(rec: TrackRecord,
                    H: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (floor_positions (N,2), normalized_speeds (N,))."""
    floor_pos = _to_floor(rec.positions, H)
    speeds = rec.speeds.astype(np.float32)
    return floor_pos, speeds


def _poly_to_floor(polygon: np.ndarray,
                   H: Optional[np.ndarray]) -> list[list[float]]:
    """Convert a (V,2) int32 pixel polygon to floor coords list-of-pairs."""
    if polygon.size == 0:
        return []
    pts = _to_floor(polygon.astype(np.float32), H)
    return [[float(x), float(y)] for x, y in pts]


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _compute_heatmap(all_floor_positions: np.ndarray,
                     grid_m: float = 0.5) -> Optional[dict]:
    """Build a 2D histogram with 0.5 m cells, Gaussian blur, log1p normalize."""
    if all_floor_positions.shape[0] < 10:
        return None

    xs = all_floor_positions[:, 0]
    ys = all_floor_positions[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    # Ensure at least 1-cell extent
    if x_max - x_min < grid_m:
        x_max = x_min + grid_m
    if y_max - y_min < grid_m:
        y_max = y_min + grid_m

    w = max(1, int(np.ceil((x_max - x_min) / grid_m)))
    h = max(1, int(np.ceil((y_max - y_min) / grid_m)))

    grid = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(((xs - x_min) / grid_m).astype(int), 0, w - 1)
    yi = np.clip(((ys - y_min) / grid_m).astype(int), 0, h - 1)
    np.add.at(grid, (yi, xi), 1.0)

    # Gaussian blur — sigma in cells
    sigma = 1.5
    if _HAS_SCIPY:
        grid = _scipy_gaussian(grid, sigma=sigma)
    else:
        # Simple box approximation when scipy is absent
        ksize = max(3, int(sigma * 3) * 2 + 1)
        import cv2
        grid = cv2.GaussianBlur(grid, (ksize, ksize), sigma)

    # Log1p normalize
    grid = np.log1p(grid)
    peak = float(grid.max())
    if peak > 0:
        grid = grid / peak

    return {
        "grid": grid.tolist(),
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "w": w, "h": h,
    }


# ---------------------------------------------------------------------------
# Journey builder
# ---------------------------------------------------------------------------

def _build_journeys(journey_edges, dwell_rows) -> dict:
    """Returns {(label, display_id): {"zones": [...], "dwell": {z: avg}}}.

    Source of truth is **dwell_rows**, not journey_edges. ``compute_journeys``
    in analytics_builder skips any track that never transitions between zones
    (a person who walks in and stays → 0 edges) and filters carts out
    entirely. Building from dwell_rows fixes both: every track that spent
    at least ``min_dwell_s`` in any zone gets a journey entry.

    Keyed by ``(label, display_id)`` because person and cart numbering are
    independent — P3 and C3 are different tracks.
    """
    # Group dwell rows per (label, display_id), preserving timing for ordering.
    grouped: dict[tuple[str, int], list] = defaultdict(list)
    for row in dwell_rows:
        grouped[(row.track_label, row.display_id)].append(row)

    result: dict[tuple[str, int], dict] = {}
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: r.enter_t)

        # Zone sequence in order of first entry; collapse consecutive duplicates.
        zones: list[str] = []
        for r in rows_sorted:
            if not zones or zones[-1] != r.zone_name:
                zones.append(r.zone_name)

        # Average dwell per zone across all visits.
        per_zone: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            per_zone[r.zone_name].append(r.dwell_seconds)
        dwell: dict[str, float] = {}
        for z in zones:
            vals = per_zone.get(z, [])
            if vals:
                dwell[z] = round(float(np.mean(vals)), 1)

        result[key] = {"zones": zones, "dwell": dwell}

    return result


# ---------------------------------------------------------------------------
# Congestion data helper
# ---------------------------------------------------------------------------

def _severity_for_zone(zone_id: str, analytics: AnalyticsResult) -> str:
    for spike in analytics.queue_spikes:
        if spike.zone_id == zone_id:
            return spike.severity
    return "NORMAL"


def _zone_analytics(zone_id: str, analytics: AnalyticsResult) -> dict:
    """Returns score, severity, avg_dwell_s, n_visits, peak_occupancy for a zone."""
    for spike in analytics.queue_spikes:
        if spike.zone_id == zone_id:
            return {
                "severity": spike.severity,
                "score": round(spike.score, 1),
                "avg_dwell_s": round(spike.avg_dwell_s, 1),
                "n_visits": spike.n_visits,
                "peak_occupancy": spike.peak_occupancy,
            }
    # Zone not in spikes → look in dwell_summary
    for s in analytics.dwell_summary:
        if s["zone_id"] == zone_id:
            return {
                "severity": "NORMAL",
                "score": 0.0,
                "avg_dwell_s": round(s.get("avg_dwell_s", 0.0), 1),
                "n_visits": s.get("n_visits", 0),
                "peak_occupancy": 0,
            }
    return {"severity": "NORMAL", "score": 0.0, "avg_dwell_s": 0.0,
            "n_visits": 0, "peak_occupancy": 0}


# ---------------------------------------------------------------------------
# Main payload builder
# ---------------------------------------------------------------------------

def _build_payload(bundle: TrajectoryBundle,
                   zones: list[Zone],
                   analytics: AnalyticsResult,
                   H: Optional[np.ndarray]) -> dict:
    fps = bundle.fps or 25.0
    total_frames = bundle.total_frames

    n_people = sum(1 for r in bundle.tracks.values() if r.label == "person")
    n_carts  = sum(1 for r in bundle.tracks.values() if r.label == "cart")

    # --- Collect all floor positions for bounds + heatmap ---
    all_floor: list[np.ndarray] = []
    track_payloads: list[dict] = []
    journeys = _build_journeys(analytics.journey_edges, analytics.dwell_rows)

    # Compute per-track 95th-percentile speed for normalization
    all_speeds: list[float] = []
    for rec in bundle.tracks.values():
        if rec.speeds.size:
            all_speeds.extend(rec.speeds.tolist())
    all_speeds.sort()
    speed_95 = float(all_speeds[max(0, int(len(all_speeds) * 0.95) - 1)]) if all_speeds else 1.0
    speed_95 = max(speed_95, 1.0)

    for rec in bundle.tracks.values():
        floor_pos, speeds = _track_to_floor(rec, H)
        if floor_pos.shape[0] == 0:
            continue
        all_floor.append(floor_pos)

        norm_speeds = (speeds / speed_95).clip(0.0, 1.0).tolist()
        journey = journeys.get((rec.label, rec.display_id), {"zones": [], "dwell": {}})

        track_payloads.append({
            "id": rec.display_id,
            "label": rec.label,
            "frames": rec.frames.tolist(),
            "x": floor_pos[:, 0].tolist(),
            "y": floor_pos[:, 1].tolist(),
            "speed": norm_speeds,
            "journey": journey,
        })

    # --- Floor bounds ---
    if all_floor:
        all_pos = np.concatenate(all_floor, axis=0)
        x_min = float(all_pos[:, 0].min())
        x_max = float(all_pos[:, 0].max())
        y_min = float(all_pos[:, 1].min())
        y_max = float(all_pos[:, 1].max())
    else:
        x_min, x_max, y_min, y_max = 0.0, 10.0, 0.0, 10.0

    # Pad slightly
    pad = max(1.0, (x_max - x_min) * 0.05)
    x_min -= pad; x_max += pad
    pad = max(1.0, (y_max - y_min) * 0.05)
    y_min -= pad; y_max += pad

    # --- Heatmap (people only) ---
    person_positions: list[np.ndarray] = []
    for rec in bundle.tracks.values():
        if rec.label != "person":
            continue
        floor_pos, _ = _track_to_floor(rec, H)
        if floor_pos.shape[0]:
            person_positions.append(floor_pos)

    heatmap_data = None
    if person_positions:
        heatmap_data = _compute_heatmap(
            np.concatenate(person_positions, axis=0), grid_m=0.5
        )

    # --- Zone payloads ---
    zone_payloads: list[dict] = []
    for z in zones:
        poly_floor = _poly_to_floor(z.polygon, H)
        kind = getattr(z, "kind", "analytics")
        # Layout zones don't get analytics — fill defaults so JS can read uniformly.
        if kind == "analytics":
            z_ana = _zone_analytics(z.zone_id, analytics)
        else:
            z_ana = {"severity": "NORMAL", "score": 0.0,
                     "avg_dwell_s": 0.0, "n_visits": 0, "peak_occupancy": 0}
        # color: BGR tuple → RGB for JS
        b, g, r = z.color
        zone_payloads.append({
            "id": z.zone_id,
            "name": z.name,
            "kind": kind,
            "polygon": poly_floor,
            "color_r": r, "color_g": g, "color_b": b,
            **z_ana,
        })

    duration_s = total_frames / fps if fps > 0 else 0.0

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 1),
        "has_homography": H is not None,
        "floor_x_min": x_min, "floor_x_max": x_max,
        "floor_y_min": y_min, "floor_y_max": y_max,
        "n_people": n_people,
        "n_carts": n_carts,
        "insight": analytics.insight_text or "",
    }

    return {
        "meta": meta,
        "tracks": track_payloads,
        "zones": zone_payloads,
        "heatmap": heatmap_data,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_html(payload_json: str) -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{
  background:#0f172a;font-family:'Segoe UI',Tahoma,sans-serif;
  color:#e2e8f0;display:flex;flex-direction:row;
  width:100%;height:100vh;overflow:hidden;
}}
/* ── Canvas panel ──────────────────────────────────── */
#canvas-panel{{
  flex:1;display:flex;flex-direction:column;
  position:relative;min-width:0;
}}
#canvas-wrap{{
  flex:1;position:relative;overflow:hidden;
  background:#1d2231;
}}
canvas#floor-bev{{
  display:block;
}}
/* ── Controls bar ──────────────────────────────────── */
#controls{{
  position:absolute;bottom:0;left:0;right:0;
  background:rgba(10,14,26,0.93);padding:8px 14px;
  display:flex;align-items:center;gap:10px;z-index:10;
  border-top:1px solid rgba(255,255,255,0.07);flex-wrap:wrap;
}}
#play-btn{{
  background:#3b82f6;color:#fff;border:none;border-radius:6px;
  padding:6px 15px;font-weight:700;cursor:pointer;font-size:13px;
  white-space:nowrap;
}}
#play-btn:hover{{background:#2563eb;}}
#frame-slider{{flex:1;min-width:80px;accent-color:#3b82f6;cursor:pointer;}}
#speed-select{{
  background:#1e293b;color:#e2e8f0;border:1px solid #334155;
  border-radius:4px;padding:4px 6px;font-size:12px;
}}
.ctrl-label{{font-size:11px;color:#64748b;white-space:nowrap;}}
.ctrl-check{{accent-color:#3b82f6;cursor:pointer;}}
#time-display{{
  color:#94a3b8;font-size:12px;min-width:110px;text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap;
}}
/* ── Sidebar ───────────────────────────────────────── */
#sidebar{{
  width:290px;flex-shrink:0;background:#1d2231;
  border-left:1px solid rgba(255,255,255,0.07);
  overflow-y:auto;display:flex;flex-direction:column;
}}
.sb-section{{border-bottom:1px solid rgba(255,255,255,0.06);padding:12px 14px;}}
.sb-hdr{{
  font-size:0.6rem;font-weight:800;letter-spacing:1.8px;
  text-transform:uppercase;color:#93c5fd;
  margin-bottom:8px;padding-bottom:4px;
  border-bottom:1px solid rgba(255,255,255,0.07);
}}
.sb-row{{
  display:flex;justify-content:space-between;align-items:baseline;
  font-size:12px;color:#94a3b8;padding:2px 0;
}}
.sb-row b{{color:#e2e8f0;font-weight:700;}}
/* zone card */
.zone-card{{
  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:6px;padding:8px 10px;margin-bottom:7px;
}}
.zone-card-name{{font-weight:700;font-size:13px;color:#e2e8f0;}}
.sev-badge{{
  display:inline-block;padding:1px 7px;border-radius:10px;
  font-size:10px;font-weight:800;letter-spacing:0.5px;
  text-transform:uppercase;margin-left:6px;
}}
.sev-NORMAL{{background:rgba(95,220,156,0.18);color:#5fdc9c;}}
.sev-WATCH{{background:rgba(107,155,255,0.18);color:#6b9bff;}}
.sev-QUEUE_FORMING{{background:rgba(240,200,112,0.18);color:#f0c870;}}
.sev-BACKED_UP{{background:rgba(255,129,112,0.18);color:#ff8170;}}
.zone-meta{{font-size:11px;color:#64748b;margin-top:4px;display:flex;gap:10px;flex-wrap:wrap;}}
/* journey */
#journey-panel .journey-step{{
  display:flex;align-items:center;gap:6px;
  font-size:12px;padding:4px 0;
  border-bottom:1px solid rgba(255,255,255,0.04);
}}
.step-num{{
  display:inline-flex;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:50%;
  background:#3b82f6;color:#fff;font-size:10px;font-weight:800;
  flex-shrink:0;
}}
.step-zone{{flex:1;color:#e2e8f0;}}
.step-dwell{{color:#64748b;font-size:11px;white-space:nowrap;}}
#journey-empty{{color:#475569;font-style:italic;font-size:12px;padding:4px 0;}}
/* insight */
#insight-text{{
  font-size:11.5px;color:#cbd5e1;line-height:1.55;
}}
#insight-text ul{{
  list-style:none;margin:0;padding:0;
  display:flex;flex-direction:column;gap:6px;
}}
#insight-text li{{
  position:relative;padding:6px 8px 6px 22px;
  background:rgba(255,255,255,0.025);
  border-left:2px solid rgba(96,165,250,0.45);
  border-radius:4px;
  color:#cbd5e1;font-size:11.5px;line-height:1.4;
}}
#insight-text li::before{{
  content:"";position:absolute;left:9px;top:11px;
  width:5px;height:5px;border-radius:50%;
  background:#60a5fa;
}}
/* Severity-tinted bullets for queue alerts */
#insight-text li.sev-watch        {{border-left-color:#60a5fa;}}
#insight-text li.sev-watch::before{{background:#60a5fa;}}
#insight-text li.sev-queue        {{border-left-color:#f59e0b;}}
#insight-text li.sev-queue::before{{background:#f59e0b;}}
#insight-text li.sev-backed       {{border-left-color:#ef4444;}}
#insight-text li.sev-backed::before{{background:#ef4444;}}
#insight-text li.sev-good         {{border-left-color:#10b981;}}
#insight-text li.sev-good::before {{background:#10b981;}}
#insight-text li.sev-info         {{border-left-color:#94a3b8;opacity:0.85;}}
#insight-text li.sev-info::before {{background:#94a3b8;}}
#insight-text li b{{color:#f1f5f9;font-weight:700;}}
/* scrollbar */
#sidebar::-webkit-scrollbar{{width:5px;}}
#sidebar::-webkit-scrollbar-track{{background:transparent;}}
#sidebar::-webkit-scrollbar-thumb{{background:#334155;border-radius:3px;}}
</style>
</head>
<body>

<!-- ══ Canvas panel ══════════════════════════════════════════ -->
<div id="canvas-panel">
  <div id="canvas-wrap">
    <canvas id="floor-bev"></canvas>
  </div>
  <div id="controls">
    <button id="play-btn">Pause</button>
    <input id="frame-slider" type="range" min="0" max="1" value="0">
    <select id="speed-select">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <label class="ctrl-label"><input class="ctrl-check" type="checkbox" id="heatmap-toggle" checked> Heatmap</label>
    <label class="ctrl-label"><input class="ctrl-check" type="checkbox" id="trails-toggle" checked> Trails</label>
    <span id="time-display">0.0s / 0.0s</span>
  </div>
</div>

<!-- ══ Sidebar ═══════════════════════════════════════════════ -->
<div id="sidebar">

  <!-- Overview -->
  <div class="sb-section" id="sb-overview">
    <div class="sb-hdr">Overview</div>
    <div class="sb-row"><span>People</span><b id="ov-people">n/a</b></div>
    <div class="sb-row"><span>Carts</span><b id="ov-carts">n/a</b></div>
    <div class="sb-row"><span>Duration</span><b id="ov-dur">n/a</b></div>
    <div class="sb-row"><span>Mode</span><b id="ov-mode">n/a</b></div>
  </div>

  <!-- Zones -->
  <div class="sb-section">
    <div class="sb-hdr">Zones</div>
    <div id="zones-list"></div>
  </div>

  <!-- Journey -->
  <div class="sb-section" id="journey-panel">
    <div class="sb-hdr" id="journey-hdr">Journey</div>
    <div id="journey-empty">Click a person dot to see their journey.</div>
    <div id="journey-steps"></div>
  </div>

  <!-- Insight -->
  <div class="sb-section">
    <div class="sb-hdr">Auto Insight</div>
    <div id="insight-text"></div>
  </div>

</div>

<script id="floor-payload" type="application/json">{payload_json}</script>
<script>
(function(){{
  const DATA  = JSON.parse(document.getElementById('floor-payload').textContent || '{{}}');
  const META  = DATA.meta  || {{}};
  const TRACKS= DATA.tracks|| [];
  const ZONES = DATA.zones || [];
  const HM    = DATA.heatmap;   // null or {{grid,x_min,x_max,y_min,y_max,w,h}}

  /* ── Color helpers ─────────────────────────────────────── */
  const VIRIDIS=[
    [68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],
    [39,173,129],[92,200,99],[170,220,50],[253,231,37]
  ];
  function velocityColor(t){{
    if(!isFinite(t))t=0;
    t=Math.max(0,Math.min(1,t));
    const f=t*(VIRIDIS.length-1);
    const i=Math.floor(f);
    const a=VIRIDIS[i];
    const b=VIRIDIS[Math.min(i+1,VIRIDIS.length-1)];
    const u=f-i;
    return [
      Math.round(a[0]+(b[0]-a[0])*u),
      Math.round(a[1]+(b[1]-a[1])*u),
      Math.round(a[2]+(b[2]-a[2])*u)
    ];
  }}
  function velocityCSS(t){{const[r,g,b]=velocityColor(t);return `rgb(${{r}},${{g}},${{b}})`;}}

  function jetColor(t){{
    const r=Math.max(0,Math.min(255,Math.round(255*(1.5-Math.abs(4*t-3)))));
    const g=Math.max(0,Math.min(255,Math.round(255*(1.5-Math.abs(4*t-2)))));
    const b2=Math.max(0,Math.min(255,Math.round(255*(1.5-Math.abs(4*t-1)))));
    return[r,g,b2];
  }}

  const SEV_COLORS={{
    NORMAL:[95,220,156],
    WATCH:[107,155,255],
    QUEUE_FORMING:[240,200,112],
    BACKED_UP:[255,129,112],
  }};
  function sevCSS(s){{const c=SEV_COLORS[s]||[150,150,150];return`rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;}}

  /* ── Canvas / layout ───────────────────────────────────── */
  const canvas=document.getElementById('floor-bev');
  const ctx=canvas.getContext('2d');
  const CONTROLS_H=42;
  const MARGIN=36;
  let scaleX=1,scaleY=1;
  let originX=MARGIN,originY=400;

  function resize(){{
    const wrap=document.getElementById('canvas-wrap');
    const W=wrap.clientWidth||800;
    const H=wrap.clientHeight||500;
    canvas.width=W; canvas.height=H;
    const fw=META.floor_x_max-META.floor_x_min||10;
    const fh=META.floor_y_max-META.floor_y_min||10;
    const availW=W-MARGIN*2;
    const availH=H-CONTROLS_H-MARGIN*2;
    scaleX=availW/fw;
    scaleY=availH/fh;
    const sc=Math.min(scaleX,scaleY);
    scaleX=sc; scaleY=sc;
    originX=MARGIN;
    originY=H-CONTROLS_H-MARGIN;
    buildHeatmapCanvas();
    draw(currentFrame);
  }}

  function floorToCanvas(fx,fy){{
    // Vertical flip only — keep image Y-down convention so top-of-frame
    // ends up at top-of-canvas. X is left untouched (left stays left).
    const cx=originX+(fx-META.floor_x_min)*scaleX;
    const cy=originY-(META.floor_y_max-fy)*scaleY;
    return[cx,cy];
  }}

  /* ── Heatmap offscreen canvas ─────────────────────────── */
  let hmCanvas=null;
  let hmBounds=null;   /* {{cx,cy,pw,ph}} in canvas pixels */

  function buildHeatmapCanvas(){{
    hmCanvas=null; hmBounds=null;
    if(!HM||!HM.grid||HM.grid.length===0)return;
    const g=HM.grid;
    const rows=g.length, cols=HM.w;
    // Build ImageData
    const cellW=(HM.x_max-HM.x_min)/HM.w;
    const cellH=(HM.y_max-HM.y_min)/HM.h;
    // Convert to canvas pixels for the bounding rect
    const [cx0,cy0]=floorToCanvas(HM.x_min,HM.y_max); // top-left on canvas (y_max→smallest cy)
    const [cx1,cy1]=floorToCanvas(HM.x_max,HM.y_min); // bottom-right
    const pw=Math.max(1,Math.abs(cx1-cx0));
    const ph=Math.max(1,Math.abs(cy1-cy0));
    hmBounds={{cx:Math.min(cx0,cx1),cy:Math.min(cy0,cy1),pw,ph}};

    const oc=document.createElement('canvas');
    oc.width=cols; oc.height=rows;
    const octx=oc.getContext('2d');
    const id=octx.createImageData(cols,rows);
    for(let r=0;r<rows;r++){{
      for(let c=0;c<cols;c++){{
        const v=(g[r]&&g[r][c])||0;
        const [jr,jg,jb]=jetColor(v);
        const idx=(r*cols+c)*4;
        id.data[idx]=jr;
        id.data[idx+1]=jg;
        id.data[idx+2]=jb;
        id.data[idx+3]=Math.round(v*220);
      }}
    }}
    octx.putImageData(id,0,0);
    hmCanvas=oc;
  }}

  /* ── Grid ──────────────────────────────────────────────── */
  function drawGrid(){{
    ctx.save();
    ctx.strokeStyle='rgba(255,255,255,0.04)';
    ctx.lineWidth=1;
    ctx.font='9px Segoe UI,sans-serif';
    ctx.fillStyle='rgba(255,255,255,0.2)';
    const xStep=1; // 1 metre
    const yStep=1;
    const xStart=Math.floor(META.floor_x_min);
    const xEnd  =Math.ceil(META.floor_x_max);
    const yStart=Math.floor(META.floor_y_min);
    const yEnd  =Math.ceil(META.floor_y_max);
    for(let x=xStart;x<=xEnd;x+=xStep){{
      const[cx,cy0]=floorToCanvas(x,META.floor_y_min);
      const[,cy1]=floorToCanvas(x,META.floor_y_max);
      ctx.beginPath();ctx.moveTo(cx,cy0);ctx.lineTo(cx,cy1);ctx.stroke();
      ctx.fillText(x+'m',cx+2,originY+12);
    }}
    for(let y=yStart;y<=yEnd;y+=yStep){{
      const[cx0,cy]=floorToCanvas(META.floor_x_min,y);
      const[cx1, ]=floorToCanvas(META.floor_x_max,y);
      ctx.beginPath();ctx.moveTo(cx0,cy);ctx.lineTo(cx1,cy);ctx.stroke();
      if(y!==yStart)ctx.fillText(y+'m',originX-26,cy+4);
    }}
    ctx.restore();
  }}

  /* ── Zones ─────────────────────────────────────────────── */
  function _zonePath(z){{
    ctx.beginPath();
    const[x0,y0]=floorToCanvas(z.polygon[0][0],z.polygon[0][1]);
    ctx.moveTo(x0,y0);
    for(let i=1;i<z.polygon.length;i++){{
      const[xi,yi]=floorToCanvas(z.polygon[i][0],z.polygon[i][1]);
      ctx.lineTo(xi,yi);
    }}
    ctx.closePath();
  }}
  function _zoneCentroid(z){{
    let sx=0,sy=0;
    for(const[px,py]of z.polygon){{const[cx,cy]=floorToCanvas(px,py);sx+=cx;sy+=cy;}}
    return[sx/z.polygon.length,sy/z.polygon.length];
  }}
  function _drawAnalyticsZone(z){{
    const sev=z.severity||'NORMAL';
    const sc=SEV_COLORS[sev]||[150,150,150];
    ctx.save();
    _zonePath(z);
    ctx.fillStyle=`rgba(${{sc[0]}},${{sc[1]}},${{sc[2]}},0.12)`;
    ctx.fill();
    ctx.strokeStyle=`rgba(${{sc[0]}},${{sc[1]}},${{sc[2]}},0.65)`;
    ctx.lineWidth=1.5;
    ctx.stroke();
    const[sx,sy]=_zoneCentroid(z);
    ctx.font='bold 11px Segoe UI,sans-serif';
    ctx.fillStyle=`rgba(${{sc[0]}},${{sc[1]}},${{sc[2]}},0.95)`;
    ctx.textAlign='center';
    ctx.fillText(z.name,sx,sy-6);
    if(z.avg_dwell_s>0){{
      ctx.font='10px Segoe UI,sans-serif';
      ctx.fillStyle='rgba(255,255,255,0.6)';
      ctx.fillText(z.avg_dwell_s.toFixed(1)+'s avg',sx,sy+8);
    }}
    ctx.textAlign='left';
    ctx.restore();
  }}
  function _drawWall(z){{
    ctx.save();
    _zonePath(z);
    ctx.fillStyle='rgba(71,85,105,0.85)';
    ctx.fill();
    ctx.strokeStyle='#94a3b8';
    ctx.lineWidth=3;
    ctx.stroke();
    const[sx,sy]=_zoneCentroid(z);
    ctx.font='bold 10px Segoe UI,sans-serif';
    ctx.fillStyle='rgba(226,232,240,0.85)';
    ctx.textAlign='center';
    ctx.fillText(z.name||'Wall',sx,sy);
    ctx.textAlign='left';
    ctx.restore();
  }}
  function _drawAisle(z){{
    ctx.save();
    _zonePath(z);
    ctx.fillStyle='rgba(255,255,255,0.04)';
    ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.25)';
    ctx.lineWidth=1.2;
    ctx.setLineDash([6,4]);
    ctx.stroke();
    ctx.setLineDash([]);
    const[sx,sy]=_zoneCentroid(z);
    ctx.font='600 11px Segoe UI,sans-serif';
    ctx.fillStyle='rgba(148,163,184,0.85)';
    ctx.textAlign='center';
    ctx.fillText(z.name||'Aisle',sx,sy);
    ctx.textAlign='left';
    ctx.restore();
  }}
  function _drawFixture(z){{
    ctx.save();
    _zonePath(z);
    ctx.fillStyle='rgba(125,211,252,0.18)';
    ctx.fill();
    ctx.strokeStyle='#7dd3fc';
    ctx.lineWidth=2;
    ctx.stroke();
    const[sx,sy]=_zoneCentroid(z);
    ctx.font='bold 11px Segoe UI,sans-serif';
    ctx.fillStyle='rgba(186,230,253,0.95)';
    ctx.textAlign='center';
    ctx.fillText('📦 '+(z.name||'Fixture'),sx,sy);
    ctx.textAlign='left';
    ctx.restore();
  }}
  function _drawDoor(z){{
    ctx.save();
    _zonePath(z);
    ctx.strokeStyle='#22c55e';
    ctx.lineWidth=2.5;
    ctx.setLineDash([8,5]);
    ctx.stroke();
    ctx.setLineDash([]);
    const[sx,sy]=_zoneCentroid(z);
    ctx.font='bold 11px Segoe UI,sans-serif';
    ctx.fillStyle='rgba(134,239,172,0.95)';
    ctx.textAlign='center';
    ctx.fillText('🚪 '+(z.name||'Door'),sx,sy);
    ctx.textAlign='left';
    ctx.restore();
  }}

  // Z-order: walls + aisles render BEHIND the heatmap; fixtures + doors +
  // analytics zones render in FRONT of the heatmap (but behind trails/dots).
  function drawLayoutBack(){{
    for(const z of ZONES){{
      if(!z.polygon||z.polygon.length<3)continue;
      if(z.kind==='wall')_drawWall(z);
      else if(z.kind==='aisle')_drawAisle(z);
    }}
  }}
  function drawLayoutFrontAndAnalytics(){{
    for(const z of ZONES){{
      if(!z.polygon||z.polygon.length<3)continue;
      if(z.kind==='fixture')_drawFixture(z);
      else if(z.kind==='door')_drawDoor(z);
      else if(!z.kind||z.kind==='analytics')_drawAnalyticsZone(z);
    }}
  }}

  /* ── Track lookup at frame f ───────────────────────────── */
  function sampleAtFrame(track,f){{
    // Binary search for last frames[i] <= f within a 3-second window
    const frames=track.frames;
    const fps=META.fps||25;
    const windowBack=fps*3;
    let lo=0,hi=frames.length-1,found=-1;
    while(lo<=hi){{
      const mid=(lo+hi)>>1;
      if(frames[mid]<=f){{found=mid;lo=mid+1;}}
      else hi=mid-1;
    }}
    if(found<0)return null;
    if(f-frames[found]>windowBack)return null;
    return found;
  }}

  /* ── Trails ─────────────────────────────────────────────── */
  function drawTrails(f){{
    if(!showTrails)return;
    const TRAIL=60;
    for(const t of TRACKS){{
      const isSelected=(t.id===selectedTrackId);
      const frames=t.frames;
      // collect samples in [f-TRAIL, f]
      const pts=[];
      for(let i=0;i<frames.length;i++){{
        if(frames[i]<f-TRAIL)continue;
        if(frames[i]>f)break;
        pts.push(i);
      }}
      if(pts.length<2)continue;
      ctx.save();
      ctx.lineWidth=isSelected?3:2;
      for(let k=1;k<pts.length;k++){{
        const i0=pts[k-1],i1=pts[k];
        const[cx0,cy0]=floorToCanvas(t.x[i0],t.y[i0]);
        const[cx1,cy1]=floorToCanvas(t.x[i1],t.y[i1]);
        const alpha=0.15+0.85*(k/pts.length);
        const spd=t.speed[i1]||0;
        const[r,g,b]=velocityColor(spd);
        ctx.strokeStyle=`rgba(${{r}},${{g}},${{b}},${{alpha}})`;
        ctx.globalAlpha=1;
        ctx.beginPath();ctx.moveTo(cx0,cy0);ctx.lineTo(cx1,cy1);ctx.stroke();
      }}
      ctx.restore();
    }}
  }}

  /* ── Dots ───────────────────────────────────────────────── */
  let dotHitlist=[];  /* {{id,label,cx,cy,r}} for click testing */
  function drawDots(f){{
    dotHitlist=[];
    for(const t of TRACKS){{
      const si=sampleAtFrame(t,f);
      if(si===null)continue;
      const[cx,cy]=floorToCanvas(t.x[si],t.y[si]);
      const isSelected=(t.id===selectedTrackId);
      const isPerson=(t.label==='person');
      const radius=isSelected?10:7;
      const spd=t.speed[si]||0;

      ctx.save();
      // Dot fill: viridis for person, amber for cart
      if(isPerson){{
        const[r,g,b]=velocityColor(spd);
        ctx.fillStyle=`rgb(${{r}},${{g}},${{b}})`;
      }}else{{
        ctx.fillStyle='#f59e0b';
      }}
      ctx.strokeStyle=isSelected?'#fff':'rgba(255,255,255,0.4)';
      ctx.lineWidth=isSelected?2:1;
      ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.fill();ctx.stroke();

      // Direction tick
      if(si>0){{
        const dx=t.x[si]-t.x[si-1];
        const dy=t.y[si]-t.y[si-1];
        const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist>0.01){{
          const tickLen=radius+6;
          ctx.strokeStyle='rgba(255,255,255,0.6)';
          ctx.lineWidth=1.5;
          ctx.beginPath();
          ctx.moveTo(cx,cy);
          ctx.lineTo(cx+(dx/dist)*tickLen,cy-(dy/dist)*tickLen);
          ctx.stroke();
        }}
      }}

      // Label
      const lbl=(isPerson?'P':'C')+t.id;
      ctx.font=`bold ${{isSelected?12:10}}px Segoe UI,sans-serif`;
      ctx.fillStyle='#fff';
      ctx.strokeStyle='rgba(0,0,0,0.6)';
      ctx.lineWidth=3;
      ctx.strokeText(lbl,cx+radius+2,cy-radius);
      ctx.fillText(lbl,cx+radius+2,cy-radius);
      ctx.restore();

      dotHitlist.push({{id:t.id,label:t.label,cx,cy,r:radius+8}});
    }}
  }}

  /* ── HUD ────────────────────────────────────────────────── */
  function drawHUD(f){{
    const fps=META.fps||25;
    const t=(f/fps).toFixed(1);
    const dur=(META.total_frames/fps).toFixed(1);
    const txt=`Frame ${{f}} / ${{META.total_frames}}  |  ${{META.n_people||0}} people  ${{META.n_carts||0}} carts`;
    ctx.save();
    ctx.font='11px Segoe UI,sans-serif';
    ctx.fillStyle='rgba(0,0,0,0.6)';
    const tw=ctx.measureText(txt).width;
    ctx.fillRect(8,8,tw+14,20);
    ctx.fillStyle='rgba(200,220,255,0.85)';
    ctx.fillText(txt,14,22);
    ctx.restore();
    // Update time display
    document.getElementById('time-display').textContent=t+'s / '+dur+'s';
  }}

  /* ── Main draw ──────────────────────────────────────────── */
  let showHeatmap=true, showTrails=true;
  function draw(f){{
    ctx.clearRect(0,0,canvas.width,canvas.height);
    drawGrid();
    drawLayoutBack();      // walls + aisles behind the heatmap
    if(showHeatmap&&hmCanvas&&hmBounds){{
      ctx.save();
      ctx.globalAlpha=0.65;
      ctx.drawImage(hmCanvas,hmBounds.cx,hmBounds.cy,hmBounds.pw,hmBounds.ph);
      ctx.globalAlpha=1;
      ctx.restore();
    }}
    drawLayoutFrontAndAnalytics();   // fixtures, doors, analytics zones
    drawTrails(f);
    drawDots(f);
    drawHUD(f);
  }}

  /* ── Sidebar population ─────────────────────────────────── */
  function populateSidebar(){{
    document.getElementById('ov-people').textContent=META.n_people||0;
    document.getElementById('ov-carts').textContent=META.n_carts||0;
    const fps=META.fps||25;
    const dur=META.total_frames/fps;
    document.getElementById('ov-dur').textContent=dur.toFixed(1)+'s';
    if(META.has_homography){{
      document.getElementById('ov-mode').innerHTML='<b style="color:#4ade80">Floor space (calibrated)</b>';
    }}else{{
      document.getElementById('ov-mode').innerHTML='<b style="color:#f59e0b">Pixel space (no calibration)</b>';
    }}

    // Zones — only analytics zones get a metrics card; layout zones (wall,
    // aisle, fixture, door) are visible on the canvas and skipped here.
    const zl=document.getElementById('zones-list');
    zl.innerHTML='';
    const analyticsZones=ZONES.filter(z=>!z.kind||z.kind==='analytics');
    const layoutZones=ZONES.filter(z=>z.kind&&z.kind!=='analytics');
    if(!analyticsZones.length){{
      zl.innerHTML='<div style="color:#475569;font-size:12px;font-style:italic">No analytics zones defined.</div>';
    }}
    for(const z of analyticsZones){{
      const sev=z.severity||'NORMAL';
      const div=document.createElement('div');
      div.className='zone-card';
      div.innerHTML=`
        <div class="zone-card-name">${{z.name}}
          <span class="sev-badge sev-${{sev}}">${{sev.replace('_',' ')}}</span>
        </div>
        <div class="zone-meta">
          <span>Avg dwell: <b style="color:#e2e8f0">${{z.avg_dwell_s}}s</b></span>
          <span>Peak: <b style="color:#e2e8f0">${{z.peak_occupancy}}</b></span>
          <span>Score: <b style="color:#e2e8f0">${{z.score}}/100</b></span>
        </div>`;
      zl.appendChild(div);
    }}
    if(layoutZones.length){{
      const counts={{wall:0,aisle:0,fixture:0,door:0}};
      for(const z of layoutZones)counts[z.kind]=(counts[z.kind]||0)+1;
      const parts=[];
      if(counts.wall)parts.push(counts.wall+' wall'+(counts.wall>1?'s':''));
      if(counts.aisle)parts.push(counts.aisle+' aisle'+(counts.aisle>1?'s':''));
      if(counts.fixture)parts.push(counts.fixture+' fixture'+(counts.fixture>1?'s':''));
      if(counts.door)parts.push(counts.door+' door'+(counts.door>1?'s':''));
      const note=document.createElement('div');
      note.style.cssText='color:#64748b;font-size:11px;font-style:italic;margin-top:6px;padding-top:6px;border-top:1px dashed rgba(255,255,255,0.06);';
      note.textContent='Layout: '+parts.join(', ');
      zl.appendChild(note);
    }}

    // Insight — render as a proper <ul> with one <li> per bullet, and
    // tag each li with a severity class so the left-border / dot picks
    // up a meaningful color.
    (function(){{
      var el=document.getElementById('insight-text');
      var raw=META.insight||'';
      el.innerHTML='';
      if(!raw){{ el.textContent='No insight available.'; return; }}
      // Use the literal escape '\\n' so JS sees the two-char escape, not a
      // raw newline (which was an unterminated-string SyntaxError before).
      var lines=raw.split('\\n').map(function(s){{return s.trim();}}).filter(Boolean);
      if(!lines.length){{ el.textContent='No insight available.'; return; }}

      function _classify(line){{
        var l=line.toLowerCase();
        if(l.indexOf('backed_up')>=0||l.indexOf('backed up')>=0) return 'sev-backed';
        if(l.indexOf('queue_forming')>=0||l.indexOf('queue forming')>=0) return 'sev-queue';
        if(l.indexOf('watch')>=0||l.indexOf('alert')>=0||l.indexOf('crowd cluster')>=0) return 'sev-watch';
        if(l.indexOf('no queue spikes')>=0||l.indexOf('no zones defined')>=0) return 'sev-info';
        if(l.indexOf('clip duration')>=0||l.indexOf('tracked ')>=0||l.indexOf('zone transition')>=0) return 'sev-info';
        if(l.indexOf('longest average')>=0||l.indexOf('shortest average')>=0) return 'sev-good';
        return '';
      }}
      // Bold the leading "Label:" prefix when present (e.g. "Exit Lane: 10 visit(s)…")
      function _format(line){{
        var m=line.match(/^([^:]{{1,40}}):\\s+(.+)$/);
        if(!m) return line.replace(/[<>&]/g,function(c){{return{{"<":"&lt;",">":"&gt;","&":"&amp;"}}[c];}});
        var head=m[1].replace(/[<>&]/g,function(c){{return{{"<":"&lt;",">":"&gt;","&":"&amp;"}}[c];}});
        var tail=m[2].replace(/[<>&]/g,function(c){{return{{"<":"&lt;",">":"&gt;","&":"&amp;"}}[c];}});
        return '<b>'+head+':</b> '+tail;
      }}

      var ul=document.createElement('ul');
      lines.forEach(function(line){{
        var li=document.createElement('li');
        var cls=_classify(line);
        if(cls) li.className=cls;
        li.innerHTML=_format(line);
        ul.appendChild(li);
      }});
      el.appendChild(ul);
    }})();
  }}

  /* ── Journey panel ──────────────────────────────────────── */
  let selectedTrackId=null;
  // build journey lookup
  const JOURNEY_MAP={{}};
  for(const t of TRACKS){{
    if(t.journey&&t.journey.zones&&t.journey.zones.length)
      JOURNEY_MAP[t.id]=t.journey;
  }}

  function showJourney(trackId,label){{
    selectedTrackId=trackId;
    const isPerson=(label==='person');
    const prefix=isPerson?'Person':'Cart';
    document.getElementById('journey-hdr').textContent=`${{prefix}} ${{trackId}} - Journey`;
    const jdata=JOURNEY_MAP[trackId];
    const empty=document.getElementById('journey-empty');
    const steps=document.getElementById('journey-steps');
    if(!jdata||!jdata.zones.length){{
      empty.textContent='No zone visits recorded for this track.';
      empty.style.display='';
      steps.innerHTML='';
      return;
    }}
    empty.style.display='none';
    let html='';
    jdata.zones.forEach(function(z,i){{
      const dw=jdata.dwell[z];
      const dwStr=dw!=null?dw.toFixed(1)+'s':'n/a';
      html+=`<div class="journey-step">
        <span class="step-num">${{i+1}}</span>
        <span class="step-zone">${{z}}</span>
        <span class="step-dwell">${{dwStr}}</span>
      </div>`;
    }});
    steps.innerHTML=html;
  }}

  /* ── Click handler ──────────────────────────────────────── */
  canvas.addEventListener('click',function(ev){{
    const rect=canvas.getBoundingClientRect();
    const mx=ev.clientX-rect.left;
    const my=ev.clientY-rect.top;
    let best=null,bestD=Infinity;
    for(const d of dotHitlist){{
      const dist=Math.sqrt((d.cx-mx)**2+(d.cy-my)**2);
      if(dist<d.r+12&&dist<bestD){{bestD=dist;best=d;}}
    }}
    if(best){{
      const t=TRACKS.find(t=>t.id===best.id);
      showJourney(best.id,best.label);
    }}else{{
      selectedTrackId=null;
      document.getElementById('journey-hdr').textContent='Journey';
      document.getElementById('journey-empty').textContent='Click a person dot to see their journey.';
      document.getElementById('journey-empty').style.display='';
      document.getElementById('journey-steps').innerHTML='';
    }}
    draw(currentFrame);
  }});

  /* ── Animation loop ─────────────────────────────────────── */
  let currentFrame=0;
  let playing=true;
  let speed=1;

  const slider=document.getElementById('frame-slider');
  slider.max=Math.max(0,META.total_frames-1);

  document.getElementById('play-btn').addEventListener('click',function(){{
    playing=!playing;
    this.textContent=playing?'Pause':'Play';
  }});
  document.getElementById('speed-select').addEventListener('change',function(){{
    speed=parseFloat(this.value)||1;
  }});
  slider.addEventListener('input',function(){{
    currentFrame=parseInt(this.value,10)||0;
    draw(currentFrame);
  }});
  document.getElementById('heatmap-toggle').addEventListener('change',function(){{
    showHeatmap=this.checked;draw(currentFrame);
  }});
  document.getElementById('trails-toggle').addEventListener('change',function(){{
    showTrails=this.checked;draw(currentFrame);
  }});

  let lastTS=0;
  function tick(ts){{
    requestAnimationFrame(tick);
    if(playing&&META.total_frames>1){{
      const interval=1000/((META.fps||25)*speed);
      if(ts-lastTS>=interval){{
        currentFrame=(currentFrame+1)%META.total_frames;
        slider.value=currentFrame;
        draw(currentFrame);
        lastTS=ts;
      }}
    }}
  }}

  window.addEventListener('resize',resize);
  populateSidebar();
  resize();
  requestAnimationFrame(tick);

}})();

</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_floor_2d_html(bundle: TrajectoryBundle,
                        zones: list[Zone],
                        analytics: AnalyticsResult,
                        H: Optional[np.ndarray] = None) -> str:
    """Build and return a self-contained animated floor-map HTML document.

    Args:
        bundle:    TrajectoryBundle from the tracker.
        zones:     List of Zone objects (pixel-space polygons).
        analytics: AnalyticsResult from analytics_builder.run_all().
        H:         Optional 3x3 homography (pixel→floor metres). If None,
                   coordinates are divided by 100 and displayed with a warning.

    Returns:
        Complete HTML string suitable for gr.HTML / iframe injection.
    """
    payload = _build_payload(bundle, zones, analytics, H)
    payload_json = json.dumps(payload, separators=(',', ':'))
    return _build_html(payload_json)
