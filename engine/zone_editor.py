# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Pure helpers for the Gradio zone editor.

Kept Gradio-free on purpose — the UI layer just shuttles data between
gr.State, gr.Image, and these functions.  That makes the editor logic
unit-testable and lets the same helpers be reused (e.g., to bake a
"draw zones over BEV" preview into the case report).
"""
from __future__ import annotations

import dataclasses
import uuid
from typing import Iterable, Optional

import cv2
import numpy as np

from .analytics_models import Zone, ZoneAppliesTo, ZoneKind

#: Kinds that describe a PLACE on the floor rather than an analytics region.
LAYOUT_KINDS: tuple[ZoneKind, ...] = ("wall", "aisle", "fixture", "door")


# Distinct, well-saturated BGR palette (chosen to be visible over both light
# and dark video backgrounds; cycles when more zones than colors).
_PALETTE_BGR = [
    (255, 152,   0),   # blue-ish
    (  0, 200, 255),   # cyan/orange
    (147, 112, 219),   # purple
    ( 80, 200,  80),   # green
    (220,  60, 200),   # magenta
    ( 60, 200, 220),   # gold
    (255, 100, 100),   # light red
    (160, 160,  60),   # teal
]

VERTEX_DOT_RADIUS = 6
EDGE_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.55
LABEL_THICK = 1
FILL_ALPHA = 0.25


def polygon_color(idx: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """Open the video, read the first decodable frame, return as BGR uint8.
    Returns None if the file is unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        for _ in range(5):                          # tolerate a couple of bad frames
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        return None
    finally:
        cap.release()


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True when segment p1p2 properly crosses p3p4 (shared endpoints excluded)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if v == 0 else (1 if v > 0 else -1)

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    return ((d1 * d2 < 0) and (d3 * d4 < 0))


def _is_self_intersecting(arr: np.ndarray) -> bool:
    """Brute-force edge-pair test. Polygons here are hand-drawn (a handful of
    vertices), so O(V^2) is free and exact beats clever."""
    n = len(arr)
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = arr[i], arr[(i + 1) % n]
        for j in range(i + 1, n):
            # Skip adjacent edges (they legitimately share a vertex)
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = arr[j], arr[(j + 1) % n]
            if _segments_cross(a1, a2, b1, b2):
                return True
    return False


def validate_polygon(pts: list[tuple[int, int]]) -> tuple[bool, str]:
    """Cheap sanity checks before promoting an in-progress polygon to a Zone.
    Returns (ok, message)."""
    if len(pts) < 3:
        return False, "Need at least 3 vertices to close a polygon."
    arr = np.asarray(pts, dtype=np.int32)
    if cv2.contourArea(arr) < 16:                   # ~4×4 px patch
        return False, "Polygon area is too small."
    # Reject degenerate consecutive duplicates
    diffs = np.diff(arr, axis=0, append=arr[:1])
    if np.any(np.all(diffs == 0, axis=1)):
        return False, "Polygon has duplicate consecutive vertices."
    # A figure-of-eight polygon is filled by fillPoly's even-odd rule, which
    # punches holes where the loops overlap. Zone membership then reports
    # "outside" for pixels the user clearly meant to include, so reject it at
    # draw time rather than letting a rule silently under-fire later.
    if _is_self_intersecting(arr):
        return False, ("Polygon edges cross each other - redraw it without "
                       "self-intersections.")
    return True, ""


_LAYOUT_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    # All in BGR for OpenCV. Match the JS hex chosen for the Floor BEV.
    "wall":    (105,  85,  71),    # slate-500-ish
    "aisle":   (200, 200, 200),    # subtle grey
    "fixture": (252, 211, 125),    # sky-300 (#7dd3fc → BGR)
    "door":    ( 94, 197,  34),    # green-500 (#22c55e → BGR)
}


def zone_color_for(kind: ZoneKind, idx: int) -> tuple[int, int, int]:
    """Colour a zone gets for a given kind. Layout kinds are colour-coded by
    type so the overlay reads as a floor plan; analytics zones just cycle the
    palette. Exposed because `kind` is editable after creation — the colour is
    derived from it, so retyping a zone has to recompute this or the overlay
    and the sidebar swatch disagree with the zone's actual type."""
    return _LAYOUT_COLORS_BGR[kind] if kind != "analytics" else polygon_color(idx)


def make_zone(name: str, polygon_pts: Iterable[tuple[int, int]],
              applies_to: ZoneAppliesTo, idx: int,
              kind: ZoneKind = "analytics") -> Zone:
    poly = np.asarray(list(polygon_pts), dtype=np.int32)
    color = zone_color_for(kind, idx)
    default_name = f"Zone {idx + 1}" if kind == "analytics" else f"{kind.capitalize()} {idx + 1}"
    return Zone(
        zone_id=str(uuid.uuid4())[:8],
        name=name.strip() or default_name,
        polygon=poly,
        applies_to=applies_to,
        kind=kind,
        color=color,
    )


# ---------------------------------------------------------------------------
# Editing zones after they are drawn
#
# Every function here is (zones, ...) -> (new_zones, changed) and never mutates
# the list it is given. `changed` exists because the UI drives these from
# Gradio events that fire more than once per user action — the caller uses it
# to decide whether to redraw and whether to rebuild the row list.
# ---------------------------------------------------------------------------
def coerce_applies_to(kind: ZoneKind,
                      applies_to: ZoneAppliesTo) -> ZoneAppliesTo:
    """Layout zones are PLACES, not track-type filters.

    `applies_to` defaults to "person", so a door or aisle left at that default
    matches zero cart tracks — and the blocked-door / static-cart rules over it
    then report nothing while the zone looks perfectly correct on screen. Force
    "both" for every layout kind, at draw time and at edit time alike.
    """
    return "both" if kind in LAYOUT_KINDS else (applies_to or "person")


def find_zone(zones: list[Zone], zone_id: str) -> int:
    """Index of zone_id, or -1. Zones are addressed by id rather than position
    because the list can be rebuilt between a click and its handler."""
    for i, z in enumerate(zones):
        if z.zone_id == zone_id:
            return i
    return -1


def _replaced(zones: list[Zone], i: int, **fields) -> list[Zone]:
    out = list(zones)
    out[i] = dataclasses.replace(out[i], **fields)     # Zone is frozen
    return out


def rename_zone(zones: list[Zone], zone_id: str,
                new_name: str) -> tuple[list[Zone], bool]:
    """Rename in place. Blank names are ignored rather than accepted — an
    unnamed zone is unreadable in the overlay, the dwell table and the rule
    output alike."""
    zones = list(zones or [])
    i = find_zone(zones, zone_id)
    clean = (new_name or "").strip()
    if i < 0 or not clean or clean == zones[i].name:
        return zones, False
    return _replaced(zones, i, name=clean), True


def retype_zone(zones: list[Zone], zone_id: str,
                kind: ZoneKind) -> tuple[list[Zone], bool]:
    """Change a zone's kind, re-deriving everything that hangs off it.

    `color` is derived from the kind and `applies_to` is constrained by it, so
    a retype that only wrote `kind` would leave the overlay drawing a door in
    an analytics colour and, worse, leave a door filtering on people only.
    """
    zones = list(zones or [])
    i = find_zone(zones, zone_id)
    if i < 0 or zones[i].kind == kind:
        return zones, False
    return _replaced(zones, i,
                     kind=kind,
                     applies_to=coerce_applies_to(kind, zones[i].applies_to),
                     color=zone_color_for(kind, i)), True


def set_zone_applies_to(zones: list[Zone], zone_id: str,
                        applies_to: ZoneAppliesTo) -> tuple[list[Zone], bool]:
    zones = list(zones or [])
    i = find_zone(zones, zone_id)
    if i < 0:
        return zones, False
    wanted = coerce_applies_to(zones[i].kind, applies_to)
    if wanted == zones[i].applies_to:
        return zones, False
    return _replaced(zones, i, applies_to=wanted), True


def remove_zone(zones: list[Zone],
                zone_id: str) -> tuple[list[Zone], Optional[Zone]]:
    """Drop a zone. Survivors keep their colours on purpose — re-running the
    palette over the remaining zones would recolour the whole overlay on every
    delete, which reads as though the wrong zone went."""
    zones = list(zones or [])
    i = find_zone(zones, zone_id)
    if i < 0:
        return zones, None
    removed = zones.pop(i)
    return zones, removed


def _zone_label(z: Zone) -> str:
    kind = getattr(z, "kind", "analytics")
    if kind == "analytics":
        return f"{z.name} [{z.applies_to}]"
    if kind == "wall":
        return f"WALL: {z.name}"
    if kind == "aisle":
        return f"AISLE: {z.name}"
    if kind == "fixture":
        return f"FIXTURE: {z.name}"
    if kind == "door":
        return f"DOOR: {z.name}"
    return z.name


def _draw_dashed_polyline(img: np.ndarray, pts: np.ndarray,
                          color: tuple[int, int, int],
                          thickness: int = 2,
                          dash_len: int = 10, gap_len: int = 6) -> None:
    """Draw a dashed closed polyline by walking each edge in dash/gap segments."""
    n = len(pts)
    if n < 2:
        return
    for i in range(n):
        p0 = pts[i].astype(np.float32)
        p1 = pts[(i + 1) % n].astype(np.float32)
        seg = p1 - p0
        L = float(np.hypot(seg[0], seg[1]))
        if L < 1e-3:
            continue
        u = seg / L
        d = 0.0
        on = True
        while d < L:
            stride = dash_len if on else gap_len
            d_end = min(d + stride, L)
            if on:
                a = (p0 + u * d).astype(int)
                b = (p0 + u * d_end).astype(int)
                cv2.line(img, tuple(a), tuple(b), color,
                         thickness, lineType=cv2.LINE_AA)
            d = d_end
            on = not on


def render_zone_overlay(frame: np.ndarray,
                        zones: list[Zone],
                        in_progress: Optional[list[tuple[int, int]]] = None,
                        ) -> np.ndarray:
    """Return a new BGR frame with all closed zones (per-kind styling) and the
    in-progress polygon (dots + line) drawn on top. Does not mutate the input."""
    if frame is None:
        return frame
    base = frame.copy()
    h, w = base.shape[:2]

    # 1) Translucent fills, per-kind alpha
    if zones:
        fill_layer = base.copy()
        # Per-kind fill alpha (analytics keeps the original FILL_ALPHA = 0.25)
        kind_alpha = {
            "analytics": 0.25,
            "wall": 0.55,
            "aisle": 0.08,
            "fixture": 0.30,
            "door": 0.0,        # door is dashed line only, no fill
        }
        # We blend each kind separately so different kinds can have different alphas.
        for kind, alpha in kind_alpha.items():
            if alpha <= 0:
                continue
            kind_zones = [z for z in zones
                          if getattr(z, "kind", "analytics") == kind]
            if not kind_zones:
                continue
            layer = base.copy()
            for z in kind_zones:
                cv2.fillPoly(layer, [z.polygon.astype(np.int32)], z.color)
            cv2.addWeighted(layer, alpha, base, 1.0 - alpha, 0, base)

    # 2) Borders + labels
    for z in zones:
        pts = z.polygon.astype(np.int32)
        kind = getattr(z, "kind", "analytics")
        if kind == "door":
            _draw_dashed_polyline(base, pts, z.color, thickness=2,
                                  dash_len=12, gap_len=6)
        elif kind == "aisle":
            _draw_dashed_polyline(base, pts, z.color, thickness=1,
                                  dash_len=8, gap_len=5)
        else:
            edge_thick = 3 if kind == "wall" else EDGE_THICKNESS
            cv2.polylines(base, [pts], isClosed=True, color=z.color,
                          thickness=edge_thick, lineType=cv2.LINE_AA)
        # Label near the top-most vertex (centroid for aisle so it reads inline)
        if kind == "aisle":
            cx = int(pts[:, 0].mean()); cy = int(pts[:, 1].mean())
            anchor = (cx, cy)
        else:
            anchor = tuple(pts[np.argmin(pts[:, 1])])
        label = _zone_label(z)
        (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICK)
        x = max(4, min(anchor[0], w - tw - 8))
        y = max(th + 6, anchor[1] - 6) if kind != "aisle" else anchor[1] + th // 2
        cv2.rectangle(base, (x - 4, y - th - 4), (x + tw + 4, y + 4),
                      (10, 10, 10), -1)
        cv2.putText(base, label, (x, y), LABEL_FONT, LABEL_SCALE,
                    (255, 255, 255), LABEL_THICK, cv2.LINE_AA)

    # 3) In-progress polygon
    if in_progress:
        ip_color = (0, 255, 255)                   # bright yellow
        for px, py in in_progress:
            cv2.circle(base, (int(px), int(py)), VERTEX_DOT_RADIUS,
                       ip_color, -1, lineType=cv2.LINE_AA)
        if len(in_progress) >= 2:
            arr = np.asarray(in_progress, dtype=np.int32)
            cv2.polylines(base, [arr], isClosed=False, color=ip_color,
                          thickness=EDGE_THICKNESS, lineType=cv2.LINE_AA)

    return base
