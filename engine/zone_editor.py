# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Pure helpers for the Gradio zone editor.

Kept Gradio-free on purpose — the UI layer just shuttles data between
gr.State, gr.Image, and these functions.  That makes the editor logic
unit-testable and lets the same helpers be reused (e.g., to bake a
"draw zones over BEV" preview into the case report).
"""
from __future__ import annotations

import uuid
from typing import Iterable, Optional

import cv2
import numpy as np

from .analytics_models import Zone, ZoneAppliesTo, ZoneKind


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
    return True, ""


_LAYOUT_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    # All in BGR for OpenCV. Match the JS hex chosen for the Floor BEV.
    "wall":    (105,  85,  71),    # slate-500-ish
    "aisle":   (200, 200, 200),    # subtle grey
    "fixture": (252, 211, 125),    # sky-300 (#7dd3fc → BGR)
    "door":    ( 94, 197,  34),    # green-500 (#22c55e → BGR)
}


def make_zone(name: str, polygon_pts: Iterable[tuple[int, int]],
              applies_to: ZoneAppliesTo, idx: int,
              kind: ZoneKind = "analytics") -> Zone:
    poly = np.asarray(list(polygon_pts), dtype=np.int32)
    color = _LAYOUT_COLORS_BGR[kind] if kind != "analytics" else polygon_color(idx)
    default_name = f"Zone {idx + 1}" if kind == "analytics" else f"{kind.capitalize()} {idx + 1}"
    return Zone(
        zone_id=str(uuid.uuid4())[:8],
        name=name.strip() or default_name,
        polygon=poly,
        applies_to=applies_to,
        kind=kind,
        color=color,
    )


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
