# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
One-shot store layout detection using MobileSAM auto-mask on the first video frame.
Returns a list of SceneElement (bbox + color + label) for BEV overlay rendering.
No new dependencies — ultralytics is already installed for YOLO.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Muted BGR palette — distinct but not distracting on the dark BEV canvas
_PALETTE = [
    ( 60, 120, 200),   # blue       — shelving / rack
    ( 60, 180, 120),   # green      — entrance / exit
    (200, 120,  60),   # amber      — aisle / floor section
    (180,  60, 180),   # purple     — checkout
    ( 60, 200, 200),   # cyan       — display / ATM
    (200, 200,  60),   # yellow     — misc
    (140,  80, 200),   # violet     — misc
    (120, 120, 120),   # grey       — misc
]

_STORE_KEYWORDS = ["aisle", "shelf", "checkout", "entrance", "exit",
                   "display", "atm", "counter", "floor", "rack"]


@dataclass
class SceneElement:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: tuple        # BGR
    area_frac: float


def detect_scene_elements(
    frame_bgr: np.ndarray,
    vlm_backend: str = "None (skip)",
    max_elements: int = 6,
) -> list[SceneElement]:
    """
    Run MobileSAM auto-mask on frame_bgr, return top N major scene elements.
    Returns [] on any failure so the BEV renders as today.
    """
    try:
        return _detect(frame_bgr, vlm_backend, max_elements)
    except Exception as exc:
        print(f"[SCENE] detection skipped: {exc}")
        return []


def _detect(frame_bgr, vlm_backend, max_elements):
    from ultralytics import SAM

    h, w = frame_bgr.shape[:2]
    total_px = h * w

    model = SAM("mobile_sam.pt")
    results = model(frame_bgr, verbose=False)

    if not results or results[0].masks is None:
        return []

    masks = results[0].masks.data.cpu().numpy()   # (N, H, W)
    boxes = results[0].boxes.xyxy.cpu().numpy()   # (N, 4)

    # Score by area, filter too-small and too-large masks
    scored: list[tuple[float, int, np.ndarray]] = []
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        area_frac = float(mask.sum()) / total_px
        if 0.02 < area_frac < 0.75:
            scored.append((area_frac, i, box))

    # Largest first, take top N
    scored.sort(reverse=True)
    selected = scored[:max_elements]

    elements: list[SceneElement] = []
    for slot, (area_frac, _, box) in enumerate(selected):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = _PALETTE[slot % len(_PALETTE)]
        label = _label_element(frame_bgr, x1, y1, x2, y2, slot, vlm_backend)
        elements.append(SceneElement(
            label=label, x1=x1, y1=y1, x2=x2, y2=y2,
            color=color, area_frac=area_frac,
        ))

    print(f"[SCENE] detected {len(elements)} major element(s) from first frame")
    return elements


def _label_element(frame_bgr, x1, y1, x2, y2, idx, vlm_backend) -> str:
    if vlm_backend == "Moondream2 (local)":
        try:
            from engine.vlm_analyzer import caption_crop_moondream
            crop = frame_bgr[y1:y2, x1:x2]
            kw = caption_crop_moondream(crop)
            if kw:
                return kw.capitalize()
        except Exception:
            pass
    return f"Area {idx + 1}"
