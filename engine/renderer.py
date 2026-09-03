# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
OpenCV drawing primitives — bounding boxes, text overlays, link lines, HUD,
and bird's-eye-view (BEV) panel rendering.

All functions mutate im0 in-place and return nothing (zero allocation).
"""
import math

import cv2
import numpy as np

from .config import (
    COLOR_PERSON, COLOR_CART, COLOR_LINK,
    CLR_UNCLEAR, CLR_NA, FILL_COLOR_MAP,
    BEV_BG_COLOR, BEV_GRID_COLOR, BEV_DOT_RADIUS,
    BEV_TRAIL_THICKNESS, BEV_ARROW_LENGTH, BEV_LABEL_SCALE, BEV_LEGEND_BG,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def outlined_text(im0, text, pos, scale, color, thickness=1):
    cv2.putText(im0, text, pos, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(im0, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bbox(im0, box, display_id, cls_id, names, class_colors):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = class_colors.get(int(cls_id), (255, 255, 255))
    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
    label = f"{names[int(cls_id)]}:{display_id}"
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.8, 2)
    bg_x1 = x1
    bg_x2 = bg_x1 + tw + 20
    bg_y2 = y1
    bg_y1 = bg_y2 - th - 16
    cv2.rectangle(im0, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    tx = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
    ty = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2
    cv2.putText(im0, label, (tx, ty), _FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


# COCO 17-keypoint skeleton: pairs of keypoint indices defining bones.
_COCO_SKELETON: tuple[tuple[int, int], ...] = (
    # face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # left arm
    (5, 7), (7, 9),
    # right arm
    (6, 8), (8, 10),
    # left leg
    (11, 13), (13, 15),
    # right leg
    (12, 14), (14, 16),
)
# Per-bone color: warm tones for left side, cool tones for right side, yellow for face/torso.
_COCO_SKELETON_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),    # face — yellow
    (255, 200, 0), (255, 200, 0), (255, 200, 0), (255, 200, 0),    # torso — cyan
    (180, 105, 255), (180, 105, 255),                              # left arm — pink
    (255, 180, 105), (255, 180, 105),                              # right arm — orange
    (180, 105, 255), (180, 105, 255),                              # left leg — pink
    (255, 180, 105), (255, 180, 105),                              # right leg — orange
)


def draw_pose_skeleton(im0, raw_to_kps,
                       point_radius: int = 3,
                       bone_thickness: int = 2) -> None:
    """Draw COCO-17 skeletons on im0 in-place.

    raw_to_kps : dict[int, list]   Output of TrackingEngine._match_pose_to_persons.
                                    Each value is a list of 17 entries; each entry
                                    is either [x, y] (visible keypoint) or None.
    """
    if not raw_to_kps:
        return
    for kps in raw_to_kps.values():
        if not kps:
            continue
        # Bones — only drawn when both endpoints are visible.
        for (a, b), color in zip(_COCO_SKELETON, _COCO_SKELETON_COLORS_BGR):
            pa = kps[a] if a < len(kps) else None
            pb = kps[b] if b < len(kps) else None
            if pa is None or pb is None:
                continue
            cv2.line(im0, (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])), color,
                     bone_thickness, cv2.LINE_AA)
        # Keypoint dots on top of bones.
        for kp in kps:
            if kp is None:
                continue
            cv2.circle(im0, (int(kp[0]), int(kp[1])),
                       point_radius, (50, 220, 255), -1, cv2.LINE_AA)


def draw_centroid_trail(im0, track, cx, cy, color):
    cv2.circle(im0, (int(cx), int(cy)), 5, color, -1)
    if len(track) >= 2:
        pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(im0, [pts], False, color, 2)


def draw_classification_overlay(im0, bbox, cls_result, pops_info):
    """Draw quality / fill / bag / POPS below cart bbox."""
    if cls_result is None:
        return
    x1 = int(bbox[0])
    oy = int(bbox[3]) + 18

    if cls_result["quality"] == "unclear":
        outlined_text(im0, "UNCLEAR", (x1, oy), 0.45, CLR_UNCLEAR)
        oy += 16
        outlined_text(im0, "Fill: N/A | Bag: N/A", (x1, oy), 0.4, CLR_NA)
        oy += 16
    elif cls_result["quality"] == "valid_cart":
        fill_lbl = cls_result["fill"].upper()
        bag_lbl  = cls_result["bag"].replace("_", " ").upper()
        fc = FILL_COLOR_MAP.get(fill_lbl, CLR_NA)
        outlined_text(im0, f"{fill_lbl} | {bag_lbl}", (x1, oy), 0.45, fc)
        oy += 16

    if pops_info:
        score = pops_info["score"]
        event = pops_info["event"]
        color = pops_info["color"]
        outlined_text(im0, f"POPS:{score} {event}", (x1, oy), 0.45, color)


#: Rule-badge fill per severity, BGR. Solid chip, dark text — the same shape
#: draw_bbox() gives a "cart:N" label, because these ARE labels and an overlay
#: that draws one kind of label as a dark card and another as a colour chip
#: reads as two unrelated systems.
#:
#: Sharing the shape means the COLOUR has to carry the whole distinction, so
#: the family sits in a band nothing else in the frame occupies: rose to
#: violet to sky. POPS runs red / dark-orange / yellow / green, carts are
#: orange, people green, links magenta — a solid amber chip beside a solid
#: orange "cart:1" chip is the one pairing a viewer will misread, and this
#: avoids it outright rather than hoping the shade is far enough apart.
#:
#: Warm to cool still ranks them: rose demands a person, violet wants one,
#: sky is a note, slate is background. Every stop is light enough for dark
#: text, which is what keeps them readable over bright store footage.
_RULE_SEVERITY_COLORS = {
    "SAFETY": (133, 113, 251),    # rose-400    #fb7185
    "ACTION": (250, 139, 167),    # violet-400  #a78bfa
    "WATCH":  (252, 211, 125),    # sky-300     #7dd3fc
    "INFO":   (225, 213, 203),    # slate-300   #cbd5e1
}
#: Near-black rather than pure: matches the text draw_bbox puts on its own
#: chips, and a hard 0 on a lossy encode fringes worse.
_RULE_BADGE_FG = (20, 20, 20)
_RULE_BADGE_H = 24


def _darkened(color, amount=0.35):
    return tuple(int(c * amount) for c in color)


def _overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def draw_rule_badges(im0, badges):
    """Draw operational rule findings ABOVE a cart's bbox.

    Above, because everything below a cart box is taken: fill/bag, POPS, and
    the link label already stack down from y2+18 (see
    draw_classification_overlay and the cart loop in tracker.process_video).
    draw_bbox's own "cart:N" chip sits immediately above y1, so these start
    above that and stack upwards.

    Placement is collision-driven, not per-cart. Two carts parked side by side
    in one doorway are the normal case for these rules, not the exception, and
    their badges are wide enough to overlap even when the carts barely do —
    whichever drew last simply erased the other, so one of two flagged carts
    silently had no label at all. Each badge instead climbs until it finds
    clear air.

    `badges` is the per-frame list built by rules.overlay_index(), already
    ordered most-severe-first, so the badge a viewer most needs is the one that
    gets the row closest to its cart.
    """
    im_h, im_w = im0.shape[:2]
    # Seeded with the "cart:N" chip of every badged cart: a badge that climbs
    # clear of its neighbour's badge must not land on its neighbour's label.
    placed = [(int(b["bbox"][0]), int(b["bbox"][1]) - 26,
               int(b["bbox"][0]) + 110, int(b["bbox"][1]))
              for b in badges]

    for b in badges:
        bbox = b["bbox"]
        x1, y1 = int(bbox[0]), int(bbox[1])

        severity = b.get("severity")
        accent = _RULE_SEVERITY_COLORS.get(severity,
                                           _RULE_SEVERITY_COLORS["INFO"])
        label = b["text"]
        zone = b.get("zone_name")
        if zone:
            # Colon, not a dash. FONT_HERSHEY_SIMPLEX draws a hyphen long
            # enough to read as an em dash on the frame, and it sat between two
            # names where a separator is all that was wanted. Matches the
            # "Cart:N" form the rest of the overlay already uses.
            label = f"{label}: {zone}"
        cart = b.get("cart_display_id")
        if cart is not None:
            # Same vocabulary the link labels use ("-> Cart:2"), so a displaced
            # badge still says who it is about without a leader line.
            label = f"Cart:{cart}  {label}"
        # Split so the two halves can be weighted differently: WHAT is wrong
        # reads first at full contrast, HOW LONG follows in a darkened tint of
        # the fill. One undifferentiated string makes the reader parse the
        # whole chip to find the rule name.
        tail = f"  {b.get('elapsed_s', 0.0):.0f}s"
        if b.get("degraded"):
            # Sparsely observed, or a synthesised clock. The finding still
            # stands, but it must not read as identical to a clean one.
            tail += " (?)"

        (lw, _th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        (tw, _th2), _ = cv2.getTextSize(tail, _FONT, 0.5, 1)
        w_px = lw + tw + 18
        # A cart at the frame edge is exactly the cart worth badging — one
        # parked half out of a doorway — so the badge slides back into frame
        # rather than being clipped to "UNATTENDED CAR". The 2px inset keeps
        # the chip's own border visible instead of merging into the edge.
        x = max(2, min(x1, im_w - w_px - 2))

        rect = None
        # Climb above the cart first, then hang below it, then sit inside it.
        # Each candidate is one badge-height further from the box.
        for y_bot in ([y1 - 26 - r * (_RULE_BADGE_H + 3) for r in range(8)]
                      + [int(bbox[3]) + 4 + r * (_RULE_BADGE_H + 3) + _RULE_BADGE_H
                         for r in range(4)]
                      + [min(im_h, int(bbox[3])) - 2 - r * (_RULE_BADGE_H + 3)
                         for r in range(4)]):
            y_top = y_bot - _RULE_BADGE_H
            if y_top < 0 or y_bot > im_h:
                continue
            cand = (x, y_top, x + w_px, y_bot)
            if not any(_overlaps(cand, q) for q in placed):
                rect = cand
                break
        if rect is None:
            # Nowhere clear anywhere on the frame. Better an overlapped badge
            # than a cart the video never flags at all, so take the first slot
            # that at least fits, and only give up if even that is impossible.
            y_bot = max(_RULE_BADGE_H, min(im_h, y1 - 26))
            if y_bot - _RULE_BADGE_H < 0:
                continue
            rect = (x, y_bot - _RULE_BADGE_H, x + w_px, y_bot)
        placed.append(rect)

        bx1, y_top, bx2, y_bot = rect
        cv2.rectangle(im0, (bx1, y_top), (bx2, y_bot), accent, -1)
        # A heavier border for SAFETY. Severity is then legible even to a
        # viewer who cannot separate the hues — a blocked fire exit should not
        # depend on colour vision to stand out from a parked cart.
        cv2.rectangle(im0, (bx1, y_top), (bx2, y_bot), _RULE_BADGE_FG,
                      2 if severity == "SAFETY" else 1)
        tx = bx1 + 9
        cv2.putText(im0, label, (tx, y_bot - 8), _FONT, 0.5,
                    _RULE_BADGE_FG, 1, cv2.LINE_AA)
        cv2.putText(im0, tail, (tx + lw, y_bot - 8), _FONT, 0.5,
                    _darkened(accent), 1, cv2.LINE_AA)


def draw_person_overlay(im0, bbox, speed_status, dir_label, link_label):
    """Draw speed / direction / link info below person bbox."""
    if dir_label == "OUTBOUND" and speed_status in ("MEDIUM", "FAST"):
        tc = (0, 0, 255)
    elif dir_label == "OUTBOUND":
        tc = (0, 165, 255)
    elif dir_label == "INBOUND":
        tc = (200, 200, 0)
    else:
        tc = (200, 200, 200)

    x1 = int(bbox[0])
    oy = int(bbox[3]) + 18

    lines = [speed_status]
    if dir_label != "UNKNOWN":
        lines.append(dir_label)
    if link_label:
        lines.append(link_label)

    for line in lines:
        c = (0, 255, 0) if "->" in line else tc
        outlined_text(im0, line, (x1, oy), 0.45, c)
        oy += 16


def draw_link_lines(im0, links, det_centroids, get_display_id,
                    person_disp_to_raw, cart_disp_to_raw):
    """Draw magenta lines between linked person-cart pairs."""
    drawn = set()
    count = 0
    for cart_raw, person_raw in links.items():
        cd = get_display_id('cart', cart_raw)
        pd = get_display_id('person', person_raw)
        if (pd, cd) in drawn:
            continue
        c_raw = cart_raw if cart_raw in det_centroids else cart_disp_to_raw.get(cd)
        p_raw = person_raw if person_raw in det_centroids else person_disp_to_raw.get(pd)
        if c_raw and c_raw in det_centroids and p_raw and p_raw in det_centroids:
            cpt = det_centroids[c_raw]
            ppt = det_centroids[p_raw]
            cv2.line(im0, cpt, ppt, COLOR_LINK, 2, cv2.LINE_AA)
            mx = (cpt[0] + ppt[0]) // 2
            my = (cpt[1] + ppt[1]) // 2
            outlined_text(im0, "Linked", (mx - 25, my - 8), 0.5, COLOR_LINK)
            count += 1
        elif c_raw or p_raw:
            count += 1
        drawn.add((pd, cd))
    return count


def draw_hud(im0, person_count, cart_count, link_count,
             frame_idx, total_frames, w):
    """Draw frame counter, person/cart/link counts."""
    items = [
        (f"Persons: {person_count}", COLOR_PERSON),
        (f"Carts: {cart_count}", COLOR_CART),
        (f"Links: {link_count}", COLOR_LINK),
    ]
    y_off = 30
    for text, color in items:
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.8, 2)
        cv2.rectangle(im0, (10, y_off - th - 5), (10 + tw + 10, y_off + 5), (0, 0, 0), -1)
        cv2.putText(im0, text, (15, y_off), _FONT, 0.8, color, 2, cv2.LINE_AA)
        y_off += th + 20
    outlined_text(im0, f"Frame: {frame_idx}/{total_frames}", (w - 250, 30), 0.6, (255, 255, 255))


# =========================================================================
# Bird's Eye View (BEV) panel
# =========================================================================

def _bev_xy(cx, cy, orig_w, orig_h, bev_w, bev_h):
    """Map camera-space centroid to BEV pixel position."""
    return int(cx / orig_w * bev_w), int(cy / orig_h * bev_h)


def _draw_bev_grid(bev, bev_w, bev_h):
    for frac in (0.25, 0.50, 0.75):
        x = int(bev_w * frac)
        y = int(bev_h * frac)
        cv2.line(bev, (x, 0), (x, bev_h), BEV_GRID_COLOR, 1)
        cv2.line(bev, (0, y), (bev_w, y), BEV_GRID_COLOR, 1)


def _draw_bev_trails(bev, track_history, colors_by_id,
                     orig_w, orig_h, bev_w, bev_h):
    for raw_id, track in track_history.items():
        if len(track) < 2:
            continue
        color = colors_by_id.get(raw_id, (180, 180, 180))
        pts = []
        for cx, cy in track:
            bx, by = _bev_xy(cx, cy, orig_w, orig_h, bev_w, bev_h)
            pts.append([bx, by])
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(bev, [arr], False, color, BEV_TRAIL_THICKNESS)


def _draw_bev_links(bev, links, det_centroids,
                    orig_w, orig_h, bev_w, bev_h):
    for cart_raw, person_raw in links.items():
        if cart_raw not in det_centroids or person_raw not in det_centroids:
            continue
        c_cx, c_cy = det_centroids[cart_raw]
        p_cx, p_cy = det_centroids[person_raw]
        c_bev = _bev_xy(c_cx, c_cy, orig_w, orig_h, bev_w, bev_h)
        p_bev = _bev_xy(p_cx, p_cy, orig_w, orig_h, bev_w, bev_h)
        cv2.line(bev, p_bev, c_bev, COLOR_LINK, 2, cv2.LINE_AA)
        mx = (c_bev[0] + p_bev[0]) // 2
        my = (c_bev[1] + p_bev[1]) // 2
        outlined_text(bev, "Linked", (mx - 20, my - 6), 0.35, COLOR_LINK)


def _draw_bev_dots(bev, det_centroids, colors_by_id,
                   orig_w, orig_h, bev_w, bev_h):
    for raw_id, (cx, cy) in det_centroids.items():
        color = colors_by_id.get(raw_id, (180, 180, 180))
        bx, by = _bev_xy(cx, cy, orig_w, orig_h, bev_w, bev_h)
        cv2.circle(bev, (bx, by), BEV_DOT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(bev, (bx, by), BEV_DOT_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_bev_direction_arrows(bev, det_centroids, motion_cache,
                               orig_w, orig_h, bev_w, bev_h):
    for raw_id, (cx, cy) in det_centroids.items():
        if raw_id not in motion_cache:
            continue
        _, direction_deg, speed_status, _, _ = motion_cache[raw_id]
        if speed_status == "STATIC":
            continue
        bx, by = _bev_xy(cx, cy, orig_w, orig_h, bev_w, bev_h)
        rad = math.radians(direction_deg)
        ex = int(bx + BEV_ARROW_LENGTH * math.cos(rad))
        ey = int(by + BEV_ARROW_LENGTH * math.sin(rad))
        cv2.arrowedLine(bev, (bx, by), (ex, ey), (220, 220, 220), 2,
                        cv2.LINE_AA, tipLength=0.35)


def _draw_bev_labels(bev, det_centroids, obj_labels, gdi, pops_cache,
                     orig_w, orig_h, bev_w, bev_h):
    for raw_id, (cx, cy) in det_centroids.items():
        label_name = obj_labels.get(raw_id, "?")
        disp = gdi(label_name, raw_id)
        prefix = "P" if label_name == "person" else "C"
        tag = f"{prefix}{disp}"

        bx, by = _bev_xy(cx, cy, orig_w, orig_h, bev_w, bev_h)
        tx = max(5, min(bx + BEV_DOT_RADIUS + 3, bev_w - 50))
        ty = max(15, min(by - BEV_DOT_RADIUS - 2, bev_h - 5))
        outlined_text(bev, tag, (tx, ty), BEV_LABEL_SCALE, (255, 255, 255))

        if label_name == "cart" and disp in pops_cache:
            info = pops_cache[disp]
            score_txt = f"POPS:{info['score']}"
            badge_y = ty + 14
            (tw, th), _ = cv2.getTextSize(score_txt, _FONT, 0.35, 1)
            cv2.rectangle(bev, (tx - 2, badge_y - th - 2),
                          (tx + tw + 4, badge_y + 3), info["color"], -1)
            cv2.putText(bev, score_txt, (tx, badge_y), _FONT, 0.35,
                        (0, 0, 0), 1, cv2.LINE_AA)


def _draw_bev_legend(bev, bev_w, bev_h):
    lx, ly = 10, bev_h - 100
    lw, lh = 140, 90
    cv2.rectangle(bev, (lx, ly), (lx + lw, ly + lh), BEV_LEGEND_BG, -1)
    cv2.rectangle(bev, (lx, ly), (lx + lw, ly + lh), (80, 80, 85), 1)

    row = ly + 16
    cv2.circle(bev, (lx + 12, row - 4), 5, COLOR_PERSON, -1)
    cv2.putText(bev, "Person", (lx + 22, row), _FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    row += 18
    cv2.circle(bev, (lx + 12, row - 4), 5, COLOR_CART, -1)
    cv2.putText(bev, "Cart", (lx + 22, row), _FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    row += 18
    cv2.line(bev, (lx + 6, row - 4), (lx + 18, row - 4), COLOR_LINK, 2)
    cv2.putText(bev, "Linked", (lx + 22, row), _FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    row += 18
    cv2.arrowedLine(bev, (lx + 6, row - 4), (lx + 18, row - 4),
                    (220, 220, 220), 2, tipLength=0.5)
    cv2.putText(bev, "Direction", (lx + 22, row), _FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_bev_scene_elements(bev, scene_elements, orig_w, orig_h, bev_w, bev_h):
    """Draw detected store layout regions as semi-transparent fills behind the grid."""
    if not scene_elements:
        return
    overlay = bev.copy()
    for el in scene_elements:
        bx1, by1 = _bev_xy(el.x1, el.y1, orig_w, orig_h, bev_w, bev_h)
        bx2, by2 = _bev_xy(el.x2, el.y2, orig_w, orig_h, bev_w, bev_h)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), el.color, -1)
    cv2.addWeighted(overlay, 0.18, bev, 0.82, 0, bev)
    for el in scene_elements:
        bx1, by1 = _bev_xy(el.x1, el.y1, orig_w, orig_h, bev_w, bev_h)
        bx2, by2 = _bev_xy(el.x2, el.y2, orig_w, orig_h, bev_w, bev_h)
        cv2.rectangle(bev, (bx1, by1), (bx2, by2), el.color, 1)
        cv2.putText(bev, el.label, (bx1 + 4, by1 + 14), _FONT, 0.35, el.color, 1, cv2.LINE_AA)


def render_bev(bev_w, bev_h, orig_w, orig_h,
               det_centroids, obj_labels, track_history,
               links, gdi, motion_cache, pops_cache, class_colors,
               scene_elements=None):
    """Render a complete bird's-eye-view panel. Returns np.ndarray (bev_h, bev_w, 3)."""
    # obj_labels maps raw_id -> 'person'|'cart', but class_colors maps cls_id -> color.
    # Build a raw_id -> color lookup using the label name.
    _name_to_color = {"person": COLOR_PERSON, "cart": COLOR_CART}
    _label_colors = {rid: _name_to_color.get(lbl, (180, 180, 180))
                     for rid, lbl in obj_labels.items()}

    bev = np.full((bev_h, bev_w, 3), BEV_BG_COLOR, dtype=np.uint8)
    _draw_bev_scene_elements(bev, scene_elements or [], orig_w, orig_h, bev_w, bev_h)
    _draw_bev_grid(bev, bev_w, bev_h)
    _draw_bev_trails(bev, track_history, _label_colors, orig_w, orig_h, bev_w, bev_h)
    _draw_bev_links(bev, links, det_centroids, orig_w, orig_h, bev_w, bev_h)
    _draw_bev_dots(bev, det_centroids, _label_colors, orig_w, orig_h, bev_w, bev_h)
    _draw_bev_direction_arrows(bev, det_centroids, motion_cache, orig_w, orig_h, bev_w, bev_h)
    _draw_bev_labels(bev, det_centroids, obj_labels, gdi, pops_cache, orig_w, orig_h, bev_w, bev_h)
    _draw_bev_legend(bev, bev_w, bev_h)
    outlined_text(bev, "Bird's Eye View", (bev_w // 2 - 75, 28), 0.6, (255, 255, 255), 2)
    return bev
