# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
TrackingEngine — orchestrates detection, tracking, linking, classification,
scoring, rendering, and JSON export for a single video.

This is the only class that touches YOLO / BoTSORT.  Everything else is
delegated to the focused modules in this package.
"""
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    MODEL_PATH, TRACKER_CONFIG,
    COLOR_PERSON, COLOR_CART,
    YOLO_IMGSZ, CLASSIFY_EVERY_N_FRAMES, JSON_EVERY_N_FRAMES,
    LINK_CONFIRM_FRAMES, LINK_GRACE_FRAMES, ABANDON_FRAMES,
    QUALITY_WEIGHT_PATH, FILL_WEIGHT_PATH, QUALITY_THRESHOLD,
    WALKAWAY_DIST_THRESH,
)
from .classifier import CartClassifier
from .linker import PersonCartLinker
from .motion import compute_motion, compute_direction_label
from .scoring import (
    compute_pops, classify_event,
    LOGGABLE_EVENTS, HIGH_EVENTS, MEDIUM_EVENTS,
)
from .renderer import (
    draw_bbox, draw_centroid_trail, draw_classification_overlay,
    draw_person_overlay, draw_link_lines, draw_hud, outlined_text,
)
from .video_io import open_video, create_writer, reencode_to_mp4
from . import ui_builder


class TrackingEngine:
    def __init__(self, model_path=MODEL_PATH, tracker_config=TRACKER_CONFIG, device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.names = self.model.names
        self.tracker_config = tracker_config

        # Build colour map once
        self._class_colors = {}
        for cid, name in self.names.items():
            if name == 'person':
                self._class_colors[cid] = COLOR_PERSON
            elif name == 'cart':
                self._class_colors[cid] = COLOR_CART
            else:
                self._class_colors[cid] = (255, 255, 255)

        # Sub-systems
        self._classifier = CartClassifier(self.device)
        self._linker: PersonCartLinker = None  # created in _reset

        self._reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _reset(self):
        self._display_map = {}
        self._next_display = {}
        self.track_history = defaultdict(list)

        self._obj_positions    = defaultdict(list)
        self._obj_timestamps   = defaultdict(list)
        self._obj_speeds       = defaultdict(list)
        self._obj_labels       = {}
        self._obj_confs        = {}
        self._obj_bboxes       = {}
        self._obj_first_frame  = {}
        self._obj_disappeared  = defaultdict(int)

        self._linker = PersonCartLinker(self._get_display_id)

        self._json_frames       = {}
        self._all_people_seen   = set()
        self._all_carts_seen    = set()

        self._cart_cls_cache    = {}
        self._pops_cache        = {}
        self._event_log         = []
        self._max_pops_per_cart = {}
        self._peak_pops_snapshot= {}
        self._cart_cls_history  = defaultdict(list)  # cd -> [(fill, bag), ...]
        self._motion_cache      = {}  # raw_id -> (speed, direction, status, accel, dir_label)
        self._walkaway_frames   = {}  # cd -> consecutive frames person is far from cart

    def _get_display_id(self, label, raw_id):
        if label not in self._display_map:
            self._display_map[label] = {}
            self._next_display[label] = 1
        m = self._display_map[label]
        if raw_id not in m:
            m[raw_id] = self._next_display[label]
            self._next_display[label] += 1
        return m[raw_id]

    # ------------------------------------------------------------------
    # Link-info helper (used for overlay text)
    # ------------------------------------------------------------------
    def _get_link_info(self, raw_id, is_person):
        links = self._linker.links
        perm_p = self._linker.permanently_linked_persons
        gdi = self._get_display_id
        if is_person:
            pd = gdi('person', raw_id)
            for cid, pid in links.items():
                if pid == raw_id:
                    return True, f"-> Cart:{gdi('cart', cid)}"
            if pd in perm_p:
                for cid, pid in links.items():
                    if gdi('person', pid) == pd:
                        return True, f"-> Cart:{gdi('cart', cid)}"
        else:
            cd = gdi('cart', raw_id)
            if raw_id in links:
                return True, f"-> Person:{gdi('person', links[raw_id])}"
            if cd in self._linker.permanently_linked_carts:
                for cid, pid in links.items():
                    if gdi('cart', cid) == cd:
                        return True, f"-> Person:{gdi('person', pid)}"
        return False, None

    # ------------------------------------------------------------------
    # Per-frame JSON builder
    # ------------------------------------------------------------------
    def _build_frame_json(self, frame_idx, timestamp, frame_detections, fps):
        people, carts = {}, {}
        frame_persons, frame_carts = {}, {}
        gdi = self._get_display_id
        links = self._linker.links

        for raw_id, cls, conf, bbox in frame_detections:
            label = self.names[int(cls)]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            is_person = label == 'person'
            is_cart = label == 'cart'
            if not is_person and not is_cart:
                continue

            # Use cached motion instead of recomputing
            cached = self._motion_cache.get(raw_id)
            if cached:
                speed, direction, speed_status, accel, dir_label = cached
            else:
                speed, direction, speed_status, accel = compute_motion(
                    self._obj_positions[raw_id], self._obj_timestamps[raw_id],
                    self._obj_speeds[raw_id], fps)
                dir_label = compute_direction_label(self._obj_positions[raw_id],
                                                    getattr(self, '_camera_placement', 'Outside (facing entrance)'))
            display_id = gdi('person' if is_person else 'cart', raw_id)
            prefix = "P" if is_person else "C"
            key = f"{prefix}{display_id}"

            pos_hist = [{"x": round(p[0], 1), "y": round(p[1], 1)}
                        for p in self._obj_positions[raw_id][-5:]]
            spd_hist = [round(s, 2) for s in self._obj_speeds[raw_id][-5:]]

            obj = {
                "id": display_id,
                "centroid": {"x": round(cx, 1), "y": round(cy, 1)},
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "width": x2 - x1, "height": y2 - y1},
                "motion": {"speed": round(speed, 2), "direction": round(direction, 2),
                           "direction_label": dir_label, "speed_status": speed_status,
                           "acceleration": round(accel, 2)},
                "tracking": {"positions_history": pos_hist, "speed_history": spd_hist,
                             "disappeared_frames": 0, "yolo_confidence": round(conf, 4)},
            }

            if is_person:
                obj["linking"] = {"is_linked": False, "linked_cart_id": None, "link_confidence": 0.0}
                people[key] = obj
                frame_persons[raw_id] = (cx, cy)
                self._all_people_seen.add(raw_id)
            else:
                cr = self._cart_cls_cache.get(display_id, {})
                pi = self._pops_cache.get(display_id, {})
                obj["classification"] = {
                    "quality": cr.get("quality", "unclassified"),
                    "fill": cr.get("fill", "unclassified"),
                    "bag": cr.get("bag", "unclassified"),
                    "quality_conf": round(cr.get("quality_conf", 0.0), 4),
                    "fill_conf": round(cr.get("fill_conf", 0.0), 4),
                    "bag_conf": round(cr.get("bag_conf", 0.0), 4),
                }
                obj["pops"] = {"score": pi.get("score", 0), "event": pi.get("event", "CLEAR")}
                obj["linking"] = {"is_linked": False, "linked_person_id": None, "link_confidence": 0.0}
                carts[key] = obj
                frame_carts[raw_id] = (cx, cy)
                self._all_carts_seen.add(raw_id)

        # Populate link info
        link_data = {}
        active = 0
        seen_pairs = set()
        for cart_raw, person_raw in links.items():
            cd = gdi('cart', cart_raw)
            pd = gdi('person', person_raw)
            ck, pk = f"C{cd}", f"P{pd}"
            if (pk, ck) in seen_pairs:
                continue
            c_in, p_in = ck in carts, pk in people
            if c_in and p_in:
                cp = frame_carts.get(cart_raw, (0, 0))
                pp = frame_persons.get(person_raw, (0, 0))
                if cart_raw not in frame_carts:
                    for r, pos in frame_carts.items():
                        if gdi('cart', r) == cd:
                            cp = pos; break
                if person_raw not in frame_persons:
                    for r, pos in frame_persons.items():
                        if gdi('person', r) == pd:
                            pp = pos; break
                dist = math.sqrt((cp[0] - pp[0]) ** 2 + (cp[1] - pp[1]) ** 2)
                carts[ck]["linking"].update({"is_linked": True, "linked_person_id": pd})
                people[pk]["linking"].update({"is_linked": True, "linked_cart_id": cd})
                sf = self._linker.link_start_frames.get(cart_raw, frame_idx)
                link_data[f"{pk}_{ck}"] = {
                    "person_id": pd, "cart_id": cd, "distance": round(dist, 2),
                    "established_frame": sf, "duration_frames": frame_idx - sf,
                }
                active += 1
            elif c_in or p_in:
                if c_in: carts[ck]["linking"].update({"is_linked": True, "linked_person_id": pd})
                if p_in: people[pk]["linking"].update({"is_linked": True, "linked_cart_id": cd})
                active += 1
            seen_pairs.add((pk, ck))

        p_dis = sum(1 for r, l in self._obj_labels.items() if l == 'person' and 0 < self._obj_disappeared[r] < 90)
        c_dis = sum(1 for r, l in self._obj_labels.items() if l == 'cart' and 0 < self._obj_disappeared[r] < 90)

        return {
            "frame_number": frame_idx, "timestamp": round(timestamp, 4),
            "people": people, "carts": carts, "links": link_data,
            "statistics": {"total_people": len(people), "total_carts": len(carts),
                           "active_links": active,
                           "people_disappeared": p_dis, "carts_disappeared": c_dis},
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def process_video(self, source_path,
                      camera_placement="Outside (facing entrance)",
                      progress=gr.Progress()):
        self._reset()
        self._camera_placement = camera_placement
        self._classifier.set_quality_threshold(QUALITY_THRESHOLD)
        t_start = time.perf_counter()

        # Load classifiers from fixed weight paths
        self._classifier.load_quality(QUALITY_WEIGHT_PATH)
        self._classifier.load_fill(FILL_WEIGHT_PATH)

        cap, w, h, fps, total_frames = open_video(source_path)
        writer, avi_path = create_writer(w, h, fps)
        names = self.names
        gdi = self._get_display_id
        links = self._linker.links
        MIN_CART_FRAMES_FOR_POPS = 10

        # Timing accumulators
        _t_yolo = 0.0; _t_cls = 0.0; _t_draw = 0.0; _t_json = 0.0; _t_other = 0.0

        frame_idx = 0
        for _ in progress.tqdm(range(total_frames), desc="Processing frames"):
            ok, im0 = cap.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # --- YOLO + BoTSORT ---
            _t0 = time.perf_counter()
            results = self.model.track(im0, persist=True, tracker=self.tracker_config,
                                       imgsz=YOLO_IMGSZ, verbose=False)
            _t_yolo += time.perf_counter() - _t0

            person_count = 0
            cart_count = 0
            frame_detections = []

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                r = results[0]
                boxes = r.boxes.xyxy.cpu()
                ids   = r.boxes.id.cpu().tolist()
                clss  = r.boxes.cls.tolist()
                confs = r.boxes.conf.cpu().tolist()

                # Count
                for c in clss:
                    lbl = names[int(c)]
                    if lbl == 'person':   person_count += 1
                    elif lbl == 'cart':   cart_count += 1

                # Cart re-ID
                cur_cart_raws = {int(id_) for box, id_, c, _ in zip(boxes, ids, clss, confs) if names[int(c)] == 'cart'}
                for box, id_, c, conf in zip(boxes, ids, clss, confs):
                    if names[int(c)] == 'cart':
                        raw = int(id_)
                        bb = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                        self._linker.try_reidentify_cart(
                            raw, bb, cur_cart_raws, self._display_map,
                            self._obj_positions, self._obj_timestamps,
                            self._obj_speeds, self._obj_disappeared)

                # Draw + collect detections
                for box, id_, c, conf in zip(boxes, ids, clss, confs):
                    raw = int(id_)
                    label = names[int(c)]
                    disp = gdi(label, raw)
                    draw_bbox(im0, box, disp, c, names, self._class_colors)

                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

                    track = self.track_history[raw]
                    track.append((cx, cy))
                    if len(track) > 50:
                        track.pop(0)
                    tc = self._class_colors.get(int(c), (255, 255, 255))
                    draw_centroid_trail(im0, track, cx, cy, tc)

                    bb_int = (int(x1), int(y1), int(x2), int(y2))
                    frame_detections.append((raw, c, conf, bb_int))

                    self._obj_positions[raw].append((cx, cy))
                    self._obj_timestamps[raw].append(timestamp)
                    self._obj_labels[raw] = label
                    self._obj_confs[raw] = conf
                    self._obj_bboxes[raw] = bb_int
                    if raw not in self._obj_first_frame:
                        self._obj_first_frame[raw] = frame_idx
                    self._obj_disappeared[raw] = 0

            # Disappearance tracking
            seen = {d[0] for d in frame_detections}
            for rid in self._obj_labels:
                if rid not in seen:
                    self._obj_disappeared[rid] += 1

            # --- Linking ---
            person_bb = {}
            cart_bb = {}
            for raw, c, _, bb in frame_detections:
                lbl = names[int(c)]
                if lbl == 'person': person_bb[raw] = bb
                elif lbl == 'cart': cart_bb[raw] = bb
            self._linker.update(person_bb, cart_bb, frame_idx,
                                self._obj_disappeared, self._obj_positions,
                                self._obj_first_frame)

            # --- Classification (every N frames) ---
            _t0 = time.perf_counter()
            if frame_idx % CLASSIFY_EVERY_N_FRAMES == 0 and self._classifier.has_quality_model:
                for raw, c, _, bb in frame_detections:
                    if names[int(c)] != 'cart':
                        continue
                    cd = gdi('cart', raw)
                    result = self._classifier.classify(im0, bb)
                    self._cart_cls_cache[cd] = result
                    if result.get("quality") == "valid_cart":
                        self._cart_cls_history[cd].append(
                            (result["fill"], result["bag"],
                             result.get("fill_conf", 0.0), result.get("bag_conf", 0.0))
                        )

            _t_cls += time.perf_counter() - _t0

            # --- Compute motion once per object, cache for reuse ---
            _t0 = time.perf_counter()
            self._motion_cache.clear()
            for raw, c, _, bb in frame_detections:
                speed, direction, speed_status, accel = compute_motion(
                    self._obj_positions[raw], self._obj_timestamps[raw],
                    self._obj_speeds[raw], fps)
                dir_label = compute_direction_label(
                    self._obj_positions[raw], camera_placement)
                self._motion_cache[raw] = (speed, direction, speed_status, accel, dir_label)
                self._obj_speeds[raw].append(speed)

            # Sync linked cart direction with person direction.
            # A linked person+cart move together — direction must match.
            for cart_raw, person_raw in links.items():
                if cart_raw in self._motion_cache and person_raw in self._motion_cache:
                    person_dir = self._motion_cache[person_raw][4]
                    if person_dir in ("INBOUND", "OUTBOUND"):
                        old = self._motion_cache[cart_raw]
                        self._motion_cache[cart_raw] = (old[0], old[1], old[2], old[3], person_dir)

            # --- POPS scoring for each cart ---
            for raw, c, _, bb in frame_detections:
                if names[int(c)] != 'cart':
                    continue
                cd = gdi('cart', raw)
                cart_age = frame_idx - self._obj_first_frame.get(raw, frame_idx)
                if cart_age < MIN_CART_FRAMES_FOR_POPS:
                    continue  # too new — might be a flicker
                speed, _, speed_status, _, dir_label = self._motion_cache[raw]

                # Link / abandonment
                linked = False
                linked_person_raw = None
                for cid, pid in links.items():
                    if gdi('cart', cid) == cd:
                        linked = True
                        linked_person_raw = pid
                        break
                # Classic: person gone from frame for N frames
                person_gone = (linked and linked_person_raw is not None
                               and self._obj_disappeared.get(linked_person_raw, 0) > ABANDON_FRAMES)
                # Walkaway: person visible but far from cart for N consecutive frames
                person_far = False
                if linked and linked_person_raw is not None and linked_person_raw in person_bb:
                    pb = person_bb[linked_person_raw]
                    pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    ccx, ccy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    dist = ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5
                    if dist > WALKAWAY_DIST_THRESH:
                        self._walkaway_frames[cd] = self._walkaway_frames.get(cd, 0) + 1
                    else:
                        self._walkaway_frames.pop(cd, None)
                    person_far = self._walkaway_frames.get(cd, 0) > ABANDON_FRAMES
                abandoned = person_gone or person_far

                cr = self._cart_cls_cache.get(cd, {})
                is_valid = cr.get("is_valid", True)
                fill_lbl = cr.get("fill", "unclassified")
                bag_lbl  = cr.get("bag", "not_applicable")

                # For abandoned carts: if currently empty but previously had
                # items, use the peak fill from history.  A cart that went from
                # partial/full → empty means someone grabbed items and ran.
                if abandoned and fill_lbl == "empty" and cd in self._cart_cls_history:
                    _FILL_RANK = {"empty": 0, "partial": 1, "full": 2}
                    for h_fill, h_bag, _, _ in self._cart_cls_history[cd]:
                        if _FILL_RANK.get(h_fill, 0) > _FILL_RANK.get(fill_lbl, 0):
                            fill_lbl = h_fill
                            bag_lbl = h_bag

                pops_score = compute_pops(dir_label, speed_status, is_valid, fill_lbl,
                                          bag_label=bag_lbl, cart_detected=True,
                                          abandoned=abandoned, linked=linked)
                event_name, event_color = classify_event(pops_score, linked, dir_label, abandoned=abandoned)

                self._pops_cache[cd] = {"score": pops_score, "event": event_name, "color": event_color}

                prev_max = self._max_pops_per_cart.get(cd, 0)
                quality_lbl = cr.get("quality", "unclassified")
                is_valid_cls = quality_lbl not in ("unclear", "unclassified")

                if pops_score >= prev_max:
                    self._max_pops_per_cart[cd] = pops_score
                    # Freeze score/event/color at peak so "HIGH PRIORITY"
                    # doesn't get downgraded to "MONITORING" later.
                    self._peak_pops_snapshot[cd] = {
                        "score": pops_score,
                        "event": event_name, "color": event_color,
                        "fill": fill_lbl, "bag": bag_lbl, "direction": dir_label,
                        "quality": quality_lbl,
                        "speed_status": speed_status,
                        "linked": linked, "abandoned": abandoned,
                    }


                # Log significant events (once per cart per event type).
                # Skip lower-severity events if a HIGH event was already
                # logged for this cart — the pushout already happened.
                # Also skip events where the cart was not validly classified
                # (unclear/unclassified) — these are noise, not actionable.
                if event_name in LOGGABLE_EVENTS:
                    skip = False
                    if quality_lbl in ("unclear", "unclassified") and event_name not in HIGH_EVENTS:
                        skip = True  # don't log noise from unclassified carts
                    cart_has_high = any(
                        e["cart_id"] == cd and e["event"] in HIGH_EVENTS
                        for e in self._event_log
                    )
                    already_logged = any(
                        e["cart_id"] == cd and e["event"] == event_name
                        for e in self._event_log
                    )
                    if not skip and not already_logged and not (cart_has_high and event_name not in HIGH_EVENTS):
                        self._event_log.append({
                            "frame": frame_idx, "timestamp": round(timestamp, 2),
                            "cart_id": cd, "event": event_name, "pops_score": pops_score,
                            "fill": fill_lbl, "bag": bag_lbl,
                            "direction": dir_label, "linked": linked,
                            "speed_status": speed_status, "abandoned": abandoned,
                        })

            _t_other += time.perf_counter() - _t0

            # --- Overlays ---
            _t0 = time.perf_counter()
            # Disp -> raw maps for link-line drawing
            p_d2r, c_d2r = {}, {}
            for raw, c, _, _ in frame_detections:
                lbl = names[int(c)]
                if lbl == 'person': p_d2r[gdi('person', raw)] = raw
                elif lbl == 'cart': c_d2r[gdi('cart', raw)] = raw

            # Person overlays (use cached motion)
            for raw, c, _, bb in frame_detections:
                if names[int(c)] != 'person':
                    continue
                _, _, status, _, dlbl = self._motion_cache[raw]
                _, lp = self._get_link_info(raw, True)
                draw_person_overlay(im0, bb, status, dlbl, lp)

            # Cart overlays
            for raw, c, _, bb in frame_detections:
                if names[int(c)] != 'cart':
                    continue
                cd = gdi('cart', raw)
                draw_classification_overlay(im0, bb, self._cart_cls_cache.get(cd), self._pops_cache.get(cd))
                _, lp = self._get_link_info(raw, False)
                if lp:
                    oy = int(bb[3]) + 18 + 16 * 3
                    outlined_text(im0, lp, (int(bb[0]), oy), 0.45, (0, 255, 0))

            # Link lines
            det_centroids = {}
            for raw, _, _, bb in frame_detections:
                det_centroids[raw] = (int((bb[0] + bb[2]) // 2), int((bb[1] + bb[3]) // 2))
            active_link_count = draw_link_lines(im0, links, det_centroids, gdi, p_d2r, c_d2r)

            # HUD
            draw_hud(im0, person_count, cart_count, active_link_count, frame_idx, total_frames, w)

            _t_draw += time.perf_counter() - _t0

            writer.write(im0)
            # Build per-frame JSON every N frames (configured in config.py)
            if frame_idx % JSON_EVERY_N_FRAMES == 0 or frame_idx == 1:
                _t0 = time.perf_counter()
                self._json_frames[str(frame_idx)] = self._build_frame_json(
                    frame_idx, timestamp, frame_detections, fps)
                _t_json += time.perf_counter() - _t0

        cap.release()
        writer.release()

        # --- Unified POPS summary reconciliation ---
        # Pick authoritative fill/bag, then RECOMPUTE score so everything
        # (score, event, fill, bag, direction) tells a coherent story.
        _EVENT_SEVERITY = {
            "PUSHOUT ALERT": 5, "HIGH PRIORITY": 4,
            "ABANDONED CART": 3, "MEDIUM PRIORITY": 3,
            "UNLINKED EXIT": 2, "LOW PRIORITY": 1,
        }
        _best_event = {}
        for ev in self._event_log:
            cd = ev["cart_id"]
            sev = _EVENT_SEVERITY.get(ev["event"], 0)
            prev = _best_event.get(cd)
            prev_sev = _EVENT_SEVERITY.get(prev["event"], 0) if prev else -1
            if sev > prev_sev or (sev == prev_sev and ev["frame"] > (prev or {}).get("frame", 0)):
                _best_event[cd] = ev

        for cd in set(list(self._peak_pops_snapshot) + list(self._cart_cls_history)):
            if cd not in self._peak_pops_snapshot:
                continue
            snap = self._peak_pops_snapshot[cd]
            original_score = snap["score"]

            # Defaults from peak snapshot
            best_fill = None
            best_bag = None
            direction = snap.get("direction", "UNKNOWN")
            speed_status = snap.get("speed_status", "STATIC")
            linked = snap.get("linked", False)
            abandoned = snap.get("abandoned", False)
            source = "snapshot"

            # Fill/bag: ALWAYS use confidence-weighted vote from full history.
            # Events can be logged at early frames with wrong predictions;
            # the vote across all frames is more reliable.
            if cd in self._cart_cls_history:
                history = self._cart_cls_history[cd]
                if history:
                    fill_conf = defaultdict(float)
                    fill_count = defaultdict(int)
                    bag_conf = defaultdict(float)
                    bag_count = defaultdict(int)
                    for fill, bag, fc, bc in history:
                        fill_conf[fill] += fc
                        fill_count[fill] += 1
                        bag_conf[bag] += bc
                        bag_count[bag] += 1
                    # confidence_sum × frame_count — rewards both high confidence and consistency
                    fill_scores = {f: fill_conf[f] * fill_count[f] for f in fill_count}
                    bag_scores = {b: bag_conf[b] * bag_count[b] for b in bag_count}
                    best_fill = max(fill_scores, key=fill_scores.get)
                    best_bag = max(bag_scores, key=bag_scores.get)
                    print(f"[VOTE] Cart {cd}: fill_conf={dict(fill_conf)} fill_count={dict(fill_count)} fill_scores={dict(fill_scores)} → {best_fill}")
                    print(f"[VOTE] Cart {cd}: bag_conf={dict(bag_conf)} bag_count={dict(bag_count)} bag_scores={dict(bag_scores)} → {best_bag}")

                    # # [OLD] Abandoned cart override (grab-and-run) — no threshold,
                    # # fires on ANY non-empty frame in early 30%. Too aggressive:
                    # # classifier noise in early frames wrongly overrides the vote.
                    # if best_fill == "empty" and abandoned:
                    #     n = len(history)
                    #     early_end = max(1, n * 30 // 100)
                    #     early_history = history[:early_end]
                    #     for candidate in ("full", "partial"):
                    #         if any(f == candidate for f, b, fc, bc in early_history):
                    #             best_fill = candidate
                    #             paired_bags = defaultdict(float)
                    #             for f, b, fc, bc in early_history:
                    #                 if f == candidate:
                    #                     paired_bags[b] += bc
                    #             if paired_bags:
                    #                 best_bag = max(paired_bags, key=paired_bags.get)
                    #             break

                    # [NEW] Abandoned cart override (grab-and-run) with temporal
                    # ordering check. Compare first half vs second half of the
                    # classification history. A real grab-and-run shows items in
                    # the first half and empty in the second half. Classifier
                    # noise is distributed evenly across both halves.
                    # Override only if: first half had >50% items AND second
                    # half has >70% empty — confirms a clear transition.
                    if best_fill == "empty" and abandoned:
                        n = len(history)
                        print(f"[DEBUG] Cart {cd}: history order = {[f for f, b, fc, bc in history]}")
                        mid = max(1, n // 2)
                        first_half = history[:mid]
                        second_half = history[mid:]

                        first_fill_count = defaultdict(int)
                        for f, b, fc, bc in first_half:
                            first_fill_count[f] += 1
                        second_fill_count = defaultdict(int)
                        for f, b, fc, bc in second_half:
                            second_fill_count[f] += 1

                        first_had_items = ((first_fill_count.get("full", 0)
                                            + first_fill_count.get("partial", 0))
                                           >= len(first_half) * 0.5)
                        second_is_empty = (second_fill_count.get("empty", 0)
                                           > len(second_half) * 0.7)

                        if first_had_items and second_is_empty:
                            for candidate in ("full", "partial"):
                                if first_fill_count.get(candidate, 0) > 0:
                                    best_fill = candidate
                                    paired_bags = defaultdict(float)
                                    for f, b, fc, bc in first_half:
                                        if f == candidate:
                                            paired_bags[b] += bc
                                    if paired_bags:
                                        best_bag = max(paired_bags, key=paired_bags.get)
                                    break

                    source = "conf-vote"

            # Context (direction, speed, linked, abandoned): from best event
            if cd in _best_event:
                ev = _best_event[cd]
                direction = ev["direction"]
                linked = ev["linked"]
                speed_status = ev.get("speed_status", speed_status)
                abandoned = ev.get("abandoned", abandoned)
                if source == "conf-vote":
                    source = "conf-vote+event-ctx"

            if best_fill is None:
                continue

            # Constraint: partial/full → bag cannot be not_applicable
            if best_fill in ("partial", "full") and best_bag == "not_applicable":
                if cd in self._cart_cls_history:
                    bag_scores = defaultdict(float)
                    for _, bag, _, bc in self._cart_cls_history[cd]:
                        if bag != "not_applicable":
                            bag_scores[bag] += bc
                    best_bag = max(bag_scores, key=bag_scores.get) if bag_scores else "unbagged"
                else:
                    best_bag = "unbagged"

            # RECOMPUTE score with finalized, consistent inputs
            recomputed = compute_pops(
                direction, speed_status, True, best_fill,
                bag_label=best_bag, cart_detected=True,
                abandoned=abandoned, linked=linked,
            )
            final_score = recomputed
            # Re-apply caps based on final fill/bag
            if best_fill == "partial" and best_bag == "bagged":
                final_score = min(final_score, 55)
            final_event, final_color = classify_event(
                final_score, linked, direction, abandoned=abandoned,
            )

            # Write back ALL fields consistently
            snap.update({
                "fill": best_fill, "bag": best_bag, "quality": "valid_cart",
                "score": final_score, "event": final_event, "color": final_color,
                "direction": direction, "speed_status": speed_status,
                "linked": linked, "abandoned": abandoned,
            })
            self._max_pops_per_cart[cd] = final_score
            print(f"[POPS] Cart {cd}: {best_fill}|{best_bag} {direction} "
                  f"score={final_score} (orig={original_score} recomp={recomputed}) "
                  f"[{source}]")

        # --- Sync last event per cart with POPS table (both directions) ---
        # For abandonment events: POPS copies from Events (Events is truth).
        # For all other carts: the last event copies score from POPS table
        # so the Events table shows the reconciled score.
        _ABANDON_EVENTS = {"ABANDONED CART"}

        # Find last event per cart
        _last_event = {}
        for ev in self._event_log:
            _last_event[ev["cart_id"]] = ev

        for cd, ev in _last_event.items():
            if cd not in self._peak_pops_snapshot:
                continue
            snap = self._peak_pops_snapshot[cd]
            if ev["event"] in _ABANDON_EVENTS:
                # Events → POPS (Events is truth for abandonment)
                snap["fill"] = ev["fill"]
                snap["bag"] = ev["bag"]
                snap["score"] = ev["pops_score"]
                snap["event"] = ev["event"]
                self._max_pops_per_cart[cd] = ev["pops_score"]
            else:
                # POPS → last Event (POPS has reconciled score)
                ev["fill"] = snap["fill"]
                ev["bag"] = snap["bag"]
                ev["pops_score"] = snap["score"]
                ev["event"] = snap["event"]

        t_frames = time.perf_counter()
        print(f"[PERF] Breakdown over {frame_idx} frames:")
        print(f"  YOLO+track : {_t_yolo:.2f}s ({_t_yolo/(t_frames-t_start)*100:.0f}%)")
        print(f"  Classify   : {_t_cls:.2f}s ({_t_cls/(t_frames-t_start)*100:.0f}%)")
        print(f"  Motion+POPS: {_t_other:.2f}s ({_t_other/(t_frames-t_start)*100:.0f}%)")
        print(f"  Drawing    : {_t_draw:.2f}s ({_t_draw/(t_frames-t_start)*100:.0f}%)")
        print(f"  Frame JSON : {_t_json:.2f}s ({_t_json/(t_frames-t_start)*100:.0f}%)")

        out_path = reencode_to_mp4(avi_path)
        t_encode = time.perf_counter()
        video_duration = total_frames / fps if fps > 0 else 0
        print(f"[PERF] Frame processing: {t_frames - t_start:.1f}s | "
              f"Video encoding: {t_encode - t_frames:.1f}s | "
              f"Total: {t_encode - t_start:.1f}s | "
              f"Video duration: {video_duration:.1f}s | "
              f"Speed: {video_duration / (t_encode - t_start):.2f}x realtime")

        # --- Build JSON ---
        t_json_start = time.perf_counter()
        full_json = {
            "video_info": {
                "video_name": os.path.basename(source_path),
                "width": w, "height": h, "fps": float(fps),
                "total_frames": total_frames,
                "processing_timestamp": datetime.now().isoformat(),
            },
            "frames": self._json_frames,
            "events": self._event_log,
            "cart_classifications": {f"C{cid}": self._cart_cls_cache.get(cid, {}) for cid in self._cart_cls_cache},
            "pops_summary": {
                f"C{cid}": {
                    "max_score": self._max_pops_per_cart.get(cid, 0),
                    "peak_event": self._peak_pops_snapshot.get(cid, {}).get("event", "CLEAR"),
                }
                for cid in set(list(self._max_pops_per_cart) + list(self._pops_cache))
            },
            "summary": {
                "total_people_seen": len(self._all_people_seen),
                "total_carts_seen": len(self._all_carts_seen),
                "total_links_established": self._linker.total_links,
                "total_events": len(self._event_log),
                "high_priority": sum(1 for e in self._event_log if e["event"] in HIGH_EVENTS),
                "medium_priority": sum(1 for e in self._event_log if e["event"] in MEDIUM_EVENTS),
            },
            "processing_info": {
                "total_frames_processed": frame_idx,
                "json_sampled_frames": len(self._json_frames),
                "json_every_n": JSON_EVERY_N_FRAMES,
                "device": self.device, "model": "YOLOv26m", "tracker": "BoTSORT",
                "quality_model": self._classifier.quality_pt or "None",
                "fill_model": self._classifier.fill_pt or "None",
                "quality_threshold": QUALITY_THRESHOLD,
            },
        }

        json_filename = os.path.splitext(os.path.basename(source_path))[0] + "_tracking.json"
        json_path = os.path.join(tempfile.gettempdir(), json_filename)
        with open(json_path, 'w') as f:
            json.dump(full_json, f, indent=2)
        json_str = json.dumps(full_json, indent=2)
        t_json_end = time.perf_counter()
        print(f"[PERF] JSON build: {t_json_end - t_json_start:.2f}s | "
              f"{len(self._json_frames)} sampled frames (every {JSON_EVERY_N_FRAMES}) | "
              f"JSON size: {len(json_str) / 1024:.0f} KB")

        # --- Build HTML ---
        video_html  = ui_builder.build_video_info(source_path, w, h, fps, total_frames, frame_idx)
        det_html    = ui_builder.build_detection_info(
            len(self._all_people_seen), len(self._all_carts_seen),
            self._linker.total_links)
        config_html = ui_builder.build_config_info(
            LINK_CONFIRM_FRAMES, LINK_GRACE_FRAMES, camera_placement,
            self._classifier.quality_pt, self._classifier.fill_pt, QUALITY_THRESHOLD)
        legend_html = ui_builder.build_legend()
        pops_html   = ui_builder.build_pops_summary(self._max_pops_per_cart, self._peak_pops_snapshot)
        events_html = ui_builder.build_events_timeline(self._event_log)

        return (out_path, json_path, json_str,
                video_html, det_html, config_html, legend_html, pops_html, events_html)
