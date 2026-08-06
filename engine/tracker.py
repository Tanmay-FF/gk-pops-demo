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
import traceback
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
    POSE_MODEL_PATH, POSE_IMGSZ, POSE_CONF_THRESHOLD,
    POSE_KP_CONF_THRESHOLD, POSE_MATCH_IOU_MIN,
    RULE_BLOCKED_DOOR_S, RULE_STATIC_CART_S, RULE_ABANDONED_CART_S,
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
    draw_pose_skeleton,
)
from .video_io import open_video, create_writer, reencode_to_mp4
from .bev3d_builder import build_3d_bev_html, slim_frame_for_3d
from .bev2d_builder import build_2d_bev_html, slim_frame_for_2d
from .bev2d_orientation import attach_orientations
from .frame_capturer import FrameCapturer
from .vlm_analyzer import VLMAnalyzer
from .case_report_builder import build_case_report_html
from . import ui_builder
from . import analytics_ui
from .analytics_builder import run_all as run_analytics, compute_impressions
from .analytics_models import (
    AnalyticsResult, CartFactSample, TrackRecord, TrajectoryBundle, Zone,
    FACTS_SCHEMA_VERSION,
)
from .trajectory_cache import TrajectoryCache, make_video_key


class TrackingEngine:
    def __init__(self, model_path=MODEL_PATH, tracker_config=TRACKER_CONFIG, device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device == "cuda":
            # cudnn.benchmark is deliberately OFF. It is normally a free win for
            # fixed input shapes, but measured on this stack (RTX 3070 Ti Laptop,
            # torch 2.11+cu128) it is a large net loss on BOTH axes:
            #
            #                     benchmark=True   benchmark=False
            #   detector 1st call     63,393 ms          842 ms
            #   pose     1st call     73,505 ms          342 ms
            #   detector steady        49.2 ms         37.3 ms
            #   pose     steady        27.1 ms         25.2 ms
            #
            # The exhaustive per-conv algorithm search costs ~137 s of one-time
            # stall across the two YOLO models (plus ~15 s for the classifiers)
            # and still picks slower kernels than the default heuristic — the
            # trial allocations thrash on a memory-constrained laptop card. That
            # stall was the whole of the "nothing happens for ~140 s after
            # clicking Run Analysis" symptom; steady state never needed work.
            #
            # Re-measure before turning this back on; on a desktop card with
            # memory headroom the trade may well go the other way.
            torch.backends.cudnn.benchmark = False
        self.names = self.model.names
        self.tracker_config = tracker_config

        # Pose model is loaded lazily on first run with pose enabled.
        self._pose_model = None

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

        # Trajectory cache survives across runs so YOLO doesn't re-execute
        # when the user only changes zones.
        self._trajectory_cache = TrajectoryCache()
        self._last_bundle: TrajectoryBundle | None = None
        self._scene_elements: list = []

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
        self._obj_bboxes       = {}          # raw_id -> LATEST bbox (overwritten each frame)
        self._obj_bbox_history = defaultdict(list)  # raw_id -> [bbox, ...] parallel to _obj_positions
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
        # Rule-engine fact timeline: cart display_id -> [CartFactSample, ...].
        # Recorded before the MIN_CART_FRAMES_FOR_POPS guard so brief carts
        # still have facts even when POPS declines to score them.
        self._cart_facts        = defaultdict(list)
        self._cls_last_frame    = {}  # cd -> frame_idx of the last real classification
        self._scene_elements    = []
        # Inputs stashed for finalize_case_report() when process_video is
        # called with defer_case_report=True. Cleared after use.
        self._pending_case_report: dict | None = None

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
    # Trajectory bundle builder + analytics recompute
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_timestamps(ts_arr: np.ndarray, first_frame: int,
                            fps: float) -> tuple[np.ndarray, bool]:
        """Return (timestamps, was_synthesized).

        Every duration the rule engine reports derives from these values, and
        OpenCV's CAP_PROP_POS_MSEC returns 0.0 for every frame on some
        containers/codecs.  Left unchecked, that makes all durations zero, so
        no rule ever crosses its threshold and the output is an empty alert
        list indistinguishable from "nothing happened" — a silent wrong answer.
        Fall back to frame-derived time and let the caller flag the run.
        """
        if ts_arr.size == 0:
            return ts_arr, False
        span = float(ts_arr[-1] - ts_arr[0])
        monotonic = bool(np.all(np.diff(ts_arr) >= -1e-6))
        if span > 1e-6 and monotonic:
            return ts_arr, False           # usable as-is
        safe_fps = fps if fps and fps > 0 else 30.0
        synth = (np.arange(ts_arr.size, dtype=np.float32) + float(first_frame)) / safe_fps
        return synth.astype(np.float32), True

    def _build_trajectory_bundle(self, source_path, w, h, fps, total_frames,
                                 representative_frame):
        """Pack the per-track state collected during process_video() into the
        TrajectoryBundle consumed by analytics_builder."""
        tracks: dict[int, TrackRecord] = {}
        any_synth = False
        for raw_id, label in self._obj_labels.items():
            positions = self._obj_positions.get(raw_id) or []
            timestamps = self._obj_timestamps.get(raw_id) or []
            speeds = self._obj_speeds.get(raw_id) or []
            bboxes = self._obj_bbox_history.get(raw_id) or []
            n = min(len(positions), len(timestamps))
            if n == 0:
                continue
            pos_arr = np.asarray(positions[:n], dtype=np.float32)
            ts_arr  = np.asarray(timestamps[:n], dtype=np.float32)
            # _obj_speeds may be slightly shorter or longer than positions
            spd = list(speeds[:n]) + [0.0] * max(0, n - len(speeds))
            spd_arr = np.asarray(spd[:n], dtype=np.float32)
            first_f = self._obj_first_frame.get(raw_id, 1)
            ts_arr, synth = self._sanitize_timestamps(ts_arr, first_f, fps)
            any_synth = any_synth or synth
            # bbox history is appended in lockstep with positions; a short
            # array means a re-identified track whose history did not carry
            # over, in which case leave it empty rather than misaligned.
            if len(bboxes) >= n:
                bbox_arr = np.asarray(bboxes[:n], dtype=np.float32)
            else:
                bbox_arr = np.empty((0, 4), dtype=np.float32)
            # `frames` is DISPLAY ONLY — derive it from the (possibly
            # synthesised) timestamps rather than assuming the track was
            # detected in every consecutive frame, which arange() did.
            safe_fps = fps if fps and fps > 0 else 30.0
            frames_arr = np.rint(ts_arr * safe_fps).astype(np.int32)
            display_id = self._display_map.get(label, {}).get(raw_id, raw_id)
            tracks[raw_id] = TrackRecord(
                raw_id=raw_id, label=label, display_id=int(display_id),
                positions=pos_arr, timestamps=ts_arr,
                frames=frames_arr, speeds=spd_arr, bboxes=bbox_arr,
            )

        bundle = TrajectoryBundle(
            video_key=make_video_key(source_path),
            video_path=os.path.abspath(source_path),
            width=int(w), height=int(h), fps=float(fps),
            total_frames=int(total_frames),
            tracks=tracks,
            cart_pops=dict(self._peak_pops_snapshot),
            event_log=list(self._event_log),
            representative_frame=representative_frame,
            cart_facts={cd: list(v) for cd, v in self._cart_facts.items()},
            facts_schema_version=FACTS_SCHEMA_VERSION,
            timestamps_synthesized=any_synth,
        )
        return bundle

    def recompute_analytics(self, source_path, zones, *,
                            analytics_out_dir=None,
                            dwell_threshold_s=30.0,
                            camera_placement=None):
        """Skip detection — pull a cached TrajectoryBundle and re-run analytics
        with the supplied zones.

        This is the path that makes the operational rule engine cheap: editing
        a door zone or retuning a threshold re-evaluates every rule over the
        cached facts with no GPU work at all.
        """
        if analytics_out_dir is None:
            analytics_out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        os.makedirs(analytics_out_dir, exist_ok=True)
        zones = list(zones or [])
        if camera_placement is None:
            camera_placement = getattr(self, "_camera_placement",
                                       "Outside (facing entrance)")

        bundle = None
        if source_path:
            video_key = make_video_key(source_path)
            bundle = self._trajectory_cache.get(video_key)
        if bundle is None and self._last_bundle is not None:
            bundle = self._last_bundle

        if bundle is None:
            empty_msg = analytics_ui.build_analytics_empty_state(has_video=bool(source_path))
            return ("", empty_msg, empty_msg, empty_msg, None, None,
                    ui_builder.build_operational_alerts(
                        [], "Run an analysis first — there are no cached "
                            "trajectories to evaluate."))

        result = run_analytics(
            bundle, zones,
            dwell_threshold_s=dwell_threshold_s,
            heatmap_background=bundle.representative_frame,
            out_dir=analytics_out_dir,
            camera_placement=camera_placement,
        )
        summary = analytics_ui.build_analytics_summary(zones, result)
        spikes  = analytics_ui.build_queue_spikes_banner(result.queue_spikes)
        dwell   = analytics_ui.build_dwell_table(zones, result.dwell_summary, result.dwell_rows)
        journey = analytics_ui.build_journey_table(result.journey_matrix, result.journey_labels)
        ops     = ui_builder.build_operational_alerts(
            result.rule_findings, result.rules_unavailable_reason)
        # Return both the ndarray (gr.Image) and the path (gr.File).
        return (summary, spikes, dwell, journey,
                result.heatmap_composite, result.heatmap_png_path, ops)

    def _ensure_pose_model(self):
        if self._pose_model is None:
            print(f"[POSE] loading {POSE_MODEL_PATH} on {self.device} ...")
            self._pose_model = YOLO(POSE_MODEL_PATH)
            self._pose_model.to(self.device)
        return self._pose_model

    @staticmethod
    def _bbox_iou(a, b):
        """IoU of two (x1,y1,x2,y2) boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return (inter / union) if union > 0 else 0.0

    def _match_pose_to_persons(self, pose_boxes, pose_kps, person_raw_to_bbox):
        """Greedy IoU match: for each tracked person, pick the pose result
        with the highest IoU above POSE_MATCH_IOU_MIN. Returns
        {raw_id: keypoints_list} where keypoints_list is 17 [x, y] pairs
        (None for low-confidence keypoints)."""
        out = {}
        used = set()
        for raw, pbb in person_raw_to_bbox.items():
            best_i = -1
            best_iou = POSE_MATCH_IOU_MIN
            for i, kbb in enumerate(pose_boxes):
                if i in used:
                    continue
                iou = self._bbox_iou(pbb, kbb)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_i >= 0:
                used.add(best_i)
                kps = pose_kps[best_i]  # (17, 3): x, y, conf
                kp_list = []
                for x, y, conf in kps:
                    if conf >= POSE_KP_CONF_THRESHOLD:
                        kp_list.append([int(x), int(y)])
                    else:
                        kp_list.append(None)
                out[raw] = kp_list
        return out

    def invalidate_cache(self, source_path=None):
        """Drop the cached trajectory for a specific video, or all of them."""
        if source_path:
            self._trajectory_cache.invalidate(make_video_key(source_path))
        else:
            self._trajectory_cache.clear()
        self._last_bundle = None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def process_video(self, source_path,
                      camera_placement="Outside (facing entrance)",
                      vlm_backend="Claude (API)", vlm_api_key="",
                      zones=None,
                      analytics_out_dir=None,
                      defer_case_report: bool = False,
                      progress=gr.Progress()):
        self._reset()
        self._camera_placement = camera_placement
        self._classifier.set_quality_threshold(QUALITY_THRESHOLD)
        t_start = time.perf_counter()
        zones = zones or []
        if analytics_out_dir is None:
            analytics_out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        os.makedirs(analytics_out_dir, exist_ok=True)

        # Load classifiers from fixed weight paths
        self._classifier.load_quality(QUALITY_WEIGHT_PATH)
        self._classifier.load_fill(FILL_WEIGHT_PATH)

        cap, w, h, fps, total_frames = open_video(source_path)

        # SAM-based one-shot scene layout detection is disabled — the user
        # draws zones explicitly in the Zone Editor, and the 2D BEV reads
        # only those (any analytics-kind zone with applies_to in person/both
        # is auto-promoted to a fixture in the 2D builder loop below).
        _ok, _first_frame = cap.read()
        self._scene_elements = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind before main loop
        # Tracked output is camera-only — BEV lives in dedicated tabs
        # (Bird's-Eye 2D / 3D View / Floor Map).
        writer, avi_path = create_writer(w, h, fps)
        names = self.names
        gdi = self._get_display_id
        links = self._linker.links
        MIN_CART_FRAMES_FOR_POPS = 10

        # Timing accumulators
        _t_yolo = 0.0; _t_cls = 0.0; _t_draw = 0.0; _t_json = 0.0; _t_other = 0.0
        _t_pose = 0.0
        slim_3d_frames = []
        frame_capturer = FrameCapturer()

        # Pose is now always-on. Eagerly load so the first frame doesn't stall.
        self._ensure_pose_model()

        frame_idx = 0
        for _ in progress.tqdm(range(total_frames), desc="Processing frames"):
            ok, im0 = cap.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # --- YOLO + BoTSORT ---
            _t0 = time.perf_counter()
            with torch.no_grad():
                results = self.model.track(im0, persist=True, tracker=self.tracker_config,
                                           imgsz=YOLO_IMGSZ, verbose=False)
            _t_yolo += time.perf_counter() - _t0

            # --- Pose estimation (always on) ---
            # Run once per frame on the full image; results are matched to
            # tracked persons by bbox IoU below.
            pose_boxes = []
            pose_kps_arr = None
            _t0 = time.perf_counter()
            with torch.no_grad():
                pres = self._pose_model.predict(
                    im0, imgsz=POSE_IMGSZ, conf=POSE_CONF_THRESHOLD,
                    device=self.device, verbose=False,
                )
            if pres and pres[0].boxes is not None and pres[0].keypoints is not None:
                pb = pres[0].boxes.xyxy.cpu().numpy()
                pk = pres[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
                pose_boxes = [tuple(map(float, row)) for row in pb]
                pose_kps_arr = pk
            _t_pose += time.perf_counter() - _t0

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
                            self._obj_speeds, self._obj_disappeared,
                            self._obj_bbox_history)

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
                    # Parallel to _obj_positions — the rule engine needs bbox
                    # extent per sample (a cart can block a doorway while its
                    # centroid sits outside the polygon). _obj_bboxes below is
                    # only the latest frame, overwritten each iteration.
                    self._obj_bbox_history[raw].append((x1, y1, x2, y2))
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
                    self._cls_last_frame[cd] = frame_idx
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

            # --- Per-cart facts + POPS scoring ---
            # Fact recording for the rule engine happens BEFORE the cart-age
            # guard, so brief carts still get a fact timeline even when POPS
            # declines to score them.  Everything hoisted above the guard is a
            # PURE READ — the _walkaway_frames counter is still mutated below
            # it, because a re-identified cart inherits a link while its
            # cart_age resets to ~0, and incrementing the counter during that
            # window would let `abandoned` fire earlier than it does today.
            for raw, c, _, bb in frame_detections:
                if names[int(c)] != 'cart':
                    continue
                cd = gdi('cart', raw)
                speed, _, speed_status, _, dir_label = self._motion_cache[raw]

                # Link state — pure read of the linker's map
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

                cr = self._cart_cls_cache.get(cd, {})
                is_valid = cr.get("is_valid", True)
                fill_lbl = cr.get("fill", "unclassified")
                bag_lbl  = cr.get("bag", "not_applicable")

                # Rule-engine fact sample — raw observations only.  Recorded
                # before the grab-and-run fill override below, which is
                # POPS-specific reasoning rather than an observation.
                self._cart_facts[cd].append(CartFactSample(
                    t=timestamp, frame=frame_idx,
                    fill=fill_lbl, bag=bag_lbl,
                    fill_conf=float(cr.get("fill_conf", 0.0)),
                    quality=cr.get("quality", "unclassified"),
                    fill_stale_frames=frame_idx - self._cls_last_frame.get(cd, frame_idx),
                    linked=linked,
                    linked_person_display=(gdi('person', linked_person_raw)
                                           if linked_person_raw is not None else None),
                    abandoned_linker=bool(
                        person_gone
                        or self._walkaway_frames.get(cd, 0) > ABANDON_FRAMES),
                ))

                cart_age = frame_idx - self._obj_first_frame.get(raw, frame_idx)
                if cart_age < MIN_CART_FRAMES_FOR_POPS:
                    continue  # too new — might be a flicker

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

                self._pops_cache[cd] = {
                    "score": pops_score, "event": event_name, "color": event_color,
                    "fill": fill_lbl, "bag": bag_lbl, "direction": dir_label,
                    "speed_status": speed_status, "linked": linked,
                }

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

            # Match pose detections to tracked persons every frame so we can
            # both draw the skeleton on the annotated video AND reuse the
            # match for JSON-sampled frames below (avoids matching twice).
            person_raw_to_bbox = None
            raw_to_kps = None
            if pose_kps_arr is not None and len(pose_boxes):
                person_raw_to_bbox = {
                    raw: tuple(self._obj_bboxes[raw])
                    for raw, c, _, _ in frame_detections
                    if names[int(c)] == 'person' and raw in self._obj_bboxes
                }
                raw_to_kps = self._match_pose_to_persons(
                    pose_boxes, pose_kps_arr, person_raw_to_bbox)
                if raw_to_kps:
                    draw_pose_skeleton(im0, raw_to_kps)

            # Capture key frames for VLM case report
            frame_capturer.check_and_capture(
                im0, frame_idx, timestamp, self._event_log,
                self._pops_cache, links, person_count, cart_count,
                gdi, self._obj_labels)

            _t_draw += time.perf_counter() - _t0

            writer.write(im0)
            # Build per-frame JSON every N frames (configured in config.py)
            if frame_idx % JSON_EVERY_N_FRAMES == 0 or frame_idx == 1:
                _t0 = time.perf_counter()
                frame_json = self._build_frame_json(
                    frame_idx, timestamp, frame_detections, fps)
                self._json_frames[str(frame_idx)] = frame_json

                # Reuse the per-frame pose match for the slim 3D frame.
                pose_for_frame = None
                bbox_for_frame = None
                if raw_to_kps:
                    pose_for_frame = {
                        f"P{gdi('person', raw)}": kps
                        for raw, kps in raw_to_kps.items()
                    }
                    bbox_for_frame = {
                        f"P{gdi('person', raw)}": list(person_raw_to_bbox[raw])
                        for raw in raw_to_kps.keys()
                    }
                slim_3d_frames.append(slim_frame_for_3d(
                    frame_json, pose_kps=pose_for_frame, person_bboxes=bbox_for_frame))
                _t_json += time.perf_counter() - _t0

        # Capture final frame for case report
        frame_capturer.capture_final_frame(
            im0, frame_idx, timestamp,
            self._pops_cache, self._cart_cls_cache)

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

            # Context (direction, speed, linked, abandoned): from best event.
            # Loaded BEFORE the vote so grab-and-run override has correct
            # abandoned state (peak snapshot may not have captured it).
            if cd in _best_event:
                ev = _best_event[cd]
                direction = ev["direction"]
                linked = ev["linked"]
                speed_status = ev.get("speed_status", speed_status)
                abandoned = ev.get("abandoned", abandoned)
                source = "event-ctx"

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

                    if source == "event-ctx":
                        source = "conf-vote+event-ctx"
                    else:
                        source = "conf-vote"

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
        print(f"  Pose       : {_t_pose:.2f}s ({_t_pose/(t_frames-t_start)*100:.0f}%)")
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

        # --- Build TrajectoryBundle (for analytics + cache reuse) ---
        rep_frame = im0.copy() if isinstance(im0, np.ndarray) else None
        bundle = self._build_trajectory_bundle(
            source_path, w, h, fps, total_frames, rep_frame,
        )
        self._last_bundle = bundle
        self._trajectory_cache.put(bundle)

        # --- Run analytics over the bundle ---
        analytics_result: AnalyticsResult = run_analytics(
            bundle, list(zones), out_dir=analytics_out_dir,
            heatmap_background=rep_frame,
            camera_placement=camera_placement,
        )

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
        bev3d_html  = build_3d_bev_html(
            slim_3d_frames, w, h, fps, total_frames,
            zones=list(zones or []), enable_pose=True,
        )

        # --- 2D BEV (velocity / orientation / proximity / impressions) ---
        slim_2d_frames = [
            slim_frame_for_2d(self._json_frames[k])
            for k in sorted(self._json_frames.keys(), key=int)
        ]
        attach_orientations(slim_2d_frames)

        fixtures = []
        for z in zones:
            if (getattr(z, "kind", "analytics") == "analytics"
                    and z.applies_to in ("person", "both") and z.polygon.size):
                xs = z.polygon[:, 0]; ys = z.polygon[:, 1]
                fixtures.append({
                    "id": z.zone_id, "label": z.name,
                    "x1": int(xs.min()), "y1": int(ys.min()),
                    "x2": int(xs.max()), "y2": int(ys.max()),
                })
        for i, se in enumerate(self._scene_elements):
            fixtures.append({
                "id": f"scene_{i}", "label": se.label,
                "x1": int(se.x1), "y1": int(se.y1),
                "x2": int(se.x2), "y2": int(se.y2),
            })

        impressions = compute_impressions(bundle, fixtures)
        bev2d_html  = build_2d_bev_html(
            slim_2d_frames, fixtures, impressions,
            w, h, fps, total_frames,
        )

        # --- Top-of-page alert banner (high-priority events + severe spikes) ---
        alert_banner_html = ui_builder.build_alert_banner(
            self._event_log, analytics_result.queue_spikes,
            analytics_result.spike_events,
            analytics_result.rule_findings,
        )
        ops_alerts_html = ui_builder.build_operational_alerts(
            analytics_result.rule_findings,
            analytics_result.rules_unavailable_reason,
        )

        # Operational findings go in the JSON under their OWN key, never spliced
        # into "events". The event log drives FrameCapturer, which JPEG-encodes
        # a full frame for every new event name and hands those frames to the
        # VLM case report — so anything added to "events" leaves the device by
        # default. Keeping rules separate is what makes the Phase-2
        # child-in-cart work safe to add here later.
        full_json["rule_findings"] = [
            {
                "rule_id": f.rule_id, "label": f.label, "severity": f.severity,
                "cart_id": f.cart_display_id,
                "zone_id": f.zone_id, "zone_name": f.zone_name,
                "start_t": f.start_t, "end_t": f.end_t,
                "duration_s": f.duration_s, "threshold_s": f.threshold_s,
                "ongoing_at_end_of_video": f.ongoing_at_eov,
                "confidence": f.confidence, "n_samples": f.n_samples,
                "reasons": list(f.reasons), "evidence": dict(f.evidence),
            }
            for f in analytics_result.rule_findings
        ]
        full_json["rule_engine"] = {
            "unavailable_reason": analytics_result.rules_unavailable_reason,
            "timestamps_synthesized": bundle.timestamps_synthesized,
            "thresholds_s": {
                "blocked_door": RULE_BLOCKED_DOOR_S,
                "static_cart": RULE_STATIC_CART_S,
                "abandoned_cart": RULE_ABANDONED_CART_S,
            },
        }
        # Re-emit now that the rule keys exist (the first write happened before
        # analytics ran, since analytics consumes the bundle built from it).
        with open(json_path, 'w') as f_json:
            json.dump(full_json, f_json, indent=2)
        json_str = json.dumps(full_json, indent=2)

        # --- Analytics HTML ---
        analytics_summary_html = analytics_ui.build_analytics_summary(list(zones), analytics_result)
        spikes_html  = analytics_ui.build_queue_spikes_banner(analytics_result.queue_spikes)
        dwell_html   = analytics_ui.build_dwell_table(
            list(zones), analytics_result.dwell_summary, analytics_result.dwell_rows)
        journey_html = analytics_ui.build_journey_table(
            analytics_result.journey_matrix, analytics_result.journey_labels)
        heatmap_path = analytics_result.heatmap_png_path
        # Composite is the colored + alpha-blended heatmap as a BGR ndarray.
        # Surfaced separately so the UI can hand it straight to gr.Image
        # without going through Gradio's flaky path-string handler.
        heatmap_img_bgr = analytics_result.heatmap_composite

        # --- Case Report (VLM analysis) ---
        case_report_html = ""
        case_report_file = None

        if defer_case_report and frame_capturer.captures:
            # Stash everything finalize_case_report() needs and return a
            # placeholder. The caller (Gradio handler) yields the rest of the
            # pipeline output immediately, then calls finalize_case_report()
            # to fill in the report.
            self._pending_case_report = {
                "captures": list(frame_capturer.captures),
                "full_json": full_json,
                "event_log": list(self._event_log),
                "peak_snapshots": dict(self._peak_pops_snapshot),
                "vlm_backend": vlm_backend,
                "vlm_api_key": vlm_api_key,
                "analytics_result": analytics_result,
            }
            case_report_html = (
                "<div style='padding:20px;color:#94a3b8;font-family:Nunito Sans,sans-serif;'>"
                "<div style='display:flex;align-items:center;gap:10px;'>"
                "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                "background:#3b82f6;animation:pulse 1.4s ease-in-out infinite;'></span>"
                "<span style='font-weight:600;color:#1e3a5f;'>Generating case report…</span>"
                "</div>"
                "<div style='font-size:0.85rem;margin-top:6px;'>"
                "Pipeline finished — the VLM is now analysing captured frames. "
                "This tab will refresh automatically when ready.</div>"
                "<style>@keyframes pulse {0%,100%{opacity:1}50%{opacity:0.3}}</style>"
                "</div>"
            )
        elif frame_capturer.captures:
            try:
                case_report_html, case_report_file = self._run_case_report(
                    captures=frame_capturer.captures,
                    full_json=full_json,
                    event_log=self._event_log,
                    peak_snapshots=self._peak_pops_snapshot,
                    vlm_backend=vlm_backend,
                    vlm_api_key=vlm_api_key,
                    analytics_result=analytics_result,
                )
            except Exception as e:
                print(f"[WARN] Case report generation failed: {e}")
                traceback.print_exc()
                case_report_html = (
                    f"<p style='color:#ef4444;padding:20px;'>Case report generation failed: {e}</p>"
                )
        print(f"[HEATMAP→UI] composite={None if heatmap_img_bgr is None else (heatmap_img_bgr.shape, heatmap_img_bgr.dtype)} png={heatmap_path}")

        # Note: emit BOTH the composite ndarray (for gr.Image) and the path
        # (for gr.File). The caller splits them into the two output slots.
        return (out_path, json_path, json_str,
                video_html, det_html, config_html, legend_html, pops_html, events_html,
                bev3d_html, bev2d_html, case_report_html, case_report_file,
                analytics_summary_html, spikes_html, dwell_html, journey_html,
                heatmap_img_bgr, heatmap_path,
                alert_banner_html, ops_alerts_html)

    # ------------------------------------------------------------------
    # Case-report generation (extracted so it can run synchronously inside
    # process_video, OR deferred and run via finalize_case_report() while
    # the rest of the dashboard is already visible).
    # ------------------------------------------------------------------
    def _run_case_report(self, *, captures, full_json, event_log,
                         peak_snapshots, vlm_backend, vlm_api_key,
                         analytics_result) -> tuple[str, str | None]:
        vlm = VLMAnalyzer(backend=vlm_backend, api_key=vlm_api_key,
                          device=self.device)
        report_data = vlm.analyze_incident(
            captures=captures,
            pops_data=full_json,
            event_log=event_log,
            peak_snapshots=peak_snapshots,
            video_info=full_json["video_info"],
            analytics_result=analytics_result,
        )
        vlm.unload_model()

        gradio_html, standalone_html = build_case_report_html(
            report_data, captures, full_json, event_log,
            peak_snapshots, full_json["video_info"],
        )

        report_name = f"pops_case_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = os.path.join(tempfile.gettempdir(), report_name)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(standalone_html)

        return gradio_html, report_path

    def finalize_case_report(self) -> tuple[str, str | None]:
        """Run the deferred VLM case-report pass.

        Consumes state stashed by `process_video(defer_case_report=True)`.
        Returns (case_report_html, case_report_file_path). Returns ("", None)
        when there's nothing pending — safe to call unconditionally.
        Errors are caught and surfaced as an inline error banner; the file
        is None in that case so gr.File renders empty.
        """
        pending = self._pending_case_report
        self._pending_case_report = None  # consume regardless of outcome
        if not pending or not pending.get("captures"):
            return "", None
        try:
            return self._run_case_report(**pending)
        except Exception as e:
            print(f"[WARN] Case report generation failed: {e}")
            traceback.print_exc()
            err_html = (
                f"<p style='color:#ef4444;padding:20px;'>"
                f"Case report generation failed: {e}</p>"
            )
            return err_html, None
