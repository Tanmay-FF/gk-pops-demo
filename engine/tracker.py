# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
TrackingEngine — orchestrates detection, tracking, linking, classification,
scoring, rendering, and JSON export for a single video.

This is the only class that touches YOLO / BoTSORT.  Everything else is
delegated to the focused modules in this package.
"""
import gc
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
    PROGRESS_MAX_UPDATES, PROGRESS_MIN_INTERVAL_S,
    LINK_CONFIRM_FRAMES, LINK_GRACE_FRAMES, ABANDON_FRAMES,
    QUALITY_WEIGHT_PATH, FILL_WEIGHT_PATH, QUALITY_THRESHOLD,
    WALKAWAY_DIST_THRESH, GRABRUN_MIN_RUN_OBS, DIRECTION_WINDOW_S,
    POSE_MODEL_PATH, POSE_IMGSZ, POSE_CONF_THRESHOLD,
    POSE_KP_CONF_THRESHOLD, POSE_MATCH_IOU_MIN,
    RULE_BLOCKED_DOOR_S, RULE_STATIC_CART_S, RULE_ABANDONED_CART_S,
)
from .ultralytics_compat import apply as _apply_ultralytics_compat
from .classifier import CartClassifier
from .linker import PersonCartLinker
from .motion import compute_motion, compute_direction_label
from .scoring import (
    compute_pops, classify_event, peak_sustained_fill, prune_event_log,
    inbound_suppression_note,
    LOGGABLE_EVENTS, HIGH_EVENTS, MEDIUM_EVENTS,
)
from .renderer import (
    draw_bbox, draw_centroid_trail, draw_classification_overlay,
    draw_person_overlay, draw_link_lines, draw_hud, outlined_text,
    draw_pose_skeleton,
)
from .video_io import open_video, create_writer, reencode_to_mp4
# The 3D View, Bird's-Eye 2D and Floor Map surfaces were removed from this
# build. They embedded PER-FRAME data in a full HTML document that the browser
# received in the same message as everything else, and the page froze at 100%.
# bev3d_builder / bev2d_builder / bev2d_orientation / floor_bev2d_builder are
# still in the tree, just unimported.
from .frame_capturer import FrameCapturer
from .vlm_analyzer import VLMAnalyzer
from .case_report_builder import build_case_report_html

# Restores ByteTrack's low-confidence second association on ultralytics
# >= 8.4.40, where fuse_score is applied to it and makes every match in
# that pass mathematically impossible. Must run before any tracker is
# constructed. See engine/ultralytics_compat.py for the full analysis.
if _apply_ultralytics_compat():
    print("[INFO] ultralytics compat: re-enabled ByteTrack's "
          "low-confidence second association")
from . import ui_builder
from . import analytics_ui
from . import highlights
from . import rules as rule_engine
from .analytics_builder import run_all as run_analytics
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
            # clicking Run Analysis" symptom.
            #
            # Re-measure before turning this back on; on a desktop card with
            # headroom the trade may well go the other way.
            torch.backends.cudnn.benchmark = False
        self.names = self.model.names
        self.tracker_config = tracker_config

        # Pose model is loaded lazily on first run with pose enabled.
        self._pose_model = None
        #: Predictor parked across a GPU offload round-trip — see
        #: _release_detection_gpu_memory() for why losing it corrupts tracking.
        self._held_predictor = None
        #: True while the detection stack is deliberately parked on the CPU for
        #: a local-VLM pass. _ensure_on_device() must not "repair" that.
        self._offloaded = False

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
        # raw_id -> [frame_idx, ...] parallel to _obj_positions. The REAL frame
        # each sample was detected on; _sanitize_timestamps() needs it to build a
        # fallback clock that does not assume gap-free detection.
        self._obj_frames       = defaultdict(list)
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
                dir_label = compute_direction_label(
                    self._obj_positions[raw_id],
                    getattr(self, '_camera_placement', 'Outside (facing entrance)'),
                    self._obj_timestamps[raw_id], DIRECTION_WINDOW_S)
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
    def _sanitize_timestamps(ts_arr: np.ndarray, frames_arr: np.ndarray,
                            fps: float) -> tuple[np.ndarray, bool]:
        """Return (timestamps, was_synthesized).

        Every duration the rule engine reports derives from these values, and
        OpenCV's CAP_PROP_POS_MSEC returns 0.0 for every frame on some
        containers/codecs.  Left unchecked, that makes all durations zero, so
        no rule ever crosses its threshold and the output is an empty alert
        list indistinguishable from "nothing happened" — a silent wrong answer.
        Fall back to frame-derived time and let the caller flag the run.

        The fallback derives from `frames_arr`, the REAL frame index of each
        sample. It used to be `arange(n) + first_frame`, which counts SAMPLES —
        and samples only exist where the track was detected. A cart parked in a
        doorway for 4 minutes but detected one frame in eight came out as a 30s
        track, so no threshold could be crossed and the density gate saw a
        sampling rate 8x denser than reality. Both failures were silent.
        """
        # Fewer than two samples has no MEASURABLE span, which is not the same
        # as a broken clock. Treating it as one was a false positive with real
        # consequences: the caller ORs this flag across every track, so a
        # single 1-sample person track (a one-frame detection, of which a busy
        # clip has several) marked the whole run degraded — stamping every
        # finding "confidence: degraded" and printing "CAP_PROP_POS_MSEC was
        # unusable for this video" on a video whose timing was perfect.
        if ts_arr.size < 2:
            return ts_arr, False
        span = float(ts_arr[-1] - ts_arr[0])
        monotonic = bool(np.all(np.diff(ts_arr) >= -1e-6))
        if span > 1e-6 and monotonic:
            return ts_arr, False           # usable as-is
        safe_fps = fps if fps and fps > 0 else 30.0
        if frames_arr.size == ts_arr.size and frames_arr.size:
            # frame_idx is 1-based; CAP_PROP_POS_MSEC is 0.0 on the first frame,
            # so subtract one to keep the two clocks on the same origin.
            synth = (frames_arr.astype(np.float64) - 1.0) / safe_fps
        else:                              # no frame record (very old state)
            synth = np.arange(ts_arr.size, dtype=np.float64) / safe_fps
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
            frames = self._obj_frames.get(raw_id) or []
            n = min(len(positions), len(timestamps))
            if n == 0:
                continue
            pos_arr = np.asarray(positions[:n], dtype=np.float32)
            ts_arr  = np.asarray(timestamps[:n], dtype=np.float32)
            # _obj_speeds may be slightly shorter or longer than positions
            spd = list(speeds[:n]) + [0.0] * max(0, n - len(speeds))
            spd_arr = np.asarray(spd[:n], dtype=np.float32)
            # Recorded per detection, so it is parallel to positions. A short
            # array means state from before _obj_frames existed; fall back to
            # first_frame + arange rather than misaligning the two.
            if len(frames) >= n:
                frames_arr = np.asarray(frames[:n], dtype=np.int32)
            else:
                first_f = self._obj_first_frame.get(raw_id, 1)
                frames_arr = (np.arange(n, dtype=np.int32) + int(first_f))
            ts_arr, synth = self._sanitize_timestamps(ts_arr, frames_arr, fps)
            any_synth = any_synth or synth
            # bbox history is appended in lockstep with positions; a short
            # array means a re-identified track whose history did not carry
            # over, in which case leave it empty rather than misaligned.
            if len(bboxes) >= n:
                bbox_arr = np.asarray(bboxes[:n], dtype=np.float32)
            else:
                bbox_arr = np.empty((0, 4), dtype=np.float32)
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
                            camera_placement=None,
                            rule_thresholds=None):
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
                        [], "Run an analysis first - there are no cached "
                            "trajectories to evaluate."),
                    "", ui_builder.build_tab_counts({}), "")

        result = run_analytics(
            bundle, zones,
            dwell_threshold_s=dwell_threshold_s,
            heatmap_background=bundle.representative_frame,
            out_dir=analytics_out_dir,
            camera_placement=camera_placement,
            rule_thresholds=rule_thresholds,
        )
        # Same note on the zero-GPU path. The POPS state still lives on this
        # engine, so retuning a zone or a threshold must not drop it - the panel
        # is rebuilt from scratch here and would otherwise lose it.
        _inbound_note = inbound_suppression_note(
            self._peak_pops_snapshot, camera_placement)
        if _inbound_note:
            result.rule_diagnostics.append(_inbound_note)
        summary = analytics_ui.build_analytics_summary(zones, result)
        spikes  = analytics_ui.build_queue_spikes_banner(result.queue_spikes)
        dwell   = analytics_ui.build_dwell_table(zones, result.dwell_summary, result.dwell_rows)
        journey = analytics_ui.build_journey_table(result.journey_matrix, result.journey_labels)
        ops     = ui_builder.build_operational_alerts(
            result.rule_findings, result.rules_unavailable_reason,
            result.rule_diagnostics)
        # Retuning a threshold changes which rules fired, so the sticky banner
        # and the tab badges have to move with it — otherwise the page keeps
        # advertising findings the user just tuned away.
        alert = ui_builder.build_alert_banner(
            self._event_log, result.queue_spikes, result.spike_events,
            result.rule_findings)
        counts = ui_builder.build_tab_counts(
            self._tab_counts(result))
        # The POPS table now carries the operational categories, so retuning a
        # threshold has to rebuild it as well — otherwise that tab keeps
        # advertising categories the user just tuned away. "" when this process
        # holds no POPS state; the caller reads that as "leave the tab alone"
        # rather than overwriting a good table with an empty one.
        pops = (ui_builder.build_pops_summary(
                    self._max_pops_per_cart, self._peak_pops_snapshot,
                    result.rule_findings)
                if self._max_pops_per_cart else "")
        # Return both the ndarray (gr.Image) and the path (gr.File).
        return (summary, spikes, dwell, journey,
                result.heatmap_composite, result.heatmap_png_path, ops,
                alert, counts, pops)

    def _tab_counts(self, analytics_result) -> dict:
        """Badge counts for the tab nav — how many findings live behind each
        tab, so the user can see where the content is without clicking."""
        findings = list(getattr(analytics_result, "rule_findings", []) or [])
        n_high = sum(1 for e in self._event_log if e.get("event") in HIGH_EVENTS)
        n_sev = sum(1 for f in findings if f.severity in ("SAFETY", "ACTION"))
        return {
            "Events": {
                "n": len(self._event_log),
                "tone": "DANGER" if n_high else "INFO",
            },
            "Operational Alerts": {
                "n": len(findings),
                "tone": "DANGER" if n_sev else "INFO",
            },
            "POPS": {
                # The POPS table also lists carts the rule engine flagged but
                # that were never scored, so the badge counts the union — a
                # badge of 3 over a 4-row table reads as a bug. Take the flagged
                # set from cart_flag_index(), which is what actually decides the
                # rows: a cart folded in via evidence["also_carts"] gets a row
                # while never appearing as any finding's own cart_display_id.
                "n": len(set(self._max_pops_per_cart)
                         | set(ui_builder.cart_flag_index(findings))),
                "tone": "DANGER" if n_high else "INFO",
            },
            "Analytics": {
                "n": len(getattr(analytics_result, "queue_spikes", []) or []),
                "tone": "WARN",
            },
        }

    def _ensure_pose_model(self):
        if self._pose_model is None:
            print(f"[POSE] loading {POSE_MODEL_PATH} on {self.device} ...")
            self._pose_model = YOLO(POSE_MODEL_PATH)
            self._pose_model.to(self.device)
        return self._pose_model

    # ------------------------------------------------------------------
    # Detection-stack GPU memory is only needed during the frame loop.
    # A local VLM case-report pass runs after that loop finishes, so we
    # temporarily move YOLO/pose/classifier off the GPU to give the VLM
    # the full card — otherwise the detection stack's ~5-6 GB footprint
    # plus a VLM's own weights can exceed an 8 GB card, forcing slow
    # CPU-offload or (on Windows/WDDM) shared-memory paging instead of a
    # clean OOM.
    # ------------------------------------------------------------------
    def _release_detection_gpu_memory(self):
        if self.device != "cuda":
            return
        # Hold the predictor across the move. ultralytics' Model._apply() runs on
        # every .to() and does `self.predictor = None`, and Model.track() reacts
        # to a missing predictor by calling register_tracker() again — which
        # APPENDS a second on_predict_postprocess_end callback instead of
        # replacing the first. There is still only ONE tracker instance, and
        # every callback in that list calls its update() again on the same
        # frame, each pass seeing only the boxes the previous one kept. So the
        # Kalman predict and the tracker's internal frame counter advance N
        # times per real frame while detections shrink: track confirmation and
        # track_buffer (120) are effectively divided by N, weak detections
        # (people at door distance) never survive to be confirmed, and the
        # association cost is paid N times.
        #
        # Measured on 1764005712780 …Inside (facing exit).mp4, run 2 of the same
        # process after one round-trip: frame 52 went from 3 people to 1, cart 2's
        # classification history from 28 observations to 6, YOLO+track from 11.8s
        # to 17.8s. That is the "detections are missing in the enhanced build"
        # report — it needs a session where the VLM case report has already run
        # once, which is why no first run and no headless run ever showed it.
        #
        # The AutoBackend inside the predictor wraps this same nn.Module
        # (verified `predictor.model.model is self.model.model`), so putting the
        # predictor back after the module returns to CUDA is sound, and it keeps
        # `predictor.trackers` present so Model.track() has nothing to re-register.
        self._held_predictor = getattr(self.model, "predictor", None)
        self.model.to("cpu")
        if self._pose_model is not None:
            self._pose_model.to("cpu")
        if self._classifier._quality_model is not None:
            self._classifier._quality_model.to("cpu")
        if self._classifier._fill_model is not None:
            self._classifier._fill_model.to("cpu")
        self._offloaded = True
        torch.cuda.empty_cache()

    def _restore_detection_gpu_memory(self):
        if self.device != "cuda":
            return
        # Cleared before the moves so the verification at the end of this
        # method is not itself skipped by the guard.
        self._offloaded = False
        try:
            self.model.to(self.device)
            # See _release_detection_gpu_memory() — without this the next
            # .track() call registers a duplicate tracking callback and
            # silently doubles the association pass for the rest of the
            # process.
            if getattr(self, "_held_predictor", None) is not None:
                self.model.predictor = self._held_predictor
                self._held_predictor = None
            if self._pose_model is not None:
                self._pose_model.to(self.device)
            if self._classifier._quality_model is not None:
                self._classifier._quality_model.to(self.device)
            if self._classifier._fill_model is not None:
                self._classifier._fill_model.to(self.device)
        except Exception as e:
            # Typically CUDA OOM because the VLM has not finished giving its
            # VRAM back. Never let this escape: it would propagate out of
            # _run_case_report's finally, replacing the real error, and it
            # leaves a module half-moved either way. Free what we can and let
            # _ensure_on_device() below retry the move.
            print(f"[WARN] restoring the detection stack to {self.device} failed: {e}")
            gc.collect()
            torch.cuda.empty_cache()
        self._ensure_on_device(context="after the local-VLM offload")

    # ------------------------------------------------------------------
    # Device-drift guard.
    #
    # The GPU round-trip above is the known way a model can end up on the
    # wrong device: torch's Module._apply moves parameters one at a time, so
    # a CUDA OOM partway through `.to("cuda")` leaves a module with some
    # weights on CUDA and some on CPU, and the exception is swallowed by
    # finalize_case_report()'s handler as an inline banner. Nothing reloads
    # them afterwards — CartClassifier.load_quality/load_fill early-return on
    # an unchanged pt_path — so every later run raises
    #   RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
    #   (torch.FloatTensor) should be the same
    # until the process restarts. Checking at the top of every run turns that
    # into a one-run problem, and names the model that drifted.
    # ------------------------------------------------------------------
    @staticmethod
    def _param_device(module) -> str | None:
        """The device type every parameter of `module` is on, or None when the
        module is absent / has no parameters. Returns "mixed" when they
        disagree, which is what a `.to()` that OOM'd partway through leaves
        behind — so this walks all parameters, not just the first."""
        if module is None:
            return None
        # Unwrap only an ultralytics Model (identified by `predictor`, which
        # nn.Module never has). A bare getattr("model") would silently scan
        # one submodule of a plain nn.Module that happens to name a child
        # `model`, so a partially-moved sibling would read clean.
        inner = module.model if hasattr(module, "predictor") else module
        try:
            devs = {p.device.type for p in inner.parameters()}
        except Exception:
            return None
        if not devs:
            return None
        return devs.pop() if len(devs) == 1 else "mixed"

    def _move_detector(self, dev: str):
        """`.to()` the detector while preserving its predictor — see
        _release_detection_gpu_memory() for why losing it corrupts tracking.

        Falls back to the predictor parked by _release_detection_gpu_memory():
        a `.to()` that raised has already run Model._apply(), which nulls
        `predictor`, so on the retry path the live attribute is gone and only
        the parked one is left."""
        held = (getattr(self.model, "predictor", None)
                or getattr(self, "_held_predictor", None))
        self.model.to(dev)
        if held is not None:
            self.model.predictor = held
            self._held_predictor = None

    def _ensure_on_device(self, context: str = ""):
        """Repair any model whose weights are not on self.device.

        Returns the list of model names that were wrong (empty on the happy
        path). If the repair itself fails, the whole stack is dropped to CPU
        and self.device is rewritten so the run still produces correct output
        — slowly, and visibly, via the device pill in the run summary.
        """
        # A deliberate offload for a local-VLM pass also reads as "on the wrong
        # device". The case report is a separate .then()-chained Gradio event,
        # so a second Run Analysis can start while it is still running; pulling
        # the detector back onto the GPU underneath the VLM would cause exactly
        # the OOM this guard exists to clean up after.
        if getattr(self, "_offloaded", False):
            return []

        want = self.device
        drifted = [name for name, dev in (
            ("detector", self._param_device(self.model)),
            ("pose", self._param_device(self._pose_model)),
            ("quality", self._param_device(self._classifier._quality_model)),
            ("fill", self._param_device(self._classifier._fill_model)),
        ) if dev is not None and dev != want]
        if not drifted:
            return []

        where = f" ({context})" if context else ""
        print(f"[WARN] models not on {want}{where}: {', '.join(drifted)} — moving back")
        try:
            self._move_detector(want)
            if self._pose_model is not None:
                self._pose_model.to(want)
            if self._classifier._quality_model is not None:
                self._classifier._quality_model.to(want)
            if self._classifier._fill_model is not None:
                self._classifier._fill_model.to(want)
        except Exception as e:
            print(f"[ERROR] could not put the model stack back on {want}: {e}")
            print("[ERROR] falling back to CPU for the rest of this process — "
                  "runs will be much slower. Restart to get the GPU back.")
            self.device = "cpu"
            self._classifier.device = "cpu"
            try:
                self._move_detector("cpu")
                if self._pose_model is not None:
                    self._pose_model.to("cpu")
                if self._classifier._quality_model is not None:
                    self._classifier._quality_model.to("cpu")
                if self._classifier._fill_model is not None:
                    self._classifier._fill_model.to("cpu")
            except Exception as e2:
                print(f"[ERROR] CPU fallback also failed: {e2}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return drifted

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
                      rule_thresholds=None,
                      enable_pose: bool = True,
                      progress=None):
        # A FRESH progress tracker per run, never a default argument.
        #
        # `progress=gr.Progress()` in this signature was evaluated ONCE at
        # import, so a single Progress object served every run for the lifetime
        # of the process — and gradio.helpers.Progress keeps mutable state on
        # the instance (`self.iterables`) that only unwinds when a tracked
        # iterator raises StopIteration. The frame loop below breaks out early
        # on the last decodable frame, so every run leaked one entry, and
        # Progress reports the WHOLE list on every step: run 2 drew two
        # progress bars, run 3 drew three, each frozen at the frame its run
        # gave up on. Gradio only injects a bound tracker for a parameter it
        # sees on the EVENT function, and this is not one, so nothing was ever
        # replacing it. Instantiating here costs nothing and cannot accumulate.
        progress = progress if progress is not None else gr.Progress()
        self._reset()
        self._camera_placement = camera_placement
        self._classifier.set_quality_threshold(QUALITY_THRESHOLD)
        t_start = time.perf_counter()
        zones = zones or []
        if analytics_out_dir is None:
            analytics_out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        os.makedirs(analytics_out_dir, exist_ok=True)

        # A previous run's local-VLM offload can have left a model on the
        # wrong device; the cached-path early-return in load_quality/load_fill
        # below will not fix that, so check first. See _ensure_on_device().
        self._ensure_on_device(context="start of run")

        # Load classifiers from fixed weight paths
        self._classifier.load_quality(QUALITY_WEIGHT_PATH)
        self._classifier.load_fill(FILL_WEIGHT_PATH)

        cap, w, h, fps, total_frames = open_video(source_path)

        # SAM-based one-shot scene layout detection is disabled — the user
        # draws zones explicitly in the Zone Editor. `_scene_elements` is kept
        # (always empty) because renderer.render_bev() still reads it; its only
        # consumer in this pipeline, the 2D BEV fixture rollup, is gone.
        _ok, _first_frame = cap.read()
        self._scene_elements = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind before main loop
        # Tracked output is camera-only.
        writer, avi_path = create_writer(w, h, fps)
        names = self.names
        gdi = self._get_display_id
        links = self._linker.links
        MIN_CART_FRAMES_FOR_POPS = 10

        # Timing accumulators
        _t_yolo = 0.0; _t_cls = 0.0; _t_draw = 0.0; _t_json = 0.0; _t_other = 0.0
        _t_pose = 0.0
        frame_capturer = FrameCapturer()

        # Eagerly load so the first frame doesn't stall — but only when pose
        # is actually on. The cold load is ~73s (see the timing table above),
        # and charging that to a run that never reads a keypoint is the whole
        # thing the toggle exists to avoid.
        if enable_pose:
            self._ensure_pose_model()

        frame_idx = 0
        # progress(...) rather than progress.tqdm(...) on purpose. tqdm() APPENDS
        # a tracked iterable to the Progress instance and pops it only when the
        # iterator raises StopIteration — which `break` below never does, since
        # CAP_PROP_FRAME_COUNT routinely over-reports and the decoder runs dry
        # first. __call__ builds `self.iterables + [one]` without appending, so
        # no amount of breaking can leave anything behind.
        # Report at most PROGRESS_MAX_UPDATES times, first and last always.
        # Every frame below is still fully processed and logged — this throttles
        # only the SSE notification. Gradio re-renders a status tracker for each
        # of the run event's 22 output components on every progress message, so
        # per-frame reporting was ~9,200 client-side component updates per clip
        # and drove Svelte into `effect_update_depth_exceeded` after the run.
        _prog_every = max(1, total_frames // max(1, PROGRESS_MAX_UPDATES))
        _prog_last_t = 0.0
        for _frame_no in range(total_frames):
            ok, im0 = cap.read()
            if not ok:
                break
            _is_edge = (_frame_no == 0 or _frame_no == total_frames - 1)
            _now = time.perf_counter()
            if _is_edge or (_frame_no % _prog_every == 0
                            and _now - _prog_last_t >= PROGRESS_MIN_INTERVAL_S):
                _prog_last_t = _now
                # Tuple form keeps the existing "N/N steps" readout; a bare float
                # would switch the bar to a bare percentage.
                progress((_frame_no + 1, total_frames), desc="Processing frames")
            frame_idx += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # --- YOLO + BoTSORT ---
            _t0 = time.perf_counter()
            with torch.no_grad():
                results = self.model.track(im0, persist=True, tracker=self.tracker_config,
                                           imgsz=YOLO_IMGSZ, verbose=False)
            _t_yolo += time.perf_counter() - _t0

            # --- Pose estimation (optional, toggled in the UI) ---
            # Run once per frame on the full image; results are matched to
            # tracked persons by bbox IoU below.
            #
            # When off, these stay at their initial values and the match/draw
            # block further down is skipped by its existing
            # `pose_kps_arr is not None` guard — pose feeds ONLY the skeleton
            # overlay, so nothing in POPS, linking, the JSON or analytics
            # changes. Every frame is still fully processed either way; this is
            # a per-run choice, not per-frame sampling.
            pose_boxes = []
            pose_kps_arr = None
            if enable_pose:
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
                            self._obj_bbox_history, self._obj_frames)

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
                    self._obj_frames[raw].append(frame_idx)
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
                    self._obj_positions[raw], camera_placement,
                    self._obj_timestamps[raw], DIRECTION_WINDOW_S)
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
                        # When the peak happened — lets the POPS table row seek
                        # the tracked video straight to the moment.
                        "timestamp": round(timestamp, 2), "frame": frame_idx,
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
                # The slim 3D frame (per-person pose keypoints + bbox) used to
                # be accumulated here for the 3D View tab. That tab is gone;
                # the full per-frame record above is unchanged and still lands
                # in the tracking JSON.
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

                    # [NEW] Abandoned cart override (grab-and-run), gated on a
                    # SUSTAINED RUN of loaded observations. See
                    # peak_sustained_fill() for why the previous
                    # first-half/second-half proportion test could not fire on
                    # a real history: a confirmed pushout on the 2026-08-13
                    # HANNAFORD clip reads partial(9) -> empty(24) ->
                    # partial(12), and neither half qualified, so the finalised
                    # score contradicted the live one (orig=75 recomp=60).
                    if best_fill == "empty" and abandoned:
                        fills = [f for f, b, fc, bc in history]
                        print(f"[DEBUG] Cart {cd}: history order = {fills}")
                        sustained = peak_sustained_fill(fills, GRABRUN_MIN_RUN_OBS)
                        if sustained:
                            best_fill = sustained
                            # best_bag is deliberately LEFT ALONE. Every "empty"
                            # observation contributes bag_conf 1.0 to
                            # not_applicable, so best_bag is not_applicable here
                            # and the partial/full constraint below re-votes it
                            # across the whole history. Voting the bag inside the
                            # run instead lands on "bagged" whenever that run's
                            # frames are mixed (that clip's early run: 4 unbagged
                            # / 5 bagged, bagged winning on confidence 4.22 vs
                            # 3.01) — and partial+bagged caps at 55, under the 71
                            # PUSHOUT threshold. Across the whole history the
                            # same vote gives unbagged 11.69 vs bagged 4.97,
                            # which is the correct read. See
                            # tests/test_grabrun_override.py.
                            print(f"[GRAB-RUN] Cart {cd}: empty vote overridden to "
                                  f"'{sustained}' | sustained run >= "
                                  f"{GRABRUN_MIN_RUN_OBS} observations")

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
                # POPS → Events (POPS has the reconciled score).
                #
                # EVERY row for this cart, not just the last one. Rewriting
                # only `_last_event[cd]` left a cart's earlier rows carrying
                # the un-reconciled score, so one incident showed up twice with
                # two different numbers and no way to tell which was current.
                for row in self._event_log:
                    if row["cart_id"] != cd:
                        continue
                    row["fill"] = snap["fill"]
                    row["bag"] = snap["bag"]
                    row["pops_score"] = snap["score"]
                    row["event"] = snap["event"]

        # The rewrite above assigns whatever classify_event() returns for the
        # reconciled score, and that is not necessarily an EVENT: a row logged
        # live as MEDIUM PRIORITY (33) can reconcile to LOW PRIORITY (16) or
        # MONITORING. Those names are not in LOGGABLE_EVENTS and nothing was
        # dropping them, so the Events tab rendered non-events as events under a
        # header reading "3 event(s) logged - no high-risk events", and they
        # shipped in full_json["events"] to every downstream consumer.
        #
        # Rewriting all of a cart's rows also makes them identical, so collapse
        # to the earliest frame — the log records when a cart FIRST reached an
        # event, and `already_logged` in the frame loop enforces exactly that.
        #
        # A dropped row can orphan a FrameCapturer capture, which was keyed to
        # the live event name during the loop and cannot be re-keyed from here.
        # An evidence frame with no matching row is a far smaller lie than a
        # "LOW PRIORITY" row presented as an incident.
        self._event_log, _dropped = prune_event_log(self._event_log)
        if _dropped:
            print(f"[EVENTS] dropped {_dropped} row(s) that reconciliation "
                  f"demoted out of LOGGABLE_EVENTS or duplicated")

        t_frames = time.perf_counter()
        print(f"[PERF] Breakdown over {frame_idx} frames:")
        print(f"  YOLO+track : {_t_yolo:.2f}s ({_t_yolo/(t_frames-t_start)*100:.0f}%)")
        print(f"  Pose       : {_t_pose:.2f}s ({_t_pose/(t_frames-t_start)*100:.0f}%)")
        print(f"  Classify   : {_t_cls:.2f}s ({_t_cls/(t_frames-t_start)*100:.0f}%)")
        print(f"  Motion+POPS: {_t_other:.2f}s ({_t_other/(t_frames-t_start)*100:.0f}%)")
        print(f"  Drawing    : {_t_draw:.2f}s ({_t_draw/(t_frames-t_start)*100:.0f}%)")
        print(f"  Frame JSON : {_t_json:.2f}s ({_t_json/(t_frames-t_start)*100:.0f}%)")

        # The frame loop is only part of the wait. Everything from here to the
        # return happens with the bar already at 100%, so without these the UI
        # reads as hung for the whole tail — which on a long clip is what the
        # "stuck at 100%" report was. A bare float switches the readout from
        # "N/N steps" to a plain percentage, which is what we want now that
        # there are no frames left to count.
        progress(1.0, desc="Encoding video")
        out_path = reencode_to_mp4(avi_path)
        t_encode = time.perf_counter()
        video_duration = total_frames / fps if fps > 0 else 0
        print(f"[PERF] Frame processing: {t_frames - t_start:.1f}s | "
              f"Video encoding: {t_encode - t_frames:.1f}s | "
              f"Total: {t_encode - t_start:.1f}s | "
              f"Video duration: {video_duration:.1f}s | "
              f"Speed: {video_duration / (t_encode - t_start):.2f}x realtime")

        # --- Build JSON ---
        progress(1.0, desc="Building tracking JSON")
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
        progress(1.0, desc="Computing analytics")
        analytics_result: AnalyticsResult = run_analytics(
            bundle, list(zones), out_dir=analytics_out_dir,
            heatmap_background=rep_frame,
            camera_placement=camera_placement,
            rule_thresholds=rule_thresholds,
        )
        # Report carts the INBOUND kill switch scored out, on the same channel
        # as the rule-coverage notes. Appended HERE rather than inside
        # evaluate_rules() because this is POPS reasoning, and rules.py is
        # deliberately independent of POPS scoring (see its module docstring) -
        # but it belongs in the same notice box, because from the reader's side
        # it answers the identical question: is this quiet run actually quiet?
        _inbound_note = inbound_suppression_note(
            self._peak_pops_snapshot, camera_placement)
        if _inbound_note:
            analytics_result.rule_diagnostics.append(_inbound_note)

        # --- Build HTML ---
        video_html  = ui_builder.build_video_info(source_path, w, h, fps, total_frames, frame_idx)
        det_html    = ui_builder.build_detection_info(
            len(self._all_people_seen), len(self._all_carts_seen),
            self._linker.total_links)
        config_html = ui_builder.build_config_info(
            LINK_CONFIRM_FRAMES, LINK_GRACE_FRAMES, camera_placement,
            self._classifier.quality_pt, self._classifier.fill_pt, QUALITY_THRESHOLD,
            enable_pose=enable_pose)
        legend_html = ui_builder.build_legend()
        # Rule findings go into the POPS table too: it is the only per-cart
        # surface, so it is the one place a cart can be shown carrying several
        # operational categories at once (unattended AND blocking the exit).
        pops_html   = ui_builder.build_pops_summary(
            self._max_pops_per_cart, self._peak_pops_snapshot,
            analytics_result.rule_findings)
        events_html = ui_builder.build_events_timeline(self._event_log)
        # The 3D and 2D BEV documents were built here. Both embedded every
        # frame; the zone-fixture rollup and compute_impressions() existed only
        # to feed the 2D one, so they went with it.

        # --- Top-of-page alert banner (high-priority events + severe spikes) ---
        alert_banner_html = ui_builder.build_alert_banner(
            self._event_log, analytics_result.queue_spikes,
            analytics_result.spike_events,
            analytics_result.rule_findings,
        )
        ops_alerts_html = ui_builder.build_operational_alerts(
            analytics_result.rule_findings,
            analytics_result.rules_unavailable_reason,
            analytics_result.rule_diagnostics,
        )

        # Operational findings go in the JSON under their OWN key, never spliced
        # into "events". The event log drives FrameCapturer, which JPEG-encodes
        # a full frame for every new event name and hands those frames to the
        # VLM case report — so anything added to "events" leaves the device by
        # default. Keeping rules separate is what makes the Phase-2
        # child-in-cart work safe to add here later.
        full_json["rule_findings"] = [
            highlights.finding_to_dict(f) for f in analytics_result.rule_findings
        ]
        # Record the thresholds ACTUALLY used, not the config defaults — with
        # the sidebar sliders those can differ, and a report that names the
        # wrong fuse length is worse than one that names none. Key names are
        # the original JSON schema's, not resolve_thresholds()' internal ones.
        _th_used = rule_engine.resolve_thresholds(rule_thresholds)
        full_json["rule_engine"] = {
            "unavailable_reason": analytics_result.rules_unavailable_reason,
            # Same list the UI panel and the case report render, so the three
            # artifacts cannot disagree about what was skipped or degraded.
            "diagnostics": highlights.ops_diagnostics(
                analytics_result.rule_diagnostics),
            "timestamps_synthesized": bundle.timestamps_synthesized,
            "thresholds_s": {
                "blocked_door": _th_used["blocked_door_s"],
                "static_cart": _th_used["static_cart_s"],
                "abandoned_cart": _th_used["abandoned_cart_s"],
            },
        }
        # Curated view of the same rule/congestion data, selected by the exact
        # logic engine.highlights shares with the case report's Operations
        # Highlights section — so a human reading the HTML/PDF and a machine
        # reading this JSON never see different "top" findings for one clip.
        _ops_state, _ops_shown, _ops_remainder = highlights.ops_findings_state(
            analytics_result.rules_unavailable_reason, analytics_result.rule_findings)
        _flag_idx = ui_builder.cart_flag_index(analytics_result.rule_findings)
        _severe_spikes, _top_dwell = highlights.select_congestion(
            analytics_result.queue_spikes, analytics_result.dwell_summary)
        full_json["operational_highlights"] = {
            "status": _ops_state,          # "unavailable" | "clean" | "findings"
            "unavailable_reason": analytics_result.rules_unavailable_reason,
            "insight_text": (analytics_result.insight_text or "").strip(),
            "top_findings": [highlights.finding_to_dict(f) for f in _ops_shown],
            "additional_findings_count": _ops_remainder,
            "n_carts_flagged": len(_flag_idx),
            "category_counts": ui_builder.category_counts_from_index(_flag_idx),
            "congestion": {
                "severe_spikes": [highlights.spike_to_dict(s) for s in _severe_spikes],
                "top_dwell_zones": list(_top_dwell),
            },
        }
        # Full congestion lists behind the curated view above — computed by
        # run_analytics() every run but, until now, never reaching the JSON at
        # all. Same "full list + curated highlights" shape as
        # rule_findings/operational_highlights.
        full_json["queue_spikes"] = [
            highlights.spike_to_dict(s) for s in analytics_result.queue_spikes
        ]
        full_json["dwell_summary"] = list(analytics_result.dwell_summary)
        # Re-emit now that the rule keys exist (the first write happened before
        # analytics ran, since analytics consumes the bundle built from it).
        with open(json_path, 'w') as f_json:
            json.dump(full_json, f_json, indent=2)
        json_str = json.dumps(full_json, indent=2)

        # --- Analytics HTML ---
        progress(1.0, desc="Rendering panels")
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
            # placeholder. The caller returns the rest of the pipeline output
            # immediately and calls finalize_case_report() from a SEPARATE
            # Gradio event — not a later yield of the same one, which would
            # keep a pending overlay over the whole dashboard until the VLM
            # finished (see run_analysis in app_poc_v2.py).
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
                "Pipeline finished - the VLM is now analysing captured frames. "
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

        # --- Run summary -------------------------------------------------
        # Every number here was already being computed and then thrown away
        # into a console print.
        run_summary_html = ui_builder.build_run_summary(
            frames=frame_idx,
            wall_s=t_encode - t_start,
            encode_s=t_encode - t_frames,
            device=self.device,
            video_duration_s=video_duration,
            n_people=len(self._all_people_seen),
            n_carts=len(self._all_carts_seen),
            n_links=self._linker.total_links,
            timings=[("YOLO", _t_yolo), ("pose", _t_pose), ("classify", _t_cls),
                     ("POPS", _t_other), ("draw", _t_draw)],
        )
        tab_counts_html = ui_builder.build_tab_counts(
            self._tab_counts(analytics_result))

        # Deliberately NOT logging a "browser payload" size here: json_str is
        # the FULL document and only a capped preview of it reaches the browser
        # (the caller substitutes it), so any total computed at this point would
        # overstate the real payload by orders of magnitude. The measurement
        # lives in app_poc_v2.run_analysis, at the seam where it is final.
        print(f"[PERF] engine outputs: json {len(json_str) / 1048576:.2f} MB "
              f"(full document, written to {os.path.basename(json_path)}), "
              f"events {len(events_html) / 1024:.0f} KB, "
              f"pops {len(pops_html) / 1024:.0f} KB")

        # Note: emit BOTH the composite ndarray (for gr.Image) and the path
        # (for gr.File). The caller splits them into the two output slots.
        return (out_path, json_path, json_str,
                video_html, det_html, config_html, legend_html, pops_html, events_html,
                case_report_html, case_report_file,
                analytics_summary_html, spikes_html, dwell_html, journey_html,
                heatmap_img_bgr, heatmap_path,
                alert_banner_html, ops_alerts_html,
                run_summary_html, tab_counts_html)

    # ------------------------------------------------------------------
    # Case-report generation (extracted so it can run synchronously inside
    # process_video, OR deferred and run via finalize_case_report() while
    # the rest of the dashboard is already visible).
    # ------------------------------------------------------------------
    def _run_case_report(self, *, captures, full_json, event_log,
                         peak_snapshots, vlm_backend, vlm_api_key,
                         analytics_result) -> tuple[str, str | None]:
        is_local_vlm = "Claude" not in vlm_backend
        if is_local_vlm:
            self._release_detection_gpu_memory()
        vlm = None
        try:
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
        finally:
            # unload BEFORE restore, and unconditionally. If analyze_incident
            # raises (a VLM OOM, a load failure, a generation error), the old
            # in-try unload was skipped, so the VLM's weights were still
            # resident when _restore_detection_gpu_memory() asked for the
            # detection stack's ~5-6 GB back — the restore then OOMs partway
            # through and leaves the detector half on CUDA, half on CPU.
            if vlm is not None:
                vlm.unload_model()
            if is_local_vlm:
                self._restore_detection_gpu_memory()

        gradio_html, standalone_html = build_case_report_html(
            report_data, captures, full_json, event_log,
            peak_snapshots, full_json["video_info"],
            analytics_result=analytics_result,
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
