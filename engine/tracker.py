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
import threading
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
    LINK_CONFIRM_FRAMES, LINK_GRACE_FRAMES, ABANDON_FRAMES, STALE_CART_FRAMES,
    QUALITY_WEIGHT_PATH, FILL_WEIGHT_PATH, QUALITY_THRESHOLD,
    WALKAWAY_GAP_FRAC, WALKAWAY_MIN_GAP_PX, GRABRUN_MIN_RUN_OBS, DIRECTION_WINDOW_S,
    DIRECTION_LATCH_FRAMES,
    LINK_DRIFT_IOU,
    POSE_MODEL_PATH, POSE_IMGSZ, POSE_CONF_THRESHOLD,
    POSE_KP_CONF_THRESHOLD, POSE_MATCH_IOU_MIN,
    RULE_BLOCKED_DOOR_S, RULE_STATIC_CART_S, RULE_ABANDONED_CART_S,
    PENDING_CASE_REPORTS_MAX,
)
from . import ultralytics_compat as _ultralytics_compat
from .cancellation import CancelToken, RunCancelled
from .classifier import CartClassifier
from .linker import PersonCartLinker
from .motion import compute_motion, compute_direction_label, DirectionLatch
from .scoring import (
    compute_pops, classify_event, peak_sustained_fill, merchandise_removed,
    vote_classification,
    prune_event_log,
    inbound_suppression_note, unassessed_cart_note, vote_bag_for_loaded_cart,
    sync_events_with_snapshots,
    LOGGABLE_EVENTS, HIGH_EVENTS, MEDIUM_EVENTS, MEDIUM_SCORE,
)
from .renderer import (
    draw_bbox, draw_centroid_trail, draw_classification_overlay,
    draw_person_overlay, draw_link_lines, draw_hud, outlined_text,
    draw_pose_skeleton,
)
from .video_io import (open_video, create_writer, reencode_to_mp4,
                       discard_run_output)
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
if _ultralytics_compat.install():
    print(f"[INFO] tracker compat: {_ultralytics_compat.describe()} — "
          f"fusing detection score into the first association only")
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
        #: True once a deliberate CPU park for a local-VLM pass has been
        #: ATTEMPTED, until the matching restore clears it. "Attempted", not
        #: "completed": _release_detection_gpu_memory() sets it before its first
        #: .to("cpu") so a release that raises partway is still owed a restore.
        #: _ensure_on_device() must not "repair" either state.
        self._offloaded = False
        #: Serialises the two halves of the Gradio event chain against each
        #: other: process_video() and finalize_case_report() are separate
        #: dependencies over one engine and one GPU, and the case report parks
        #: the detection stack on the CPU for its whole duration. Held by those
        #: two methods ONLY — never by _release/_restore/_run_case_report, so
        #: that "does this nest?" stays answerable by reading two call sites.
        #: A plain Lock, not an RLock, because it provably does not nest: the
        #: synchronous case-report branch in _process_video() calls
        #: _run_case_report() directly rather than finalize_case_report().
        self._gpu_lock = threading.Lock()
        #: The Cancel button's signal. Sequence-scoped so cancelling a run
        #: cannot discard an EARLIER run's still-pending case report — see
        #: engine/cancellation.py.
        self._cancel = CancelToken()
        #: (cap, writer, avi_path) while a frame loop is running, None
        #: otherwise — the registration `_release_run_handles()` reads.
        self._run_handles = None
        #: Payloads stashed by _process_video(defer_case_report=True), oldest
        #: first, one per run still awaiting its finalize_case_report() call.
        #:
        #: A LIST, and declared HERE rather than in _reset(), both for the same
        #: reason: the finalize event for run N is a separate Gradio dependency
        #: submitted only after run N's process_video returns, so run N+1 can
        #: legitimately start — and reset — before run N's report has been
        #: consumed. As a single slot cleared by _reset() this silently served
        #: run 2's payload to run 1's finalize, or served it None and rendered
        #: "No case report for this clip" for a run that had captures. Popping
        #: FIFO gives the Nth finalize the Nth payload. See
        #: docs/device_drift_fix_plan.md §9.4.2.
        self._pending_case_reports: list[dict] = []

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
        # DISPLAY ids, not raw tracker ids. Raw ids inflate on every tracker ID
        # switch and every re-ID, so counting them reported 18 carts for a clip
        # that had 9 — while the per-frame records, which are keyed by display id,
        # showed 9. One clip, two answers, and the larger one on the summary line.
        self._all_people_seen   = set()
        self._all_carts_seen    = set()

        self._cart_cls_cache    = {}
        self._pops_cache        = {}
        self._event_log         = []
        self._max_pops_per_cart = {}
        self._peak_pops_snapshot= {}
        self._cart_cls_history  = defaultdict(list)  # cd -> [(fill, bag), ...]
        self._motion_cache      = {}  # raw_id -> (speed, direction, status, accel, dir_label)
        # cart display_id -> last sustained OUTBOUND heading, held through the
        # UNKNOWN frames a parked cart produces. See motion.DirectionLatch.
        self._dir_latch         = DirectionLatch(DIRECTION_LATCH_FRAMES)
        self._walkaway_frames   = {}  # cd -> consecutive frames person is far from cart
        # cd -> raw id of the last person the linker gave this cart. Abandonment
        # used to be readable ONLY while the link was live, so the moment a cart
        # lost its owner it became permanently un-abandonable — the one state in
        # which abandonment is the interesting question. classify_event()'s
        # invariant is preserved by this being a memory of a REAL link: a cart
        # that never had an owner has no entry and still cannot be abandoned.
        self._last_owner_raw    = {}
        # cd -> the last frame this cart was seen, so the owner memory above can
        # be aged out on the same rule the linker purges links with
        # (STALE_CART_FRAMES). Without it a display ID reused much later would
        # inherit a stranger's departure as its own abandonment evidence.
        self._last_owner_frame  = {}
        # Rule-engine fact timeline: cart display_id -> [CartFactSample, ...].
        # Recorded before the MIN_CART_FRAMES_FOR_POPS guard so brief carts
        # still have facts even when POPS declines to score them.
        self._cart_facts        = defaultdict(list)
        self._cls_last_frame    = {}  # cd -> frame_idx of the last real classification
        # Classification coverage. Counted rather than derived at the end because
        # the per-frame records are sampled and the caches only hold the LAST
        # reading, so neither can say how much of the run was unreadable.
        self._cart_frames_total = 0
        self._cart_frames_unassessed = 0
        self._scene_elements    = []

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
                self._all_people_seen.add(display_id)
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
                # `classification` above is this frame's OBSERVATION; `pops`
                # below is scored from the VOTE over the cart's whole history
                # (see the frame loop), so the two can legitimately disagree on
                # a noisy frame. That is the point — a tier must not turn on one
                # observation — but a reader comparing them needs to know which
                # is which.
                obj["pops"] = {
                    "score": pi.get("score", 0), "event": pi.get("event", "CLEAR"),
                    "abandoned": bool(pi.get("abandoned", False)),
                    "merch_removed": bool(pi.get("merch_removed", False)),
                }
                obj["linking"] = {"is_linked": False, "linked_person_id": None, "link_confidence": 0.0}
                carts[key] = obj
                frame_carts[raw_id] = (cx, cy)
                self._all_carts_seen.add(display_id)

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
        # Set BEFORE the first move, not after the last one. The flag means "a
        # release was attempted", which is what the caller needs to decide
        # whether a restore is owed: _run_case_report() runs this inside its
        # try/finally, so a raise anywhere below must still leave the finally
        # able to tell "half-moved, put it back" from "never moved, leave it
        # alone". Setting it at the end could only ever describe a release that
        # completed, and the failure it left behind — a stranded flag with no
        # matching restore — is Phase 8.1's permanent-poison shape:
        # _ensure_on_device() early-returns on this flag forever.
        self._offloaded = True
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
            # Buffers as well as parameters. Module._apply() moves children,
            # then its own parameters, then its own buffers, so a move that
            # failed at the tail can leave a BatchNorm running_mean behind with
            # every parameter already correct — and a buffer on the wrong
            # device raises the same conv2d type error a parameter does.
            # Confirmed harmless on the real stack: every buffer tracks .to()
            # (detector 231, pose 573, quality 138, fill 138), so this never
            # reports phantom drift.
            devs |= {b.device.type for b in inner.buffers()}
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
        the parked one is left.

        The re-attach is in a `finally` because this `.to()` is itself allowed
        to fail: _ensure_on_device() catches that and retries with "cpu", and
        that retry must still find a predictor. Model._apply() nulls
        `predictor` BEFORE it can raise, so without the finally the reference
        is dropped on exactly the path that needs it most. Re-attaching a
        predictor whose module is half-moved is safe — it wraps that same
        module, and the caller's CPU fallback makes it consistent again."""
        held = (getattr(self.model, "predictor", None)
                or getattr(self, "_held_predictor", None))
        try:
            self.model.to(dev)
        finally:
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

        # Past the offload check, a parked predictor is a leftover, not a
        # deliberate state: _release_detection_gpu_memory() stashes it and
        # .to("cpu") nulls it, so a restore that raised AFTER the detector's
        # own move already succeeded leaves the drift check reading clean
        # while Model.track() sees no predictor and registers a SECOND
        # tracking callback — the silent detection-degradation regression
        # documented in _release_detection_gpu_memory(). Nothing else is left
        # to put it back, so do it here regardless of drift.
        if getattr(self, "_held_predictor", None) is not None:
            if getattr(self.model, "predictor", None) is None:
                self.model.predictor = self._held_predictor
            self._held_predictor = None

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

    def _reset_trackers(self):
        """Clear the tracker's state so a run starts from a clean slate.

        `model.track(persist=True)` is right WITHIN a run — it is what keeps
        ids stable frame to frame — but the predictor and the tracker hanging
        off it survive between runs, and nothing in `_reset()` reaches them:
        that method only clears this engine's own bookkeeping. So a second Run
        Analysis in the same process used to inherit run 1's tracked and lost
        tracks, its Kalman filter, its GMC warp estimate and its frame counter,
        from a DIFFERENT video.

        Measured on the golden clip, same engine, no case report in between:
        run 1 reproduced the baseline exactly, run 2 lost cart 1 on frame 116,
        and run 3 repeated run 2. With this reset all three match.

        `BYTETracker.reset()` also calls `reset_id()`, which zeroes the
        process-global `BaseTrack._count`. That is safe here because display
        ids are remapped from raw ids per run — `_reset()` clears
        `_display_map`/`_next_display` — so a fresh `track_id=1` cannot collide
        with a previous run's entry. `TrajectoryCache` is the other state that
        outlives a run, and it is keyed per video (`make_video_key()`), never by
        a raw track id, so a restarted id counter cannot reach another video's
        entry either. This touches the tracker only, never `predictor` itself,
        so the parked-predictor contract in `_release_detection_gpu_memory()` is
        unaffected.

        The parked predictor is included deliberately. During a local-VLM pass
        `.to("cpu")` has nulled `self.model.predictor`, so an overlapping run —
        the second Run Analysis that §7 of docs/device_drift_fix_plan.md leaves
        deferred — would otherwise find nothing to reset and keep run 1's
        tracks. Resetting the parked one is safe: run 1's frame loop is long
        finished by the time the case report starts, and the VLM never touches
        trackers.
        """
        predictor = (getattr(self.model, "predictor", None)
                     or getattr(self, "_held_predictor", None))
        for tracker in (getattr(predictor, "trackers", None) or []):
            try:
                tracker.reset()
            except Exception as e:
                # Stale tracker state is a wrong-output problem, not a
                # crash-the-run problem: an ultralytics without
                # BYTETracker.reset() must degrade, not abort.
                print(f"[WARN] could not reset the tracker between runs: {e}")

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

    @staticmethod
    def _bbox_gap(a, b):
        """Shortest distance between two (x1,y1,x2,y2) boxes; 0.0 if they touch.

        Unlike a centroid distance this does not grow with the size of either
        box, so a person standing against a cart measures zero however large the
        cart is drawn. See WALKAWAY_GAP_FRAC in config.py.
        """
        dx = max(a[0] - b[2], b[0] - a[2], 0.0)
        dy = max(a[1] - b[3], b[1] - a[3], 0.0)
        return (dx * dx + dy * dy) ** 0.5

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

    def _release_run_handles(self):
        """Close whatever video handles the last run left open, and bin its
        partial output.

        Called from `process_video`'s finally, so it runs on every exit path
        including the ones nobody planned: a CUDA error mid-loop, a cancel, a
        bug in the POPS reconciliation. `_process_video` clears the registration
        once it has released the handles itself, so the normal path reaches here
        with nothing to do.

        Never raises. It runs while an exception is already propagating, and
        replacing a real error with an OSError from a tidy-up would hide the
        thing worth reading.
        """
        handles = getattr(self, "_run_handles", None)
        if handles is None:
            return
        self._run_handles = None
        cap, writer, avi_path = handles
        for name, handle in (("capture", cap), ("writer", writer)):
            try:
                handle.release()
            except Exception as e:
                print(f"[WARN] releasing the video {name} failed: {e}")
        # After both releases, never before: the writer holds a lock on the file
        # under Windows.
        discard_run_output(avi_path)

    def request_cancel(self) -> int | None:
        """Ask the newest run (or its case report) to stop. Returns the
        sequence it targeted, or None when nothing has run yet.

        Called from the Cancel button's own Gradio event, on another thread,
        while the run being cancelled holds `_gpu_lock` and is inside its frame
        loop. So it must not take that lock and must not block: all it does is
        set a flag the loop already checks every frame.

        Gradio's `cancels=` cannot do this job — it explicitly lets a running
        function finish — so the button needs both: `cancels=` to drop queued
        jobs and clear the client spinner, and this to stop the work.
        """
        seq = self._cancel.request()
        if seq is None:
            print("[CANCEL] nothing to cancel — no run has started yet")
        else:
            print(f"[CANCEL] stopping run {seq} at the next checkpoint")
        return seq

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
    def process_video(self, *args, progress=None, **kwargs):
        """Serialised public entry point — see `_process_video` for the pipeline.

        A thin wrapper so the whole pipeline runs under `_gpu_lock` without
        re-indenting it. `process_video` and `finalize_case_report` are separate
        Gradio dependencies over one engine and one GPU (app_poc_v2.py:1508 and
        :1567), and a local-VLM case report parks the detection stack on the CPU
        for its whole duration. A second Run click during that window used to
        start this pipeline against CPU-resident weights while
        `self._classifier.device` still said cuda, which is:

            RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                          (torch.FloatTensor) should be the same

        raised from the first classify() call. The start-of-run device guard
        cannot fix it — it early-returns on `_offloaded` precisely so it does
        not drag the detector back onto the card underneath a live VLM — so the
        only fix is to not run the two at once. See
        docs/device_drift_fix_plan.md §9.

        The wait is announced BEFORE the acquire, and only when the lock is
        actually held: after it, the message describes a wait that is already
        over. `locked()` is advisory here — a false negative costs a missing
        status line, never correctness, because the acquire below is what
        serialises.
        """
        # BEFORE the wait, not after: a run blocked behind a case report has
        # begun as far as the user is concerned, and pressing Cancel during that
        # wait has to reach it. Claiming the sequence here is what lets it —
        # see CancelToken.begin().
        run_seq = self._cancel.begin()
        if self._gpu_lock.locked():
            progress = progress if progress is not None else gr.Progress()
            progress(0, desc="Waiting for the previous case report to finish…")
        with self._gpu_lock:
            # The wait is the most likely place for a cancel to land: it is the
            # only part of a run that can take minutes before a single frame has
            # been read.
            self._cancel.raise_if_cancelled(run_seq, "before the run started")
            try:
                return self._process_video(*args, run_seq=run_seq,
                                           progress=progress, **kwargs)
            finally:
                # The only place that closes the frame loop's video handles on
                # an unplanned exit. Here rather than around the loop itself
                # because that would mean re-indenting ~600 lines of pipeline
                # into a `with`, and this scope is the one that already owns the
                # run: it takes the lock, so it can also guarantee the run gives
                # its file handles back before the next one takes it.
                self._release_run_handles()

    def _process_video(self, source_path,
                      camera_placement="Outside (facing entrance)",
                      vlm_backend="Claude (API)", vlm_api_key="",
                      zones=None,
                      analytics_out_dir=None,
                      defer_case_report: bool = False,
                      rule_thresholds=None,
                      enable_pose: bool = True,
                      progress=None,
                      run_seq: int | None = None):
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

        # Holding _gpu_lock (see process_video), no case report can be in
        # flight, so _offloaded True here cannot mean "deliberately parked right
        # now" — it means a restore was skipped entirely and the flag was
        # stranded. Left set, it turns the guard on the next line into a
        # permanent no-op and makes the device-mismatch error permanent for the
        # life of the process. Clearing it lets the guard do its job, and the
        # message names the cause instead of leaving a silent CPU-speed run.
        if self._offloaded:
            print("[WARN] the detection stack was still parked on the CPU at "
                  "the start of a run — a case report did not restore it. "
                  "Repairing; the run continues.")
            self._offloaded = False

        # A previous run's local-VLM offload can have left a model on the
        # wrong device; the cached-path early-return in load_quality/load_fill
        # below will not fix that, so check first. See _ensure_on_device().
        self._ensure_on_device(context="start of run")
        # After the device check, not before: _ensure_on_device() can put a
        # parked predictor back, and the trackers to reset are the ones on
        # whatever predictor is live at the start of this run.
        self._reset_trackers()
        # Leftover duplicate tracking callbacks from an earlier run cannot be
        # left in place: each one re-runs the tracker on every frame and
        # silently starves detection. See dedupe_tracking_callbacks(). Checked
        # again after this run's first .track() call, which is where a lost
        # predictor makes ultralytics append a fresh one.
        _stale_cbs = _ultralytics_compat.dedupe_tracking_callbacks(self.model)
        if _stale_cbs:
            print(f"[WARN] dropped {_stale_cbs} duplicate tracking callback(s) "
                  f"left over from an earlier run")

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
        # Registered so the wrapper's finally can close both on ANY exit path.
        # Before this, `cap` and `writer` were released at exactly two points —
        # the end of the loop and the cancel branch — so every other way out of
        # this method leaked both: the device-mismatch RuntimeError that started
        # this whole investigation left an open VideoCapture and a VideoWriter
        # holding a lock on its AVI, and on Windows that lock is what makes the
        # next run's writer fail with "being used by another process".
        #
        # AFTER create_writer, obviously: registering above it referenced
        # `writer` before it existed, which broke every run with an
        # UnboundLocalError. A tuple rather than three attributes — they are
        # acquired and released together, and one name is one thing to reset.
        self._run_handles = (cap, writer, avi_path)
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
        _cancelled = False
        for _frame_no in range(total_frames):
            # Checked EVERY frame, and before the frame is read. This is the
            # Cancel button's only way in: Gradio's own `cancels=` lets a
            # running function finish, so a run that does not look at the flag
            # cannot be stopped at all. The check is an Event.is_set() plus an
            # int compare — cheaper than the frame read below it — and nothing
            # about the frame is processed once it fires, so a cancelled run
            # never emits a half-processed frame.
            if self._cancel.is_cancelled(run_seq):
                _cancelled = True
                break
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

            # THE call that can duplicate the tracking callback: Model.track()
            # registers one whenever it finds no predictor, and appends rather
            # than replaces. Every copy calls tracker.update() again on the same
            # frame, which divides track confirmation and track_buffer by the
            # number of copies — the "carts and people stopped being detected"
            # failure. Checked here so a degraded run is impossible rather than
            # merely unlikely: the predictor-parking contract in
            # _release_detection_gpu_memory() has to hold across a Gradio event
            # chain in which a case report and the next run can overlap, and
            # this does not depend on it holding.
            #
            # EVERY frame, not just the first. The previous run's case report is
            # a chained Gradio event that can call
            # _release_detection_gpu_memory() while this loop is running: .to()
            # nulls `predictor` mid-run, the next frame re-registers, and a
            # first-frame-only check has already passed by then. (Such a run has
            # a second problem — its detector is now on the CPU — but that one
            # is loud and self-repairing, and this one is neither.) The check is
            # two list scans of two or three entries; Motion+POPS, which does
            # far more, measures 0.03s across 415 frames.
            _dup_cbs = _ultralytics_compat.dedupe_tracking_callbacks(self.model)
            if _dup_cbs:
                print(f"[WARN] dropped {_dup_cbs} duplicate tracking "
                      f"callback(s) at frame {frame_idx} — the predictor was "
                      f"re-registered; detection would have degraded from "
                      f"here on")

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
                    # The device the pose weights are actually on, not
                    # self.device. Handing predict() an explicit "cuda" while
                    # the module is parked on the CPU for a local-VLM pass makes
                    # ultralytics re-initialise AutoBackend on the card and take
                    # VRAM back from a generating VLM — a path fighting the
                    # offload rather than merely a victim of it.
                    #
                    # Passing the real device, NOT dropping the argument:
                    # predict() with no device reaches select_device(""), which
                    # per its own docstring "defaults to auto-selecting the
                    # first available GPU" (ultralytics 8.4.50,
                    # utils/torch_utils.py), so it would take the VRAM anyway.
                    # self.model.track() a few lines up passes no device and is
                    # still safe, but for a different reason — it reuses the
                    # existing predictor instead of re-selecting. The two calls
                    # are not symmetric.
                    # "mixed" (a restore that OOM'd partway) is not a device
                    # ultralytics can be handed; CPU is the one that cannot
                    # raise, and the start-of-run guard repairs the module.
                    _pose_dev = self._param_device(self._pose_model) or self.device
                    pres = self._pose_model.predict(
                        im0, imgsz=POSE_IMGSZ, conf=POSE_CONF_THRESHOLD,
                        device="cpu" if _pose_dev == "mixed" else _pose_dev,
                        verbose=False,
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

            # A link the linker has just disowned was never real — a person who
            # brushed past a parked cart, not one who took it. `_last_owner_raw`
            # deliberately OUTLIVES a link so a departing owner can still be
            # scored as abandonment, and that is exactly wrong here: it is aged
            # out on STALE_CART_FRAMES since the cart was last SEEN, so for a
            # cart that stays in frame it never expires at all. On the
            # 1764200318790 clip that kept a passer-by on record as the owner of
            # a parked cart for the rest of the run and scored it ABANDONED CART
            # (65) from the frame she left the doorway.
            for cd in self._linker.disowned_carts:
                self._last_owner_raw.pop(cd, None)
                self._last_owner_frame.pop(cd, None)
                self._walkaway_frames.pop(cd, None)

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

            # Hold a cart's last SUSTAINED outbound heading through the UNKNOWN
            # frames it produces once it stops. Applied AFTER the sync above so
            # a heading inherited from the cart's owner latches too, and so a
            # resolved reversal — the owner, or a staff member, wheeling the
            # cart back inside — clears the latch on the frame it happens.
            # Keyed by display id: see DirectionLatch.
            # Resolved once per DISPLAY id, not once per detection: two raw
            # boxes can carry the same display id on one frame (a nested
            # duplicate the deduper let through), and calling resolve() twice
            # would double-count that frame toward the latch run.
            _latched_dirs = {}
            for raw, c, _, _ in frame_detections:
                if names[int(c)] != 'cart':
                    continue
                cd = gdi('cart', raw)
                if cd not in _latched_dirs:
                    _latched_dirs[cd] = self._dir_latch.resolve(
                        cd, self._motion_cache[raw][4])
                old = self._motion_cache[raw]
                self._motion_cache[raw] = (
                    old[0], old[1], old[2], old[3], _latched_dirs[cd])

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
                # Remember who the linker gave this cart, and forget the previous
                # owner's walkaway progress when it changes hands: that counter
                # measures ONE person's distance, and inheriting it would let a
                # cart that just found a new owner be called abandoned on the
                # strength of the old one's departure. Until the `linked` gate
                # came off `person_far` below this was safe by accident.
                if linked and linked_person_raw is not None:
                    if self._last_owner_raw.get(cd) != linked_person_raw:
                        self._walkaway_frames.pop(cd, None)
                    self._last_owner_raw[cd] = linked_person_raw
                if cd in self._last_owner_raw:
                    last_seen = self._last_owner_frame.get(cd, frame_idx)
                    if frame_idx - last_seen > STALE_CART_FRAMES:
                        self._last_owner_raw.pop(cd, None)
                        self._walkaway_frames.pop(cd, None)
                    self._last_owner_frame[cd] = frame_idx

                # Whose departure counts as this cart being abandoned. The live
                # link when there is one, otherwise the last person who held it.
                #
                # Reading only the live link is what closed the whole abandonment
                # route on the 1764099569430 clip: a released cart is exactly the
                # cart whose owner may have walked off, and it was the one state
                # in which the question could not be asked. A cart with no
                # remembered owner still returns None here, so classify_event()'s
                # invariant — abandonment is unavailable to a cart that never had
                # an owner — holds unchanged.
                owner_raw = (linked_person_raw if linked and linked_person_raw is not None
                             else self._last_owner_raw.get(cd))

                # A cart someone is currently engaged with is attended, whoever
                # that is. This is what keeps a genuine handover from reading as
                # abandonment in the frames between one link ending and the next
                # being confirmed.
                attended_now = any(
                    self._bbox_iou(bb, pb) >= LINK_DRIFT_IOU
                    for pb in person_bb.values()
                )
                if attended_now:
                    self._walkaway_frames.pop(cd, None)

                # Classic: person gone from frame for N frames
                person_gone = (not attended_now and owner_raw is not None
                               and self._obj_disappeared.get(owner_raw, 0) > ABANDON_FRAMES)

                cr = self._cart_cls_cache.get(cd, {})
                is_valid = cr.get("is_valid", True)
                fill_lbl = cr.get("fill", "unclassified")
                bag_lbl  = cr.get("bag", "not_applicable")

                # Coverage tally, before the cart-age guard below: a cart the
                # classifier could not read is scored INBOUND_SCORE without being
                # assessed, which looks exactly like a clean result.
                self._cart_frames_total += 1
                if cr.get("quality", "unclassified") in ("unclear", "unclassified"):
                    self._cart_frames_unassessed += 1

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

                # Walkaway: the owner is visible but far from the cart for N
                # consecutive frames. Judged against `owner_raw`, so it survives
                # the link being released — a cart standing at the door while the
                # person who brought it walks away is the case this measures, and
                # gating it on the live link meant it could not see it.
                #
                # Distance is measured between the box EDGES and scaled by the
                # cart's own size. Measuring it centroid to centroid against a
                # flat pixel bar is what scored Cart 1 of the 1764173272870
                # static-cart clip 65 ABANDONED CART while Person 4 stood
                # against it: the two boxes were in contact (edge gap 0.0 px)
                # yet their centroids were 232 px apart on every frame, past the
                # old 200 px bar, so the counter ran to 31 and the finaliser
                # latched the flag over the rest of the run. See
                # WALKAWAY_GAP_FRAC in config.py.
                person_far = False
                if (not attended_now and owner_raw is not None
                        and owner_raw in person_bb):
                    pb = person_bb[owner_raw]
                    gap = self._bbox_gap(bb, pb)
                    cart_diag = ((bb[2] - bb[0]) ** 2 + (bb[3] - bb[1]) ** 2) ** 0.5
                    far_gap = max(WALKAWAY_GAP_FRAC * cart_diag, WALKAWAY_MIN_GAP_PX)
                    if gap > far_gap:
                        self._walkaway_frames[cd] = self._walkaway_frames.get(cd, 0) + 1
                    else:
                        self._walkaway_frames.pop(cd, None)
                    person_far = self._walkaway_frames.get(cd, 0) > ABANDON_FRAMES
                abandoned = person_gone or person_far

                # Goods gone from a cart that was carrying them, owner gone
                # too. Read off the classification HISTORY, not off the current
                # cached frame and not off whether the fill override below
                # fired: classification runs every CLASSIFY_EVERY_N_FRAMES, so
                # on frames where the stale cache still reads partial this
                # would flicker back to False and the live tier would depend on
                # classifier cadence. The finaliser derives it the same way
                # from the same history, which is what keeps the two paths
                # agreeing.
                #
                # Gated on GRABRUN_MIN_RUN_OBS rather than the any-frame peak
                # used by the fill override, because this decides a tier: one
                # noisy "partial" on a genuinely empty cart must not float it
                # to a pushout.
                merch_removed = bool(
                    abandoned and cd in self._cart_cls_history
                    and merchandise_removed(
                        [h_fill for h_fill, _, _, _ in self._cart_cls_history[cd]],
                        GRABRUN_MIN_RUN_OBS))

                # Score the VOTE over the cart's whole classification
                # history, not the labels of the latest classified frame.
                #
                # A tier used to be decidable by one observation. On the
                # FF1763940475070 INSIDE clip Cart 3 was read full|unbagged
                # once, at bag confidence 0.679, against 28 bagged
                # observations totalling 23.21 and 21 partial reads against 8
                # full. OUTBOUND + full + unbagged + moving scores 75, so that
                # single frame logged a HIGH PRIORITY event and froze a 75 peak
                # — and because both peak-as-floor rules keep the higher live
                # reading (below, and sync_events_with_snapshots), the run
                # finished at 75 even though the finaliser's own vote said
                # partial|bagged / 30. Voting here is what makes the two paths
                # structurally incapable of disagreeing about labels; the
                # floors then only ever arbitrate CONTEXT (direction, speed,
                # abandonment), which is what they were written for.
                #
                # `is_valid` and `quality` still come from the current frame:
                # they are the kill switch and the event-logging gate, and both
                # ask "can the classifier read this cart right now", which is a
                # question about this frame and not about the history.
                #
                # Costs nothing per frame that the finaliser was not already
                # paying once per cart — the history is a few dozen tuples.
                history = self._cart_cls_history.get(cd, ())
                voted_fill, voted_bag, _vote_detail = vote_classification(history)
                if voted_fill is not None:
                    fill_lbl, bag_lbl = voted_fill, voted_bag
                    # Grab-and-run: the vote lands on "empty" because the empty
                    # tail outnumbers the loaded run, but the goods were there.
                    # Gated on a SUSTAINED run, exactly as the finaliser gates
                    # it — the any-frame peak this replaces let one noisy
                    # "partial" on a genuinely empty cart float it to a
                    # pushout, which is the same single-observation failure the
                    # vote above exists to stop.
                    if abandoned and fill_lbl == "empty":
                        sustained = peak_sustained_fill(
                            [h_fill for h_fill, _, _, _ in history],
                            GRABRUN_MIN_RUN_OBS)
                        if sustained:
                            fill_lbl = sustained
                    # A loaded cart cannot have bag "not_applicable": every
                    # empty observation votes 1.0 for it, so it wins the bag
                    # vote on any history with an empty majority. Re-vote
                    # across the loaded observations only.
                    if fill_lbl in ("partial", "full") and bag_lbl == "not_applicable":
                        bag_lbl = vote_bag_for_loaded_cart(history)

                pops_score = compute_pops(dir_label, speed_status, is_valid, fill_lbl,
                                          bag_label=bag_lbl, cart_detected=True,
                                          abandoned=abandoned, linked=linked,
                                          merch_removed=merch_removed)
                event_name, event_color = classify_event(pops_score, linked, dir_label, abandoned=abandoned)

                self._pops_cache[cd] = {
                    "score": pops_score, "event": event_name, "color": event_color,
                    "fill": fill_lbl, "bag": bag_lbl, "direction": dir_label,
                    "speed_status": speed_status, "linked": linked,
                    # Exported to the per-frame JSON. `abandoned` is the
                    # strongest term in compute_pops() — it floors the score at
                    # 60/75 — and `merch_removed` is what separates a parked
                    # cart from a grab-and-run. Neither was visible anywhere in
                    # the output, so a run that scored a cart 45 instead of 75
                    # could not be attributed without re-deriving both by hand
                    # from the classification history.
                    "abandoned": abandoned, "merch_removed": merch_removed,
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
                        "merch_removed": merch_removed,
                        # Who the cart was linked to at its peak. Recorded
                        # because abandonment is reasoned about per-owner, so a
                        # cart that scored high for the right reason and one
                        # that scored high off a mislinked bystander are
                        # otherwise indistinguishable in the output.
                        "owner": (gdi('person', linked_person_raw)
                                  if linked_person_raw is not None else None),
                        # When the peak happened — lets the POPS table row seek
                        # the tracked video straight to the moment.
                        "timestamp": round(timestamp, 2), "frame": frame_idx,
                    }


                # Log significant events (once per cart per event type).
                # Skip lower-severity events if a HIGH event was already
                # logged for this cart — the pushout already happened.
                # Also skip events where the cart was not validly classified
                # (unclear/unclassified) — these are noise, not actionable.
                # The score check is not redundant with the name: "UNLINKED EXIT"
                # is also returned below every tier for any unlinked outbound
                # cart, so an empty cart drifting doorward at POPS 0 carried a
                # loggable name. See prune_event_log(), which applies the same
                # rule to the finalised log.
                if event_name in LOGGABLE_EVENTS and pops_score >= MEDIUM_SCORE:
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

        if _cancelled:
            # No release here: the wrapper's finally owns the handles now, and
            # it does exactly what this used to — release both, then delete the
            # partial file (that order matters, an open VideoWriter keeps a lock
            # on it under Windows). One owner, so the two paths cannot drift.
            print(f"[CANCEL] run {run_seq} stopped at frame {frame_idx} of "
                  f"{total_frames}")
            raise RunCancelled(f"cancelled at frame {frame_idx} of "
                               f"{total_frames}")

        # Capture final frame for case report
        frame_capturer.capture_final_frame(
            im0, frame_idx, timestamp,
            self._pops_cache, self._cart_cls_cache)

        cap.release()
        writer.release()
        # Deregistered, so the finally does not delete an AVI the re-encode
        # below still needs. Everything after this point is post-processing:
        # an exception there leaves a complete AVI and its MP4 in a run
        # directory the pruner owns, which is the pre-existing behaviour and
        # not something a cleanup path should start second-guessing.
        self._run_handles = None

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
            merch_removed = False
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
            #
            # vote_classification() is the SAME function the frame loop scores
            # with, which is the point: the two paths can no longer land on
            # different labels for one cart, so a live peak that scored higher
            # than the reconciliation below differs only in CONTEXT — direction,
            # pace, abandonment — never in what was in the cart.
            if cd in self._cart_cls_history:
                history = self._cart_cls_history[cd]
                if history:
                    best_fill, best_bag, _vd = vote_classification(history)
                    print(f"[VOTE] Cart {cd}: fill_conf={_vd['fill_conf']} "
                          f"fill_count={_vd['fill_count']} "
                          f"fill_scores={_vd['fill_scores']} → {best_fill}")
                    print(f"[VOTE] Cart {cd}: bag_conf={_vd['bag_conf']} "
                          f"bag_count={_vd['bag_count']} "
                          f"bag_scores={_vd['bag_scores']} → {best_bag}")

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
                    fills = [f for f, b, fc, bc in history]
                    # Derived from the history rather than from whether the
                    # override below fired: the evidence is "loaded run, then
                    # an empty tail", and that is true whichever label the
                    # confidence vote happens to land on. Reading it off the
                    # override would make the finalised tier depend on the
                    # vote, so a cart whose vote lands on 'partial' directly
                    # would lose the escalation the live path already gave it.
                    merch_removed = abandoned and merchandise_removed(
                        fills, GRABRUN_MIN_RUN_OBS)

                    if best_fill == "empty" and abandoned:
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
                best_bag = vote_bag_for_loaded_cart(
                    self._cart_cls_history.get(cd, ()))

            # RECOMPUTE score with finalized, consistent inputs
            recomputed = compute_pops(
                direction, speed_status, True, best_fill,
                bag_label=best_bag, cart_detected=True,
                abandoned=abandoned, linked=linked,
                merch_removed=merch_removed,
            )
            final_score = recomputed
            # Re-apply caps based on final fill/bag
            if best_fill == "partial" and best_bag == "bagged":
                final_score = min(final_score, 55)
            final_event, final_color = classify_event(
                final_score, linked, direction, abandoned=abandoned,
            )

            # The reconciliation is the ONLY authority, and it writes back
            # unconditionally — including when it scores LOWER than the live peak
            # this cart reached mid-run.
            #
            # The live peak used to be a floor: whichever of the two scored
            # higher won, with all of its own fields. That is what put a cart in
            # the POPS table under a fill its own history voted against. On the
            # FF1763940475070 INSIDE clip Cart 3 was voted partial|bagged (fill
            # score 320 partial against 47 full) and displayed full|bagged,
            # because one full-reading frame had scored 40 against the
            # reconciled 35 and carried its labels with it.
            #
            # The cost is real and is accepted deliberately: a reconciliation
            # that lands lower now demotes the cart, which on the 1764099569430
            # OUTSIDE clip reports a live peak of 55 as 45. The vote is the
            # better evidence about what was in the cart — it is the whole
            # classification history against one frame — and a fill the table
            # shows has to be the fill the vote reached. Anything else is a
            # number from one reading beside a label from another.
            #
            # This is gk-pops-code's rule, taken as it stands there: see its
            # engine/tracker.py, where snap.update() is likewise unconditional.
            # sync_events_with_snapshots() drops its matching floor for the same
            # reason, and the two have to agree or the POPS table and the Events
            # tab tell different stories about one cart.
            snap.update({
                "fill": best_fill, "bag": best_bag, "quality": "valid_cart",
                "score": final_score, "event": final_event, "color": final_color,
                "direction": direction, "speed_status": speed_status,
                "linked": linked, "abandoned": abandoned,
                "merch_removed": merch_removed,
            })
            self._max_pops_per_cart[cd] = final_score
            print(f"[POPS] Cart {cd}: {best_fill}|{best_bag} {direction} "
                  f"score={final_score} (orig={original_score} recomp={recomputed}) "
                  f"[{source}]"
                  + (" merch_removed" if merch_removed else ""))

        # --- Sync last event per cart with POPS table ---
        # The reconciled POPS snapshot is the single source of truth and every
        # Events row is rewritten from it, downward included. See
        # sync_events_with_snapshots() for why the old "Events is truth for
        # abandonment" direction was wrong: it
        # discarded the reconciliation that had just been PRINTED, so the
        # 1764099569430 OUTSIDE clip logged `Cart 1: partial|unbagged score=65`
        # and showed partial|bagged 55 in the UI, and no cart could ever be
        # reconciled UP out of the partial+bagged cap of 55.
        for _note in sync_events_with_snapshots(
                self._event_log, self._peak_pops_snapshot,
                self._max_pops_per_cart):
            print(f"[POPS] {_note}")

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
        # A broken encoder must not throw away a good run. reencode_to_mp4()
        # raises now instead of silently returning a path to a file it failed to
        # write, and everything downstream of here — POPS reconciliation,
        # analytics, the heat-map, the case report — is worth having without a
        # playable video. So this is the one place that swallows it: the video
        # panel comes back empty, and the reason (ffmpeg's own last lines, and
        # where the raw AVI was kept) is already in the log above.
        try:
            out_path = reencode_to_mp4(avi_path)
        except Exception as e:
            print(f"[ERROR] the tracked video could not be encoded: {e}")
            print("[ERROR] the rest of the run is unaffected — POPS, analytics "
                  "and the case report below are complete; only the video "
                  "player will be empty.")
            out_path = None
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
            # Beyond max_score/peak_event: the reconciled snapshot's own reading
            # of WHY the cart scored what it did. Without these a run cannot be
            # attributed after the fact — see docs/missed_pushout_fix_plan.md,
            # where a cart's 45 could only be explained by re-deriving
            # abandonment and the owner link by hand from the per-frame records.
            "pops_summary": {
                f"C{cid}": {
                    "max_score": self._max_pops_per_cart.get(cid, 0),
                    "peak_event": self._peak_pops_snapshot.get(cid, {}).get("event", "CLEAR"),
                    "peak_frame": self._peak_pops_snapshot.get(cid, {}).get("frame"),
                    "peak_timestamp": self._peak_pops_snapshot.get(cid, {}).get("timestamp"),
                    "owner": self._peak_pops_snapshot.get(cid, {}).get("owner"),
                    "abandoned": bool(self._peak_pops_snapshot.get(cid, {}).get("abandoned", False)),
                    "merch_removed": bool(self._peak_pops_snapshot.get(cid, {}).get("merch_removed", False)),
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
        # Same channel, same question: was this quiet run actually quiet, or just
        # unreadable? A cart the quality head declined to classify is scored
        # without being assessed, and that was reported nowhere.
        _unassessed_note = unassessed_cart_note(
            self._cart_frames_total, self._cart_frames_unassessed,
            [cd for cd, snap in self._peak_pops_snapshot.items()
             if (snap or {}).get("quality", "unclassified")
             in ("unclear", "unclassified")])
        if _unassessed_note:
            analytics_result.rule_diagnostics.append(_unassessed_note)

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
            self._stash_pending_case_report({
                "captures": list(frame_capturer.captures),
                "full_json": full_json,
                "event_log": list(self._event_log),
                "peak_snapshots": dict(self._peak_pops_snapshot),
                "vlm_backend": vlm_backend,
                "vlm_api_key": vlm_api_key,
                "analytics_result": analytics_result,
                # Which run this report belongs to. finalize_case_report()
                # compares it against the cancel token so a Cancel aimed at a
                # LATER run cannot discard this one — see
                # engine/cancellation.py.
                "run_seq": run_seq,
            })
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
            # The same checkpoint the deferred path gets in
            # finalize_case_report(): a cancel that landed during the post-loop
            # tail (encode, JSON, analytics, heat-map) must not be followed by a
            # VLM pass.
            self._cancel.raise_if_cancelled(run_seq, "before the case report")
            try:
                case_report_html, case_report_file = self._run_case_report(
                    run_seq=run_seq,
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
                         analytics_result,
                         run_seq: int | None = None) -> tuple[str, str | None]:
        is_local_vlm = "Claude" not in vlm_backend
        vlm = None
        try:
            # INSIDE the try. Outside it, anything raising in here — the
            # empty_cache(), or a .to("cpu") after a partial move — stranded
            # self._offloaded True with no matching restore, because the finally
            # that clears it had not been entered yet. finalize_case_report()
            # swallows the exception into an inline banner, so the only visible
            # sign was that every LATER run raised the device-mismatch error
            # this whole document-length guard exists to prevent. Same shape as
            # Phase 8.1; see docs/device_drift_fix_plan.md §9.3.
            if is_local_vlm:
                self._release_detection_gpu_memory()
            # should_cancel, not the token itself: the analyzer has no
            # business knowing about run sequences, and a plain callable is what
            # a transformers StoppingCriteria wants anyway.
            vlm = VLMAnalyzer(backend=vlm_backend, api_key=vlm_api_key,
                              device=self.device,
                              should_cancel=(
                                  lambda: self._cancel.is_cancelled(run_seq)))
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
                try:
                    vlm.unload_model()
                except Exception as e:
                    # Never let this escape. It would replace the real error,
                    # and — worse — it would skip the restore below, leaving
                    # self._offloaded True for the life of the process. The
                    # start-of-run guard early-returns on that flag, so the
                    # stack would stay on the CPU with self.device == "cuda"
                    # and every later run would raise the device-mismatch
                    # error this whole guard exists to prevent.
                    print(f"[WARN] unloading the VLM failed: {e}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            # Gated on _offloaded, not on is_local_vlm alone. With the release
            # now inside the try (above), it may not have run at all — and an
            # unconditional restore in that case is NOT a no-op: it calls
            # self.model.to(self.device), and Model._apply() nulls `predictor`
            # on every .to(), same-device included. With no release there is no
            # parked _held_predictor to put back, _ensure_on_device() sees no
            # drift and returns early, and the next .track() appends a second
            # tracking callback — the silent detection-degradation regression
            # _release_detection_gpu_memory() is written to prevent.
            if is_local_vlm and self._offloaded:
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

    def _stash_pending_case_report(self, payload: dict):
        """Queue a deferred case report for finalize_case_report() to consume.

        Bounded. One finalize event is chained per run, so the queue normally
        holds one entry and never more than a couple; it can only grow if a
        chained event never fires at all — a browser reload, or a dropped SSE
        stream mid-chain. Each entry pins a full JSON document plus the captured
        frames, so an unbounded queue would leak memory and eventually hand a
        stale report to a later run. Dropping the OLDEST is what keeps the newest
        runs paired with their own reports.
        """
        self._pending_case_reports.append(payload)
        while len(self._pending_case_reports) > PENDING_CASE_REPORTS_MAX:
            self._pending_case_reports.pop(0)
            print("[WARN] dropped an orphaned pending case report — its "
                  "finalize event never ran (browser reload mid-run?)")

    def finalize_case_report(self) -> tuple[str, str | None]:
        """Run the deferred VLM case-report pass.

        Consumes state stashed by `process_video(defer_case_report=True)`.
        Returns (case_report_html, case_report_file_path). Returns ("", None)
        when there's nothing pending — safe to call unconditionally.
        Errors are caught and surfaced as an inline error banner; the file
        is None in that case so gr.File renders empty.

        Holds `_gpu_lock` for the whole pass, which is the other half of the
        serialisation described in `process_video`: a local-VLM report parks the
        detection stack on the CPU, so a run must not be in its frame loop while
        this is running. The pop is inside the hold too — not because the lock
        fixes the pairing (it cannot; see §9.4.2 of
        docs/device_drift_fix_plan.md), but because there is no reason for it to
        be outside.
        """
        with self._gpu_lock:
            return self._finalize_case_report_locked()

    def _finalize_case_report_locked(self) -> tuple[str, str | None]:
        # Popped, not read-then-cleared, and popped BEFORE the try: consuming
        # regardless of outcome is deliberate (a failed report must not be
        # served again to the next finalize), and taking the OLDEST entry is
        # what pairs this call with the run it was chained to.
        pending = (self._pending_case_reports.pop(0)
                   if self._pending_case_reports else None)
        if not pending or not pending.get("captures"):
            return "", None
        # Scoped to the popped payload's OWN run, which is the whole reason
        # CancelToken counts sequences. This event is chained after a run, so it
        # also fires after a run that was itself cancelled — and the payload it
        # finds may belong to an EARLIER run that completed normally and is
        # entitled to its report. Comparing sequences is what tells those apart;
        # a bare flag would discard the innocent one.
        if self._cancel.is_cancelled(pending.get("run_seq")):
            print(f"[CANCEL] dropped the case report for run "
                  f"{pending.get('run_seq')}")
            return ("<p style='color:#94a3b8;padding:20px;'>"
                    "Case report cancelled.</p>", None)
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
