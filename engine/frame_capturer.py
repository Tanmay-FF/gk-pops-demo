# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Event-driven frame capturer — grabs key video frames at POPS event
moments for VLM analysis and case report evidence.
"""
import cv2

from .config import (
    FRAME_CAPTURE_POPS_MEDIUM, FRAME_CAPTURE_POPS_HIGH, FRAME_CAPTURE_MAX,
)


class FrameCapturer:
    """Captures key video frames at POPS event moments.

    Instantiated once per ``process_video()`` call.  Called at specific
    points in the frame loop.  Returns captured frames after processing.
    """

    def __init__(self, max_frames: int = FRAME_CAPTURE_MAX):
        self._captures: list[dict] = []
        self._triggers_fired: dict[int, set] = {}   # cart display_id -> set of trigger names
        self._first_codetection_done = False
        self._seen_link_carts: set = set()
        self._max = max_frames

    # ------------------------------------------------------------------
    def _store(self, im0, frame_idx, timestamp, trigger, cart_id, pops_context):
        if len(self._captures) >= self._max:
            return
        ok, buf = cv2.imencode('.jpg', im0, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        self._captures.append({
            "image_bytes": buf.tobytes(),
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 2),
            "trigger": trigger,
            "cart_id": cart_id,
            "pops_context": pops_context or {},
        })

    def _cart_triggers(self, cd):
        if cd not in self._triggers_fired:
            self._triggers_fired[cd] = set()
        return self._triggers_fired[cd]

    # ------------------------------------------------------------------
    def check_and_capture(self, im0, frame_idx, timestamp,
                          event_log, pops_cache, links,
                          person_count, cart_count,
                          get_display_id, obj_labels):
        """Check all trigger conditions and capture if any fires.

        Called once per frame, after overlays are drawn but before BEV
        composition.
        """
        if len(self._captures) >= self._max:
            return

        # 1. First person+cart co-detection (global, once)
        if not self._first_codetection_done and person_count > 0 and cart_count > 0:
            self._first_codetection_done = True
            self._store(im0, frame_idx, timestamp,
                        "first_codetection", None, {})

        # 2. New person-cart link established
        for cart_raw in links:
            label = obj_labels.get(cart_raw, "cart")
            cd = get_display_id(label, cart_raw)
            if cd not in self._seen_link_carts:
                self._seen_link_carts.add(cd)
                ctx = pops_cache.get(cd, {})
                self._store(im0, frame_idx, timestamp,
                            "link_established", cd, ctx)

        # 3-4-5. Per-cart POPS triggers
        for cd, info in pops_cache.items():
            score = info.get("score", 0)
            triggers = self._cart_triggers(cd)

            if score >= FRAME_CAPTURE_POPS_MEDIUM and "medium" not in triggers:
                triggers.add("medium")
                self._store(im0, frame_idx, timestamp,
                            "pops_medium", cd, info)

            if score >= FRAME_CAPTURE_POPS_HIGH and "high" not in triggers:
                triggers.add("high")
                self._store(im0, frame_idx, timestamp,
                            "pops_high", cd, info)

        # 5. Event just logged (pushout / abandoned / high priority)
        if event_log:
            last = event_log[-1]
            if last["frame"] == frame_idx:
                cd = last["cart_id"]
                triggers = self._cart_triggers(cd)
                evt_key = f"event_{last['event']}"
                if evt_key not in triggers:
                    triggers.add(evt_key)
                    self._store(im0, frame_idx, timestamp,
                                f"event:{last['event']}", cd,
                                pops_cache.get(cd, {}))

    # ------------------------------------------------------------------
    def capture_final_frame(self, im0, frame_idx, timestamp,
                            pops_cache, cart_cls_cache):
        """Always capture the last frame of the video."""
        ctx = {}
        if pops_cache:
            # Pick highest-score cart for context
            best_cd = max(pops_cache, key=lambda c: pops_cache[c].get("score", 0))
            ctx = pops_cache[best_cd]
        self._store(im0, frame_idx, timestamp, "final_frame", None, ctx)

    # ------------------------------------------------------------------
    @property
    def captures(self) -> list[dict]:
        return self._captures
