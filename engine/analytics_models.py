# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Dataclasses shared between the trajectory cache, analytics builder, and the
Gradio UI for the in-store retail analytics layer.

These types deliberately depend only on numpy + the stdlib so that
analytics_builder.py (which consumes them) can be unit-tested without
importing torch / ultralytics / cv2-heavy modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


ZoneAppliesTo = Literal["person", "cart", "both"]
ZoneKind = Literal["analytics", "wall", "aisle", "door", "fixture"]
SpikeSeverity = Literal["NORMAL", "WATCH", "QUEUE_FORMING", "BACKED_UP"]

# Operational rule-engine severities. Deliberately a DIFFERENT vocabulary from
# SpikeSeverity — a blocked fire exit is not a "queue forming", and collapsing
# the two would put a lie in the JSON export. build_alert_banner() filters each
# family on its own vocabulary.
RuleSeverity = Literal["INFO", "WATCH", "ACTION", "SAFETY"]
RuleId = Literal["blocked_door", "static_cart", "abandoned_cart",
                 "incoming_empty", "child_in_cart"]

# Bumped whenever CartFactSample / TrackRecord gain or lose a field that the
# rule engine depends on. Bundles cached by an older build carry a lower value
# (or none at all) and must be reported as "rules unavailable" rather than
# silently yielding zero findings — an empty list reads as "no issues found",
# which is a wrong answer rather than a missing one.
FACTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Zone:
    """A named polygon in the source video's pixel coordinates."""
    zone_id: str
    name: str
    polygon: np.ndarray            # shape (V, 2), int32 — pixel coords on the source frame
    applies_to: ZoneAppliesTo = "person"
    kind: ZoneKind = "analytics"
    color: tuple[int, int, int] = (0, 200, 255)   # BGR for cv2 overlays


@dataclass
class TrackRecord:
    """All per-frame samples for a single raw track id, packed as numpy arrays."""
    raw_id: int
    label: str                     # "person" | "cart"
    display_id: int                # human-friendly id from TrackingEngine._display_map
    positions: np.ndarray          # shape (N, 2) float32 — pixel-space centroids
    timestamps: np.ndarray         # shape (N,) float32 — seconds (CAP_PROP_POS_MSEC / 1000)
    frames: np.ndarray             # shape (N,) int32 — DISPLAY ONLY, see warning below
    speeds: np.ndarray             # shape (N,) float32
    # shape (N, 4) float32 — (x1, y1, x2, y2) per sample. Needed for
    # overlap-fraction zone tests (a cart can block a doorway while its
    # centroid sits outside the polygon). Empty array when unavailable.
    bboxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))

    # WARNING on `frames`: it is synthesised as arange(first_frame, first_frame + N),
    # which assumes the track was detected in every consecutive frame. That is
    # false whenever detection drops out, and false after cart re-identification.
    # Never derive durations or ordering from it — use `timestamps`.

    @property
    def n_samples(self) -> int:
        return int(self.positions.shape[0])

    @property
    def has_bboxes(self) -> bool:
        return self.bboxes.shape[0] == self.positions.shape[0] and self.bboxes.size > 0

    @property
    def duration_s(self) -> float:
        if self.timestamps.size < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass
class CartFactSample:
    """One observation of a cart's classified + link state.

    Recorded every frame the cart is detected, BEFORE the
    MIN_CART_FRAMES_FOR_POPS guard, so brief carts still have facts even when
    POPS declines to score them.

    `fill` / `bag` only refresh every CLASSIFY_EVERY_N_FRAMES; `fill_stale_frames`
    says how many frames old the reading is, so rules can count *classified
    observations* (stale == 0) rather than frames.
    """
    t: float                              # seconds — same clock as TrackRecord.timestamps
    frame: int
    fill: str                             # empty | partial | full | unclassified | non-applicable
    bag: str
    fill_conf: float
    quality: str
    fill_stale_frames: int
    linked: bool
    linked_person_display: Optional[int]
    abandoned_linker: bool                # the live linker's person_gone or person_far


@dataclass
class TrajectoryBundle:
    """Everything analytics needs, captured once at the end of process_video()."""
    video_key: str
    video_path: str
    width: int
    height: int
    fps: float
    total_frames: int
    tracks: dict[int, TrackRecord] = field(default_factory=dict)
    cart_pops: dict[int, dict] = field(default_factory=dict)        # display_id -> peak snapshot
    event_log: list[dict] = field(default_factory=list)
    representative_frame: Optional[np.ndarray] = None               # last decoded frame; used as heatmap background
    # --- Rule-engine inputs ------------------------------------------------
    # cart display_id -> chronological fact samples. Keyed by DISPLAY id while
    # tracks are keyed by RAW id, and the two have different sample counts —
    # join them with rules.join_on_time(), never by parallel indexing.
    cart_facts: dict[int, list[CartFactSample]] = field(default_factory=dict)
    facts_schema_version: int = 0          # 0 = written before the rule engine existed
    # True when CAP_PROP_POS_MSEC was unusable and timestamps were derived from
    # frame_index / fps instead. Surfaced in the UI so a degraded run is visible.
    timestamps_synthesized: bool = False


@dataclass
class DwellRow:
    """One zone visit by one person/cart."""
    track_label: str               # "person" | "cart"
    display_id: int
    zone_id: str
    zone_name: str
    enter_t: float
    exit_t: float
    dwell_seconds: float
    visit_index: int               # 1-based, per (display_id, zone)


@dataclass
class JourneyEdge:
    """A single zone-to-zone transition by one track."""
    track_label: str
    display_id: int
    src_zone: str                  # zone_id or "__OUTSIDE__"
    dst_zone: str
    transition_t: float


@dataclass
class QueueSpike:
    """A zone flagged by the multi-signal congestion model."""
    zone_id: str
    zone_name: str
    avg_dwell_s: float
    max_dwell_s: float
    n_visits: int
    threshold_s: float
    severity: SpikeSeverity
    # Multi-signal evidence — populated by compute_zone_congestion().
    # Defaulted so older callers / tests that built QueueSpike directly
    # remain valid.
    score: float = 0.0
    peak_occupancy: int = 0
    mean_occupancy: float = 0.0
    static_fraction: float = 0.0
    avg_speed_px_s: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class RuleFinding:
    """One operational rule outcome over a closed time interval.

    Deliberately separate from the POPS event log: "ABANDONED CART" already
    exists in scoring.py as a theft-risk label gated behind pops_score >= 31,
    and an INBOUND cart is floored to 5 by the kill switch — so an operational
    abandonment routed through POPS would be unreachable.
    """
    rule_id: str                   # see RuleId
    label: str                     # human-facing, e.g. "UNATTENDED CART (OPS)"
    severity: str                  # see RuleSeverity
    cart_display_id: int
    zone_id: Optional[str]
    zone_name: Optional[str]
    start_t: float
    end_t: float
    duration_s: float
    threshold_s: float
    ongoing_at_eov: bool = False   # interval still open when the video ended
    confidence: str = "high"       # "high" | "degraded" (sparse samples / synthesised ts)
    n_samples: int = 0
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass
class AnalyticsResult:
    """One-shot output of analytics_builder.run_all()."""
    dwell_rows: list[DwellRow] = field(default_factory=list)
    dwell_summary: list[dict] = field(default_factory=list)         # per-zone aggregates
    heatmap_png_path: Optional[str] = None
    heatmap_array: Optional[np.ndarray] = None                      # (H, W) float32, 0..1
    heatmap_composite: Optional[np.ndarray] = None                  # (H, W, 3) uint8 BGR — colormapped + alpha-blended ready for gr.Image
    journey_edges: list[JourneyEdge] = field(default_factory=list)
    journey_matrix: Optional[np.ndarray] = None                     # NxN int counts, includes __OUTSIDE__
    journey_labels: list[str] = field(default_factory=list)
    queue_spikes: list[QueueSpike] = field(default_factory=list)
    spike_events: list[dict] = field(default_factory=list)          # ready to splice into _event_log
    insight_text: str = ""                                           # auto-generated narrative summary
    # --- Operational rule engine -------------------------------------------
    rule_findings: list[RuleFinding] = field(default_factory=list)
    # None  = rules ran normally.
    # str   = why they could not run (stale cached bundle, no monitored zones,
    #         video shorter than every threshold). Rendered as an explicit
    #         "did not run" state so it never reads as "no issues found".
    rules_unavailable_reason: Optional[str] = None
