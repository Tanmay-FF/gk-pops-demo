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
    frames: np.ndarray             # shape (N,) int32 — frame indices
    speeds: np.ndarray             # shape (N,) float32

    @property
    def n_samples(self) -> int:
        return int(self.positions.shape[0])

    @property
    def duration_s(self) -> float:
        if self.timestamps.size < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


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
