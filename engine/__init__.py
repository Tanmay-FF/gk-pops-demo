# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""POPS Engine — modular cart tracking + classification + scoring pipeline."""
from .tracker import TrackingEngine
from .scoring import compute_pops, classify_event
from .config import SAMPLE_VIDEOS
from .analytics_models import (
    AnalyticsResult, DwellRow, JourneyEdge, QueueSpike,
    TrackRecord, TrajectoryBundle, Zone,
)
from . import analytics_builder, analytics_ui, zone_editor
from .scene_detector import SceneElement, detect_scene_elements

__all__ = [
    "TrackingEngine",
    "compute_pops", "classify_event",
    "SAMPLE_VIDEOS",
    "Zone", "TrajectoryBundle", "AnalyticsResult",
    "DwellRow", "JourneyEdge", "QueueSpike", "TrackRecord",
    "analytics_builder", "analytics_ui", "zone_editor",
    "SceneElement", "detect_scene_elements",
]
