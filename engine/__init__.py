# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""POPS Engine — modular cart tracking + classification + scoring pipeline."""
from .tracker import TrackingEngine
from .scoring import compute_pops, classify_event
from .config import SAMPLE_VIDEOS

__all__ = [
    "TrackingEngine",
    "compute_pops", "classify_event",
    "SAMPLE_VIDEOS",
]
