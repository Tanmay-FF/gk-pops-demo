# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Pydantic request/response models for the POPS FastAPI layer."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class ZoneIn(BaseModel):
    name: str
    points: list[list[float]]   # [[x, y], ...]  source-pixel coords
    applies_to: str = "person"
    zone_index: int = 0         # for palette color assignment


class ZoneOut(BaseModel):
    zone_id: str
    name: str
    polygon: list[list[int]]    # [[x, y], ...]
    applies_to: str
    color: list[int]            # [B, G, R]


class RunRequest(BaseModel):
    video_id: str
    camera_placement: str = "Outside (facing entrance)"
    vlm_backend: str = "None (skip)"
    vlm_api_key: str = ""
    zones: list[ZoneOut] = []


class RecomputeRequest(BaseModel):
    video_id: str
    zones: list[ZoneOut] = []


class InvalidateRequest(BaseModel):
    video_id: Optional[str] = None


class POPSCart(BaseModel):
    id: str
    max_pops: int
    peak_event: str
    quality: str
    fill: str
    bag: str
    direction: str
    speed_status: str
    linked: bool


class AnalyticsHTML(BaseModel):
    summary_html: str
    spikes_html: str
    dwell_html: str
    journey_html: str


class RunResult(BaseModel):
    session_id: str
    video_url: str
    heatmap_url: Optional[str]
    tracking_json: dict
    pops_carts: list[POPSCart]
    case_report_html: str
    bev3d_html: str
    bev2d_html: str = ""
    alert_banner_html: str = ""
    analytics: AnalyticsHTML


class RecomputeResult(BaseModel):
    analytics: AnalyticsHTML
    heatmap_url: Optional[str]


class UploadResult(BaseModel):
    video_id: str
    frame_url: str
    width: int
    height: int


class MakeZoneResult(BaseModel):
    zone: ZoneOut
