# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
FastAPI layer for GK-POPS — exposes TrackingEngine.process_video,
recompute_analytics, and invalidate_cache over HTTP.

Run with:
    cd D:/gatekeeper_projects/gk-pops-enhanced
    venv_gk-pops-enhanced/Scripts/uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse

# Add repo root to path so `engine` resolves
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import TrackingEngine, zone_editor
from engine.analytics_models import Zone
from engine.trajectory_cache import make_video_key

from .models import (
    RunRequest, RunResult, RecomputeRequest, RecomputeResult,
    InvalidateRequest, UploadResult, ZoneIn, MakeZoneResult,
    ZoneOut, POPSCart, AnalyticsHTML,
)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="GK-POPS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_engine = TrackingEngine(device="auto")
# Single-worker executor pins all CUDA work to one OS thread for the lifetime
# of the server. This matches Gradio's single-thread behaviour and avoids the
# CUDA stream re-sync overhead that occurs when the default ThreadPoolExecutor
# schedules inference on a different thread each request.
_inference_executor = ThreadPoolExecutor(max_workers=1)
_sessions: dict[str, dict] = {}         # video_id -> session data
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "pops_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zone_out_to_engine(z: ZoneOut) -> Zone:
    polygon = np.array(z.polygon, dtype=np.int32)
    return Zone(
        zone_id=z.zone_id,
        name=z.name,
        polygon=polygon,
        applies_to=z.applies_to,
        color=tuple(z.color),
    )


def _extract_pops_carts(engine: TrackingEngine) -> list[POPSCart]:
    """Merge peak_pops_snapshot + display_map into a typed cart list."""
    carts: list[POPSCart] = []
    snap_map = getattr(engine, "_peak_pops_snapshot", {})
    display_map = getattr(engine, "_display_map", {})
    max_pops = getattr(engine, "_max_pops_per_cart", {})

    for raw_id, snap in snap_map.items():
        disp = display_map.get(raw_id)
        if disp is None:
            disp = f"C-{int(raw_id):03d}" if str(raw_id).isdigit() else str(raw_id)
        carts.append(POPSCart(
            id=disp,
            max_pops=int(max_pops.get(raw_id, snap.get("score", 0))),
            peak_event=snap.get("event", "CLEAR"),
            quality=snap.get("quality", "valid_cart"),
            fill=snap.get("fill", "unknown"),
            bag=snap.get("bag", "unknown"),
            direction=snap.get("direction", "UNKNOWN"),
            speed_status=snap.get("speed_status", "NORMAL"),
            linked=bool(snap.get("linked", False)),
        ))
    carts.sort(key=lambda c: c.max_pops, reverse=True)
    return carts


def _serve_heatmap(heatmap_path: Optional[str], video_id: str) -> Optional[str]:
    """Copy heatmap PNG into the session dir and return its URL."""
    if not heatmap_path or not os.path.exists(heatmap_path):
        return None
    session_dir = _UPLOAD_DIR / video_id
    session_dir.mkdir(exist_ok=True)
    dest = session_dir / "heatmap.png"
    shutil.copy2(heatmap_path, dest)
    return f"/api/files/{video_id}/heatmap.png"


def _serve_video(out_path: Optional[str], video_id: str) -> str:
    if not out_path or not os.path.exists(out_path):
        return ""
    session_dir = _UPLOAD_DIR / video_id
    session_dir.mkdir(exist_ok=True)
    ext = Path(out_path).suffix or ".mp4"
    dest = session_dir / f"tracked{ext}"
    shutil.copy2(out_path, dest)
    return f"/api/files/{video_id}/tracked{ext}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=UploadResult)
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file, save it, return video_id + first-frame URL."""
    video_id = str(uuid.uuid4())
    session_dir = _UPLOAD_DIR / video_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_path = session_dir / (file.filename or "video.mp4")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract first frame for zone editor
    frame = zone_editor.extract_first_frame(str(dest_path))
    h, w = (frame.shape[:2] if frame is not None else (1080, 1920))

    frame_url = ""
    if frame is not None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            frame_path = session_dir / "frame.jpg"
            frame_path.write_bytes(buf.tobytes())
            frame_url = f"/api/files/{video_id}/frame.jpg"

    _sessions[video_id] = {
        "video_path": str(dest_path),
        "frame_shape": (h, w),
    }
    return UploadResult(video_id=video_id, frame_url=frame_url, width=w, height=h)


@app.post("/api/run", response_model=RunResult)
async def run_analysis(req: RunRequest):
    """Run the full POPS pipeline on the uploaded video."""
    session = _sessions.get(req.video_id)
    if not session:
        raise HTTPException(status_code=404, detail="video_id not found — upload first")

    video_path = session["video_path"]
    engine_zones = [_zone_out_to_engine(z) for z in req.zones]

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _inference_executor,
        lambda: _engine.process_video(
            video_path,
            camera_placement=req.camera_placement,
            vlm_backend=req.vlm_backend,
            vlm_api_key=req.vlm_api_key,
            zones=engine_zones,
        ),
    )

    (out_path, json_path, json_str,
     video_html, det_html, config_html, legend_html, pops_html, events_html,
     bev3d_html, bev2d_html, case_report_html, case_report_file,
     analytics_summary_html, spikes_html, dwell_html, journey_html,
     heatmap_path,
     alert_banner_html) = result

    tracking_json = json.loads(json_str) if json_str else {}
    # Inject pre-rendered HTML blobs into tracking_json for tabs that need it
    tracking_json["_html"] = {
        "video_info": video_html,
        "detection": det_html,
        "config": config_html,
        "legend": legend_html,
    }

    pops_carts = _extract_pops_carts(_engine)
    video_url = _serve_video(out_path, req.video_id)
    heatmap_url = _serve_heatmap(heatmap_path, req.video_id)

    # Store result for recompute
    session["last_result"] = {
        "out_path": out_path,
        "heatmap_path": heatmap_path,
    }

    return RunResult(
        session_id=req.video_id,
        video_url=video_url,
        heatmap_url=heatmap_url,
        tracking_json=tracking_json,
        pops_carts=pops_carts,
        case_report_html=case_report_html or "",
        bev3d_html=bev3d_html or "",
        bev2d_html=bev2d_html or "",
        alert_banner_html=alert_banner_html or "",
        analytics=AnalyticsHTML(
            summary_html=analytics_summary_html or "",
            spikes_html=spikes_html or "",
            dwell_html=dwell_html or "",
            journey_html=journey_html or "",
        ),
    )


@app.post("/api/recompute", response_model=RecomputeResult)
async def recompute_analytics(req: RecomputeRequest):
    """Recompute analytics from cached trajectories without re-running YOLO."""
    session = _sessions.get(req.video_id)
    if not session:
        raise HTTPException(status_code=404, detail="video_id not found")

    video_path = session["video_path"]
    engine_zones = [_zone_out_to_engine(z) for z in req.zones]

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _inference_executor,
        lambda: _engine.recompute_analytics(video_path, engine_zones),
    )

    # recompute_analytics returns (analytics_summary_html, spikes_html, dwell_html, journey_html, heatmap_image)
    analytics_summary_html, spikes_html, dwell_html, journey_html, heatmap_img = result

    # heatmap_img may be a numpy array or path
    heatmap_url: Optional[str] = None
    if isinstance(heatmap_img, np.ndarray):
        ok, buf = cv2.imencode(".png", heatmap_img)
        if ok:
            session_dir = _UPLOAD_DIR / req.video_id
            session_dir.mkdir(exist_ok=True)
            (session_dir / "heatmap.png").write_bytes(buf.tobytes())
            heatmap_url = f"/api/files/{req.video_id}/heatmap.png"
    elif isinstance(heatmap_img, str) and os.path.exists(heatmap_img):
        heatmap_url = _serve_heatmap(heatmap_img, req.video_id)

    return RecomputeResult(
        analytics=AnalyticsHTML(
            summary_html=analytics_summary_html or "",
            spikes_html=spikes_html or "",
            dwell_html=dwell_html or "",
            journey_html=journey_html or "",
        ),
        heatmap_url=heatmap_url,
    )


@app.post("/api/invalidate")
async def invalidate_cache(req: InvalidateRequest):
    """Clear detection cache for a specific video or all videos."""
    video_path: Optional[str] = None
    if req.video_id:
        session = _sessions.get(req.video_id)
        if session:
            video_path = session.get("video_path")
    _engine.invalidate_cache(video_path)
    return {"ok": True}


@app.post("/api/zones/make", response_model=MakeZoneResult)
async def make_zone(z: ZoneIn):
    """Validate a polygon and return a fully-formed Zone object."""
    pts = [(int(p[0]), int(p[1])) for p in z.points]
    ok, msg = zone_editor.validate_polygon(pts)
    if not ok:
        raise HTTPException(status_code=422, detail=msg)
    zone = zone_editor.make_zone(z.name, pts, z.applies_to, z.zone_index)
    b, g, r = zone.color
    return MakeZoneResult(zone=ZoneOut(
        zone_id=zone.zone_id,
        name=zone.name,
        polygon=zone.polygon.tolist(),
        applies_to=zone.applies_to,
        color=[b, g, r],
    ))


@app.get("/api/files/{video_id}/{filename}")
async def serve_file(video_id: str, filename: str):
    """Serve a session file (tracked video, heatmap PNG, first frame)."""
    path = _UPLOAD_DIR / video_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "video/mp4" if filename.endswith(".mp4") else (
        "image/jpeg" if filename.endswith(".jpg") else "image/png"
    )
    return FileResponse(str(path), media_type=media_type)


@app.get("/api/health")
async def health():
    return {"ok": True, "model": "YOLOv26m", "device": _engine.device}
