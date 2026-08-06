"""
POPS - Push-Out Probability Score: Interactive Demo (v2 — modular)
===================================================================
Thin Gradio UI wrapper.  All logic lives in engine/*.py.
python code/demo_app_v2.py
"""
import os
import tempfile
import traceback
import cv2
import gradio as gr

from engine import TrackingEngine, SAMPLE_VIDEOS, analytics_ui, zone_editor
from engine import ui_builder
from engine.config import TEST_VIDEO_DIR
from engine.config import VLM_BACKENDS, VLM_DEFAULT_BACKEND
from engine.trajectory_cache import make_video_key
from engine.floor_bev2d_builder import build_floor_2d_html


def _bgr_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _bgr_ndarray_to_rgb(img):
    """Convert the engine's BGR composite to RGB for gr.Image.

    gr.Image accepts ndarrays directly, but Gradio 6 silently mishandles
    string paths (the gr.File chip gets stuck on "Uploading…" and the
    gr.Image stays blank). The engine now returns the heatmap as a BGR
    ndarray; we just flip channel order here.
    """
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------
engine = TrackingEngine(device='auto')

ZONE_APPLIES_OPTIONS = ["person", "cart", "both"]
ZONE_KIND_OPTIONS = [
    ("Analytics zone (Entrance/Exit/Checkout)", "analytics"),
    ("Wall",     "wall"),
    ("Aisle",    "aisle"),
    ("Fixture",  "fixture"),
    ("Door",     "door"),
]
EMPTY_RUN_RETURN_LEN = 21     # outputs of run_analysis (matches process_video tuple + duplicated heatmap path + alert banner + operational alerts)


def _empty_run_outputs(message: str = "No video selected"):
    """Shape-matched stub used on early-exit paths so Gradio doesn't choke."""
    err_html = f"<p style='color:#ef4444;padding:20px;'>{message}</p>"
    err_json = '{{"error": "{}"}}'.format(message.replace('"', "'"))
    return (
        None, None, err_json,                               # video, json file, json str
        err_html, "", "", "", "", "", "", "", "", None,    # 9 HTML panels + case file
        "", "", "", "",                                     # analytics summary/spikes/dwell/journey
        None, None,                                          # heatmap_image, heatmap_file
        gr.update(value="", visible=False),                 # alert banner — hidden when no run
        "",                                                  # operational alerts table
    )


def run_analysis(video_path, camera_placement, vlm_backend, vlm_api_key,
                 zones_state):
    """Generator handler — yields TWICE so the dashboard renders the fast
    pipeline first, then refreshes the case report when the VLM finishes.

    Yield 1: full 20-tuple with case_report_html as a "Generating…"
             placeholder and case_report_download as None.
    Yield 2: gr.update() no-ops for everything except the two case-report
             components, which receive the final VLM output.
    """
    if video_path is None:
        gr.Warning("Please upload or select a video first.")
        yield _empty_run_outputs("No video selected")
        return
    try:
        # ----- Phase 1: fast pipeline, VLM deferred -----
        result = engine.process_video(
            video_path,
            camera_placement=camera_placement,
            vlm_backend=vlm_backend,
            vlm_api_key=vlm_api_key,
            zones=list(zones_state or []),
            defer_case_report=True,
        )
        # Engine returns 21 outputs:
        #   0..8   misc html / video / json
        #   9      bev3d_html (full <!DOCTYPE> document)
        #   10     bev2d_html (full <!DOCTYPE> document)
        #   11,12  case_report
        #   13..16 analytics html
        #   17     heatmap_img_bgr (ndarray)
        #   18     heatmap_path
        #   19     alert_banner_html
        #   20     ops_alerts_html (operational rule findings)
        out = list(result)

        # Wrap the 3D and 2D BEV documents in an iframe srcdoc so each new
        # run forces the browser to reload the iframe content. Without this,
        # gr.HTML.value updates the surrounding div but a stale iframe from
        # the previous run keeps showing — that's why running a second video
        # still showed the first video's BEVs.
        if out[9]:
            out[9] = _wrap_in_iframe(out[9], height_px=620)
        if out[10]:
            out[10] = _wrap_in_iframe(out[10], height_px=620)

        ops_html     = out[-1] or ""
        alert_html   = out[-2] or ""
        heatmap_path = out[-3]
        heatmap_bgr  = out[-4]
        heatmap_rgb  = _bgr_ndarray_to_rgb(heatmap_bgr)
        alert_update = gr.update(value=alert_html, visible=bool(alert_html.strip()))
        if alert_html.strip():
            gr.Warning("Alerts detected — review the banner at the top of the page.")
        # Replace the trailing 4-tuple (BGR, path, alert HTML, ops HTML) with
        # (RGB, path, alert update, ops HTML). Slots 9 / 10 already swapped to
        # iframes.
        yield tuple(out[:-4]) + (heatmap_rgb, heatmap_path, alert_update, ops_html)

        # ----- Phase 2: run deferred VLM and refresh case-report tab only -----
        try:
            case_html, case_file = engine.finalize_case_report()
        except Exception as e:
            traceback.print_exc()
            case_html = (
                f"<p style='color:#ef4444;padding:20px;'>Case report failed: {e}</p>"
            )
            case_file = None
        if not case_html and not case_file:
            # No captures were taken (rare) — nothing to refresh.
            return
        no_op = gr.update()
        # Outputs slot 11 = case_report_html, slot 12 = case_report_download.
        # 11 leading no-ops, then the two updates, then 8 trailing no-ops.
        yield (no_op,) * 11 + (case_html, case_file) + (no_op,) * 8
    except Exception as e:
        gr.Warning(f"Error: {str(e)}")
        traceback.print_exc()
        yield _empty_run_outputs(f"Error: {e}")


# ---------------------------------------------------------------------------
# Zone editor handlers
# ---------------------------------------------------------------------------
def on_video_upload(video_path, prev_zones_state, prev_video_key):
    """When the user picks a video, reset zones (if it's a new file) and
    extract the first frame so they can draw zones on it."""
    if not video_path:
        return None, prev_zones_state or [], [], None, "", _zone_summary_html([])

    new_key = make_video_key(video_path)
    if new_key != prev_video_key:
        zones_state = []
        if prev_video_key is not None:
            gr.Info("New video — zones reset.")
    else:
        zones_state = list(prev_zones_state or [])

    frame = zone_editor.extract_first_frame(video_path)
    if frame is None:
        gr.Warning("Could not decode the first frame of this video.")
        return None, zones_state, [], new_key, None, _zone_summary_html(zones_state)

    overlay = zone_editor.render_zone_overlay(frame, zones_state, in_progress=[])
    return frame, zones_state, [], new_key, _bgr_to_rgb(overlay), _zone_summary_html(zones_state)


def on_canvas_click(evt: gr.SelectData, current_poly, first_frame, zones_state):
    """Append a vertex on every click; redraw the overlay live."""
    if first_frame is None:
        gr.Warning("Upload a video first.")
        return current_poly or [], None
    pts = list(current_poly or [])
    x, y = int(evt.index[0]), int(evt.index[1])
    pts.append((x, y))
    overlay = zone_editor.render_zone_overlay(first_frame, zones_state or [], in_progress=pts)
    return pts, _bgr_to_rgb(overlay)


def close_polygon(current_poly, zone_name, applies_to, zone_kind, zones_state, first_frame):
    """Validate the in-progress polygon and promote it to a Zone."""
    if first_frame is None:
        gr.Warning("Upload a video first.")
        return zones_state or [], current_poly or [], None, _zone_summary_html(zones_state or []), gr.update()

    pts = list(current_poly or [])
    ok, msg = zone_editor.validate_polygon(pts)
    if not ok:
        gr.Warning(msg)
        overlay = zone_editor.render_zone_overlay(first_frame, zones_state or [], in_progress=pts)
        return zones_state or [], pts, _bgr_to_rgb(overlay), _zone_summary_html(zones_state or []), gr.update()

    zones_state = list(zones_state or [])
    kind = zone_kind or "analytics"
    # Layout zones are PLACES, not track-type filters. "Track type" defaults to
    # "person", so a hand-drawn door or aisle left at that default would match
    # zero cart tracks and the operational rules over it would silently never
    # fire. Force "both" for layout kinds.
    if kind in ("door", "aisle", "fixture", "wall"):
        if applies_to != "both":
            gr.Info(f"{kind.capitalize()} zones apply to people and carts — "
                    f'"Track type" set to "both".')
        applies_to = "both"
    new_zone = zone_editor.make_zone(
        zone_name, pts, applies_to, len(zones_state),
        kind=kind,
    )
    zones_state.append(new_zone)

    overlay = zone_editor.render_zone_overlay(first_frame, zones_state, in_progress=[])
    delete_choices = [(z.name, z.zone_id) for z in zones_state]
    return zones_state, [], _bgr_to_rgb(overlay), _zone_summary_html(zones_state), gr.update(choices=delete_choices, value=None)


def undo_vertex(current_poly, first_frame, zones_state):
    pts = list(current_poly or [])
    if pts:
        pts.pop()
    overlay = (_bgr_to_rgb(zone_editor.render_zone_overlay(first_frame, zones_state or [], in_progress=pts))
               if first_frame is not None else None)
    return pts, overlay


def clear_inprogress(first_frame, zones_state):
    overlay = (_bgr_to_rgb(zone_editor.render_zone_overlay(first_frame, zones_state or [], in_progress=[]))
               if first_frame is not None else None)
    return [], overlay


def clear_all_zones(first_frame):
    overlay = (_bgr_to_rgb(zone_editor.render_zone_overlay(first_frame, [], in_progress=[]))
               if first_frame is not None else None)
    return [], [], overlay, _zone_summary_html([]), gr.update(choices=[], value=None)


def delete_zone(zone_id, zones_state, first_frame, current_poly):
    if not zone_id or not zones_state:
        return zones_state or [], None, _zone_summary_html(zones_state or []), gr.update()
    new_state = [z for z in zones_state if z.zone_id != zone_id]
    overlay = (_bgr_to_rgb(zone_editor.render_zone_overlay(first_frame, new_state, in_progress=current_poly or []))
               if first_frame is not None else None)
    delete_choices = [(z.name, z.zone_id) for z in new_state]
    return new_state, overlay, _zone_summary_html(new_state), gr.update(choices=delete_choices, value=None)


def recompute_analytics_handler(video_path, zones_state, camera_placement):
    def _empty(has_video):
        empty = analytics_ui.build_analytics_empty_state(has_video=has_video)
        return ("", empty, empty, empty, None, None,
                ui_builder.build_operational_alerts(
                    [], "Analytics could not be recomputed."))

    if not video_path:
        gr.Warning("Upload a video first.")
        return _empty(False)
    try:
        # Engine returns (summary, spikes, dwell, journey, heatmap_bgr,
        # heatmap_path, ops_alerts). heatmap_bgr is a BGR ndarray ready for
        # gr.Image after channel-flip.
        #
        # This is the cheap path for the rule engine: zone edits and threshold
        # changes re-evaluate every operational rule off the cached facts
        # without touching the GPU.
        result = engine.recompute_analytics(
            video_path, list(zones_state or []),
            camera_placement=camera_placement)
        summary, spikes, dwell, journey, heatmap_bgr, heatmap_path, ops = result
        heatmap_rgb = _bgr_ndarray_to_rgb(heatmap_bgr)
        return (summary, spikes, dwell, journey, heatmap_rgb, heatmap_path, ops)
    except Exception as e:
        gr.Warning(f"Recompute failed: {e}")
        traceback.print_exc()
        return _empty(True)


def invalidate_cache_handler(video_path):
    if video_path:
        engine.invalidate_cache(video_path)
        gr.Info("Detection cache cleared. Next Run Analysis will re-run YOLO.")
    else:
        engine.invalidate_cache()
        gr.Info("All cached trajectories cleared.")


def _zone_summary_html(zones_state):
    """Compact list of currently-defined zones for the Zone Editor sidebar."""
    if not zones_state:
        return ("<div style='color:#94a3b8;padding:10px;font-size:0.9rem;'>"
                "No zones yet. Click on the image to add vertices, then "
                "press <b>Close polygon</b>.</div>")
    rows = ""
    _KIND_TAG_BG = {
        "analytics": "#dbeafe", "wall": "#e2e8f0", "aisle": "#f1f5f9",
        "fixture": "#cffafe", "door": "#dcfce7",
    }
    _KIND_TAG_FG = {
        "analytics": "#1d4ed8", "wall": "#475569", "aisle": "#64748b",
        "fixture": "#0e7490", "door": "#15803d",
    }
    for i, z in enumerate(zones_state):
        b, g, r = z.color
        css = f"rgb({r},{g},{b})"
        kind = getattr(z, "kind", "analytics")
        tag_bg = _KIND_TAG_BG.get(kind, "#e2e8f0")
        tag_fg = _KIND_TAG_FG.get(kind, "#475569")
        # For analytics zones show applies_to; for layout kinds show the kind itself.
        right_tag = z.applies_to if kind == "analytics" else kind
        rows += (
            f"<div style='display:flex;align-items:center;gap:8px;padding:6px 10px;"
            f"border-bottom:1px solid #e2e8f0;'>"
            f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
            f"background:{css};border:1px solid rgba(0,0,0,0.15);'></span>"
            f"<span style='font-weight:700;color:#111827;'>{z.name}</span>"
            f"<span style='background:{tag_bg};color:{tag_fg};font-size:0.7rem;"
            f"font-weight:700;padding:1px 7px;border-radius:8px;text-transform:uppercase;"
            f"letter-spacing:0.4px;'>{right_tag}</span>"
            f"<span style='color:#94a3b8;font-size:0.78rem;margin-left:auto;'>"
            f"{len(z.polygon)} pts</span></div>"
        )
    return (
        f"<div style='border:1px solid #e2e8f0;border-radius:8px;background:#fff;"
        f"font-family:Nunito Sans,sans-serif;font-size:0.92rem;'>"
        f"<div style='padding:8px 12px;background:#f1f5f9;border-bottom:1px solid #e2e8f0;"
        f"font-weight:800;color:#1e3a5f;letter-spacing:0.3px;'>Defined Zones "
        f"({len(zones_state)})</div>"
        f"{rows}</div>"
    )


# ---------------------------------------------------------------------------
# Floor Map handler
# ---------------------------------------------------------------------------
def _wrap_in_iframe(html_doc: str, height_px: int = 700) -> str:
    """Wrap a full HTML document in an iframe via srcdoc so styles don't leak
    into the Gradio parent page. Works with plain gr.HTML — no custom JS."""
    import html as _html
    escaped = _html.escape(html_doc, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{height_px}px;border:none;border-radius:10px;'
        f'background:#0f172a;" allow="autoplay"></iframe>'
    )


def generate_floor_map(video_path, zones_state, camera_name):
    """Build the animated 2D floor-plan HTML for the Floor Map tab."""
    print(f"[floor_map] generate_floor_map called video={video_path} cam={camera_name!r}")
    if not video_path:
        return "<p style='color:#ef4444;padding:20px'>Upload or pick a video first.</p>"
    try:
        from engine.trajectory_cache import make_video_key
        video_key = make_video_key(video_path)
        bundle = engine._trajectory_cache.get(video_key)
        if bundle is None:
            bundle = engine._last_bundle
        if bundle is None:
            return ("<p style='color:#ef4444;padding:20px'>"
                    "No cached trajectory — click <b>Run Analysis</b> first.</p>")
        print(f"[floor_map] bundle loaded: {len(bundle.tracks)} tracks, "
              f"{bundle.total_frames} frames @ {bundle.fps:.1f} fps")

        # Load homography from a camera YAML if the user typed a name.
        # Search both gk-pops-enhanced/configs/cameras/ and
        # retail_store_analytics/configs/cameras/.
        H = None
        if camera_name and camera_name.strip():
            import yaml
            import numpy as np
            from pathlib import Path
            name = camera_name.strip()
            search_paths = [
                Path("configs/cameras") / f"{name}.yaml",
                Path("retail_store_analytics/configs/cameras") / f"{name}.yaml",
            ]
            yaml_path = next((p for p in search_paths if p.exists()), None)
            if yaml_path is None:
                print(f"[floor_map] no calibration YAML for '{name}' — "
                      f"falling back to pixel/100. Searched: {[str(p) for p in search_paths]}")
            else:
                with open(yaml_path) as fh:
                    cam_data = yaml.safe_load(fh) or {}
                if cam_data.get("homography"):
                    H = np.asarray(cam_data["homography"], dtype=np.float64)
                    print(f"[floor_map] loaded homography from {yaml_path}")

        from engine.analytics_builder import run_all
        analytics = run_all(bundle, list(zones_state or []))
        print(f"[floor_map] analytics: {len(analytics.dwell_rows)} dwell rows, "
              f"{len(analytics.queue_spikes)} queue spikes")

        html_doc = build_floor_2d_html(
            bundle, list(zones_state or []), analytics, H=H,
        )
        print(f"[floor_map] built HTML ({len(html_doc):,} chars), wrapping in iframe")
        return _wrap_in_iframe(html_doc, height_px=700)
    except Exception as e:
        traceback.print_exc()
        return f"<p style='color:#ef4444;padding:20px'>Error: {e}</p>"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─────────────────────────────────────────────────────────────────────
   GLOBAL — typography, page background, kill Gradio chrome we don't want
   ───────────────────────────────────────────────────────────────────── */
:root {
    --gk-bg: #f1f5f9;          /* page background */
    --gk-card: #ffffff;        /* panels */
    --gk-sidebar: #ffffff;
    --gk-border: #e2e8f0;
    --gk-border-strong: #cbd5e1;
    --gk-ink: #0f172a;         /* primary text */
    --gk-ink-2: #475569;       /* secondary text */
    --gk-ink-3: #94a3b8;       /* muted text */
    --gk-accent: #2563eb;
    --gk-accent-2: #1d4ed8;
    --gk-success: #059669;
    --gk-success-2: #047857;
    --gk-danger: #dc2626;
    --gk-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 16px rgba(15,23,42,0.05);
    --gk-radius: 10px;
}
/* Dark palette — applied automatically when the OS prefers dark, OR when
   the explicit toggle adds .gk-dark on <html>. */
@media (prefers-color-scheme: dark) {
    :root {
        --gk-bg: #0b1220;
        --gk-card: #161d2e;
        --gk-sidebar: #161d2e;
        --gk-border: #233047;
        --gk-border-strong: #334765;
        --gk-ink: #e2e8f0;
        --gk-ink-2: #94a3b8;
        --gk-ink-3: #64748b;
        --gk-accent: #3b82f6;
        --gk-accent-2: #2563eb;
        --gk-success: #10b981;
        --gk-success-2: #059669;
        --gk-danger: #ef4444;
        --gk-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.4);
    }
}
html.gk-dark,
html.dark,
body.dark {
    --gk-bg: #0b1220;
    --gk-card: #161d2e;
    --gk-sidebar: #161d2e;
    --gk-border: #233047;
    --gk-border-strong: #334765;
    --gk-ink: #e2e8f0;
    --gk-ink-2: #94a3b8;
    --gk-ink-3: #64748b;
    --gk-accent: #3b82f6;
    --gk-accent-2: #2563eb;
    --gk-success: #10b981;
    --gk-success-2: #059669;
    --gk-danger: #ef4444;
    --gk-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.4);
}
/* Explicit "light" override beats the prefers-color-scheme media query. */
html.gk-light {
    --gk-bg: #f1f5f9;
    --gk-card: #ffffff;
    --gk-sidebar: #ffffff;
    --gk-border: #e2e8f0;
    --gk-border-strong: #cbd5e1;
    --gk-ink: #0f172a;
    --gk-ink-2: #475569;
    --gk-ink-3: #94a3b8;
    --gk-accent: #2563eb;
    --gk-accent-2: #1d4ed8;
    --gk-success: #059669;
    --gk-success-2: #047857;
    --gk-danger: #dc2626;
    --gk-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 16px rgba(15,23,42,0.05);
}
/* Form-control re-skin for any dark mode (auto OS, or explicit). */
@media (prefers-color-scheme: dark) {
    .gradio-container input[type="text"],
    .gradio-container input[type="number"],
    .gradio-container input[type="password"],
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container .input-text,
    .gradio-container .gradio-dropdown,
    .gradio-container .wrap > .options {
        background: var(--gk-bg) !important;
        color: var(--gk-ink) !important;
        border-color: var(--gk-border) !important;
    }
    .gradio-container button { color: var(--gk-ink) !important; }
}
html.gk-dark .gradio-container input[type="text"],
html.gk-dark .gradio-container input[type="number"],
html.gk-dark .gradio-container input[type="password"],
html.gk-dark .gradio-container textarea,
html.gk-dark .gradio-container select,
html.gk-dark .gradio-container .input-text,
html.gk-dark .gradio-container .gradio-dropdown,
html.gk-dark .gradio-container .wrap > .options,
body.dark .gradio-container input[type="text"],
body.dark .gradio-container input[type="number"],
body.dark .gradio-container input[type="password"],
body.dark .gradio-container textarea,
body.dark .gradio-container select,
body.dark .gradio-container .input-text,
body.dark .gradio-container .gradio-dropdown,
body.dark .gradio-container .wrap > .options {
    background: var(--gk-bg) !important;
    color: var(--gk-ink) !important;
    border-color: var(--gk-border) !important;
}
html.gk-dark .gradio-container button,
body.dark .gradio-container button { color: var(--gk-ink) !important; }
/* Theme toggle button in the topbar */
.gk-theme-toggle {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    color: #fff; cursor: pointer;
    width: 32px; height: 32px; border-radius: 8px;
    font-size: 15px; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center;
    margin-left: 6px; padding: 0; transition: background 0.15s;
}
.gk-theme-toggle:hover { background: rgba(255,255,255,0.2); }
.gradio-container,
.gradio-container * { font-family: 'Inter','Segoe UI',Tahoma,sans-serif !important; }
.gradio-container { background: var(--gk-bg) !important; max-width:100% !important; width:100% !important; margin:0 auto !important; padding-left:16px !important; padding-right:16px !important; }

/* hide Gradio internal progress text — we have our own UX feedback */
.progress-text,.meta-text,.meta-text-center,.timer,.eta-bar { display:none !important; }
footer { display:none !important; }

/* ─────────────────────────────────────────────────────────────────────
   FORM LABELS — Gradio's default renders a coloured pill on every label.
   Replace with a clean uppercase caption that sits flush above the field.
   ───────────────────────────────────────────────────────────────────── */
.gradio-container label > .label-wrap,
.gradio-container .label-wrap > span,
.gradio-container .form .block-title {
    background: transparent !important;
    color: var(--gk-ink-2) !important;
    text-transform: uppercase !important;
    font-size: 0.66rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    padding: 0 0 4px 0 !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container .label-wrap { background: transparent !important; }

/* ─────────────────────────────────────────────────────────────────────
   INPUTS, SELECTS, TEXTAREAS — calmer borders, focus ring, even radius
   ───────────────────────────────────────────────────────────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input[type="password"],
.gradio-container textarea,
.gradio-container select,
.gradio-container .input-text,
.gradio-container .gradio-dropdown,
.gradio-container .wrap > .options {
    background: #fff !important;
    border: 1px solid var(--gk-border) !important;
    color: var(--gk-ink) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
}
.gradio-container input:focus, .gradio-container textarea:focus,
.gradio-container select:focus, .gradio-container .input-text:focus-within {
    border-color: var(--gk-accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
    outline: none !important;
}

/* ─────────────────────────────────────────────────────────────────────
   BUTTONS — neutral baseline, theme primary/stop separately
   ───────────────────────────────────────────────────────────────────── */
.gradio-container button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    transition: all 0.12s ease !important;
}
.gradio-container button.primary,
.gradio-container .gradio-button.primary,
.gradio-container button[variant="primary"] {
    background: var(--gk-accent) !important;
    color: #fff !important; border-color: var(--gk-accent) !important;
}
.gradio-container button.primary:hover { background: var(--gk-accent-2) !important; }
.gradio-container button.stop,
.gradio-container button[variant="stop"] {
    background: #fef2f2 !important; color: var(--gk-danger) !important;
    border: 1px solid #fecaca !important;
}
.gradio-container button.stop:hover { background:#fee2e2 !important; }

/* ─────────────────────────────────────────────────────────────────────
   TOP BAR
   ───────────────────────────────────────────────────────────────────── */
.gk-topbar {
    background: linear-gradient(135deg,#0b1733 0%,#162447 60%,#1e3a8a 100%);
    border-radius: var(--gk-radius); margin-bottom: 14px; padding: 14px 22px;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow: 0 4px 18px rgba(15,23,42,0.18);
}
.gk-topbar-left  { display:flex; align-items:center; gap:14px; }
.gk-logo-mark {
    width:40px; height:40px;
    background: linear-gradient(135deg,#3b82f6,#1d4ed8);
    border:1px solid rgba(255,255,255,0.18); border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.05rem; font-weight:800; color:#fff; flex-shrink:0;
    letter-spacing:-0.5px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.2);
}
.gk-brand-super  { font-size:0.6rem; font-weight:700; letter-spacing:2.4px; color:#93c5fd; text-transform:uppercase; }
.gk-brand-title  { font-size:1.05rem; font-weight:700; color:#fff; line-height:1.2; margin:2px 0; }
.gk-brand-sub    { font-size:0.72rem; color:#bfdbfe; opacity:0.85; }
.gk-topbar-chips { display:flex; gap:6px; flex-wrap:wrap; max-width:560px; justify-content:flex-end; }
.gk-chip {
    background:rgba(255,255,255,0.07); color:#cbd5e1; padding:3px 10px;
    border-radius:99px; font-size:0.66rem; font-weight:600; letter-spacing:0.3px;
    border:1px solid rgba(255,255,255,0.10);
}

/* ─────────────────────────────────────────────────────────────────────
   TWO-COLUMN LAYOUT
   ───────────────────────────────────────────────────────────────────── */
#gk-main-row { gap:14px !important; align-items:stretch !important; }
#gk-sidebar {
    flex: 0 0 296px !important;
    min-width: 296px !important;
    max-width: 296px !important;
    background: var(--gk-sidebar) !important;
    border: 1px solid var(--gk-border) !important;
    border-radius: var(--gk-radius) !important;
    padding: 0 !important;
    gap: 0 !important;
    overflow-y: auto !important;
    max-height: calc(100vh - 90px);
    position: sticky;
    top: 10px;
    align-self: flex-start;
    box-shadow: var(--gk-shadow);
}
#gk-sidebar > .form, #gk-sidebar > div { padding:0 !important; gap:0 !important; }
#gk-main {
    flex:1 !important; min-width:0 !important; padding:0 !important;
}

/* sidebar scrollbar */
#gk-sidebar::-webkit-scrollbar { width:6px; }
#gk-sidebar::-webkit-scrollbar-track { background: transparent; }
#gk-sidebar::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }

/* ─────────────────────────────────────────────────────────────────────
   SIDEBAR — section headers, padding, dividers, footer
   ───────────────────────────────────────────────────────────────────── */
.sb-hdr {
    background: transparent;
    color: var(--gk-ink-2); padding: 14px 16px 6px;
    font-size: 0.65rem; font-weight: 800; letter-spacing: 1.6px; text-transform: uppercase;
    display: flex; align-items: center; gap: 8px; user-select: none;
    border-top: 1px solid var(--gk-border);
}
.sb-hdr:first-of-type, .sb-hdr.sb-first { border-top: none; }
.sb-hdr-step {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px;
    background: var(--gk-accent); color:#fff;
    border-radius: 999px; font-size: 0.65rem; font-weight: 800; flex-shrink: 0;
    letter-spacing: 0;
}
.sb-pad { padding: 4px 16px 14px; }
.sb-div { height: 1px; background: var(--gk-border); margin: 4px 0; }
.sb-foot {
    text-align: center; color: var(--gk-ink-3); font-size: 0.68rem;
    padding: 12px 16px; border-top: 1px solid var(--gk-border);
    margin-top: 4px;
}
.sb-foot strong { color: var(--gk-accent); font-weight: 700; }

/* sidebar primary run button — green = "go" */
#gk-run-btn button {
    background: var(--gk-success) !important;
    border-color: transparent !important;
    color:#fff !important; font-size: 0.92rem !important; font-weight: 700 !important;
    height: 46px !important; letter-spacing: 0.2px !important;
    box-shadow: 0 2px 6px rgba(5,150,105,0.20) !important;
    border-radius: 9px !important;
}
#gk-run-btn button:hover {
    background: var(--gk-success-2) !important;
    box-shadow: 0 4px 10px rgba(5,150,105,0.28) !important;
}

/* compact secondary buttons inside sidebar */
#gk-sidebar .gradio-button.secondary,
#gk-sidebar button[variant="secondary"] {
    background:#f8fafc !important; color:var(--gk-ink-2) !important;
    border:1px solid var(--gk-border) !important;
    font-size:0.74rem !important; font-weight:600 !important;
    padding:6px 10px !important;
}
#gk-sidebar .gradio-button.secondary:hover { background:#f1f5f9 !important; }

/* sidebar download chips */
.sb-downloads { padding: 4px 16px 12px !important; }
.sb-downloads .file-preview, .sb-downloads .gradio-file { padding:6px 10px !important; }
.sb-downloads .label-wrap span, .sb-downloads .label-wrap {
    font-size:0.66rem !important; font-weight:700 !important;
    color:var(--gk-ink-2) !important; letter-spacing:0.6px !important;
}
.sb-downloads .form, .sb-downloads > div { gap:8px !important; }

/* ─────────────────────────────────────────────────────────────────────
   TABS — calmer, shorter, scroll on small widths instead of wrapping ugly
   ───────────────────────────────────────────────────────────────────── */
#gk-tabs .tab-nav, .tab-nav {
    border-bottom: 1px solid var(--gk-border) !important;
    gap: 0 !important;
    background: transparent !important;
    padding: 0 4px !important;
    flex-wrap: wrap !important;
    margin-bottom: 10px !important;
}
#gk-tabs .tab-nav button, .tab-nav button {
    font-size: 0.78rem !important;
    padding: 8px 14px !important;
    font-weight: 600 !important;
    color: var(--gk-ink-2) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    opacity: 1 !important;
    transition: all 0.12s ease !important;
}
#gk-tabs .tab-nav button:hover, .tab-nav button:hover {
    color: var(--gk-accent) !important;
    background: rgba(37,99,235,0.04) !important;
}
#gk-tabs .tab-nav button.selected, .tab-nav button.selected {
    color: var(--gk-accent) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--gk-accent) !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

/* main-content card around tab body */
#gk-main > .form, #gk-main > div { gap:14px !important; }

/* video output panel */
#gk-main video, #gk-main .gradio-video {
    border-radius: var(--gk-radius) !important;
    overflow: hidden !important;
    box-shadow: var(--gk-shadow);
    background: #0f172a !important;
}

/* generic group cards */
.gradio-container .gradio-group {
    background: var(--gk-card);
    border: 1px solid var(--gk-border);
    border-radius: var(--gk-radius);
    box-shadow: var(--gk-shadow);
}

/* accordion polish */
.gradio-container .gradio-accordion {
    border: 1px solid var(--gk-border) !important;
    border-radius: var(--gk-radius) !important;
    background: var(--gk-card) !important;
}
.gradio-container .gradio-accordion > .label-wrap {
    padding: 10px 14px !important;
    font-size: 0.78rem !important; font-weight: 700 !important;
    color: var(--gk-ink) !important;
    background: #f8fafc !important;
    border-bottom: 1px solid var(--gk-border) !important;
    letter-spacing: 0.2px !important;
    text-transform: none !important;
}

/* ─────────────────────────────────────────────────────────────────────
   ZONE EDITOR — header, fullscreen, toolbar
   ───────────────────────────────────────────────────────────────────── */
.zone-hdr-bar {
    display:flex; align-items:center; justify-content:space-between;
    background: linear-gradient(135deg,#0b1733,#1e3a8a);
    border-radius: var(--gk-radius) var(--gk-radius) 0 0;
    padding: 9px 16px; color:#dbeafe; font-size:0.76rem; font-weight:600;
}
.zone-hdr-bar span { opacity:0.9; }
.gk-fs-btn {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    color:#fff; border-radius:7px; padding:5px 12px;
    font-size:0.72rem; font-weight:600; cursor:pointer;
    font-family:'Inter',sans-serif; white-space:nowrap;
    transition: all 0.12s;
}
.gk-fs-btn:hover  { background: rgba(255,255,255,0.18) !important; }
.gk-fs-btn.exiting { background: rgba(220,38,38,0.85) !important; border-color: rgba(220,38,38,0.5) !important; }

#zone-editor-fs-wrap.gk-fs-active {
    position:fixed !important; inset:0 !important;
    z-index:9990 !important; background:#020617 !important;
    display:flex !important; flex-direction:column !important;
    padding:0 !important; border-radius:0 !important; gap:0 !important;
}
#zone-editor-fs-wrap.gk-fs-active .zone-canvas-area { flex:1; min-height:0; }
#zone-editor-fs-wrap.gk-fs-active .zone-canvas-area .gradio-image { height:calc(100vh - 94px) !important; }
#zone-editor-fs-wrap.gk-fs-active .zone-canvas-area img { max-height:calc(100vh - 94px) !important; object-fit:contain; }
#zone-editor-fs-wrap.gk-fs-active .zone-hdr-bar { border-radius:0; }
#zone-editor-fs-wrap.gk-fs-active .zone-toolbar-row,
#zone-editor-fs-wrap.gk-fs-active .zone-toolbar-row-secondary {
    background:#0f172a !important; border-top:1px solid #1e293b !important;
    border-radius:0 !important; flex-shrink:0;
}
#zone-editor-fs-wrap.gk-fs-active .zone-toolbar-row label,
#zone-editor-fs-wrap.gk-fs-active .zone-toolbar-row .label-wrap span { color:#94a3b8 !important; }
#zone-editor-fs-wrap.gk-fs-active .zone-toolbar-row input { color:#e2e8f0 !important; }
body.gk-zone-fs { overflow:hidden !important; }

.zone-toolbar-row {
    background: var(--gk-card); border: 1px solid var(--gk-border); border-top: none;
    border-radius: 0; padding: 10px 16px;
    display: grid !important;
    grid-template-columns: minmax(0,1fr) minmax(0,1fr) minmax(0,1fr) auto;
    align-items: center !important;
    gap: 14px !important;
}
/* Each [label · input] pair sits horizontally on a single line. */
.zone-toolbar-row .zone-field-pair {
    display: flex !important; flex-direction: row !important;
    align-items: center !important; gap: 10px !important;
    min-width: 0 !important;
    background: transparent !important; border: none !important;
    padding: 0 !important; margin: 0 !important;
}
.zone-toolbar-row .zone-field-pair > * { min-width: 0 !important; }
.zone-toolbar-row .zone-field-pair .zone-field-label {
    flex: 0 0 auto !important;
    font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--gk-ink-2); white-space: nowrap;
}
/* The form-component div fills the remaining space in the pair. */
.zone-toolbar-row .zone-field-pair > .gradio-textbox,
.zone-toolbar-row .zone-field-pair > .gradio-dropdown,
.zone-toolbar-row .zone-field-pair > .form,
.zone-toolbar-row .zone-field-pair > div:not(:first-child) {
    flex: 1 1 auto !important; min-width: 0 !important;
}
.zone-name-box, .zone-applies-dd { min-width: 0 !important; max-width: none !important; flex: none !important; }
.zone-toolbar-row .zone-save-btn { margin-left: 0 !important; flex: none !important; }
.zone-toolbar-row .zone-save-btn button {
    height: 38px !important; min-width: 120px !important;
    padding: 0 16px !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    border-radius: 8px !important;
    background: var(--gk-accent) !important;
    color:#fff !important; border-color: var(--gk-accent) !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.25) !important;
}
.zone-toolbar-row .zone-save-btn button:hover { background: var(--gk-accent-2) !important; }

/* Toolbar row 2 — Undo / Clear current / Clear all (equal width) */
.zone-toolbar-row-secondary {
    background: var(--gk-card); border: 1px solid var(--gk-border); border-top: none;
    border-radius: 0 0 var(--gk-radius) var(--gk-radius);
    padding: 8px 16px 12px;
    display:flex !important; gap: 12px !important; flex-wrap: nowrap !important;
}
.zone-toolbar-row-secondary > * { flex:1 1 0 !important; min-width:0 !important; }
.zone-toolbar-row-secondary button {
    width:100% !important; height: 38px !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ─────────────────────────────────────────────────────────────────────
   JSON code block (lives inside an Accordion now, rarely opened)
   ───────────────────────────────────────────────────────────────────── */
.json-scroll { max-height: 320px; overflow-y: auto; }
.json-scroll textarea { min-height: 240px !important; }

/* ─────────────────────────────────────────────────────────────────────
   STICKY ALERT BANNER
   ───────────────────────────────────────────────────────────────────── */
#gk-alert-host {
    position: sticky !important; top: 8px !important; z-index: 40 !important;
    padding: 0 !important; background: transparent !important; border: none !important;
}
#gk-alert-host > .form, #gk-alert-host > div { padding: 0 !important; }
.gk-alert-banner { animation: gkAlertSlide 0.45s ease-out; backdrop-filter: blur(2px); }
.gk-alert-banner .gk-alert-pulse {
    box-shadow: 0 0 0 0 rgba(255,255,255,0.7);
    animation: gkAlertPulse 1.6s infinite;
}
@keyframes gkAlertPulse {
    0%   { box-shadow: 0 0 0 0   rgba(255,255,255,0.7); }
    70%  { box-shadow: 0 0 0 12px rgba(255,255,255,0); }
    100% { box-shadow: 0 0 0 0   rgba(255,255,255,0); }
}
@keyframes gkAlertSlide {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─────────────────────────────────────────────────────────────────────
   Hide Gradio 6.8's built-in fullscreen / icon buttons inside the zone
   editor and the tracked-output video. They throw
       TypeError: i.onclick is not a function
   from FullscreenButton-DrvnVVOX.js (a Gradio bundle bug), and we provide
   our own working fullscreen toggle (#gk-zone-fs-btn).
   The selectors are intentionally broad — Gradio names these icons
   inconsistently across components — but we keep #gk-zone-fs-btn alive
   with an explicit allow-rule below.
   ───────────────────────────────────────────────────────────────────── */
#zone-canvas-img button[aria-label*="ull" i],
#zone-canvas-img button[title*="ull" i],
#zone-canvas-img [class*="fullscreen" i],
#zone-canvas-img .icon-buttons,
#zone-editor-fs-wrap .gradio-image button[aria-label*="ull" i],
#zone-editor-fs-wrap .gradio-image button[title*="ull" i],
#zone-editor-fs-wrap .gradio-image [class*="fullscreen" i] {
    display: none !important;
}
/* keep our own fullscreen button visible regardless of the rules above */
#gk-zone-fs-btn { display: inline-flex !important; }

/* ─────────────────────────────────────────────────────────────────────
   JSON FULLSCREEN OVERLAY (kept from before)
   ───────────────────────────────────────────────────────────────────── */

/* ── json fullscreen overlay ──────────────────────────────────────────── */
#json-fullscreen-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,0.88); z-index:10000; padding:28px; box-sizing:border-box;
}
#json-fullscreen-overlay.active { display:flex; flex-direction:column; }
#json-fullscreen-close {
    align-self:flex-end; background:#ef4444; color:#fff; border:none;
    padding:7px 18px; border-radius:6px; font-size:0.88rem; font-weight:700;
    cursor:pointer; margin-bottom:10px;
}
#json-fullscreen-close:hover { background:#dc2626; }
#json-fullscreen-content {
    flex:1; background:#0d1117; color:#c9d1d9; border-radius:8px;
    padding:20px; overflow:auto; font-family:monospace; font-size:0.85rem;
    white-space:pre; line-height:1.5;
}
"""

_ZONE_FS_JS = """
() => {
    // The previous error  i.onclick is not a function  came from Gradio's
    // own ScrollFade / FullscreenButton wrappers, which call
    //   button.onclick()
    // on every clicked button as part of their internal routing. Inline
    // onclick="..." attributes are stripped by Gradio 6's HTML sanitizer,
    // so the property is null and the call throws.
    //
    // Fix: assign a real function to btn.onclick (the DOM property — which
    // sanitization doesn't touch, only the HTML attribute does). Use a
    // MutationObserver to bind as soon as each button appears.

    function _toggleZoneFs(ev) {
        if (ev && ev.preventDefault) ev.preventDefault();
        if (ev && ev.stopPropagation) ev.stopPropagation();
        var wrap = document.getElementById('zone-editor-fs-wrap');
        var btn  = document.getElementById('gk-zone-fs-btn');
        console.log('[gk] toggleZoneFs fired, wrap=', !!wrap);
        if (!wrap) {
            console.warn('[gk] zone-editor-fs-wrap not found in DOM');
            return false;
        }
        var active = wrap.classList.toggle('gk-fs-active');
        document.body.classList.toggle('gk-zone-fs', active);
        if (btn) {
            btn.innerHTML = active
                ? '\\u2612\\uFE0E&nbsp; Exit fullscreen'
                : '\\u26F6\\uFE0E&nbsp; Fullscreen';
            btn.classList.toggle('exiting', active);
        }
        window.dispatchEvent(new Event('resize'));
        return false;
    }

    function _effectiveTheme() {
        var root = document.documentElement;
        var body = document.body;
        if (root.classList.contains('gk-dark') || root.classList.contains('dark')
            || (body && body.classList.contains('dark'))) return 'dark';
        if (root.classList.contains('gk-light')) return 'light';
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    function _applyTheme(t) {
        var root = document.documentElement;
        var body = document.body;
        root.classList.remove('gk-dark', 'gk-light', 'dark');
        if (body) body.classList.remove('dark');
        if (t === 'dark') {
            root.classList.add('gk-dark', 'dark');
            if (body) body.classList.add('dark');
        } else if (t === 'light') {
            root.classList.add('gk-light');
        }
    }
    function _syncThemeBtn() {
        var btn = document.getElementById('gk-theme-toggle');
        if (!btn) return;
        var cur = _effectiveTheme();
        btn.textContent = cur === 'dark' ? '\\u2600' : '\\u{1F319}';
        btn.title = cur === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
    function _toggleTheme() {
        var next = _effectiveTheme() === 'dark' ? 'light' : 'dark';
        _applyTheme(next);
        try { localStorage.setItem('gk-theme', next); } catch (e) {}
        _syncThemeBtn();
        return false;
    }
    function _closeJsonOverlay() {
        var ov = document.getElementById('json-fullscreen-overlay');
        if (ov) ov.classList.remove('active');
        return false;
    }

    // Map of element id → handler to assign to .onclick once it mounts.
    var BINDINGS = {
        'gk-zone-fs-btn':       _toggleZoneFs,
        'gk-theme-toggle':      _toggleTheme,
        'json-fullscreen-close': _closeJsonOverlay
    };

    function _bindAll() {
        for (var id in BINDINGS) {
            var el = document.getElementById(id);
            if (!el) continue;
            // Always (re)assign .onclick — Gradio's Svelte router calls
            //   element.onclick()
            // directly on the matched button, so the property MUST be set.
            if (el.onclick !== BINDINGS[id]) {
                el.onclick = BINDINGS[id];
            }
            // Also addEventListener so the click works regardless of
            // whether Svelte's router reaches the .onclick path. Guard
            // against double-binding with a dataset flag.
            if (!el.dataset.gkBound) {
                el.addEventListener('click', BINDINGS[id]);
                el.dataset.gkBound = '1';
                console.log('[gk] bound handler to #' + id);
            }
        }
        // Defensive: Gradio 6.8's FullscreenButton-DrvnVVOX.js calls
        //   button.onclick()
        // on its rendered icon, but never sets one — so it throws
        //   TypeError: i.onclick is not a function
        // and breaks the click pipeline for the entire page. Give every
        // such icon a no-op so the call succeeds silently.
        var icons = document.querySelectorAll(
            'button[aria-label*="ull" i], button[title*="ull" i], ' +
            'button[class*="fullscreen" i]'
        );
        for (var i = 0; i < icons.length; i++) {
            var b = icons[i];
            if (b.id === 'gk-zone-fs-btn') continue;   // ours — keep alive
            if (typeof b.onclick !== 'function') {
                b.onclick = function() { return false; };
            }
        }
    }

    // Try once now (in case buttons are already in the DOM), then keep watching.
    _bindAll();
    if (!window.__gkBindingObserver) {
        var obs = new MutationObserver(_bindAll);
        var start = function() {
            obs.observe(document.body, { childList: true, subtree: true });
            window.__gkBindingObserver = obs;
            _bindAll();   // catch anything added between page-load and observer start
            _syncThemeBtn();
            console.log('[gk] button binding observer installed for', Object.keys(BINDINGS));
        };
        if (document.body) start();
        else document.addEventListener('DOMContentLoaded', start);
    }

    // Restore saved theme on first load.
    try {
        var saved = localStorage.getItem('gk-theme');
        if (saved === 'dark' || saved === 'light') {
            if (document.body) {
                _applyTheme(saved);
            } else {
                document.documentElement.classList.add('gk-' + saved);
                document.addEventListener('DOMContentLoaded', function() {
                    _applyTheme(saved);
                });
            }
        }
    } catch (e) {}

    // Back-compat aliases on window in case anything still references them.
    window.gkToggleZoneFullscreen = _toggleZoneFs;
    window.gkToggleTheme           = _toggleTheme;
    window.gkSyncThemeBtn          = _syncThemeBtn;
    window.gkEffectiveTheme        = _effectiveTheme;
}
"""

with gr.Blocks(
    title="POPS — Push-Out Probability Score | Gatekeeper AI",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    css=_CSS,
    js=_ZONE_FS_JS,
) as demo:

    # ── persistent state ────────────────────────────────────────────────
    first_frame_state = gr.State(value=None)
    zones_state       = gr.State(value=[])
    current_poly      = gr.State(value=[])
    video_key_state   = gr.State(value=None)

    # ── top bar ─────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="gk-topbar">
      <div class="gk-topbar-left">
        <div class="gk-logo-mark">GK</div>
        <div>
          <div class="gk-brand-super">Gatekeeper AI</div>
          <div class="gk-brand-title">POPS &mdash; Push-Out Probability Score</div>
          <div class="gk-brand-sub">AI-Powered Retail Loss Prevention &mdash; Tracking + Classification + Scoring</div>
        </div>
      </div>
      <div class="gk-topbar-chips">
        <span class="gk-chip">YOLOv26m + BoTSORT</span>
        <span class="gk-chip">POPS Score</span>
        <span class="gk-chip">Retail Analytics</span>
        <button type="button" class="gk-theme-toggle" id="gk-theme-toggle"
                title="Toggle dark mode">🌙</button>
      </div>
    </div>
    """)

    # ── two-column layout ────────────────────────────────────────────────
    with gr.Row(elem_id="gk-main-row"):

        # ════════════════════════════════════════════════════════════════
        #  LEFT SIDEBAR
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=0, min_width=282, elem_id="gk-sidebar"):

            # ── Step 1: Video ────────────────────────────────────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">1</span>VIDEO SOURCE</div>')
            with gr.Group(elem_classes=["sb-pad"]):
                video_input = gr.Video(
                    label="Upload video", sources=["upload"], height=160,
                )
                if SAMPLE_VIDEOS:
                    sample_names = [os.path.basename(v) for v in SAMPLE_VIDEOS]
                    sample_dropdown = gr.Dropdown(
                        choices=list(zip(sample_names, SAMPLE_VIDEOS)),
                        label=f"Or pick a sample clip ({len(SAMPLE_VIDEOS)} available)",
                    )
                    sample_dropdown.change(
                        fn=lambda x: x, inputs=[sample_dropdown], outputs=[video_input])

            # ── Step 2: Camera ───────────────────────────────────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">2</span>CAMERA PLACEMENT</div>')
            with gr.Group(elem_classes=["sb-pad"]):
                camera_placement = gr.Dropdown(
                    choices=[
                        "Outside (facing entrance)",
                        "Inside (facing exit)",
                        "Inside (exit on right)",
                        "Inside (exit on left)",
                        "Inside (exit on both sides)",
                    ],
                    value="Outside (facing entrance)",
                    label="Camera angle / orientation",
                )

            # ── VLM Case Report Settings (always visible) ───────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">⚙</span>VLM CASE REPORT</div>')
            with gr.Group(elem_classes=["sb-pad"]):
                vlm_backend = gr.Dropdown(
                    choices=VLM_BACKENDS, value=VLM_DEFAULT_BACKEND,
                    label="VLM Backend",
                )
                vlm_api_key = gr.Textbox(
                    label="API Key (optional)",
                    placeholder="Leave blank to use OAuth token",
                    type="password",
                )
                vlm_backend.change(
                    fn=lambda b: gr.update(visible="Claude" in b),
                    inputs=[vlm_backend], outputs=[vlm_api_key],
                )

            # ── Step 4: Zones ────────────────────────────────────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">3</span>ZONES</div>')
            with gr.Group(elem_classes=["sb-pad"]):
                zones_summary_html = gr.HTML(_zone_summary_html([]))
                delete_zone_dd = gr.Dropdown(
                    choices=[], label="Remove a zone", interactive=True,
                )

            gr.HTML('<div class="sb-div"></div>')

            # ── Run button ───────────────────────────────────────────────
            with gr.Group(elem_classes=["sb-pad"]):
                run_btn = gr.Button(
                    "Run Analysis", variant="primary", size="lg", elem_id="gk-run-btn",
                )
                with gr.Row():
                    invalidate_btn = gr.Button(
                        "Re-run detection", size="sm", variant="secondary",
                    )
                    recompute_btn = gr.Button(
                        "Recompute analytics", size="sm", variant="secondary",
                    )

            # ── Downloads ────────────────────────────────────────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">↓</span>DOWNLOADS</div>')
            with gr.Group(elem_classes=["sb-pad", "sb-downloads"]):
                json_download = gr.File(label="Tracking JSON")
                case_report_download = gr.File(label="Case Report")

            gr.HTML('<div class="sb-foot">Built by <strong>Tanmay Thaker</strong> &mdash; Gatekeeper Systems</div>')

        # ════════════════════════════════════════════════════════════════
        #  MAIN CONTENT
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, elem_id="gk-main"):

            # Sticky top-of-page alert banner — surfaces high-priority POPS
            # events and severe zone congestion as soon as analysis finishes.
            # Hidden until run_analysis populates it.
            alert_banner_html = gr.HTML(
                value="", visible=False, elem_id="gk-alert-host",
            )

            # Tracked video output — large, at the top
            video_output = gr.Video(
                label="Tracked Output",
                autoplay=True, height=520,
            )

            # ── Result tabs ──────────────────────────────────────────────
            with gr.Tabs(elem_id="gk-tabs"):

                # ── Zone Editor ─────────────────────────────────────────
                with gr.Tab("Zone Editor"):
                    with gr.Group(elem_id="zone-editor-fs-wrap"):

                        # Header bar with fullscreen toggle. The click is bound
                        # via document-level event delegation in _ZONE_FS_JS —
                        # Gradio 6 strips inline onclick="..." attrs as XSS risk.
                        gr.HTML("""
                        <div class="zone-hdr-bar">
                          <span>Click on the frame to place polygon vertices, then Save Zone.</span>
                          <button id="gk-zone-fs-btn" type="button" class="gk-fs-btn">&#x26F6;&#xFE0E;&nbsp; Fullscreen</button>
                        </div>
                        """)

                        # Canvas — full width
                        with gr.Group(elem_classes=["zone-canvas-area"]):
                            zone_canvas = gr.Image(
                                label=None, show_label=False,
                                interactive=False, height=460,
                                elem_id="zone-canvas-img",
                            )

                        # Toolbar row 1 — each [label · input] pair on one line
                        with gr.Row(elem_classes=["zone-toolbar-row"]):
                            with gr.Column(elem_classes=["zone-field-pair"]):
                                gr.HTML('<span class="zone-field-label">Zone name</span>')
                                zone_name_in = gr.Textbox(
                                    show_label=False, value="Aisle 1",
                                    placeholder="e.g. Checkout, Entrance...",
                                    elem_classes=["zone-name-box"],
                                )
                            with gr.Column(elem_classes=["zone-field-pair"]):
                                gr.HTML('<span class="zone-field-label">Layout type</span>')
                                zone_kind = gr.Dropdown(
                                    choices=ZONE_KIND_OPTIONS, value="analytics",
                                    show_label=False, elem_classes=["zone-applies-dd"],
                                )
                            with gr.Column(elem_classes=["zone-field-pair"]):
                                gr.HTML('<span class="zone-field-label">Track type</span>')
                                zone_applies = gr.Dropdown(
                                    choices=ZONE_APPLIES_OPTIONS, value="person",
                                    show_label=False, elem_classes=["zone-applies-dd"],
                                )
                            close_btn = gr.Button(
                                "✓ Save zone", variant="primary", size="sm",
                                elem_classes=["zone-save-btn"],
                            )

                        # Toolbar row 2 — Undo / Clear current / Clear all
                        with gr.Row(elem_classes=["zone-toolbar-row-secondary"]):
                            undo_btn      = gr.Button("Undo", size="sm")
                            clear_btn     = gr.Button("Clear current", size="sm")
                            clear_all_btn = gr.Button("Clear all", size="sm",
                                                      variant="stop")

                # ── POPS ────────────────────────────────────────────────
                with gr.Tab("POPS"):
                    pops_html = gr.HTML("")

                # ── Events ──────────────────────────────────────────────
                with gr.Tab("Events"):
                    events_html = gr.HTML("")

                # ── Operational Alerts (rule engine) ────────────────────
                # Separate tab from Events on purpose: these are rule-engine
                # outcomes with their own severity vocabulary, not POPS
                # theft-risk events, and they must not share the event log.
                with gr.Tab("Operational Alerts"):
                    gr.HTML(
                        "<div style='font-family:\"Nunito Sans\",sans-serif;"
                        "font-size:0.85rem;color:#64748b;padding:4px 8px 10px;'>"
                        "Blocked doors, static carts, unattended carts and "
                        "incoming empty carts. Thresholds are configurable in "
                        "<code>engine/config.py</code> — edit a zone and hit "
                        "<b>Recompute Analytics</b> to re-evaluate without "
                        "re-running detection.</div>"
                    )
                    ops_alerts_html = gr.HTML("")

                # ── Analytics ───────────────────────────────────────────
                with gr.Tab("Analytics"):
                    analytics_summary_html = gr.HTML(
                        analytics_ui.build_analytics_empty_state(has_video=False))
                    spikes_html = gr.HTML("")
                    with gr.Row():
                        with gr.Column(scale=2):
                            heatmap_image = gr.Image(
                                label="Traffic Heatmap", interactive=False, height=360,
                            )
                        with gr.Column(scale=1, min_width=160):
                            heatmap_file = gr.File(label="Download Heatmap PNG")
                    dwell_html   = gr.HTML("")
                    journey_html = gr.HTML("")

                # ── 3D View ─────────────────────────────────────────────
                # Plain gr.HTML — run_analysis wraps the engine's full HTML
                # document in an <iframe srcdoc="..."> so each new run forces
                # the browser to reload the iframe content.
                with gr.Tab("3D View"):
                    bev3d_html = gr.HTML(
                        value=(
                            "<div style='width:100%;height:620px;background:#0f172a;"
                            "border-radius:10px;display:flex;align-items:center;"
                            "justify-content:center;color:#64748b;'>"
                            "Run analysis to see the 3D view."
                            "</div>"
                        ),
                    )

                # ── Bird's-Eye 2D ───────────────────────────────────────
                with gr.Tab("Bird's-Eye 2D"):
                    bev2d_html = gr.HTML(
                        value=(
                            "<div style='width:100%;height:620px;background:#0f172a;"
                            "border-radius:10px;display:flex;align-items:center;"
                            "justify-content:center;color:#64748b;'>"
                            "Run analysis to see the 2D bird's-eye view."
                            "</div>"
                        ),
                    )

                # ── Floor Map ───────────────────────────────────────────
                with gr.Tab("🗺 Floor Map"):
                    with gr.Row():
                        floor_camera_input = gr.Textbox(
                            label="Camera name (for calibration lookup)",
                            placeholder="e.g. cam-test  (optional)",
                            value="",
                            scale=2,
                        )
                        floor_gen_btn = gr.Button(
                            "Generate Floor Map", variant="primary", scale=1,
                        )
                    floor_map_html = gr.HTML(
                        value=(
                            "<div style='color:#94a3b8;padding:40px;text-align:center;"
                            "background:#0f172a;border-radius:10px;height:700px;"
                            "display:flex;align-items:center;justify-content:center;'>"
                            "Run analysis then click <b>Generate Floor Map</b>."
                            "</div>"
                        ),
                    )

                # ── Case Report ─────────────────────────────────────────
                with gr.Tab("Case Report"):
                    case_report_html = gr.HTML("")

                # ── Detection ───────────────────────────────────────────
                with gr.Tab("Detection"):
                    detection_html = gr.HTML("")

                # ── Video Info ──────────────────────────────────────────
                with gr.Tab("Video Info"):
                    video_info_html = gr.HTML("")

                # ── Config ──────────────────────────────────────────────
                with gr.Tab("Config"):
                    config_html = gr.HTML("")

                # ── Legend ──────────────────────────────────────────────
                with gr.Tab("Legend"):
                    legend_html = gr.HTML("")

            # ── JSON output (collapsed by default — rarely opened) ───────
            with gr.Accordion("Raw tracking JSON (advanced)", open=False):
                json_output = gr.Code(
                    label="Tracking + Classification JSON",
                    language="json", lines=10, elem_classes=["json-scroll"],
                )
                fullscreen_btn = gr.Button(
                    "View JSON Fullscreen", variant="secondary", size="sm",
                )

    # ── JSON fullscreen overlay ──────────────────────────────────────────
    gr.HTML("""
    <div id="json-fullscreen-overlay">
      <button id="json-fullscreen-close" type="button">
        Close
      </button>
      <div id="json-fullscreen-content"></div>
    </div>
    """)

    # ════════════════════════════════════════════════════════════════════
    #  EVENT WIRING  (handlers unchanged)
    # ════════════════════════════════════════════════════════════════════
    fullscreen_btn.click(
        fn=None, inputs=[json_output], outputs=[],
        js="""(t) => {
            const o = document.getElementById('json-fullscreen-overlay');
            const c = document.getElementById('json-fullscreen-content');
            if (o && c) { c.textContent = t || 'No JSON yet.'; o.classList.add('active'); }
        }""",
    )

    video_input.change(
        fn=on_video_upload,
        inputs=[video_input, zones_state, video_key_state],
        outputs=[first_frame_state, zones_state, current_poly,
                 video_key_state, zone_canvas, zones_summary_html],
    )

    zone_canvas.select(
        fn=on_canvas_click,
        inputs=[current_poly, first_frame_state, zones_state],
        outputs=[current_poly, zone_canvas],
    )

    close_btn.click(
        fn=close_polygon,
        inputs=[current_poly, zone_name_in, zone_applies, zone_kind, zones_state, first_frame_state],
        outputs=[zones_state, current_poly, zone_canvas, zones_summary_html, delete_zone_dd],
    )

    undo_btn.click(
        fn=undo_vertex,
        inputs=[current_poly, first_frame_state, zones_state],
        outputs=[current_poly, zone_canvas],
    )

    clear_btn.click(
        fn=clear_inprogress,
        inputs=[first_frame_state, zones_state],
        outputs=[current_poly, zone_canvas],
    )

    clear_all_btn.click(
        fn=clear_all_zones,
        inputs=[first_frame_state],
        outputs=[zones_state, current_poly, zone_canvas, zones_summary_html, delete_zone_dd],
    )

    delete_zone_dd.select(
        fn=delete_zone,
        inputs=[delete_zone_dd, zones_state, first_frame_state, current_poly],
        outputs=[zones_state, zone_canvas, zones_summary_html, delete_zone_dd],
    )

    invalidate_btn.click(
        fn=invalidate_cache_handler, inputs=[video_input], outputs=[],
    )

    run_btn.click(
        fn=run_analysis,
        inputs=[video_input, camera_placement, vlm_backend, vlm_api_key,
                zones_state],
        outputs=[
            video_output, json_download, json_output,
            video_info_html, detection_html, config_html, legend_html,
            pops_html, events_html, bev3d_html, bev2d_html,
            case_report_html, case_report_download,
            analytics_summary_html, spikes_html, dwell_html, journey_html,
            heatmap_image, heatmap_file,
            alert_banner_html, ops_alerts_html,
        ],
    )

    recompute_btn.click(
        fn=recompute_analytics_handler,
        inputs=[video_input, zones_state, camera_placement],
        outputs=[analytics_summary_html, spikes_html, dwell_html, journey_html,
                 heatmap_image, heatmap_file, ops_alerts_html],
    )

    floor_gen_btn.click(
        fn=generate_floor_map,
        inputs=[video_input, zones_state, floor_camera_input],
        outputs=[floor_map_html],
    )


if __name__ == "__main__":
    # `analytics_out_dir` in tracker.py defaults to <project_root>/temp —
    # Gradio 6 only serves files inside its allowlist, so without this entry
    # gr.File("Download Heatmap PNG") sticks at "Uploading…" and the heatmap
    # image stays blank even though the PNG is on disk.
    _ANALYTICS_OUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(_ANALYTICS_OUT_DIR, exist_ok=True)
    _allowed_paths = [_ANALYTICS_OUT_DIR]
    if os.path.isdir(TEST_VIDEO_DIR):
        _allowed_paths.append(TEST_VIDEO_DIR)
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True,
        allowed_paths=_allowed_paths,
    )
