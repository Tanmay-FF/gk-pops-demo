"""
POPS - Push-Out Probability Score: Interactive Demo (v2 — modular)
===================================================================
Thin Gradio UI wrapper.  All logic lives in engine/*.py.
python code/demo_app_v2.py
"""
import os

#: transformers still probes for TensorFlow: image_transforms.py does
#: `if is_tf_available(): import tensorflow as tf`, which fires when the
#: Qwen3-VL processor is built and prints two absl/oneDNN banners to stderr.
#: Nothing here uses TF. USE_TF=0 makes is_tf_available() False so the import
#: never happens; the other two only matter if something else pulls TF in.
#: All three have to be set before the first transformers import -- once TF
#: has initialised they are ignored.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import time
import traceback
from functools import partial
from pathlib import Path

import cv2
import gradio as gr

import console_noise
import console_ui as ui
from engine import (TrackingEngine, SAMPLE_VIDEOS, analytics_ui, zone_editor,
                    zone_presets)
from engine.cancellation import RunCancelled, RunSuperseded
from engine import theme as T
from engine import ui_builder
from engine.config import TEST_VIDEO_DIR
from engine.config import VLM_BACKENDS, VLM_DEFAULT_BACKEND
from engine.rules import DEFAULT_THRESHOLDS
from engine.trajectory_cache import make_video_key

# Third-party console noise, muted before the server starts. Same intent as the
# USE_TF block at the top of this file: keep the demo console readable so OUR
# lines are the ones a reader sees. Narrow by construction -- see
# console_noise.py for what each filter covers and why neither is actionable
# from this repo.
console_noise.apply()

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"


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
#: One definition, shared with the pure edit helpers.
LAYOUT_KINDS = zone_editor.LAYOUT_KINDS

#: Output slots of run_analysis, in order. Keeping the names next to the
#: component list below is what stops the two drifting — the phase-2 yield
#: has to pad with exactly the right number of no-ops.
RUN_OUTPUT_NAMES = [
    "video_output", "json_download", "json_output",
    "video_info_html", "detection_html", "config_html", "legend_html",
    "pops_html", "events_html",
    "case_report_html", "case_report_download",
    "analytics_summary_html", "spikes_html", "dwell_html", "journey_html",
    "heatmap_image", "heatmap_file",
    "alert_banner_html", "ops_alerts_html",
    "run_summary_html", "tab_counts_html", "result_tabs",
]
N_RUN_OUTPUTS = len(RUN_OUTPUT_NAMES)

#: name -> position in the engine's return tuple. Valid ONLY because
#: `result_tabs` is the single UI-added slot and it is last, so the engine's
#: tuple is this list minus its tail. Asserted rather than assumed: inserting a
#: name mid-list would otherwise shift every lookup silently, which is the exact
#: failure this table replaced (hardcoded out[17]/out[22] indices that drifted
#: when the rule engine added two outputs).
assert RUN_OUTPUT_NAMES[-1] == "result_tabs", \
    "result_tabs must stay last — _IDX maps engine outputs by position"
_IDX = {name: i for i, name in enumerate(RUN_OUTPUT_NAMES)}

#: Slots the pre-run flush leaves alone.
#:
#: `result_tabs` because pushing a gr.Tabs config through an update is the one
#: thing that reliably wedges Gradio's client here (see the run wiring at the
#: bottom of this file) — and a flush has no reason to move the selection.
#: `legend_html` because it is static reference content: clearing the colour key
#: between runs would remove information that was never run-specific.
_NO_FLUSH = ("result_tabs", "legend_html")
FLUSH_OUTPUT_NAMES = [n for n in RUN_OUTPUT_NAMES if n not in _NO_FLUSH]

#: Bytes of raw tracking JSON fed to the gr.Code viewer. The full document is
#: 5+ MB for a 22-second clip and grows ~7 KB per frame (JSON_EVERY_N_FRAMES=1),
#: so pushing all of it into a component value is what froze the tab at 100%.
#: The complete file is untouched on disk and reachable via the Download button.
#:
#: Small on purpose. gr.Code is a CodeMirror editor: it lays out and
#: syntax-highlights whatever it is given, mutating the DOM as it goes, and
#: those mutations drive the MutationObserver in static/app.js. 200 KB is
#: ~5,000 lines of that; 20 KB is a few hundred, which is all a preview is for.
JSON_PREVIEW_BYTES = 20_000


def _blank_panels(*, has_video: bool = True, placeholder: str | None = None) -> dict:
    """Every run-output slot at its idle value, keyed by RUN_OUTPUT_NAMES.

    THE one definition of "nothing has run yet", and there are three callers
    that all have to agree on it: the components' initial values, the flush that
    runs before a new analysis, and the early-exit stub below. When those were
    three separate positional tuples, "cleared" and "freshly loaded" were
    different screens — and a slot added to RUN_OUTPUT_NAMES only had to be
    forgotten in one of them to leave the previous run's data on display.

    Keyed rather than positional for the same reason `_IDX` exists: a name that
    moves in the list cannot silently take another slot's value with it.

    `placeholder` overrides the copy on every panel that a run fills, for the
    failure path — "the run did not complete" says more there than "Run
    Analysis to see detection counts".
    """
    def ph(headline: str, sub: str = "") -> str:
        return placeholder if placeholder is not None else T.empty_state(headline, sub)

    return {
        "video_output": None,
        "json_download": None,
        "json_output": "",
        "video_info_html": ph("Run Analysis to see video metadata."),
        "detection_html": ph("Run Analysis to see detection counts."),
        "config_html": ph("Run Analysis to see the model configuration."),
        # NOT `ph(...)` — see _NO_FLUSH. The legend is the same reference card
        # before, during and after a run.
        "legend_html": ui_builder.build_legend(),
        "pops_html": ph("Run Analysis to score carts.",
                        "One row per cart: peak POPS event plus every "
                        "operational category that cart fell into."),
        "events_html": ph("Run Analysis to build the event timeline."),
        "case_report_html": ph(
            "Run Analysis to generate a case report.",
            "A VLM summarises captured frames once the main pipeline "
            "finishes."),
        "case_report_download": None,
        "analytics_summary_html":
            analytics_ui.build_analytics_empty_state(has_video=has_video),
        # The three panels below the summary go BLANK, not to an empty state:
        # they sit stacked in the same tab, so an empty-state card in each drew
        # the same "nothing here yet" message three times down the page.
        "spikes_html": "",
        "dwell_html": "",
        "journey_html": "",
        "heatmap_image": None,
        "heatmap_file": None,
        # Empty string, not a visibility update — see the gr.HTML that
        # hosts this slot for why the banner is never toggled.
        "alert_banner_html": "",
        "ops_alerts_html": ph("Run Analysis to evaluate operational rules."),
        "run_summary_html": "",
        "tab_counts_html": ui_builder.build_tab_counts({}),
        "result_tabs": gr.update(),                      # selection — leave put
    }


assert set(_blank_panels()) == set(RUN_OUTPUT_NAMES), \
    "_blank_panels and RUN_OUTPUT_NAMES have drifted apart"


def _empty_run_outputs(message: str = "No video selected", detail: str = ""):
    """Shape-matched stub used on early-exit paths so Gradio doesn't choke.

    The failure message goes to the STICKY BANNER, not into a results tab.
    It previously landed in `video_info_html` — eleventh in the tab bar —
    so a failed run showed a transient toast and parked the reason where
    nobody would look.
    """
    blank = _blank_panels(
        placeholder=T.empty_state("Nothing to show - the run did not complete.",
                                  "See the banner at the top of the page."))
    blank["alert_banner_html"] = ui_builder.build_error_banner(
        message, detail)
    return tuple(blank[name] for name in RUN_OUTPUT_NAMES)


def flush_results(video_path):
    """Wipe every panel the previous run filled, before the next one starts.

    Wired as the FIRST event of the Run chain (and after a video change), so a
    second run does not sit under the first one's POPS table, event log,
    operational alerts, heat-map and alert banner while it works — and a run
    that fails leaves the previous run's findings nowhere on screen to be read
    as the new ones.

    Engine-side state needs no equivalent: `TrackingEngine.process_video` calls
    `_reset()` before it touches the first frame, so the numbers behind these
    panels are already per-run. This is only what the browser is still holding.
    """
    blank = _blank_panels(has_video=bool(video_path))
    return tuple(blank[name] for name in FLUSH_OUTPUT_NAMES)


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Bisection switches
# ---------------------------------------------------------------------------
# A tab can lock hard enough that DevTools will not open, which leaves no way
# to measure anything from inside the browser — no console, no profiler, no
# `window.__gkBindStats`. These make a locked page bisectable from OUTSIDE, one
# run per switch, reading only the server log and whether the page survives.
#
#   GK_SAFE_MODE=1        every switch below at once. If the page renders, the
#                         cause is in this app's own content; if it still locks,
#                         it is the page shell and nothing we send.
#   GK_NO_APP_JS=1        launch without static/app.js
#   GK_NO_APP_CSS=1       launch without static/app.css
#   GK_NO_JSON_VIEWER=1   send no JSON to the gr.Code editor
#   GK_NO_VIDEO=1         send no tracked video
#   GK_NO_HEATMAP=1       send no heatmap image
#   GK_NO_PANELS=1        replace every HTML table/panel with one line
#
# The three below were added after GK_SAFE_MODE still locked the page: with
# every switch above on, these were the ONLY values still reaching the browser
# when a run finished, and the page was provably fine until that response
# landed. Two of them change LAYOUT rather than just content, which is the
# shape of thing that drives Svelte's `effect_update_depth_exceeded`.
#
#   GK_NO_BANNER=1        never reveal the sticky alert banner (hidden -> visible
#                         is a layout change on an element above the fold)
#   GK_NO_TAB_SWITCH=1    do not auto-select the POPS tab (a tab switch mounts
#                         one tab's children and unmounts another's)
#   GK_NO_JSON_FILE=1     do not hand the multi-MB tracking JSON to the
#                         download component
_SAFE = _flag("GK_SAFE_MODE")
NO_APP_JS      = _SAFE or _flag("GK_NO_APP_JS")
NO_APP_CSS     = _SAFE or _flag("GK_NO_APP_CSS")
NO_JSON_VIEWER = _SAFE or _flag("GK_NO_JSON_VIEWER")
NO_VIDEO       = _SAFE or _flag("GK_NO_VIDEO")
NO_HEATMAP     = _SAFE or _flag("GK_NO_HEATMAP")
NO_PANELS      = _SAFE or _flag("GK_NO_PANELS")
NO_BANNER      = _SAFE or _flag("GK_NO_BANNER")
NO_TAB_SWITCH  = _SAFE or _flag("GK_NO_TAB_SWITCH")
NO_JSON_FILE   = _SAFE or _flag("GK_NO_JSON_FILE")

#: Output slots blanked by GK_NO_PANELS — everything that ships generated HTML.
_PANEL_SLOTS = (
    "pops_html", "events_html", "ops_alerts_html", "analytics_summary_html",
    "spikes_html", "dwell_html", "journey_html", "case_report_html",
    "video_info_html", "detection_html", "config_html", "legend_html",
    "run_summary_html", "tab_counts_html",
)


def _active_switches() -> list[str]:
    return [n for n, on in (
        ("GK_NO_APP_JS", NO_APP_JS), ("GK_NO_APP_CSS", NO_APP_CSS),
        ("GK_NO_JSON_VIEWER", NO_JSON_VIEWER), ("GK_NO_VIDEO", NO_VIDEO),
        ("GK_NO_HEATMAP", NO_HEATMAP), ("GK_NO_PANELS", NO_PANELS),
        ("GK_NO_BANNER", NO_BANNER), ("GK_NO_TAB_SWITCH", NO_TAB_SWITCH),
        ("GK_NO_JSON_FILE", NO_JSON_FILE),
    ) if on]


def _apply_switches(out: list) -> None:
    """Blank whichever outputs the bisection switches disable.

    Applied at the very end of the handler, on the finished output list, so a
    switch changes ONLY what the browser receives — never what the pipeline
    computed or wrote to disk.
    """
    active = _active_switches()
    if not active:
        return
    if NO_JSON_VIEWER:
        out[_IDX["json_output"]] = "// disabled by GK_NO_JSON_VIEWER"
    if NO_VIDEO:
        out[_IDX["video_output"]] = None
    if NO_HEATMAP:
        out[_IDX["heatmap_image"]] = None
        out[_IDX["heatmap_file"]] = None
    if NO_PANELS:
        for name in _PANEL_SLOTS:
            out[_IDX[name]] = (f"<div class='gk-section-note'>{name} disabled "
                               f"by GK_NO_PANELS</div>")
    if NO_JSON_FILE:
        out[_IDX["json_download"]] = None
    if NO_BANNER:
        # Blanks the value. The banner host is always mounted now, so there is
        # no hidden -> visible reveal left to suppress; what this switch still
        # removes is the above-the-fold DOM the banner HTML inserts.
        out[_IDX["alert_banner_html"]] = ""
    print(f"[gk] bisection: blanked outputs for {', '.join(active)}")


def _json_preview(json_str: str) -> str:
    """Head of the tracking JSON, cut on a line boundary, with a note saying
    how much was elided and where the whole thing is.

    Byte-sliced rather than re-serialised from a trimmed dict: json.loads on a
    50 MB string costs a second and doubles peak memory to produce something
    only meant to be glanced at.
    """
    if not json_str:
        return ""
    total = len(json_str)
    if total <= JSON_PREVIEW_BYTES:
        return json_str
    cut = json_str.rfind("\n", 0, JSON_PREVIEW_BYTES)
    head = json_str[:cut if cut > 0 else JSON_PREVIEW_BYTES]
    return (f"// Preview - first {len(head) / 1024:,.0f} KB of "
            f"{total / 1048576:.1f} MB.\n"
            f"// The complete per-frame JSON is on disk: use the Download "
            f"button, or 'Open full JSON' below.\n"
            f"// (Sending all of it to the browser is what froze this page.)\n"
            f"{head}\n// … truncated …\n")


def _thresholds(blocked_door_s, static_cart_s, abandoned_cart_s) -> dict:
    return {
        "blocked_door_s": blocked_door_s,
        "static_cart_s": static_cart_s,
        "abandoned_cart_s": abandoned_cart_s,
    }


def run_analysis(video_path, camera_placement, vlm_backend, vlm_api_key,
                 zones_state, blocked_door_s, static_cart_s, abandoned_cart_s,
                 enable_pose):
    """The fast pipeline. Returns once, with the case report left pending.

    NOT a generator, deliberately. This used to yield twice — dashboard first,
    case report second — on the theory that an intermediate yield renders while
    the VLM still runs. It does not: Gradio paints a pending overlay across
    every output component of an event, including the gr.Tabs, and only clears
    it when the function RETURNS. So a local VLM taking minutes left the
    finished dashboard sitting underneath "Processing frames - 100.0%" bars,
    which reads as a hung page — and no amount of payload trimming fixes it,
    because the results had already arrived.

    The case report is now a SEPARATE event chained with .then(), so this one
    completes, the overlays clear, and only the two case-report components stay
    pending while the VLM works.
    """
    if video_path is None:
        gr.Warning("Please upload or select a video first.")
        return _empty_run_outputs(
            "No video selected",
            "Pick a sample clip or upload a file in the sidebar, then run again.")
    try:
        # ----- Phase 1: fast pipeline, VLM deferred -----
        result = engine.process_video(
            video_path,
            camera_placement=camera_placement,
            vlm_backend=vlm_backend,
            vlm_api_key=vlm_api_key,
            zones=list(zones_state or []),
            defer_case_report=True,
            rule_thresholds=_thresholds(blocked_door_s, static_cart_s,
                                        abandoned_cart_s),
            enable_pose=bool(enable_pose),
        )
        out = list(result)
        # The engine produces every slot except `result_tabs`, which the UI
        # adds. A mismatch here means Gradio would map values to the wrong
        # components — silently, and with plausible-looking output. Fail loudly.
        expected = N_RUN_OUTPUTS - 1
        if len(out) != expected:
            raise RuntimeError(
                f"process_video returned {len(out)} values, UI expects "
                f"{expected}. RUN_OUTPUT_NAMES and the engine's return tuple "
                f"have drifted apart.")

        # The raw JSON goes to the browser as a PREVIEW. The full document is
        # already on disk and wired to the Download button; sending it as a
        # component value as well is what made the tab unresponsive at 100%.
        out[_IDX["json_output"]] = _json_preview(out[_IDX["json_output"]])

        # Measured HERE, after that substitution — this is the seam where the
        # payload is final. Everything below leaves in ONE websocket message
        # that the browser parses on its main thread, which is why the page
        # died at 100%. Logging the bytes makes that a number per clip instead
        # of something to infer. The case report is absent on purpose: it is a
        # separate event and measures itself.
        _sizes = [(n, len(out[_IDX[n]])) for n in
                  ("json_output", "events_html", "pops_html", "ops_alerts_html",
                   "analytics_summary_html", "dwell_html", "journey_html",
                   "alert_banner_html")
                  if isinstance(out[_IDX[n]], str)]
        _total = sum(v for _, v in _sizes)
        print(f"[PERF] browser payload: {_total / 1048576:.2f} MB  |  "
              + "  ".join(f"{k} {v / 1024:.0f}KB" for k, v in
                          sorted(_sizes, key=lambda x: -x[1])[:4])
              + "  (+ case report, phase 2)")

        # Switches FIRST, then read the locals out of the blanked list.
        #
        # This used to run after the six reads below, so every slot that leaves
        # via a local — run_summary_html, tab_counts_html, ops_alerts_html and
        # the banner — was captured before being blanked and shipped anyway.
        # GK_NO_PANELS silently did nothing for those four, which is why a
        # GK_SAFE_MODE run still painted the RUN summary strip and the alert
        # banner. A bisection harness that quietly under-reports is worse than
        # none: it clears suspects that were never actually turned off.
        _apply_switches(out)

        tab_counts   = out[_IDX["tab_counts_html"]]
        run_summary  = out[_IDX["run_summary_html"]]
        ops_html     = out[_IDX["ops_alerts_html"]] or ""
        heatmap_path = out[_IDX["heatmap_file"]]
        heatmap_bgr  = out[_IDX["heatmap_image"]]
        heatmap_rgb  = _bgr_ndarray_to_rgb(heatmap_bgr)
        # Straight through: _apply_switches has already blanked this slot if
        # GK_NO_BANNER is on, and the banner is a plain value now rather than a
        # visibility toggle, so there is no update dict to tell apart.
        alert_html = out[_IDX["alert_banner_html"]] or ""
        if alert_html.strip():
            gr.Warning("Alerts detected - see the banner at the top of the page.")

        if NO_HEATMAP:
            heatmap_rgb, heatmap_path = None, None

        # The tab switch is NOT done here — see _GOTO_RESULTS_JS, chained after
        # this event. Returning `gr.Tabs(selected=...)` from this function put a
        # component-config reconciliation in the SAME Svelte flush that replaces
        # content in a dozen components across nine tabs, which made the client
        # mount tab_pops' children while their values were still being patched.
        # That threw inside Gradio's flush (uncaught, in its own bundle), and an
        # exception mid-flush aborts the update: the pending overlays never
        # clear and toasts stop dismissing, so the page looks frozen at 100%
        # even though the server finished and the payload was only ~40 KB.
        #
        # gr.skip() leaves this slot untouched so the arity stays 22 and _IDX
        # keeps working; the failure path already returned a plain gr.update()
        # here, which is why a FAILED run never froze the page.
        return (tuple(out[:_IDX["heatmap_image"]])
                + (heatmap_rgb, heatmap_path, alert_html, ops_html,
                   run_summary, tab_counts, gr.skip()))
    except RunCancelled as e:
        # Before the generic handler, and no traceback: a cancel is the user
        # getting what they asked for, not an error. The banner says so rather
        # than reading "Analysis failed", which is what a bare `except
        # Exception` here would have produced.
        print(f"[CANCEL] run_analysis stopped: {e}")
        gr.Info("Run cancelled.")
        return _empty_run_outputs(
            "Run cancelled",
            "The run was stopped before it finished, so there are no results "
            "to show. Press Run Analysis to start again.")
    except Exception as e:
        gr.Warning(f"Error: {e}")
        traceback.print_exc()
        return _empty_run_outputs("Analysis failed",
                                  f"{type(e).__name__}: {e}")


#: Lands the user on the results after a run, by CLICKING the tab in the
#: browser. The Zone Editor is the first tab, so without this a finished run
#: still shows the polygon canvas while the user hunts for what changed.
#:
#: Not `gr.Tabs(selected=...)`. That was tried both inside run_analysis and as
#: its own chained event; each time Gradio's client wedged and the selection
#: never applied. Clicking the rendered button goes through Gradio's own tab
#: handler rather than asking it to reconcile a layout component built outside
#: the Blocks tree. Falls back to doing nothing if app.js is not loaded.
_GOTO_RESULTS_JS = ("() => {}" if NO_TAB_SWITCH else
                    "() => { try { if (window.gkGotoTab) "
                    "window.gkGotoTab('POPS'); } "
                    "catch (e) { console.warn('[gk] tab switch failed', e); } }")


def cancel_run_handler():
    """Stop the run (or case report) that is currently in flight.

    Two mechanisms, and both are needed. `cancels=` on this button's event is
    Gradio's own, and its docstring is explicit that it only drops jobs that
    have "not yet run" — a function already executing "will be allowed to
    finish". So it clears the client's spinner and any queued follow-up, and
    nothing more. `engine.request_cancel()` is what actually stops the work: the
    frame loop tests the flag every frame and the local VLM tests it every
    generated token.

    Deliberately outputs nothing. It must not be blocked behind the run it is
    cancelling, so it touches no engine state beyond the cancel token — which
    takes its own lock, never `_gpu_lock`.
    """
    try:
        seq = engine.request_cancel()
    except Exception as e:
        # A Cancel button that raises is worse than useless: the run keeps
        # going and the toast blames the button. Nothing in request_cancel()
        # should be able to fail, which is exactly why a failure here needs to
        # be visible rather than a stack trace in a terminal nobody is reading.
        traceback.print_exc()
        gr.Warning(f"Could not cancel: {e}")
        return
    if seq is None:
        gr.Info("Nothing to cancel — no run is in progress.")
    else:
        gr.Info("Cancelling — the run stops at the next frame.")


def finalize_case_report_handler():
    """Second event: run the deferred VLM and fill in the case-report tab.

    Chained off run_analysis with .then() rather than being a second yield of
    it. That is the whole point — an event holds a pending overlay over every
    one of its output components until it returns, so with both phases in one
    event a slow local VLM hid the entire finished dashboard. Here only the two
    case-report components wait.

    Runs on the engine singleton's state from the run that just finished, which
    is the same thing the previous yield-based version did.
    """
    t0 = time.perf_counter()
    try:
        case_html, case_file = engine.finalize_case_report()
    except RunSuperseded as e:
        # A NEWER run replaced this one, so its Run click already flushed
        # case_report_html and case_report_download (both are in
        # FLUSH_OUTPUT_NAMES). Returning any value -- even the "cancelled"
        # panel -- would paint this dead run's result over the live run's fresh
        # panels. gr.skip() leaves both components exactly as the flush left
        # them, so the new run's own finalize fills them in when it gets there.
        print(f"[CANCEL] case report abandoned: {e}")
        return gr.skip(), gr.skip()
    except RunCancelled as e:
        # Same reasoning as run_analysis: not a failure, so not the SAFETY
        # notice. The engine also returns a "cancelled" panel without raising
        # when the payload it popped was already cancelled before the VLM
        # started; this is the path where the cancel landed mid-generation.
        print(f"[CANCEL] case report stopped: {e}")
        return (T.empty_state("Case report cancelled.",
                              "The run was stopped before the report finished."),
                None)
    except Exception as e:
        traceback.print_exc()
        return (T.notice("Case report failed", T.esc(str(e)), tone="SAFETY"),
                None)
    if not case_html and not case_file:
        # No captures were taken (rare) — say so rather than leaving the
        # "Generating…" placeholder up forever.
        return (T.empty_state(
            "No case report for this clip.",
            "No frames were captured, so there was nothing for the VLM to "
            "summarise."), None)
    # The one payload still tied to event count: each captured frame is
    # embedded as a base64 JPEG (capped by FRAME_CAPTURE_MAX).
    print(f"[PERF] case report: {time.perf_counter() - t0:.1f}s, "
          f"{len(case_html or '') / 1048576:.2f} MB to the browser")
    return case_html, case_file


# ---------------------------------------------------------------------------
# Zone editor handlers
# ---------------------------------------------------------------------------
def on_video_upload(video_path, prev_zones_state, prev_video_key):
    """When the user picks a video, load or reset its zones (if it's a new
    file) and extract the first frame so they can draw zones on it.

    Seven outputs, and every return site emits all seven — a short tuple is
    silent in Gradio and maps the values it does have onto the wrong
    components. The seventh is which saved set the zones came from; the
    saved-set panel itself is redrawn by the refresh_preset_ui() chained onto
    this event, not from here.
    """
    if not video_path:
        # The zone canvas slot must be None, NOT "". gr.Image treats any string
        # as a filepath, and Gradio resolves a relative one against the process
        # cwd: "" becomes the project directory, which it then tries to open and
        # hash. On Windows that is PermissionError [Errno 13] on a directory,
        # raised inside postprocess, so clearing the video killed the event.
        # `prev_video_key` is carried through, NOT cleared. This branch keeps
        # the zones on purpose, and dropping the key would make the very next
        # pick of that same clip look like a new video to the branch below -
        # which would auto-load the saved preset over the zones just preserved,
        # discarding every edit made since the last save.
        return (None, prev_zones_state or [], [], prev_video_key, None,
                _zone_summary_html([]), None)

    new_key = make_video_key(video_path)
    is_new_video = new_key != prev_video_key
    zones_state = [] if is_new_video else list(prev_zones_state or [])

    frame = zone_editor.extract_first_frame(video_path)
    if frame is None:
        gr.Warning("Could not decode the first frame of this video.")
        return (None, zones_state, [], new_key, None,
                _zone_summary_html(zones_state), None)

    # The newest saved set for this clip is loaded here rather than behind a
    # button: picking the clip IS the request for its zones. The saved-set
    # list under the canvas still loads any older set, and Clear all still
    # empties the editor. Only on a NEW clip — a rerun of the same file keeps
    # whatever is on screen, edits included.
    loaded_path = None
    if is_new_video:
        zones_state, loaded_path, loaded = _autoload_preset(video_path, frame)
        if not loaded and prev_video_key is not None:
            gr.Info("New video - zones reset.")

    overlay = zone_editor.render_zone_overlay(frame, zones_state, in_progress=[])
    return (frame, zones_state, [], new_key, _bgr_to_rgb(overlay),
            _zone_summary_html(zones_state), loaded_path)


def on_canvas_click(evt: gr.SelectData, current_poly, first_frame, zones_state):
    """Append a vertex on every click; redraw the overlay live."""
    if first_frame is None:
        gr.Warning("Upload a video first.")
        return current_poly or [], None
    pts = list(current_poly or [])
    x, y = int(evt.index[0]), int(evt.index[1])
    pts.append((x, y))
    overlay = zone_editor.render_zone_overlay(
        first_frame, zones_state or [], in_progress=pts)
    return pts, _bgr_to_rgb(overlay)


def close_polygon(current_poly, zone_name, applies_to, zone_kind, zones_state,
                  first_frame):
    """Validate the in-progress polygon and promote it to a Zone."""
    if first_frame is None:
        gr.Warning("Upload a video first.")
        return (zones_state or [], current_poly or [], None,
                _zone_summary_html(zones_state or []))

    pts = list(current_poly or [])
    ok, msg = zone_editor.validate_polygon(pts)
    if not ok:
        gr.Warning(msg)
        overlay = zone_editor.render_zone_overlay(
            first_frame, zones_state or [], in_progress=pts)
        return (zones_state or [], pts, _bgr_to_rgb(overlay),
                _zone_summary_html(zones_state or []))

    zones_state = list(zones_state or [])
    kind = zone_kind or "analytics"
    applies_to = _coerce_applies_to(kind, applies_to, announce=True)
    new_zone = zone_editor.make_zone(
        zone_name, pts, applies_to, len(zones_state), kind=kind,
    )
    zones_state.append(new_zone)

    overlay = zone_editor.render_zone_overlay(first_frame, zones_state,
                                              in_progress=[])
    return (zones_state, [], _bgr_to_rgb(overlay),
            _zone_summary_html(zones_state))


def undo_vertex(current_poly, first_frame, zones_state):
    pts = list(current_poly or [])
    if pts:
        pts.pop()
    overlay = (_bgr_to_rgb(zone_editor.render_zone_overlay(
                   first_frame, zones_state or [], in_progress=pts))
               if first_frame is not None else None)
    return pts, overlay


def clear_inprogress(first_frame, zones_state):
    overlay = (_bgr_to_rgb(zone_editor.render_zone_overlay(
                   first_frame, zones_state or [], in_progress=[]))
               if first_frame is not None else None)
    return [], overlay


def clear_all_zones(zones_state, first_frame):
    prev = list(zones_state or [])
    overlay = _redraw_canvas(first_frame, [], [])
    undo = _undo_slot(f"{len(prev)} zones", prev) if prev else None
    return [], [], overlay, _zone_summary_html([]), undo


# ---------------------------------------------------------------------------
# Saved zone sets
#
# Gradio wrappers over engine/zone_presets.py. Every set is per-clip and lives
# in ZONE_PRESET_DIR as one JSON file named
# "<clip>__<set name>__<when>.json", so the folder is readable on its own.
#
# The panel is a FIXED POOL of rows, occupancy expressed in the meta cell's
# HTML rather than Gradio's `visible` flag — same construction as the zone
# manager below, and for the same reason: `visible` updates on a Row inside a
# gr.Tab did not land, while an HTML value update always does.
# ---------------------------------------------------------------------------
#: Rows in the saved-set list. Older sets past this stay on disk and the
#: header says how many are not shown.
MAX_PRESET_ROWS = 8

#: Emitted into an unused slot's meta cell. CSS hides any row containing it.
PRESET_EMPTY_SLOT = "<span class='gk-preset-slot-empty'></span>"

#: Head of the tuple `_preset_ui` returns, before the per-row values.
_PRESET_HEAD_N = 4


def _preset_folder_label():
    """The folder, shown relative to the repo when it is inside it — an
    absolute Windows path is the one thing that makes this header wrap."""
    folder = str(zone_presets.preset_dir())
    try:
        rel = os.path.relpath(folder, os.path.dirname(os.path.abspath(__file__)))
        return folder if rel.startswith("..") else rel
    except ValueError:                      # different drive
        return folder


def _preset_hdr(video_path, infos, loaded_path, total=None):
    """Title row: where the files are, how many this clip has, which one is on
    screen. This is the answer to "where did my zones go", and it is on the
    page rather than in a toast that has already faded.

    `total` is the number of files on disk, which is NOT len(infos) - the rows
    are capped at MAX_PRESET_ROWS. Counting the truncated list would report
    "8 saved" while 13 sat in the folder, and the older ones would look lost.
    """
    folder = T.esc(_preset_folder_label())
    total = len(infos) if total is None else total
    if not video_path:
        count = "no clip selected"
    elif not total:
        count = "none saved for this clip yet"
    else:
        count = f"{total} saved for this clip"
        if len(infos) < total:
            count += f" - showing the {len(infos)} newest"
    live = ""
    for i in infos:
        if loaded_path and i.path == loaded_path:
            live = (f"<span class='gk-preset-live'>on screen: "
                    f"{T.esc(i.label)}</span>")
            break
    return (f"<div class='gk-preset-hdr'>"
            f"<span class='gk-preset-title'>Saved zone sets</span>"
            f"<span class='gk-preset-count'>{T.esc(count)}</span>"
            f"{live}"
            f"<span class='gk-preset-folder'>folder: <code>{folder}</code>"
            f"</span></div>")


def _preset_empty_html(video_path, infos):
    """Shown instead of the rows when there is nothing to list. Says what to
    press and where the file will land — an empty list otherwise reads as a
    broken feature."""
    if infos:
        return ""
    if not video_path:
        return ("<div class='gk-preset-empty'>Pick a clip first - a zone set "
                "belongs to one video.</div>")
    return ("<div class='gk-preset-empty'>Nothing saved for this clip yet. "
            "Draw your zones on the frame above, type a name in "
            "<b>Set name</b> and press <b>Save current zones</b>. The file "
            f"lands in <code>{T.esc(_preset_folder_label())}</code> as "
            "<code>clip__name__when.json</code>, and the newest set is "
            "loaded back automatically the next time you pick this clip."
            "</div>")


def _preset_row_meta(info, is_loaded):
    """One saved set, described the way its filename is: name, size, when."""
    plural = "" if info.n_zones == 1 else "s"
    size = (f"{info.frame_w}x{info.frame_h}"
            if info.frame_w and info.frame_h else "size unknown")
    live = "<span class='gk-preset-badge-live'>loaded</span>" if is_loaded else ""
    return (f"<div class='gk-preset-meta-in'>"
            f"<span class='gk-preset-name'>{T.esc(info.label)}</span>{live}"
            f"<span class='gk-preset-sub'>{info.n_zones} zone{plural} - "
            f"{size} - {T.esc(info.when)}</span>"
            f"<span class='gk-preset-file'>{T.esc(info.filename)}</span>"
            f"</div>")


def _preset_rows_update(infos, loaded_path):
    """One `gr.update` per row slot — always exactly MAX_PRESET_ROWS long."""
    out = []
    for i in range(MAX_PRESET_ROWS):
        if i < len(infos):
            out.append(gr.update(
                value=_preset_row_meta(infos[i],
                                       infos[i].path == loaded_path)))
        else:
            out.append(gr.update(value=PRESET_EMPTY_SLOT))
    return out


def _preset_ui(video_path, loaded_path=None):
    """The single return shape every preset action emits.

    `paths` is the slot -> file mapping the row buttons act through: the pool
    is rebuilt from the folder on every action, so the row a button sits in
    and the file it loads or deletes cannot drift apart.
    """
    all_infos = zone_presets.list_presets(video_path) if video_path else []
    infos = all_infos[:MAX_PRESET_ROWS]
    paths = [i.path for i in infos]
    if loaded_path not in paths:
        # A set can be on screen while its row is off the end of the pool, or
        # after its file was deleted. Neither is a row the buttons can act on,
        # so the mark simply is not shown.
        loaded_path = None
    return (paths, loaded_path,
            _preset_hdr(video_path, infos, loaded_path, total=len(all_infos)),
            _preset_empty_html(video_path, infos),
            *_preset_rows_update(infos, loaded_path))


def refresh_preset_ui(video_path, loaded_path):
    """Re-list the folder. Chained after a video change, and wired to the
    Refresh button for a folder that changed outside the app."""
    return _preset_ui(video_path, loaded_path)


def _autoload_preset(video_path, frame):
    """Newest saved set for this clip, as (zones, path, loaded).

    Returns an empty list and `loaded=False` for every failure — a clip with
    no sets, a set drawn on a different frame size, a corrupt file. The caller
    carries on with an empty editor either way; losing the auto-load is not a
    reason to fail the video selection.
    """
    infos = zone_presets.list_presets(video_path)
    if not infos:
        return [], None, False
    info = infos[0]
    try:
        zones, notes = zone_presets.load_preset(
            info.path, frame_shape=frame.shape, max_zones=MAX_ZONE_ROWS)
    except zone_presets.PresetError as e:
        gr.Warning(f"Saved zones not loaded: {e}")
        return [], None, False
    for note in notes:
        gr.Warning(note)
    gr.Info(f'Loaded saved zone set "{info.label}" - {len(zones)} '
            f'zone{"" if len(zones) == 1 else "s"}.')
    return zones, info.path, True


def save_zone_set(video_path, label, zones_state, first_frame, loaded_path):
    """Write the zones on screen to a new file for this clip."""
    if not video_path:
        gr.Warning("Pick a clip before saving zones - a zone set belongs to "
                   "one video.")
        return _preset_ui(video_path, loaded_path)
    zones = list(zones_state or [])
    try:
        path = zone_presets.save_preset(
            video_path, label or zone_presets.DEFAULT_LABEL, zones,
            # The frame size the polygons were drawn against. Recorded so a
            # reload onto a differently-sized re-encode is refused instead of
            # putting every zone in the wrong place.
            frame_shape=None if first_frame is None else first_frame.shape,
        )
    except zone_presets.PresetError as e:
        gr.Warning(str(e))
        return _preset_ui(video_path, loaded_path)
    gr.Info(f'Saved {len(zones)} zone{"" if len(zones) == 1 else "s"} to '
            f'{_preset_folder_label()}{os.sep}{path.name}')
    # The set just written is what is on screen, so it is the one marked
    # "loaded" in the list.
    return _preset_ui(video_path, str(path))


def load_zone_set(slot, paths, zones_state, first_frame, loaded_path):
    """Replace the zone list with the saved set in row `slot`.

    The zones on screen go into the one-level undo slot first, so a Load fired
    over unsaved work is recoverable through the same Restore button as a
    Remove or a Clear all.
    """
    path = paths[slot] if paths and 0 <= slot < len(paths) else None
    # A refused load leaves the zones alone, so it has to leave the "loaded"
    # mark alone too - marking the set that FAILED would put the badge on a
    # row the zones on screen did not come from.
    keep = (list(zones_state or []), None, loaded_path)
    if not path:
        gr.Warning("That row is empty - press Refresh.")
        return keep
    if first_frame is None:
        gr.Warning("Load a clip first - zones are drawn on its frame.")
        return keep
    try:
        zones, notes = zone_presets.load_preset(
            path, frame_shape=first_frame.shape, max_zones=MAX_ZONE_ROWS)
    except zone_presets.PresetError as e:
        gr.Warning(str(e))
        return keep
    for note in notes:
        gr.Warning(note)
    prev = list(zones_state or [])
    undo = _undo_slot(f"{len(prev)} zones", prev) if prev else None
    gr.Info(f'Loaded {len(zones)} zone{"" if len(zones) == 1 else "s"} from '
            f'"{os.path.basename(path)}".')
    return zones, undo, path


def delete_zone_set(slot, paths, video_path, loaded_path):
    """Delete the file in row `slot`. The zones stay in the editor, which is
    what makes this recoverable — the toast says so, because the file itself
    is gone for good."""
    path = paths[slot] if paths and 0 <= slot < len(paths) else None
    if not path:
        gr.Warning("That row is empty - press Refresh.")
        return _preset_ui(video_path, loaded_path)
    name = os.path.basename(str(path))
    try:
        gone = zone_presets.delete_preset(path)
    except zone_presets.PresetError as e:
        gr.Warning(str(e))
        return _preset_ui(video_path, loaded_path)
    if gone:
        gr.Info(f'Deleted "{name}". The zones are still in the editor - '
                f'press "Save current zones" to write them back.')
    else:
        gr.Warning(f'"{name}" was already gone.')
    return _preset_ui(video_path,
                      None if path == loaded_path else loaded_path)


# ---------------------------------------------------------------------------
# Zone manager — edit / remove / restore
#
# Thin Gradio wrappers over the pure helpers in engine/zone_editor.py. These
# back the per-zone rows rendered under the canvas (see the @gr.render block in
# the Zone Editor tab); all they add is the toast, the redraw and the revision
# bump that rebuilds the list.
# ---------------------------------------------------------------------------
def _coerce_applies_to(kind, applies_to, *, announce=False):
    coerced = zone_editor.coerce_applies_to(kind, applies_to)
    if announce and coerced != applies_to and kind in LAYOUT_KINDS:
        gr.Info(f"{kind.capitalize()} zones apply to people and carts - "
                f'"Track type" set to "both".')
    return coerced


def _redraw_canvas(first_frame, zones, current_poly):
    if first_frame is None:
        return None
    return _bgr_to_rgb(zone_editor.render_zone_overlay(
        first_frame, list(zones or []), in_progress=list(current_poly or [])))


def _undo_slot(label, zones_before):
    """One-level undo. Snapshots the whole list rather than the removed zone
    so a single delete and a Clear all restore through the same path."""
    return {"label": label, "zones": list(zones_before or [])}


#: Rows in the saved-zone pool. A hand-drawn demo does not go past a handful;
#: anything beyond this stays in the list and the header says so.
MAX_ZONE_ROWS = 12

#: Emitted into an unused slot's meta cell. CSS hides any row containing it.
EMPTY_SLOT = "<span class='gk-zone-slot-empty'></span>"

#: Values `_zone_ui` produces per row slot, in order. Kept next to the builder
#: so the outputs list and the return tuple cannot drift apart.
_ZONE_ROW_FIELDS = ("meta", "name", "kind", "applies")
_ZONE_HEAD_N = 6


def _zone_rows_update(zones):
    """One `gr.update` per field of every slot — occupied slots show their
    zone, the rest are hidden. Always exactly the same length."""
    zones = list(zones or [])
    out = []
    for i in range(MAX_ZONE_ROWS):
        if i < len(zones):
            z = zones[i]
            out += [
                gr.update(value=_zone_mgr_badge(i, z)),
                gr.update(value=z.name),
                gr.update(value=z.kind),
                # Track type is meaningless on a layout zone — it is pinned to
                # "both" — so the control is shown locked rather than lying.
                gr.update(value=z.applies_to,
                          interactive=z.kind not in LAYOUT_KINDS),
            ]
        else:
            out += [gr.update(value=EMPTY_SLOT), gr.update(value=""),
                    gr.update(value="analytics"),
                    gr.update(value="person", interactive=True)]
    return out


def _undo_msg(undo):
    if not undo:
        return ""
    return (f"<span class='gk-zone-undo-msg'>Removed "
            f"{T.esc(undo['label'])}.</span>")


def _zone_ui(zones, first_frame, current_poly, undo):
    """The single return shape shared by every zone mutation.

    One shape for all of them on purpose: the alternative is each handler
    picking its own subset of outputs, which is how the tuple and the wiring
    drift apart — a mismatch Gradio reports as a warning and then papers over
    by silently mapping values to the wrong components.
    """
    zones = list(zones or [])
    return (zones,
            _redraw_canvas(first_frame, zones, current_poly),
            _zone_summary_html(zones),
            _zone_mgr_header(len(zones)),
            undo,
            _undo_msg(undo),
            *_zone_rows_update(zones))


def refresh_zone_ui(zones, first_frame, current_poly, undo):
    """Re-emit the whole zone UI from current state. Chained after the
    handlers that already existed (upload / save / clear all) so those keep
    their original signatures."""
    return _zone_ui(zones, first_frame, current_poly, undo)


def _zone_at_slot(zones, slot):
    """Row `slot` always shows `zones[slot]` — the pool is rebuilt from the
    list on every change, so position addressing cannot go stale."""
    zones = list(zones or [])
    return zones[slot] if 0 <= slot < len(zones) else None


def rename_zone(slot, new_name, zones_state, first_frame, current_poly, undo):
    z = _zone_at_slot(zones_state, slot)
    if z is None:
        return _zone_ui(zones_state, first_frame, current_poly, undo)
    zones, _ = zone_editor.rename_zone(zones_state or [], z.zone_id, new_name)
    return _zone_ui(zones, first_frame, current_poly, undo)


def retype_zone(slot, new_kind, zones_state, first_frame, current_poly, undo):
    """Change a zone's layout type. The colour is derived from the kind and
    Track type is constrained by it, so both get re-derived here."""
    z = _zone_at_slot(zones_state, slot)
    if z is None:
        return _zone_ui(zones_state, first_frame, current_poly, undo)
    kind = new_kind or "analytics"
    was = z.applies_to
    # Gradio's Dropdown fires .input twice per selection; the second pass is a
    # no-op inside retype_zone, so the toast is not doubled either.
    zones, changed = zone_editor.retype_zone(zones_state or [], z.zone_id, kind)
    if changed and kind in LAYOUT_KINDS and was != "both":
        gr.Info(f"{kind.capitalize()} zones apply to people and carts - "
                f'"Track type" set to "both".')
    return _zone_ui(zones, first_frame, current_poly, undo)


def set_zone_applies(slot, new_applies, zones_state, first_frame,
                     current_poly, undo):
    """Which track types a zone filters on. Only meaningful for analytics
    zones — the control is locked for layout kinds."""
    z = _zone_at_slot(zones_state, slot)
    if z is None:
        return _zone_ui(zones_state, first_frame, current_poly, undo)
    zones, _ = zone_editor.set_zone_applies_to(
        zones_state or [], z.zone_id, new_applies)
    # The overlay label carries applies_to, so this still redraws the canvas.
    return _zone_ui(zones, first_frame, current_poly, undo)


def remove_zone(slot, zones_state, first_frame, current_poly, undo):
    before = list(zones_state or [])
    z = _zone_at_slot(before, slot)
    if z is None:
        return _zone_ui(before, first_frame, current_poly, undo)
    zones, removed = zone_editor.remove_zone(before, z.zone_id)
    return _zone_ui(zones, first_frame, current_poly,
                    _undo_slot(f'"{removed.name}"', before))


def restore_zones(undo_slot, zones_state, first_frame, current_poly):
    if not undo_slot:
        return _zone_ui(zones_state, first_frame, current_poly, None)
    return _zone_ui(list(undo_slot.get("zones") or []),
                    first_frame, current_poly, None)


def recompute_analytics_handler(video_path, zones_state, camera_placement,
                                blocked_door_s, static_cart_s, abandoned_cart_s):
    """Re-evaluate analytics and every operational rule off the cached facts.

    No GPU work: this is what makes the threshold sliders a live control
    rather than a config-file edit plus a full re-run.
    """
    def _empty(has_video, reason):
        # Same definition of "empty" the flush and the initial page use, so a
        # failed recompute cannot leave the Analytics tab looking different from
        # a cleared one. The two `gr.update()`s are deliberate: the banner and
        # the POPS table belong to the RUN, and a recompute that failed has no
        # business blanking either.
        blank = _blank_panels(has_video=has_video)
        return (blank["analytics_summary_html"], blank["spikes_html"],
                blank["dwell_html"], blank["journey_html"],
                blank["heatmap_image"], blank["heatmap_file"],
                # Not a generic placeholder — this one carries WHY.
                ui_builder.build_operational_alerts([], reason),
                gr.update(), blank["tab_counts_html"], gr.update())

    if not video_path:
        gr.Warning("Upload a video first.")
        return _empty(False, "No video loaded.")
    try:
        result = engine.recompute_analytics(
            video_path, list(zones_state or []),
            camera_placement=camera_placement,
            rule_thresholds=_thresholds(blocked_door_s, static_cart_s,
                                        abandoned_cart_s))
        (summary, spikes, dwell, journey, heatmap_bgr, heatmap_path,
         ops, alert_html, tab_counts, pops) = result
        heatmap_rgb = _bgr_ndarray_to_rgb(heatmap_bgr)
        gr.Info("Analytics and operational rules re-evaluated.")
        return (summary, spikes, dwell, journey, heatmap_rgb, heatmap_path,
                ops, alert_html, tab_counts,
                # The POPS table carries the operational categories, so it moves
                # with the thresholds. Empty means this process has no POPS
                # state to rebuild from — leave the tab as it is.
                pops if pops else gr.update())
    except Exception as e:
        gr.Warning(f"Recompute failed: {e}")
        traceback.print_exc()
        return _empty(True, f"Recompute failed: {e}")


def reset_thresholds():
    return (DEFAULT_THRESHOLDS["blocked_door_s"],
            DEFAULT_THRESHOLDS["static_cart_s"],
            DEFAULT_THRESHOLDS["abandoned_cart_s"])


def invalidate_cache_handler(video_path):
    if video_path:
        engine.invalidate_cache(video_path)
        gr.Info("Detection cache cleared. Next Run Analysis will re-run YOLO.")
    else:
        engine.invalidate_cache()
        gr.Info("All cached trajectories cleared.")


_KIND_TONE = {
    "analytics": ("var(--gk-accent-bg)", "var(--gk-accent)"),
    "wall":      ("var(--gk-neutral-bg)", "var(--gk-neutral)"),
    "aisle":     ("var(--gk-neutral-bg)", "var(--gk-ink-2)"),
    "fixture":   ("var(--gk-cyan-bg)",    "var(--gk-cyan)"),
    "door":      ("var(--gk-good-bg)",    "var(--gk-good)"),
}


def _zone_summary_html(zones_state):
    """Compact list of currently-defined zones for the Zone Editor sidebar."""
    if not zones_state:
        return ("<div class='sb-hint'>No zones yet. Click on the frame in the "
                "Zone Editor tab to add vertices, then press "
                "<b>Save zone</b>.</div>")
    rows = ""
    for z in zones_state:
        b, g, r = z.color
        kind = getattr(z, "kind", "analytics")
        tag_bg, tag_fg = _KIND_TONE.get(kind, _KIND_TONE["wall"])
        right_tag = z.applies_to if kind == "analytics" else kind
        rows += (
            f"<div class='gk-zone-row'>"
            f"<span class='gk-swatch' style='background:rgb({r},{g},{b});'></span>"
            # A long name ellipsises in a 294px pane; the tooltip is how it
            # stays readable without letting the row grow.
            f"<span class='gk-zone-name' title='{T.esc(z.name)}'>"
            f"{T.esc(z.name)}</span>"
            f"<span class='gk-zone-tag' style='background:{tag_bg};color:{tag_fg};'>"
            f"{T.esc(right_tag)}</span>"
            f"<span class='gk-zone-pts'>{len(z.polygon)} pts</span></div>"
        )
    return (f"<div class='gk-zone-list'>"
            f"<div class='gk-zone-list-hdr'>Defined zones "
            f"({len(zones_state)})</div>{rows}"
            f"<div class='gk-zone-list-foot'>Edit or remove them in the "
            f"<b>Zone Editor</b> tab.</div></div>")


def _zone_mgr_header(n: int) -> str:
    """Heading above the editable zone list (or the empty state in its place)."""
    if not n:
        return ("<div class='gk-zone-mgr-empty'>No zones saved yet - click the "
                "frame above to place vertices, then <b>Save zone</b>. Saved "
                "zones appear here and stay editable.</div>")
    return (f"<div class='gk-zone-mgr-hdr'>"
            f"<span class='gk-zone-mgr-title'>Saved zones</span>"
            f"<span class='gk-zone-mgr-count'>{n}</span>"
            f"<span class='gk-zone-mgr-hint'>Rename inline (press Enter or "
            f"click away), change the type, or remove.</span></div>")


def _zone_mgr_badge(idx: int, z) -> str:
    """Row leader: the polygon's own colour, so the row is matchable against
    the overlay above without having to select anything."""
    b, g, r = z.color
    return (f"<span class='gk-zone-mgr-swatch' "
            f"style='background:rgb({r},{g},{b});'>{idx + 1}</span>"
            f"<span class='gk-zone-mgr-pts'>{len(z.polygon)} pts</span>")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
# CSS and JS live in static/ — 700+ lines of them embedded as Python string
# literals meant no syntax highlighting, no linting, and hand-escaped unicode.
_CSS = (_STATIC / "app.css").read_text(encoding="utf-8")
_JS = (_STATIC / "app.js").read_text(encoding="utf-8")

_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate",
                        neutral_hue="slate")

#: The components' starting values come from the SAME dict the flush re-emits,
#: so "a new run just cleared this" and "nothing has run yet" are the identical
#: screen. `has_video=False` only affects the Analytics tab's copy, which says
#: "Upload a video to begin" until one is loaded.
_INITIAL = _blank_panels(has_video=False)

# Gradio 6 moved theme/css/js from the Blocks constructor to launch(); passing
# them here only produces a deprecation warning, so they go to launch() below.
with gr.Blocks(
    title="POPS - Push-Out Probability Score | Gatekeeper AI",
) as demo:

    # ── persistent state ────────────────────────────────────────────────
    first_frame_state = gr.State(value=None)
    zones_state       = gr.State(value=[])
    current_poly      = gr.State(value=[])
    video_key_state   = gr.State(value=None)
    # One-level undo for Remove / Clear all.
    zone_undo         = gr.State(value=None)
    # Saved zone sets: slot -> file path for the row buttons, and which file
    # the zones on screen came from (marked "loaded" in the list).
    preset_paths      = gr.State(value=[])
    preset_loaded     = gr.State(value=None)

    # ── top bar ─────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="gk-topbar">
      <div class="gk-topbar-left">
        <div class="gk-logo-mark">GK</div>
        <div>
          <div class="gk-brand-super">Gatekeeper AI</div>
          <div class="gk-brand-title">POPS - Push-Out Probability Score</div>
          <div class="gk-brand-sub">AI-Powered Retail Loss Prevention - Tracking + Classification + Scoring</div>
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

    # Hidden host for the tab count badges (app.js reads the data attribute).
    tab_counts_html = gr.HTML(_INITIAL["tab_counts_html"])

    # ── two-column layout ────────────────────────────────────────────────
    with gr.Row(elem_id="gk-main-row"):

        # ════════════════════════════════════════════════════════════════
        #  LEFT SIDEBAR
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=0, min_width=282, elem_id="gk-sidebar"):

            # ── Step 1: Video ────────────────────────────────────────────
            gr.HTML('<div class="sb-hdr sb-first"><span class="sb-hdr-step">1</span>VIDEO SOURCE</div>')
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
                # Defaults ON, which is what the pipeline did before this was
                # a choice — turning it off should be a decision, not something
                # that silently happens to a rerun.
                enable_pose = gr.Checkbox(
                    value=True,
                    label="Pose estimation",
                    info="Skeleton overlay on the tracked video. Off skips a "
                         "second model per frame — noticeably faster, and "
                         "nothing else changes: POPS, linking, zones, rules "
                         "and the JSON are all unaffected.",
                )

            # ── Step 3: Zones ────────────────────────────────────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">3</span>ZONES</div>')
            # Read-only glance state — visible from whichever tab you are on.
            # Renaming, retyping and removing live next to the canvas in the
            # Zone Editor tab, where you can see which polygon you are acting on.
            with gr.Group(elem_classes=["sb-pad"]):
                zones_summary_html = gr.HTML(_zone_summary_html([]))

            # ── Step 4: Rule thresholds ──────────────────────────────────
            # These were "configurable in engine/config.py" — i.e. not
            # configurable at all for anyone driving the demo. Re-evaluating
            # rules is a post-hoc pass over cached facts, so a slider plus
            # Recompute costs no GPU time at all.
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">4</span>RULE THRESHOLDS</div>')
            # All three thresholds are in seconds and the panel says so once, as
            # a unit suffix on the value fields (.sb-thresholds in app.css), so
            # the labels no longer each carry a "(s)" — at sidebar width that
            # suffix was what pushed every label onto a second line.
            with gr.Group(elem_classes=["sb-pad", "sb-thresholds"]):
                # The unit lives in `info` as well as in the CSS suffix: if the
                # stylesheet ever fails to load, "Blocked door / 5" with no unit
                # anywhere is a worse panel than a slightly wordier caption.
                blocked_door_s = gr.Slider(
                    5, 300, value=DEFAULT_THRESHOLDS["blocked_door_s"], step=5,
                    label="Blocked door",
                    info="Seconds a cart is stationary in a door zone")
                static_cart_s = gr.Slider(
                    10, 600, value=DEFAULT_THRESHOLDS["static_cart_s"], step=10,
                    label="Static cart",
                    info="Seconds a cart is stationary in an aisle / "
                         "analytics zone")
                abandoned_cart_s = gr.Slider(
                    10, 900, value=DEFAULT_THRESHOLDS["abandoned_cart_s"], step=10,
                    label="Unattended cart",
                    info="Seconds a cart is stationary with nobody nearby")
                reset_thresholds_btn = gr.Button(
                    "Reset to defaults", size="sm", variant="secondary",
                    elem_classes=["gk-sb-btn2"])

            # ── VLM Case Report ──────────────────────────────────────────
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
                    visible="Claude" in VLM_DEFAULT_BACKEND,
                )
                vlm_backend.change(
                    fn=lambda b: gr.update(visible="Claude" in b),
                    inputs=[vlm_backend], outputs=[vlm_api_key],
                )

            gr.HTML('<div class="sb-div"></div>')

            # ── Run button ───────────────────────────────────────────────
            with gr.Group(elem_classes=["sb-pad"]):
                run_btn = gr.Button(
                    "Run Analysis", variant="primary", size="lg", elem_id="gk-run-btn",
                )
                # Always mounted, never toggled. A visibility flip would be the
                # obvious design — show it only while a run is in flight — but
                # this build has twice been bitten by Gradio 6 failing to paint
                # a hidden→visible reveal (the alert banner is permanently
                # mounted for the same reason), and a Cancel button that does
                # not appear is worse than one that is always there. Pressing it
                # with nothing running is a logged no-op: CancelToken.request()
                # returns None.
                cancel_btn = gr.Button(
                    "Cancel run", variant="stop", size="sm",
                    elem_id="gk-cancel-btn",
                )
                # Both buttons carry `gk-sb-btn2` because they are a matched
                # pair: same size, same height, same hover. `variant="secondary"`
                # alone does not get there — see the .gk-sb-btn2 rules in
                # static/app.css for why the sidebar's own secondary rule never
                # lands on them.
                with gr.Row(elem_classes=["gk-sb-btn2-row"]):
                    invalidate_btn = gr.Button(
                        "Re-run detection", size="sm", variant="secondary",
                        elem_classes=["gk-sb-btn2"],
                    )
                    recompute_btn = gr.Button(
                        "Recompute analytics", size="sm", variant="secondary",
                        elem_classes=["gk-sb-btn2"],
                    )
                gr.HTML('<div class="sb-hint">“Recompute analytics” re-applies '
                        'zones and thresholds to the cached run - no GPU work, '
                        'near-instant.</div>')

            # ── Downloads ────────────────────────────────────────────────
            gr.HTML('<div class="sb-hdr"><span class="sb-hdr-step">↓</span>DOWNLOADS</div>')
            with gr.Group(elem_classes=["sb-pad", "sb-downloads"]):
                json_download = gr.File(label="Tracking JSON")
                case_report_download = gr.File(label="Case Report")

            gr.HTML('<div class="sb-foot">Built by <strong>Tanmay Thaker</strong> - Gatekeeper Systems</div>')

        # ════════════════════════════════════════════════════════════════
        #  MAIN CONTENT
        # ════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, elem_id="gk-main"):

            # Sticky top-of-page alert banner — the ONE canonical alert
            # surface. Also where run failures are reported.
            #
            # Mounted VISIBLE with an empty value, never toggled. It used to
            # start `visible=False` and be revealed by a gr.update, and on
            # gradio 6.8.0 (the conda `all` env) that reveal never painted:
            # the server sent value + visible:true, the toast fired, every
            # other panel filled, and the banner stayed hidden. 6.12.0 (the
            # venv) painted it fine. An empty value renders nothing — the host
            # is zero-height with `container=False` and the padding/background/
            # border reset in static/app.css — so "no alerts" looks identical
            # without depending on which gradio the env resolved.
            alert_banner_html = gr.HTML(
                value="", elem_id="gk-alert-host",
            )

            video_output = gr.Video(
                label="Tracked Output",
                autoplay=True, height=520,
            )
            run_summary_html = gr.HTML("")

            # ── Result tabs ──────────────────────────────────────────────
            # One flat row, original order. Each tab carries an id so a
            # finished run can select one (see run_analysis).
            with gr.Tabs(elem_id="gk-tabs") as result_tabs:

                # ── Zone Editor ─────────────────────────────────────────
                with gr.Tab("Zone Editor", id="tab_zones"):
                    with gr.Group(elem_id="zone-editor-fs-wrap"):

                        # Header bar with fullscreen toggle. The click is bound
                        # by id in static/app.js — Gradio 6 strips inline
                        # onclick="..." attrs as an XSS risk.
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


                        # ── Saved zones: rename / retype / remove ─────────
                        # A FIXED POOL of rows, shown and hidden with
                        # `visible=`. This started out as an @gr.render block
                        # driven by a revision counter, which is the idiomatic
                        # way to build a list whose length changes — but a
                        # render block sitting inside a gr.Tab wedges the page
                        # when a finished run updates every other component at
                        # once: Svelte's scheduler throws out of its own
                        # flush(), the main thread stops responding, and the
                        # results never paint. A/B'd it — same run, same data,
                        # clean with this block removed. A demo never has more
                        # than a handful of zones, so a pool costs nothing and
                        # creates no components at request time.
                        zone_mgr_hdr = gr.HTML(_zone_mgr_header(0))
                        zone_rows = []
                        for _slot in range(MAX_ZONE_ROWS):
                            # NOT `visible=False`. Gradio ignored `visible`
                            # updates on these Rows — the `hide` class it stamps
                            # at build time simply never came off, so removing a
                            # zone left an empty row on screen. Every slot is
                            # therefore always mounted and the EMPTY-SLOT marker
                            # in its meta cell is what CSS hides it by; gr.HTML
                            # value updates are reliable in a way layout-
                            # container visibility is not.
                            with gr.Row(elem_classes=["gk-zone-mgr-row"]) as _zrow:
                                _meta = gr.HTML(
                                    EMPTY_SLOT,
                                    elem_classes=["gk-zone-mgr-meta"])
                                _name = gr.Textbox(
                                    value="", show_label=False, container=False,
                                    interactive=True, scale=4,
                                    elem_classes=["gk-zone-mgr-name"])
                                _kind = gr.Dropdown(
                                    choices=ZONE_KIND_OPTIONS, value="analytics",
                                    show_label=False, container=False,
                                    interactive=True, scale=4,
                                    elem_classes=["gk-zone-mgr-kind"])
                                _applies = gr.Dropdown(
                                    choices=ZONE_APPLIES_OPTIONS, value="person",
                                    show_label=False, container=False,
                                    interactive=True, scale=3,
                                    elem_classes=["gk-zone-mgr-applies"])
                                _rm = gr.Button("Remove", size="sm", scale=2,
                                                elem_classes=["gk-zone-mgr-del"])
                            zone_rows.append({"row": _zrow, "meta": _meta,
                                              "name": _name, "kind": _kind,
                                              "applies": _applies, "rm": _rm})


                        # Undo strip — one level, covers a single Remove and a
                        # Clear all through the same path.
                        with gr.Row(
                                elem_classes=["gk-zone-mgr-undo"]) as zone_undo_row:
                            zone_undo_msg = gr.HTML("")
                            zone_restore_btn = gr.Button(
                                "Restore", size="sm",
                                elem_classes=["gk-zone-restore"])

                        # ── Saved zone sets ──────────────────────────────
                        # Where a clip's polygons are kept between runs. The
                        # header carries the folder, so "where did it save?"
                        # is answered on the page and not only in a toast.
                        preset_hdr = gr.HTML(_preset_hdr(None, [], None))
                        with gr.Row(elem_classes=["gk-preset-savebar"]):
                            with gr.Column(elem_classes=["zone-field-pair"]):
                                gr.HTML('<span class="zone-field-label">Set name</span>')
                                preset_name_in = gr.Textbox(
                                    show_label=False,
                                    value=zone_presets.DEFAULT_LABEL,
                                    placeholder="e.g. doorway-v2",
                                    elem_classes=["zone-name-box"],
                                )
                            preset_save_btn = gr.Button(
                                "\U0001F4BE  Save current zones", size="sm",
                                variant="primary",
                                elem_classes=["gk-preset-save"])
                            preset_refresh_btn = gr.Button(
                                "\u21BB  Refresh", size="sm",
                                elem_classes=["gk-preset-refresh"])

                        # Same fixed-pool construction as the zone manager
                        # above: every slot is always mounted and the EMPTY
                        # marker in its meta cell is what CSS hides it by,
                        # because `visible` updates on a Row inside a gr.Tab
                        # did not land.
                        preset_rows = []
                        for _pslot in range(MAX_PRESET_ROWS):
                            with gr.Row(elem_classes=["gk-preset-row"]):
                                _pmeta = gr.HTML(
                                    PRESET_EMPTY_SLOT,
                                    elem_classes=["gk-preset-meta"])
                                _pload = gr.Button(
                                    "Load", size="sm", scale=0,
                                    elem_classes=["gk-preset-load"])
                                _pdel = gr.Button(
                                    "Delete", size="sm", scale=0,
                                    elem_classes=["gk-preset-del"])
                            preset_rows.append({"meta": _pmeta,
                                                "load": _pload, "rm": _pdel})
                        preset_empty = gr.HTML(_preset_empty_html(None, []))

                        # Zones feed the rule engine, so any edit above makes a
                        # finished run stale. The button that fixes that lives
                        # in the sidebar; put a second one here so the fix is
                        # where the change was made. Wired further down, once
                        # its output components exist.
                        with gr.Row(elem_classes=["gk-zone-mgr-foot"]):
                            zone_recompute_btn = gr.Button(
                                "Apply zone changes (recompute analytics)",
                                size="sm", variant="secondary",
                                elem_classes=["gk-zone-recompute"])

                # ── POPS ────────────────────────────────────────────────
                with gr.Tab("POPS", id="tab_pops"):
                    pops_html = gr.HTML(_INITIAL["pops_html"])

                # ── Events ──────────────────────────────────────────────
                with gr.Tab("Events", id="tab_events"):
                    events_html = gr.HTML(_INITIAL["events_html"])

                # ── Operational Alerts (rule engine) ────────────────────
                # Separate tab from Events on purpose: these are rule-engine
                # outcomes with their own severity vocabulary, not POPS
                # theft-risk events, and they must not share the event log.
                with gr.Tab("Operational Alerts", id="tab_ops"):
                    gr.HTML(
                        "<div class='gk-section-note'>Blocked doors, static "
                        "carts, unattended carts and incoming empty carts. "
                        "One row per incident - the same categories appear "
                        "per cart in the <b>POPS</b> tab, where a cart in "
                        "several categories shows all of them. Tune the "
                        "thresholds in the sidebar and hit <b>Recompute "
                        "analytics</b> to re-evaluate without re-running "
                        "detection.</div>")
                    gr.HTML(ui_builder.build_category_reference())
                    ops_alerts_html = gr.HTML(_INITIAL["ops_alerts_html"])

                # ── Analytics ───────────────────────────────────────────
                with gr.Tab("Analytics", id="tab_analytics"):
                    analytics_summary_html = gr.HTML(
                        _INITIAL["analytics_summary_html"])
                    spikes_html = gr.HTML(_INITIAL["spikes_html"])
                    with gr.Row():
                        with gr.Column(scale=2):
                            heatmap_image = gr.Image(
                                label="Traffic Heatmap", interactive=False, height=360,
                            )
                        with gr.Column(scale=1, min_width=160):
                            heatmap_file = gr.File(label="Download Heatmap PNG")
                    dwell_html   = gr.HTML(_INITIAL["dwell_html"])
                    journey_html = gr.HTML(_INITIAL["journey_html"])

                # The 3D View, Bird's-Eye 2D and Floor Map tabs were removed
                # here. Each rendered a full HTML document with per-frame data
                # inside an iframe srcdoc; together with the raw JSON they made
                # the finished run arrive as one payload the browser could not
                # process without going unresponsive.

                # ── Case Report ─────────────────────────────────────────
                with gr.Tab("Case Report", id="tab_case"):
                    case_report_html = gr.HTML(_INITIAL["case_report_html"])

                # ── Detection ───────────────────────────────────────────
                with gr.Tab("Detection", id="tab_detection"):
                    detection_html = gr.HTML(_INITIAL["detection_html"])

                # ── Video Info ──────────────────────────────────────────
                with gr.Tab("Video Info", id="tab_videoinfo"):
                    video_info_html = gr.HTML(_INITIAL["video_info_html"])

                # ── Config ──────────────────────────────────────────────
                with gr.Tab("Config", id="tab_config"):
                    config_html = gr.HTML(_INITIAL["config_html"])

                # ── Legend ──────────────────────────────────────────────
                with gr.Tab("Legend", id="tab_legend"):
                    legend_html = gr.HTML(_INITIAL["legend_html"])

            # ── JSON output (collapsed by default — rarely opened) ───────
            # A PREVIEW only. Every frame is logged (JSON_EVERY_N_FRAMES = 1),
            # so the full document runs ~7 KB per frame — tens of MB on a real
            # clip — and handing all of it to a component value is what made
            # the page unresponsive the instant a run finished.
            with gr.Accordion("Raw tracking JSON (advanced)", open=False):
                json_output = gr.Code(
                    label="Tracking + Classification JSON (preview)",
                    language="json", lines=10, elem_classes=["json-scroll"],
                )
                with gr.Row():
                    fullscreen_btn = gr.Button(
                        "View preview fullscreen", variant="secondary", size="sm",
                    )
                    open_json_btn = gr.Button(
                        "Open full JSON in a new tab", variant="secondary",
                        size="sm",
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
    #  EVENT WIRING
    # ════════════════════════════════════════════════════════════════════
    fullscreen_btn.click(
        fn=None, inputs=[json_output], outputs=[],
        js="""(t) => {
            const o = document.getElementById('json-fullscreen-overlay');
            const c = document.getElementById('json-fullscreen-content');
            if (o && c) { c.textContent = t || 'No JSON yet.'; o.classList.add('active'); }
        }""",
    )

    # The complete JSON is served as a FILE, opened in its own tab. It is never
    # injected into this page: a multi-MB text node blocks the main thread just
    # as hard on click as it did on load, and the browser's own viewer handles a
    # large document far better than a <div> can.
    #
    # Gradio 6 hands a js-only handler the frontend's FileData
    # ({path, url, size, orig_name, ...}), so the URL comes from the component
    # itself rather than from a hand-built /gradio_api/file= path. Shape is
    # guarded anyway — a miss falls back to a message, never a dead button.
    open_json_btn.click(
        fn=None, inputs=[json_download], outputs=[],
        js="""(f) => {
            const d = Array.isArray(f) ? f[0] : f;
            // Only `url` is usable here — `path` is a server-side filesystem
            // path the browser cannot fetch.
            const url = d && d.url;
            if (url) { window.open(url, '_blank', 'noopener'); return; }
            const o = document.getElementById('json-fullscreen-overlay');
            const c = document.getElementById('json-fullscreen-content');
            if (o && c) {
                c.textContent = d
                    ? 'Could not resolve a URL for the JSON file. Use the '
                      + 'Download tracking JSON button in the sidebar.'
                    : 'No JSON yet - run an analysis first.';
                o.classList.add('active');
            }
        }""",
    )

    # Everything the zone pool needs to redraw itself, in one place. Built from
    # the widget list rather than typed out, so a row gaining a control cannot
    # leave the outputs list one short — that mismatch is silent in Gradio.
    ZONE_UI_OUT = [zones_state, zone_canvas, zones_summary_html, zone_mgr_hdr,
                   zone_undo, zone_undo_msg]
    ZONE_UI_OUT += [w[f] for w in zone_rows for f in _ZONE_ROW_FIELDS]
    assert len(ZONE_UI_OUT) == _ZONE_HEAD_N + len(_ZONE_ROW_FIELDS) * MAX_ZONE_ROWS

    ZONE_UI_IN = [zones_state, first_frame_state, current_poly, zone_undo]

    # Row `slot` always edits `zones[slot]`: the pool is rebuilt from the list
    # on every change, so the position a control sits at and the zone it acts on
    # cannot drift apart.
    for _slot, _w in enumerate(zone_rows):
        # .blur/.submit rather than .change — .change fires per keystroke, and
        # each one would write state and rewrite the box being typed in.
        for _ev in (_w["name"].blur, _w["name"].submit):
            _ev(fn=partial(rename_zone, _slot),
                inputs=[_w["name"]] + ZONE_UI_IN, outputs=ZONE_UI_OUT)
        # .input rather than .change — .change also fires when the refresh sets
        # the value back, which is a loop.
        _w["kind"].input(fn=partial(retype_zone, _slot),
                         inputs=[_w["kind"]] + ZONE_UI_IN, outputs=ZONE_UI_OUT)
        _w["applies"].input(fn=partial(set_zone_applies, _slot),
                            inputs=[_w["applies"]] + ZONE_UI_IN,
                            outputs=ZONE_UI_OUT)
        _w["rm"].click(fn=partial(remove_zone, _slot),
                       inputs=ZONE_UI_IN, outputs=ZONE_UI_OUT)

    zone_restore_btn.click(
        fn=restore_zones,
        inputs=[zone_undo, zones_state, first_frame_state, current_poly],
        outputs=ZONE_UI_OUT,
    )

    # The flush is wired at the bottom of this block, once FLUSH_OUTPUTS exists —
    # picking a different clip makes the panels on screen belong to a video the
    # user is no longer looking at.
    # Everything the saved-set panel needs to redraw itself. Built from the
    # row list rather than typed out, so a row gaining a control cannot leave
    # the outputs list one short — that mismatch is silent in Gradio.
    PRESET_UI_OUT = [preset_paths, preset_loaded, preset_hdr, preset_empty]
    PRESET_UI_OUT += [w["meta"] for w in preset_rows]
    assert len(PRESET_UI_OUT) == _PRESET_HEAD_N + MAX_PRESET_ROWS

    _video_change_event = video_input.change(
        fn=on_video_upload,
        inputs=[video_input, zones_state, video_key_state],
        outputs=[first_frame_state, zones_state, current_poly,
                 video_key_state, zone_canvas, zones_summary_html,
                 preset_loaded],
    ).then(refresh_zone_ui, ZONE_UI_IN, ZONE_UI_OUT
    ).then(refresh_preset_ui, [video_input, preset_loaded], PRESET_UI_OUT)

    # ── Saved zone sets ──────────────────────────────────────────────────
    # Row `slot` always acts on `paths[slot]`: the pool is rebuilt from the
    # folder on every action, so the position a button sits at and the file it
    # loads or deletes cannot drift apart.
    #
    # Load writes zones_state and the undo slot, then hands off to
    # refresh_zone_ui for the canvas, the summary and the zone row pool — the
    # same chain every other zone mutation goes through, because setting
    # zones_state alone leaves the manager rows showing the previous list. The
    # second .then() re-marks which set is on screen.
    for _pslot, _pw in enumerate(preset_rows):
        _pw["load"].click(
            fn=partial(load_zone_set, _pslot),
            inputs=[preset_paths, zones_state, first_frame_state,
                    preset_loaded],
            outputs=[zones_state, zone_undo, preset_loaded],
        ).then(refresh_zone_ui, ZONE_UI_IN, ZONE_UI_OUT
        ).then(refresh_preset_ui, [video_input, preset_loaded], PRESET_UI_OUT)

        _pw["rm"].click(
            fn=partial(delete_zone_set, _pslot),
            inputs=[preset_paths, video_input, preset_loaded],
            outputs=PRESET_UI_OUT,
        )

    preset_save_btn.click(
        fn=save_zone_set,
        inputs=[video_input, preset_name_in, zones_state, first_frame_state,
                preset_loaded],
        outputs=PRESET_UI_OUT,
    )

    # For a folder changed outside the app — another window, or a file dropped
    # in by hand. Deliberately a button rather than a refresh on dropdown
    # focus: writing values back into a control the user is interacting with
    # is the same class of Gradio-6 behaviour as the `visible` toggle that
    # never repaints and the render block that wedges a tab.
    preset_refresh_btn.click(
        fn=refresh_preset_ui, inputs=[video_input, preset_loaded],
        outputs=PRESET_UI_OUT,
    )

    zone_canvas.select(
        fn=on_canvas_click,
        inputs=[current_poly, first_frame_state, zones_state],
        outputs=[current_poly, zone_canvas],
    )

    close_btn.click(
        fn=close_polygon,
        inputs=[current_poly, zone_name_in, zone_applies, zone_kind,
                zones_state, first_frame_state],
        outputs=[zones_state, current_poly, zone_canvas, zones_summary_html],
    ).then(refresh_zone_ui, ZONE_UI_IN, ZONE_UI_OUT)

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
        inputs=[zones_state, first_frame_state],
        outputs=[zones_state, current_poly, zone_canvas, zones_summary_html,
                 zone_undo],
    ).then(refresh_zone_ui, ZONE_UI_IN, ZONE_UI_OUT)

    reset_thresholds_btn.click(
        fn=reset_thresholds, inputs=[],
        outputs=[blocked_door_s, static_cart_s, abandoned_cart_s],
    )

    invalidate_btn.click(
        fn=invalidate_cache_handler, inputs=[video_input], outputs=[],
    )

    # name -> component for every run output, so the two outputs lists below are
    # DERIVED from RUN_OUTPUT_NAMES rather than typed out a second time. The
    # literal list this replaced was a positional twin of that table, and `_IDX`
    # reads values by position: a slot inserted in one and not the other mapped
    # every value after it to the wrong component, silently.
    RUN_COMPONENTS = {
        "video_output": video_output,
        "json_download": json_download,
        "json_output": json_output,
        "video_info_html": video_info_html,
        "detection_html": detection_html,
        "config_html": config_html,
        "legend_html": legend_html,
        "pops_html": pops_html,
        "events_html": events_html,
        "case_report_html": case_report_html,
        "case_report_download": case_report_download,
        "analytics_summary_html": analytics_summary_html,
        "spikes_html": spikes_html,
        "dwell_html": dwell_html,
        "journey_html": journey_html,
        "heatmap_image": heatmap_image,
        "heatmap_file": heatmap_file,
        "alert_banner_html": alert_banner_html,
        "ops_alerts_html": ops_alerts_html,
        "run_summary_html": run_summary_html,
        "tab_counts_html": tab_counts_html,
        "result_tabs": result_tabs,
    }
    assert set(RUN_COMPONENTS) == set(RUN_OUTPUT_NAMES), \
        "RUN_COMPONENTS and RUN_OUTPUT_NAMES have drifted apart"
    RUN_OUTPUTS   = [RUN_COMPONENTS[n] for n in RUN_OUTPUT_NAMES]
    FLUSH_OUTPUTS = [RUN_COMPONENTS[n] for n in FLUSH_OUTPUT_NAMES]

    # Every panel goes back to its empty state BEFORE the pipeline starts, so a
    # second run never renders on top of the first one's findings.
    #
    # Its own event, `.then()`-chained rather than a second `run_btn.click(...)`:
    # two handlers on one trigger are dispatched together, not sequenced, so the
    # flush could land after the results and wipe them. Chaining makes the order
    # structural.
    #
    # Not a first `yield` inside run_analysis either — that turns it into a
    # generator, and the docstring there records what that cost: Gradio holds a
    # pending overlay over every output component until the function RETURNS, so
    # an intermediate yield paints nothing and the finished dashboard sat under
    # "Processing frames — 100.0%" for as long as the VLM took.
    #
    # `show_progress="hidden"`: a flush is instant, and the default would paint a
    # progress tracker over all twenty components to say so.
    _flush_event = run_btn.click(
        fn=flush_results,
        inputs=[video_input],
        outputs=FLUSH_OUTPUTS,
        show_progress="hidden",
    )

    _run_event = _flush_event.then(
        fn=run_analysis,
        inputs=[video_input, camera_placement, vlm_backend, vlm_api_key,
                zones_state, blocked_door_s, static_cart_s, abandoned_cart_s,
                enable_pose],
        outputs=RUN_OUTPUTS,
        # Progress indicator on the VIDEO PANEL ONLY.
        #
        # Gradio's default paints a status tracker over EVERY output component,
        # and this event has 22. Two consequences, both reported from the UI:
        #
        #   * the alert banner — the one panel whose whole job is to be read the
        #     moment it appears — spent the run underneath a progress bar, and
        #     stayed covered while the bar sat pinned at 100% through the
        #     post-loop tail (encode, JSON, analytics, heat-map);
        #   * every progress message re-rendered 22 trackers, which is the cost
        #     PROGRESS_MAX_UPDATES in tracker.py exists to throttle.
        #
        # Pointing it at one component fixes both: the bar lands on the thing
        # actually being produced, and the throttle stops fighting 22 rerenders.
        show_progress_on=[video_output],
    )

    # BOTH follow-ups hang off `_run_event` directly. They are NOT chained to
    # each other, and that is load-bearing:
    #
    # A dependency with `fn=None, js=...` is JS-ONLY — the client evaluates it
    # in the browser and never opens a backend submission for it. Gradio's
    # client dispatches a dependency's `.then()` targets only on the SUBMISSION
    # branch, when the SSE stream reaches stage "complete" (see the `u.type`
    # switch in the frontend bundle: the `"data"` and `"void"` branches call
    # neither the `success` nor the `all` trigger list). So a js-only event is a
    # DEAD END for the chain: anything `.then()`-ed after it never runs.
    #
    # Chaining these serially — run → js tab-switch → finalize — meant
    # finalize_case_report_handler was never called at all. Whenever frames were
    # captured, the case-report tab kept the "Generating case report…"
    # placeholder that process_video returns under defer_case_report=True, and
    # kept it indefinitely. Nothing was slow and nothing errored; the second
    # phase simply never started.
    _run_event.then(
        # Tab switch done in the BROWSER, by clicking the real tab button — not
        # by returning gr.Tabs(selected=...) from Python.
        #
        # Returning the component was tried twice: once inside run_analysis, and
        # once as its own chained event. Both wedged the tab. Moving it to its own
        # event did fix the CONTENT (video, panels and badges all render now), but
        # the selection itself never applied and the page still locked — so the
        # Tabs config update is the part Gradio's client cannot reconcile here,
        # regardless of which flush it lands in.
        #
        # A click is what a user would do: it goes through Gradio's own tab
        # handler instead of asking it to diff a layout component that was
        # constructed outside the Blocks tree. No-op when app.js is disabled.
        fn=None,
        inputs=None,
        outputs=None,
        js=_GOTO_RESULTS_JS,
    )
    _finalize_event = _run_event.then(
        # Deliberately a CHAINED event, not a second yield inside run_analysis.
        # An event holds a pending overlay over each of its output components
        # until it returns; with the VLM inside the run event, that overlay sat
        # on top of the whole finished dashboard (including the tab bar) for as
        # long as a local model took to load. Here it covers only these two.
        fn=finalize_case_report_handler,
        inputs=[],
        outputs=[case_report_html, case_report_download],
    )

    # Declared HERE, after both events exist: `cancels=` takes the event objects
    # themselves, so this cannot move up next to the button.
    #
    # Both events are listed. Cancelling only the run would leave its chained
    # finalize to fire anyway and start a VLM pass on a run the user just
    # stopped; cancelling only the finalize would leave a frame loop running.
    # And `cancels=` alone stops neither once started — see cancel_run_handler.
    cancel_btn.click(
        fn=cancel_run_handler,
        inputs=None,
        outputs=None,
        cancels=[_run_event, _finalize_event],
        # A flag flip is instant, and the default would paint a progress tracker
        # over every component to say so.
        show_progress="hidden",
    )

    # Loading a different clip clears the results too, for the same reason the
    # Run chain starts with a flush: the POPS table, event log and heat-map on
    # screen describe the OLD video, and nothing about them says so. Chained
    # after the zone refresh rather than onto `video_input.change` directly, so
    # it cannot race the zone reset that shares this trigger.
    _video_change_event.then(
        fn=flush_results,
        inputs=[video_input],
        outputs=FLUSH_OUTPUTS,
        show_progress="hidden",
    )

    # Same handler behind both entry points: the sidebar button, and the one
    # under the zone list — next to the edits that made the results stale.
    for _recompute_src in (recompute_btn, zone_recompute_btn):
        _recompute_src.click(
            fn=recompute_analytics_handler,
            inputs=[video_input, zones_state, camera_placement,
                    blocked_door_s, static_cart_s, abandoned_cart_s],
            outputs=[analytics_summary_html, spikes_html, dwell_html,
                     journey_html, heatmap_image, heatmap_file, ops_alerts_html,
                     alert_banner_html, tab_counts_html, pops_html],
        )


if __name__ == "__main__":
    # `analytics_out_dir` in tracker.py defaults to <project_root>/temp —
    # Gradio 6 only serves files inside its allowlist, so without this entry
    # gr.File("Download Heatmap PNG") sticks at "Uploading…" and the heatmap
    # image stays blank even though the PNG is on disk.
    _ANALYTICS_OUT_DIR = os.path.join(_HERE, "temp")
    os.makedirs(_ANALYTICS_OUT_DIR, exist_ok=True)
    _allowed_paths = [_ANALYTICS_OUT_DIR, str(_STATIC)]
    if os.path.isdir(TEST_VIDEO_DIR):
        _allowed_paths.append(TEST_VIDEO_DIR)

    # Announce the bisection switches at startup, so the log of any run says
    # which configuration produced it. Without this, "it froze again" and "it
    # froze again, with app.js off" are indistinguishable after the fact.
    #: Printed here, not at import time: the rows below are the only thing
    #: under this heading, and everything between the import and this point
    #: is model loading. A heading that appears a minute before its own
    #: content reads as a hang.
    ui.phase_free("Runtime")
    _sw = _active_switches()
    if _sw:
        ui.warn("bisection", ", ".join(_sw))
        if NO_APP_JS:
            ui.note("launching WITHOUT static/app.js")
        if NO_APP_CSS:
            ui.note("launching WITHOUT static/app.css")
    else:
        ui.ok("bisection", "none — normal run")
    print()
#5173
    # Guarded so nothing the server does can put a raw traceback on screen
    # during a demo. Ctrl-C is a normal way to stop it, not a crash, and gets
    # a plain line; anything else goes to console_noise.report_fatal, which
    # prints a tidy block and puts the traceback in temp/server_errors.log
    # rather than on the console. The exit code stays non-zero so
    # run_demo.bat can still tell it failed.
    try:
        demo.launch( 
            server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True, 
            allowed_paths=_allowed_paths,
            theme=_THEME,
            css=(None if NO_APP_CSS else _CSS),
            js=(None if NO_APP_JS else _JS),
        )
    except KeyboardInterrupt:
        print()
        ui.info("stopped", "demo closed from the console")
    except Exception as _exc:
        console_noise.report_fatal(_exc)
        raise SystemExit(1)
