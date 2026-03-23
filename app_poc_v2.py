# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
POPS - Push-Out Probability Score: Interactive Demo (v2 — modular)
===================================================================
Thin Gradio UI wrapper.  All logic lives in engine/*.py.
python code/demo_app_v2.py
"""
import os
import traceback
import gradio as gr

from engine import TrackingEngine, SAMPLE_VIDEOS

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------
engine = TrackingEngine(device='auto')


def run_analysis(video_path, camera_placement):
    if video_path is None:
        gr.Warning("Please upload or select a video first.")
        return None, None, "", "<p>No video selected</p>", "", "", "", "", ""
    try:
        return engine.process_video(
            video_path,
            camera_placement=camera_placement,
        )
    except Exception as e:
        gr.Warning(f"Error: {str(e)}")
        traceback.print_exc()
        return None, None, f'{{"error": "{str(e)}"}}', f"<p>Error: {str(e)}</p>", "", "", "", "", ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="POPS - Push-Out Probability Score | Gatekeeper AI",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;500;600;700;800;900&display=swap');
    .gradio-container, .gradio-container * { font-family: 'Nunito Sans', 'Segoe UI', Tahoma, sans-serif !important; }
    .gk-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 1rem 2rem; border-radius: 12px; margin-bottom: 0.8rem;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.2);
    }
    .gk-header-row { display: flex; align-items: center; justify-content: space-between; }
    .gk-brand { font-size: 0.7rem; font-weight: 700; letter-spacing: 3px; color: #93c5fd; text-transform: uppercase; margin: 0 0 0.1rem 0; }
    .gk-title { font-size: 1.35rem; font-weight: 800; color: #fff; margin: 0; line-height: 1.2; }
    .gk-subtitle { font-size: 0.8rem; color: #bfdbfe; margin: 0.1rem 0 0 0; }
    .gk-info-chips { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .gk-chip { background: rgba(255,255,255,0.15); color: #dbeafe; padding: 3px 12px; border-radius: 16px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px; }
    .gk-footer { text-align: center; color: #94a3b8; font-size: 0.7rem; padding: 0.5rem 0 0.3rem 0; border-top: 1px solid #e2e8f0; margin-top: 0.5rem; }
    .gk-footer strong { color: #2563eb; }
    .tabs { min-height: 0 !important; }
    .tab-nav { border-bottom: 2px solid #e2e8f0 !important; gap: 4px !important; background: transparent !important; }
    .gradio-container .tab-nav button, .dark .tab-nav button, .tab-nav button {
        font-size: 0.85rem !important; padding: 8px 18px !important; font-weight: 700 !important;
        color: #fff !important; background: #3b82f6 !important; border: 2px solid #2563eb !important;
        border-bottom: none !important; border-radius: 8px 8px 0 0 !important; opacity: 1 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    .gradio-container .tab-nav button:hover, .dark .tab-nav button:hover, .tab-nav button:hover { background: #2563eb !important; color: #fff !important; }
    .gradio-container .tab-nav button.selected, .dark .tab-nav button.selected, .tab-nav button.selected {
        background: #1d4ed8 !important; color: #fff !important; border-color: #1d4ed8 !important;
        box-shadow: 0 -2px 6px rgba(29,78,216,0.4) !important;
    }
    .json-scroll { max-height: 300px; overflow-y: auto; }
    .json-scroll textarea { min-height: 280px !important; }
    #json-fullscreen-overlay {
        display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.85); z-index: 9999; padding: 30px; box-sizing: border-box;
    }
    #json-fullscreen-overlay.active { display: flex; flex-direction: column; }
    #json-fullscreen-close {
        align-self: flex-end; background: #ef4444; color: #fff; border: none;
        padding: 8px 20px; border-radius: 6px; font-size: 0.9rem; font-weight: 700; cursor: pointer; margin-bottom: 10px;
    }
    #json-fullscreen-close:hover { background: #dc2626; }
    #json-fullscreen-content {
        flex: 1; background: #0d1117; color: #c9d1d9; border-radius: 8px;
        padding: 20px; overflow: auto; font-family: monospace; font-size: 0.85rem; white-space: pre; line-height: 1.5;
    }
    .cls-section { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; }
    .cls-section label, .cls-section .label-wrap span { font-size: 1.25rem !important; font-weight: 700 !important; }
    .cls-section select, .cls-section input, .cls-section .wrap-inner, .cls-section .secondary-wrap,
    .cls-section .single-select, .cls-section .token { font-size: 1.15rem !important; }
    .gradio-container .gradio-dropdown label, .gradio-container .gradio-dropdown .label-wrap span { font-size: 1.25rem !important; font-weight: 700 !important; }
    .gradio-container .gradio-dropdown select, .gradio-container .gradio-dropdown input,
    .gradio-container .gradio-dropdown .wrap-inner, .gradio-container .gradio-dropdown .secondary-wrap,
    .gradio-container .gradio-dropdown .single-select, .gradio-container .gradio-dropdown .token { font-size: 1.15rem !important; }
    .progress-text, .meta-text, .meta-text-center, .timer { display: none !important; }
    .eta-bar { display: none !important; }
    """,
) as demo:

    # Header
    gr.HTML("""
    <div class="gk-header">
        <div class="gk-header-row">
            <div>
                <p class="gk-brand">GATEKEEPER AI</p>
                <h1 class="gk-title">POPS &mdash; Push-Out Probability Score</h1>
                <p class="gk-subtitle">AI-Powered Retail Loss Prevention &mdash; Tracking + Classification + Scoring</p>
                <p class="gk-subtitle" style="margin-top:4px;font-size:0.7rem;color:#93c5fd;"><strong style="color:#fff;">Engineering</strong> &mdash; Gatekeeper Systems</p>
            </div>
            <div class="gk-info-chips">
                <span class="gk-chip">YOLOv26m</span>
                <span class="gk-chip">BoTSORT</span>
                <span class="gk-chip">Cart Quality</span>
                <span class="gk-chip">Fill + Bag</span>
                <span class="gk-chip">POPS Score</span>
            </div>
        </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        # Left panel — controls
        with gr.Column(scale=1, min_width=280):
            video_input = gr.Video(label="Upload Video", sources=["upload"], height=200)
            if SAMPLE_VIDEOS:
                sample_names = [os.path.basename(v) for v in SAMPLE_VIDEOS]
                sample_dropdown = gr.Dropdown(
                    choices=list(zip(sample_names, SAMPLE_VIDEOS)),
                    label="Sample Videos",
                    info=f"{len(SAMPLE_VIDEOS)} test clips",
                )
                sample_dropdown.change(fn=lambda x: x, inputs=[sample_dropdown], outputs=[video_input])

            gr.Markdown("### Settings")
            with gr.Group(elem_classes=["cls-section"]):
                camera_placement = gr.Dropdown(
                    choices=["Outside (facing entrance)", "Inside (facing exit)",
                             "Inside (exit on right)", "Inside (exit on left)",
                             "Inside (exit on both sides)"],
                    value="Outside (facing entrance)",
                    label="Camera Placement",
                )

            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

        # Right panel — outputs
        with gr.Column(scale=2, min_width=480):
            video_output = gr.Video(label="Tracked Video", autoplay=True, height=340)
            with gr.Tabs():
                with gr.Tab("Events"):
                    events_html = gr.HTML("")
                with gr.Tab("POPS"):
                    pops_html = gr.HTML("")
                with gr.Tab("Detection"):
                    detection_html = gr.HTML("")
                with gr.Tab("Video Info"):
                    video_info_html = gr.HTML("")
                with gr.Tab("Config"):
                    config_html = gr.HTML("")
                with gr.Tab("Legend"):
                    legend_html = gr.HTML("")

    # Fullscreen JSON overlay
    gr.HTML("""
    <div id="json-fullscreen-overlay">
        <button id="json-fullscreen-close" onclick="document.getElementById('json-fullscreen-overlay').classList.remove('active');">
            Close Fullscreen
        </button>
        <div id="json-fullscreen-content"></div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            json_output = gr.Code(label="Tracking + Classification JSON", language="json", lines=14, elem_classes=["json-scroll"])
            fullscreen_btn = gr.Button("View JSON Fullscreen", variant="secondary", size="sm")
        with gr.Column(scale=1, min_width=180):
            json_download = gr.File(label="Download JSON")

    fullscreen_btn.click(
        fn=None, inputs=[json_output], outputs=[],
        js="""
        (jsonText) => {
            const overlay = document.getElementById('json-fullscreen-overlay');
            const content = document.getElementById('json-fullscreen-content');
            if (overlay && content) {
                content.textContent = jsonText || 'No JSON data yet.';
                overlay.classList.add('active');
            }
        }
        """,
    )

    run_btn.click(
        fn=run_analysis,
        inputs=[video_input, camera_placement],
        outputs=[video_output, json_download, json_output, video_info_html, detection_html,
                 config_html, legend_html, pops_html, events_html],
    )

    gr.HTML("""
    <div class="gk-footer">
        <strong>GATEKEEPER AI</strong> &bull; POPS Demo &bull; YOLOv26m + BoTSORT + Cart Quality + Fill/Bag &bull; Retail Loss Prevention
        <br>Built by <strong>Tanmay Thaker</strong>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
