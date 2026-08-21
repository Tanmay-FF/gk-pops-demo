# POPS -- Push-Out Probability Score

**AI-powered system to identify potential cart push-out theft in real time using computer vision, multi-object tracking, and behavioral scoring.**

---

## Overview

POPS processes surveillance video to detect, track, and score shopping carts and persons in a retail environment to assesss the likelihood of a push-out theft event. The system outputs an annotated video with bounding boxes, trails, classification overlays, and a detailed JSON report with per-frame and per-cart analytics. It links persons to carts, classifies cart contents, analyzes motion direction, and computes a **0–100 risk score**.

---

## Demonstration

**Double-click `run_demo.bat`.** That is the whole procedure on Windows.

Nothing has to be installed first. It finds or fetches a Python 3.11/3.12,
builds the environment, checks the model weights, and opens the demo in your
browser at **http://localhost:7860**. The first run downloads several GB and
takes 10-20 minutes; later runs start in seconds.

Before the first run, drop a video clip into `sample_videos/`. Clips are never
committed since `*.mp4` is gitignored, so whoever sends you this project sends the clip separately.

```
run_demo.bat                 # normal run
run_demo.bat --check-only    # prepare the machine, do not open the demo
run_demo.bat --no-model      # do not pre-fetch the 4 GB case-report model
```


Starting from a fresh clone instead of a copied folder:

```bash
git clone https://github.com/Tanmay-FF/gk-pops-demo
cd gk-pops-demo
git lfs install && git lfs pull    # weights are LFS objects
run_demo.bat
```

Handing this to someone non-technical: give them
[RUN_THIS_DEMO.md](RUN_THIS_DEMO.md). A shorter overview of this build is in
[QUICKSTART.md](QUICKSTART.md). Details of what `run_demo.bat` does are under
[Running it](#running-it-no-setup-knowledge-required) below.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Gradio Web Interface                         │
│              app_poc_v2.py -- Upload / Sample Videos                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TrackingEngine (tracker.py)                      │
│              Orchestrates the full per-frame pipeline                │
│                                                                     │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  YOLOv26m │  │  BoTSORT   │  │ Cart Re-ID   │  │   Motion    │  │
│  │ Detection │─▶│  Tracking  │─▶│  (distance)  │─▶│  Analysis   │  │
│  └───────────┘  └────────────┘  └──────────────┘  └──────┬──────┘  │
│                                                          │         │
│  ┌───────────────────┐  ┌──────────────────────┐         │         │
│  │  PersonCartLinker  │  │   CartClassifier     │         │         │
│  │  (overlap + co-   │  │  Stage 1: Quality    │         │         │
│  │   movement)       │  │  Stage 2: Fill + Bag │         │         │
│  └────────┬──────────┘  └──────────┬───────────┘         │         │
│           │                        │                     │         │
│           ▼                        ▼                     ▼         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    POPS Scoring Engine                       │   │
│  │         Direction + Fill + Bag + Speed + Link state          │   │
│  │                  → 0–100 risk score                          │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                      │
│  ┌──────────────┐  ┌───────┴───────┐  ┌────────────────────────┐  │
│  │   Renderer   │  │  Event Logger │  │   Post-Processing      │  │
│  │  (OpenCV)    │  │  + Classifier │  │  Reconciliation +      │  │
│  │              │  │               │  │  Confidence Voting      │  │
│  └──────┬───────┘  └───────┬───────┘  └───────────┬────────────┘  │
└─────────┼──────────────────┼──────────────────────┼────────────────┘
          ▼                  ▼                      ▼
   Annotated MP4        JSON Report          HTML Dashboards
   (NVENC/x264)      (per-frame data)     (Events, POPS, Config)
```

---

## Processing Pipeline

```
Video Frame
    │
    ├─ 1. YOLO Detection ──────────── Detect persons + carts (640px input)
    │
    ├─ 2. BoTSORT Tracking ────────── Assign persistent IDs across frames
    │
    ├─ 3. Cart Re-Identification ──── Recover identity after brief occlusion
    │        └── Distance-based matching (< 200px, < 15 frames absent)
    │
    ├─ 4. Person–Cart Linking ─────── Associate persons with carts
    │        ├── IoU overlap accumulation
    │        ├── Co-movement detection (cosine similarity)
    │        ├── Behind-the-cart bonus (1.5× for trailing person)
    │        ├── Adaptive thresholds (6 frames single, 20 contested)
    │        └── Drift detection + tracker swap recovery
    │
    ├─ 5. Classification (every N frames)
    │        ├── Stage 1 -- Cart Quality: valid_cart vs unclear
    │        │     └── Threshold: 0.50 confidence
    │        └── Stage 2 -- Fill + Bag (valid carts only)
    │              ├── Fill:  empty / partial / full
    │              ├── Bag:   bagged / unbagged / not_applicable
    │              └── Empty override: force empty if P(empty) ≥ 0.50
    │
    ├─ 6. Motion Analysis ─────────── Speed, direction, acceleration
    │        ├── Speed: STATIC < 10 < SLOW < 100 < MEDIUM < 180 < FAST
    │        └── Direction: INBOUND / OUTBOUND / UNKNOWN
    │
    ├─ 7. POPS Scoring ────────────── 0–100 risk score per cart
    │
    └─ 8. Rendering + Export ──────── Annotated video, JSON, HTML
```

---

## Event Classification

| Score Range | Event | Severity |
|:-----------:|:------|:--------:|
| 71–100 | PUSHOUT ALERT / HIGH PRIORITY | High |
| 31–70 | MEDIUM PRIORITY / UNLINKED EXIT / ABANDONED CART | Medium |
| 0–30 | MONITORING / LOW PRIORITY / INBOUND | Low |

---

## Classification Models

### Stage 1 -- Cart Quality (valid vs unclear)

| Component | Detail |
|:----------|:-------|
| Architecture | MobileNetV3 + LayerNorm + Dropout(0.3) + Linear(2) |
| Input | 224 × 224 crop, ImageNet normalization |
| Classes | `valid_cart`, `unclear` |
| Threshold | 0.50 confidence for `valid_cart` |
| Temperature | 0.779 (from calibration.json) |

### Stage 2 -- Fill + Bag (dual-head)

| Component | Detail |
|:----------|:-------|
| Architecture | Shared backbone → two heads |
| Fill Head | Linear(features → 3): `empty`, `partial`, `full` |
| Bag Head | Linear(features + fill_logits → 3): `bagged`, `unbagged`, `not_applicable` |
| Fill Temperature | 0.616 |
| Bag Temperature | 0.613 |
| Empty Override | If P(empty) ≥ 0.50, force `empty` + `not_applicable` |

The bag head receives concatenated backbone features + fill logits, so fill state informs bag prediction (empty carts have no bag state).

---

## Person–Cart Linking

The linker uses a multi-stage state machine per frame:

| Stage | What It Does |
|:-----:|:-------------|
| 0 | **Purge stale links** -- carts absent > 30 frames |
| 0.5 | **Drift detection** -- linked person drifts away while someone else overlaps then release after 6 frames |
| 1 | **Tracker swap recovery** - if linked person vanishes, find new person within 80px of last known position |
| 2 | **New links** -- accumulate IoU for overlapping + co-moving persons, adaptive threshold (6 vs 20 frames), behind-the-cart 1.5× bonus |

---

## Project Structure

```
gk-pops-code/
├── app_poc_v2.py               # Gradio web UI entry point
├── create_virtual_env.py       # Generate the virutal env that can run the demo
├── run_demo.py.py              # Overall orchestrator for demo (creates environment and activates demo)
├── requirements.txt            # Python dependencies
├── README.md
│
├── engine/                     # Core pipeline package
│   ├── __init__.py             # Public API exports
│   ├── botsort_retail.yaml     # BoTSORT tracker configuration
│   ├── config.py               # Paths, thresholds, colors, hyperparameters
│   ├── tracker.py              # TrackingEngine -- main orchestrator
│   ├── classifier.py           # CartClassifier -- batched two-stage inference
│   ├── models.py               # CartQualityModel + DualHeadModel architectures
│   ├── linker.py               # PersonCartLinker -- association state machine
│   ├── motion.py               # Speed, direction, co-movement analysis
│   ├── scoring.py              # POPS score computation + event classification
│   ├── renderer.py             # OpenCV drawing primitives
│   ├── ui_builder.py           # HTML table/dashboard generation for Gradio
│   └── video_io.py             # Video read/write + NVENC/x264 encoding
│
├── weights/                    # Pre-trained model weights
│   ├── detection/
│   │   └── weights/best.pt     # YOLOv26m person + cart detector
│   ├── cart_quality/
│   │   ├── weights/best.pt     # Stage 1 quality classifier
│   │   └── calibration.json    # Temperature: 0.779
│   └── fill_and_bag_classifier/
│       ├── weights/best.pt     # Stage 2 fill + bag classifier
│       └── calibration.json    # Fill temp: 0.616, Bag temp: 0.613
│
└── sample_videos/              # Test video clips
```

### Module Responsibilities

| Module | Role |
|:-------|:-----|
| `tracker.py` | Orchestrates the full pipeline: detection → tracking → re-ID → linking → classification → scoring → rendering → export |
| `classifier.py` | Crops carts from frames, runs batched GPU inference through both classification stages |
| `models.py` | Defines neural network architectures and checkpoint loading with temperature calibration |
| `linker.py` | Maintains person–cart associations across frames with drift detection, swap recovery, and adaptive thresholds |
| `motion.py` | Computes per-object speed/direction/acceleration and determines INBOUND/OUTBOUND/UNKNOWN labels |
| `scoring.py` | Implements the POPS formula: direction + fill + bag + speed + link state → 0–100 score |
| `renderer.py` | Draws bounding boxes, centroid trails, classification overlays, link lines, and HUD onto frames |
| `ui_builder.py` | Generates styled HTML tables for the Gradio dashboard tabs |
| `video_io.py` | Handles video input, AVI writing, and MP4 re-encoding with GPU acceleration (NVENC) or CPU fallback (libx264) |
| `config.py` | Single source of truth for all paths, thresholds, colors, and hyperparameters |

---

### Running it (no setup knowledge required)

Double-click **`run_demo.bat`**. That is the whole procedure.

It needs nothing installed beforehand. It looks for a Python 3.12 or 3.11 on
the machine, and if there isn't one it uses [uv](https://astral.sh/uv) to
download a private copy that touches nothing else on the system,
installing uv itself first if necessary. Then it builds the environment, checks the model
weights arrived intact, and opens the demo in your browser at
**http://localhost:7860**.

The first run downloads several GB and takes 10-20 minutes: the environment,
plus the 4 GB local model that writes the case report. That model is fetched
during setup on purpose; otherwise it downloads the first time someone clicks
Run Analysis, silently, while the case-report tab appears to hang. Later runs
start in seconds. An internet connection is needed the first time only.

**No environment variables are needed.** Nothing has to be set, exported, or
added to PATH. `ANTHROPIC_API_KEY` is read only if you switch the backend
dropdown to Claude instead of the local model, and there is a box in the UI for
that key. The `GK_*` variables in `app_poc_v2.py` are debugging switches that
default to off.

To check a machine is ready without starting anything:

```
run_demo.bat --check-only     # check and prepare, do not start the demo
run_demo.bat --no-model       # skip the 4 GB case-report model
```

On macOS or Linux, or if you would rather not use the `.bat`:

```bash
python run_demo.py
```

#### The case-report model

The written case report is produced by **Qwen3-VL-2B-Instruct**: Apache-2.0,
public, no Hugging Face account and no token. It is ~4 GB and is not in this
repository.

`run_demo.bat` fetches it during setup and skips instantly on later runs. Three
ways it can be satisfied, checked in this order:

| Source | When it applies |
|:--|:--|
| `models/Qwen3-VL-2B-Instruct/` | You put an offline copy there. Nothing contacts Hugging Face. |
| `%USERPROFILE%\.cache\huggingface` | It has been downloaded before on this machine. |
| Hugging Face | Neither of the above, so it is downloaded during setup. |

**Offline or blocked network.** If the machine cannot reach huggingface.co, copy
the model's 12 files into `models/Qwen3-VL-2B-Instruct/`. `engine/config.py`
checks for `config.json` there at import and loads from the folder when it
exists. No variable to set, nothing else to change. `models/README.txt` has
the copy instructions, including how to lift the files out of a working
machine's cache.

**If the download fails**, setup says so and carries on. Everything except the
written case report works; detection, tracking, scoring, the annotated video
and every dashboard tab are unaffected. `--no-model` skips the fetch
deliberately.

#### Before you send this folder to someone

1. **Git LFS.** The detection/classifier weights are LFS objects. Whoever clones needs
   `git lfs install` then `git lfs pull`, or the `.pt` files arrive as 130-byte
   pointers. `run_demo.bat` detects that and says so in plain English rather
   than dying inside torch.
2. **A video clip.** Clips are never committed. `*.mp4` is gitignored, and
   these are store recordings of identifiable people. Send one separately and
   tell the recipient to drop it into `sample_videos/`. Without one the
   dropdown is empty; they can still drag a video into the upload box.
3. **Check it.** Run `run_demo.bat --check-only` on the target machine.

### Prerequisites

None, if you use `run_demo.bat`. It obtains what it needs.

For a manual setup:

- **Python 3.11 or 3.12.** Not 3.13 (numpy 1.26.4 has no cp313 wheel) and not
  3.10 (scipy 1.17.1 needs >= 3.11). `create_virtual_env.py` refuses to run
  outside that window rather than letting pip fail halfway.
- **An NVIDIA driver**, if you want GPU inference. You do *not* need the CUDA
  toolkit; the torch wheels carry their own CUDA runtime. Without a GPU the
  demo still runs; inference is just much slower.
- **Nothing else.** ffmpeg arrives with the `imageio-ffmpeg` package, so no
  system ffmpeg and no PATH entry is required.

### Manual setup

```bash
python create_virtual_env.py            # detects the GPU, builds the venv, verifies it
python create_virtual_env.py --dry-run  # see the plan first, install nothing
python create_virtual_env.py --cpu      # force the CPU build of torch
python create_virtual_env.py --ensure   # build only if needed; verify and exit otherwise
```

The script creates `venv_gk-pops-enhanced/`, installs the CUDA build of
torch/torchvision from PyTorch's own index, then everything in
`requirements.txt`, and finishes by importing the whole stack and reporting
whether `torch.cuda.is_available()` actually came back true. It fails loudly if
a GPU was detected but CUDA is not usable, because the only other symptom of
that is a demo that runs quietly on the CPU.

Then:

```bash
venv_gk-pops-enhanced/Scripts/python app_poc_v2.py        # http://localhost:7860
venv_gk-pops-enhanced/Scripts/python tests/run_all.py --fast
```

### Model Weights

All model weights are bundled under the `weights/` directory:

| Asset | Path |
|:------|:-----|
| YOLOv26m detector | `weights/detection/weights/best.pt` |
| Cart quality classifier | `weights/cart_quality/weights/best.pt` |
| Fill + bag classifier | `weights/fill_and_bag_classifier/weights/best.pt` |
| BoTSORT tracker config | `engine/botsort_retail.yaml` |

Paths are configured in [engine/config.py](engine/config.py).

### UI

The Gradio interface launches at **http://localhost:7860**. Upload a video or pick from the sample videos dropdown, select camera placement, and click **Run Analysis**.

### Camera Placement Options

| Option | When to Use |
|:-------|:------------|
| Outside (facing entrance) | Camera is outside the store, pointing at the entrance |
| Inside (facing exit) | Camera is inside, pointing toward the exit doors |
| Inside (exit on right) | Camera is inside, exit is on the right side of the frame |
| Inside (exit on left) | Camera is inside, exit is on the left side of the frame |
| Inside (exit on both sides) | Camera is inside, exits on both sides |

Camera placement determines how INBOUND vs OUTBOUND direction is computed from object motion vectors.

---

## Outputs

| Output | Description |
|:-------|:------------|
| **Annotated Video** | MP4 with bounding boxes, centroid trails, classification labels, POPS scores, link lines, and HUD |
| **JSON Report** | Per-frame tracking data, event log, per-cart POPS summary, processing metadata |
| **Events Timeline** | HTML table of all significant events with timestamps, scores, and classifications |
| **POPS Summary** | HTML table showing final risk assessment per cart |

---

## Key Hyperparameters

All configurable in [engine/config.py](engine/config.py):

| Parameter | Value | Purpose |
|:----------|:-----:|:--------|
| `YOLO_IMGSZ` | 640 | Detection input resolution |
| `CLASSIFY_EVERY_N_FRAMES` | 8 | Classification frequency |
| `QUALITY_THRESHOLD` | 0.50 | Minimum confidence for valid cart |
| `EMPTY_OVERRIDE_THRESH` | 0.50 | Force empty prediction threshold |
| `LINK_CONFIRM_FRAMES` | 6 | Frames to confirm single-candidate link |
| `LINK_CONTESTED_FRAMES` | 20 | Frames to decide among multiple candidates |
| `LINK_GRACE_FRAMES` | 15 | Grace period before linking new carts |
| `ABANDON_FRAMES` | 30 | Person absent this many frames → abandonment |
| `REID_DIST_THRESH` | 200 px | Max distance for cart re-identification |
| `SPEED_STATIC` | 10 px/s | Below this → STATIC |
| `SPEED_SLOW` | 100 px/s | Below this → SLOW |
| `SPEED_MEDIUM` | 180 px/s | Below this → MEDIUM, above → FAST |

---

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Object Detection | YOLOv8 (Ultralytics) -- custom YOLOv26m |
| Multi-Object Tracking | BoTSORT |
| Classification | PyTorch -- MobileNetV3 / EfficientNet / ConvNeXt / ResNet backbones |
| Video Processing | OpenCV + FFmpeg (NVENC GPU / libx264 CPU) |
| Web Interface | Gradio |
| Inference | CUDA GPU with temperature-calibrated softmax |
