# POPS -- Push-Out Probability Score

**AI-powered system to identify potential cart push-out theft in real time using computer vision, multi-object tracking, and behavioral scoring.**

---

## Overview

POPS processes surveillance video to detect, track, and score shopping carts and persons in a retail environment to assess the likelihood of a push-out theft event. The system outputs an annotated video with bounding boxes, trails, classification overlays, and a detailed JSON report with per-frame and per-cart analytics. It links persons to carts, classifies cart contents, analyzes motion direction, and computes a **0–100 risk score** per cart.

On top of the per-cart score, every clip gets one overall answer (see [Clip Verdict](#clip-verdict)), and a separate set of operational rules (blocked door, static cart, unattended cart, incoming empty cart) runs over the zones you draw (see [Operational Rules](#operational-rules)).

---

## Demonstration

**Double-click `run_demo.bat`.** That is the whole procedure on Windows.

Nothing has to be installed first. It finds or fetches a Python 3.11/3.12,
builds the environment, checks the model weights, and opens the demo in your
browser at **http://localhost:7860**. The first run downloads several GB and
takes 10-20 minutes; later runs start in seconds.

Five demo clips ship in `sample_videos/` as Git LFS objects, so the clip
dropdown is filled as soon as LFS has pulled them. To try another clip, drop it
into `sample_videos/` or drag it into the upload box. No other clip is
committed: they are store recordings of identifiable people, and `*.mp4` is
gitignored everywhere except those five.

```
run_demo.bat                 # normal run
run_demo.bat --check-only    # prepare the machine, do not open the demo
run_demo.bat --no-model      # do not pre-fetch the 4 GB case-report model
```


Starting from a fresh clone instead of a copied folder:

```bash
git clone https://github.com/Tanmay-FF/gk-pops-demo
cd gk-pops-demo
git checkout pops-3class-yolo26-clip-verdict
git lfs install && git lfs pull    # weights and the demo clips are LFS objects
run_demo.bat
```

Handing this to someone non-technical: give them
[RUN_THIS_DEMO.md](RUN_THIS_DEMO.md). A shorter overview of this build is in
[QUICKSTART.md](QUICKSTART.md). Details of what `run_demo.bat` does are under
[Running it](#running-it-no-setup-knowledge-required) below.

---

## What This Branch Adds

Branch `pops-3class-yolo26-clip-verdict`, compared with `pops-enhanced`.

**Detection**

- The default detector is now a **3-class YOLOv26m** (`person`, `empty_cart`,
  `non_empty_cart`), bundled in `weights/detection_3class_yolo26/`. Its
  empty / non-empty call decides which carts reach the classifiers at all. See
  [Detector Backends](#detector-backends).
- Boxes down to confidence 0.1 reach the tracker, which keeps a briefly occluded
  person or cart on the same ID. Only cart boxes at 0.25 or above vote empty /
  non-empty.
- One cart no longer becomes two tracks. The two cart classes are merged before
  NMS, and a tight basket box nested inside a loose one is suppressed by
  containment.
- The classifiers crop the clean decoded frame, not the frame with our boxes and
  trails drawn on it. The drawn overlay had been pushing reads toward "loaded"
  and "unbagged".
- `ultralytics` is pinned to 8.4.19, with a compatibility shim that keeps
  ByteTrack's low-confidence second association working on later versions.

**POPS scoring** (details in [POPS Scoring Rules](#pops-scoring-rules))

- PUSHOUT ALERT can be reached on score alone (80 or more), not only through
  abandonment.
- Walking a loaded, unbagged cart out is HIGH PRIORITY (75). Running with it is
  a PUSHOUT ALERT.
- Bagged goods on a cart that was not abandoned are treated as paid and capped
  at 30 (LOW).
- An abandoned empty cart that never held goods adds nothing to the score.
- A grab-and-run whose direction never resolved still reaches the 75 floor when
  the goods left the cart.
- The confidence-weighted vote over a cart's whole read history is the only
  authority on its contents. No single frame overrides it.
- Direction is measured over the last 4 seconds and latched, so small drifts no
  longer flip it.

**Clip verdict.** Every clip ends with one answer: its most severe event, or
LOW PRIORITY, INSUFFICIENT EVIDENCE or NOTHING HAPPENED. See
[Clip Verdict](#clip-verdict).

**Operational rules.** Blocked door, static cart, unattended cart and incoming
empty cart, evaluated over drawn zones, shown in the Operational Alerts tab and
drawn as badges into the output MP4. Zone sets can be saved per clip. See
[Operational Rules](#operational-rules).

**Linking.** Possession is judged by how strong the link ever got, so a parked
cart's real owner is kept on record and their departure still counts as
abandonment. A cart that has never moved cannot be claimed by a passer-by.

**Running the demo**

- `run_demo.bat` sets up everything on a machine with no Python, including the
  case-report model.
- Pressing Run again replaces the run in progress. Cancel stops work inside the
  frame loop and the case-report model.
- Runs after a local case-report pass no longer degrade, and the ~140 s stall on
  the first Run is gone.

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
    ├─ 1. YOLOv26m 3-class Detection ─ person / empty_cart / non_empty_cart (640px)
    │        ├── Cart classes merged before NMS; nested duplicate boxes suppressed
    │        └── Boxes down to conf 0.1 go to the tracker
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
    ├─ 5. Cart Reads (every 8 frames, cropped from a clean copy of the frame)
    │        ├── Detector state: mean P(non-empty) since the cart's last read,
    │        │     from cart boxes with conf ≥ 0.25
    │        ├── EMPTY: recorded as empty, no classifier runs
    │        └── NON-EMPTY:
    │              ├── Stage 1 -- Cart Quality: valid_cart vs unclear (0.50)
    │              │     └── A crop that fails gives no fill reading
    │              └── Stage 2 -- Fill + Bag
    │                    ├── Fill: partial / full
    │                    └── Bag:  bagged / unbagged
    │
    ├─ 6. Motion Analysis ─────────── Speed, direction, acceleration
    │        ├── Speed (px/s): STATIC < 10 < SLOW < 100 < MEDIUM < 240 < FAST
    │        └── Direction: INBOUND / OUTBOUND / UNKNOWN over the last 4 s, latched
    │
    ├─ 7. POPS Scoring ────────────── 0–100 risk score per cart
    │
    ├─ 8. Rendering ───────────────── Boxes, trails, labels, scores per frame
    │
    └─ After the last frame
             ├── Reconcile each cart: confidence-weighted vote over its reads
             ├── Clip verdict
             ├── Operational rules over the cached trajectories and zones
             └── Export: MP4 (rule badges drawn during encode), JSON, HTML
```

On the 2-class backends, step 1 detects `person` and `cart` only, and step 5
runs both classifier stages on every read, with fill `empty / partial / full`
and an empty override at P(empty) ≥ 0.50.

---

## Event Classification

| Event | When | Severity |
|:------|:-----|:--------:|
| PUSHOUT ALERT | Score ≥ 80, or score ≥ 71 and the cart was abandoned | High |
| HIGH PRIORITY | Score 71–79, not abandoned | High |
| ABANDONED CART | Score 31–70, abandoned | Medium |
| UNLINKED EXIT | Score 31–70, outbound with no linked person | Medium |
| MEDIUM PRIORITY | Score 31–70, otherwise | Medium |
| MONITORING / LOW PRIORITY / INBOUND / UNLINKED EXIT | Score below 31 (UNLINKED EXIT here is an unlinked outbound cart with a low score) | Low |

Only the High and Medium rows are logged as events. Anything under 31 is shown
live on the video but never reaches the Events tab, and the end-of-run
reconciliation drops any row whose final score falls under 31.

PUSHOUT ALERT has two routes. At 80 or more, what the cart is doing is enough on
its own: outbound, loaded, unbagged, moving. At 71 to 79 it also needs the
linked person to have left the cart. Only outbound carts can reach 80, so an
arriving cart is never called a pushout.

---

## POPS Scoring Rules

Implemented in `compute_pops()` in [engine/scoring.py](engine/scoring.py).

**Kill switches.** A cart that was not detected scores 0. An INBOUND cart, or
one whose crop is not a valid cart, scores 5 before anything else is looked at.
When that hides a cart that read as loaded, the run says so in the Operational
Alerts panel, because a wrong camera placement looks exactly like "the
detections stopped".

**OUTBOUND**

| Term | Points |
|:-----|:-------|
| Base | +15 |
| Full | +50 unbagged, +20 bagged |
| Partial | +30 unbagged, +15 bagged |
| Empty | −15 |
| Speed | +15 FAST, +5 MEDIUM |
| Loose merchandise on the move (unbagged partial/full, not STATIC) | +10, and the MEDIUM +5 is dropped |
| FAST with loose merchandise | +15 |

Outbound carts are never damped for having a linked person, since a person
walking a cart out is the case being detected. The loose-merchandise term is
what separates walking out from running out:

| Full + unbagged, OUTBOUND | Score | Event |
|:--------------------------|:-----:|:------|
| STATIC | 65 | MEDIUM PRIORITY |
| SLOW | 75 | HIGH PRIORITY |
| MEDIUM | 75 | HIGH PRIORITY |
| FAST | 100 | PUSHOUT ALERT |

**UNKNOWN direction.** Full adds +25 unbagged or +15 bagged, partial +8, FAST
+8. A cart with its linked person still at it loses 20.

**Abandonment** (the linked person left the frame or walked away from the cart)

| Cart | OUTBOUND | UNKNOWN |
|:-----|:---------|:--------|
| Partial / full | Floor of 75 | Floor of 65, or 75 when the goods left the cart (unbagged only) |
| Empty | Floor of 60 | +25 |
| Empty, never loaded | No change | No change |
| No fill reading | +35 | +25 |

- **Goods left the cart** means the read history holds a sustained loaded run
  (5 reads in a row) and then ends empty. A shopper who parks a full cart at a
  shelf never produces that, so they stay at 65.
- **Never loaded** is judged per cart: it never had 5 reads in a row saying it
  held something. A NON-EMPTY detector call counts as "held something" even when
  its crop failed the quality gate. A cart whose track starts within 5 s of a loaded cart's track
  ending, near where it ended, is taken to be that cart picked up again under a
  new ID and is not exempt. A cart with no fill reading is never exempt: no
  reading is not an empty reading.

**Caps**

- Partial + bagged is capped at 55.
- Partial or full + bagged, when the cart was not abandoned (including a cart
  that never had a linked person), is capped at 30 (LOW). Bagged goods are
  treated as paid. The 30 cap does not apply to an abandoned bagged cart, which
  falls back to the abandonment floors and the 55 partial cap. The bag classifier cannot tell paid
  bags from unpaid ones, so a pushout with the goods already bagged is given up
  on purpose.

**What the score is computed from**

- **Contents.** A cart's fill and bag come from a confidence-weighted vote over
  its whole read history. That vote is final: the event rows and the POPS table
  are rewritten from it, downward as well as upward, and no single frame can
  override it. One stray "full" read cannot decide a tier.
- **Direction.** Measured over the last 4 seconds (`DIRECTION_WINDOW_S`), not the
  whole track, so a shopper who came in and left by the same door still reads
  OUTBOUND. Once resolved, it latches and only flips back after moving 40 px
  (`LATCH_REVERSAL_PX`) the other way.

---

## Clip Verdict

Every clip gets one answer, shown above the POPS table and written to
`clip_verdict` in the JSON. The first match wins:

| # | Condition | Verdict |
|:-:|:----------|:--------|
| 1 | Any event survived reconciliation | The most severe event's name (PUSHOUT ALERT, HIGH PRIORITY, ...) |
| 2 | A scored cart never got a usable reading | INSUFFICIENT EVIDENCE |
| 3 | A cart held merchandise | LOW PRIORITY |
| 4 | Every cart read empty, or no cart was tracked | NOTHING HAPPENED |

A cart seen for under 0.5 s (`CART_BLIP_S`) that never got a reading is a
detector blip and is left out, so it cannot hold a clip at INSUFFICIENT
EVIDENCE. Operational rule findings are listed beside the verdict and never
change it: they answer a different question.

---

## Operational Rules

Implemented in [engine/rules.py](engine/rules.py). These are separate from POPS:
they report on store operations, not theft risk, and use their own labels and
severities.

| Rule | Fires when | Zone needed | Default | Severity |
|:-----|:-----------|:------------|:-------:|:--------:|
| BLOCKED DOOR | A cart's box overlaps a door zone (at least 15% of the box) and stays still | `door` | 5 s | SAFETY |
| STATIC CART | A cart stays still inside an aisle or analytics zone | `aisle`, `analytics` | 10 s | WATCH |
| UNATTENDED CART (OPS) | A cart stays still with nobody near it (gap to the nearest person over 0.7 of the cart box's diagonal), outside any cart corral | none; `fixture` zones are corrals and are exempt | 10 s | ACTION |
| INCOMING CART WITHOUT ITEMS | A cart reads empty and moves inward in its first 4 s in view | none | n/a | INFO |

- **Rules run after the video is processed**, over the cached trajectories. That
  is what lets the **Recompute analytics** button apply a new zone or threshold
  in seconds, with no GPU pass. The trajectory cache is keyed on the detector backend, its
  weights and the library versions, so a stale bundle is not reused.
- **Blocked door and static cart need zones**; unattended cart and incoming
  empty cart run without any. Draw zones in the **Zone Editor** tab. Door,
  aisle, fixture and wall zones apply to carts as well as people. "No findings"
  and "rules could not run" are shown as different states, and the panel says
  which rule families had no zone to run on, so a quiet run is never mistaken
  for an all-clear.
- **Still means positional spread**, not speed: a parked cart's box jitters
  enough to read as moving on speed alone.
- **Sparse intervals are rejected.** An interval sampled more than 2 s apart (a
  cart hidden behind traffic, say) does not count, and the panel reports what it
  dropped.
- **Findings appear** in the Operational Alerts tab, the alert banner, the case
  report, `rule_findings` in the JSON, and as badges drawn into the output MP4.
- **Saved zone sets.** The Zone Editor saves the zones drawn for a clip to
  `zone_presets/` (gitignored) and reloads the newest set when that clip is
  selected. A set drawn for a different frame size is refused rather than
  rescaled.

**Crowd clusters.** Separately from the four rules, a group of 4 or more people
within 100 px of each other for 3 s or more raises QUEUE_FORMING, and 6 or more
for 8 s raises BACKED_UP. Both show in the Analytics tab and the alert banner.

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

On the 3-class backends the detector has already called the cart non-empty, so the fill answer is limited to `partial` / `full` (renormalised over those two) and the empty override does not apply. See `classify_loaded()` in [engine/classifier.py](engine/classifier.py).

---

## Person–Cart Linking

The linker uses a multi-stage state machine per frame:

| Stage | What It Does |
|:-----:|:-------------|
| 0 | **Purge stale links** -- carts absent > 30 frames |
| 0.5 | **Drift detection** -- linked person drifts away while someone else overlaps then release after 6 frames |
| 1 | **Tracker swap recovery** - if linked person vanishes, find new person within 80px of last known position |
| 2 | **New links** -- accumulate IoU for overlapping + co-moving persons, adaptive threshold (6 vs 20 frames), behind-the-cart 1.5× bonus |

Possession rules added on this branch:

- **A cart that has never moved cannot be claimed by a passer-by.** Brief corner
  overlap with a parked cart no longer makes the passer-by its owner.
- **A released link keeps its owner on record when the link was real.** When a
  parked cart's link is released, the cart is only disowned if the link's peak
  IoU (over the candidacy and the link) stayed under 0.15
  (`LINK_OWNED_PEAK_IOU`) and the person never pushed it anywhere. Otherwise the
  owner is remembered, and their leaving still counts as abandonment.
- **Owner selection ignores a person who only overlaps in 2D**, since in a scene
  with depth they can be metres away.

---

## Project Structure

```
gk-pops-demo/
├── app_poc_v2.py               # Gradio web UI entry point
├── run_demo.bat                # Double-click launcher (Windows)
├── run_demo.py                 # Launcher: checks weights, builds the env, fetches the case-report model, starts the demo
├── create_virtual_env.py       # Builds and verifies venv_gk-pops-enhanced/
├── console_ui.py               # Console formatting for the launchers
├── console_noise.py            # Filters third-party console noise; logs server errors to temp/
├── requirements.txt            # Python dependencies (ultralytics pinned to 8.4.19)
├── README.md, QUICKSTART.md, RUN_THIS_DEMO.md
│
├── api/                        # HTTP API over the same engine
├── static/                     # UI CSS and JS
│
├── engine/                     # Core pipeline package
│   ├── __init__.py             # Public API exports
│   ├── botsort_retail.yaml     # BoTSORT tracker configuration
│   ├── config.py               # Paths, thresholds, colors, hyperparameters
│   ├── tracker.py              # TrackingEngine -- main orchestrator
│   ├── detection_backends.py   # Detector backends (3-class YOLOv26m / YOLOv5, 2-class) and the tracker driver
│   ├── detection_dedup.py      # Suppresses nested duplicate cart boxes
│   ├── ultralytics_compat.py   # Keeps ByteTrack's low-confidence association working across ultralytics versions
│   ├── cart_state.py           # Collects the detector's P(non-empty) per cart between reads
│   ├── classifier.py           # CartClassifier -- quality gate, fill + bag
│   ├── models.py               # CartQualityModel + DualHeadModel architectures
│   ├── linker.py               # PersonCartLinker -- association state machine
│   ├── motion.py               # Speed, direction, direction latch, co-movement
│   ├── scoring.py              # POPS score, event tiers, reconciliation, clip verdict
│   ├── rules.py                # Operational rules (blocked door, static, unattended, incoming empty)
│   ├── highlights.py           # Operational findings and diagnostics for the panel, report and JSON
│   ├── analytics_builder.py    # Dwell, heatmap, crowd clusters; runs the rule engine
│   ├── trajectory_cache.py     # Cached trajectories, so Re-run analytics needs no GPU pass
│   ├── zone_editor.py          # Zone geometry and validation
│   ├── zone_presets.py         # Saved zone sets per clip
│   ├── cancellation.py         # Cancel / supersede for runs in progress
│   ├── vlm_analyzer.py         # Case-report model (local Qwen3-VL or Claude)
│   ├── case_report_builder.py  # Standalone case report
│   ├── renderer.py             # OpenCV drawing primitives, rule badges
│   ├── ui_builder.py           # HTML table/dashboard generation for Gradio
│   └── video_io.py             # Video read/write + NVENC/x264 encoding
│
├── weights/                    # Pre-trained model weights (Git LFS)
│   ├── detection_3class_yolo26/
│   │   └── weights/best.pt     # YOLOv26m person / empty_cart / non_empty_cart (default)
│   ├── detection/
│   │   └── weights/best.pt     # YOLOv26m person + cart (2-class backends only)
│   ├── pose_estimation/
│   │   └── yolo26l-pose.pt     # Person pose
│   ├── cart_quality/
│   │   ├── weights/best.pt     # Stage 1 quality classifier
│   │   └── calibration.json    # Temperature: 0.779
│   └── fill_and_bag_classifier/
│       ├── weights/best.pt     # Stage 2 fill + bag classifier
│       └── calibration.json    # Fill temp: 0.616, Bag temp: 0.613
│
├── sample_videos/              # Five demo clips (Git LFS)
└── tests/                      # python tests/run_all.py --fast
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
2. **Video clips.** The five demo clips in `sample_videos/` come with the
   checkout through LFS. Any other clip has to be sent separately: `*.mp4` is
   gitignored outside those five, and these are store recordings of
   identifiable people. The recipient drops it into `sample_videos/` or drags
   it into the upload box.
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
| YOLOv26m 3-class detector (person / empty_cart / non_empty_cart, the default backend) | `weights/detection_3class_yolo26/weights/best.pt` |
| YOLOv26m 2-class detector (person / cart, used only by the `ultralytics_*` backends) | `weights/detection/weights/best.pt` |
| Pose model | `weights/pose_estimation/yolo26l-pose.pt` |
| Cart quality classifier | `weights/cart_quality/weights/best.pt` |
| Fill + bag classifier | `weights/fill_and_bag_classifier/weights/best.pt` |
| BoTSORT tracker config | `engine/botsort_retail.yaml` |

Paths are configured in [engine/config.py](engine/config.py). `run_demo.py`
checks that every one of these weights is a real file and not an LFS pointer
before it starts the demo.

### Detector Backends

`DETECTOR_BACKEND` in [engine/config.py](engine/config.py) picks the detector:

| Backend | Detector | Notes |
|:--------|:---------|:------|
| `yolo26_3class` | YOLOv26m, 3-class | **Default.** Bundled. On 56 held-out clips it called carts correctly 92.0% of the time against 87.3% for YOLOv5 (McNemar p = 0.013), at the same speed. |
| `yolov5_3class` | YOLOv5m, 3-class | Needs a local YOLOv5 repository and weights that are **not** in this repo, so it does not run from a clone. |
| `ultralytics_track` | YOLOv26m, 2-class | The original path, `YOLO(...).track()`. |
| `ultralytics_direct` | YOLOv26m, 2-class | Same weights through this repo's own tracker driver; exists to prove the driver matches `ultralytics_track`. |

Every backend except `ultralytics_track` runs detection and tracking
separately: the detector feeds this repo's own BoTSORT driver, which on the
3-class backends carries each cart box's P(non-empty) through to its track. The trajectory cache key includes the backend and its
weights, so switching backends never reuses another backend's trajectories.

### UI

The Gradio interface launches at **http://localhost:7860**. Upload a video or pick from the sample videos dropdown, select camera placement, and click **Run Analysis**. Pressing it again while a run is in progress replaces that run, and **Cancel** stops it. After a run, **Recompute analytics** re-evaluates zones, rules and analytics over the cached trajectories without a GPU pass.

| Tab | Shows |
|:----|:------|
| Zone Editor | Draw door / aisle / fixture / wall / analytics zones; save and reload zone sets per clip |
| POPS | Clip verdict callout, then the final score, fill, bag and event per cart |
| Events | Every logged event with timestamp, score and classification |
| Operational Alerts | Rule findings, plus notes on rules that could not run and carts the INBOUND kill switch scored out |
| Analytics | Dwell, heatmap, crowd clusters |
| Case Report | Written report from the local Qwen3-VL model (or Claude, with an API key) |
| Detection, Video Info, Config, Legend | Run details and reference |

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
| **Annotated Video** | MP4 with bounding boxes, centroid trails, classification labels, POPS scores, link lines, HUD, and operational rule badges |
| **JSON Report** | Per-frame tracking data, event log, per-cart POPS summary, processing metadata (keys below) |
| **Events Timeline** | HTML table of all significant events with timestamps, scores, and classifications |
| **POPS Summary** | HTML table showing the clip verdict and final risk assessment per cart |

JSON keys added on this branch:

| Key | Holds |
|:----|:------|
| `clip_verdict` | The clip's verdict, tier, top cart and score, and the reason |
| `rule_findings` | Operational rule findings, kept apart from `events` |
| `rule_engine` | Why rules could not run (if they could not), diagnostics such as skipped rule families, dropped intervals and INBOUND suppression, and the thresholds actually used |
| `pops_summary.C<n>.fill` / `.bag` / `.quality` | The voted labels each cart's final score was computed from |
| `cart_classifications.C<n>.observed_at_frame` | Frame of the cart's last reading, which can differ from the vote |
| `field_notes` | Explains the two fields above inside the document |

---

## Key Hyperparameters

All configurable in [engine/config.py](engine/config.py):

| Parameter | Value | Purpose |
|:----------|:-----:|:--------|
| `DETECTOR_BACKEND` | `yolo26_3class` | Which detector runs (see [Detector Backends](#detector-backends)) |
| `YOLO_IMGSZ` | 640 | Detection input resolution |
| `Y26_TRACK_CONF` | 0.1 | Lowest YOLOv26 box confidence handed to the tracker |
| `CART_STATE_MIN_CONF` | 0.25 | Lowest cart box confidence that may vote empty / non-empty |
| `CART_STATE_NON_EMPTY_THRESH` | 0.5 | Mean P(non-empty) at or above this reads NON-EMPTY |
| `CLASSIFY_EVERY_N_FRAMES` | 8 | Classification frequency |
| `QUALITY_THRESHOLD` | 0.50 | Minimum confidence for valid cart |
| `EMPTY_OVERRIDE_THRESH` | 0.50 | Force empty prediction threshold (2-class backends only) |
| `LINK_CONFIRM_FRAMES` | 6 | Frames to confirm single-candidate link |
| `LINK_CONTESTED_FRAMES` | 20 | Frames to decide among multiple candidates |
| `LINK_GRACE_FRAMES` | 15 | Grace period before linking new carts |
| `LINK_OWNED_PEAK_IOU` | 0.15 | Peak link IoU at which a released parked cart keeps its owner |
| `ABANDON_FRAMES` | 30 | Person absent this many frames → abandonment |
| `REID_DIST_THRESH` | 200 px | Max distance for cart re-identification |
| `SPEED_STATIC` | 10 px/s | Below this → STATIC |
| `SPEED_SLOW` | 100 px/s | Below this → SLOW |
| `SPEED_MEDIUM` | 240 px/s | Below this → MEDIUM, above → FAST |
| `DIRECTION_WINDOW_S` | 4.0 s | Direction is measured over this much recent track |
| `LATCH_REVERSAL_PX` | 40 px | Movement needed to flip a latched direction |
| `GRABRUN_MIN_RUN_OBS` | 5 | Reads in a row that count as "held merchandise" |
| `SPLIT_TRACK_GAP_S` | 5.0 s | A new cart track this soon after a loaded one ends, nearby, is taken as the same cart |
| `CART_BLIP_S` | 0.5 s | Unread carts seen for less than this are left out of the clip verdict |
| `RULE_BLOCKED_DOOR_S` | 5.0 s | Blocked-door threshold |
| `RULE_STATIC_CART_S` | 10.0 s | Static-cart threshold |
| `RULE_ABANDONED_CART_S` | 10.0 s | Unattended-cart threshold |

The POPS tier boundaries and caps (`MEDIUM_SCORE` 31, `HIGH_SCORE` 71,
`PUSHOUT_SCORE` 80, `BAGGED_PAID_CAP` 30) live in
[engine/scoring.py](engine/scoring.py).

---

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Object Detection | Ultralytics 8.4.19, custom YOLOv26m (3-class by default) |
| Multi-Object Tracking | BoTSORT, driven by this repo's own driver on every backend except `ultralytics_track` |
| Classification | PyTorch -- MobileNetV3 / EfficientNet / ConvNeXt / ResNet backbones |
| Video Processing | OpenCV + FFmpeg (NVENC GPU / libx264 CPU) |
| Web Interface | Gradio |
| Inference | CUDA GPU with temperature-calibrated softmax |
