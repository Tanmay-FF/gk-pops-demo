# POPS Demo — Build Notes (quick read)

A single-machine demo that watches a store-entrance video clip and flags carts
that are likely being pushed out without payment. One video in; an annotated
video, a risk score per cart, an event timeline, operational alerts, analytics
and a written case report out.

Full technical detail is in [README.md](README.md). This page is the short
version, plus what is specific to this build.

---

## Running it

Double-click **`run_demo.bat`** (Windows) or run `python run_demo.py`
(macOS/Linux). Nothing needs to be installed first — it finds or fetches a
Python 3.11/3.12, builds `venv_gk-pops-enhanced/`, downloads the case-report
model, and opens the demo at **http://localhost:7860**.

```
run_demo.bat                 # normal run
run_demo.bat --check-only    # prepare the machine, do not open the demo
run_demo.bat --no-model      # do not pre-fetch the 4 GB case-report model
```

`run_demo.bat` forwards its arguments to `run_demo.py`, so both flags work
either way.

First run downloads several GB and takes 10–20 minutes. Later runs start in
seconds. Internet is needed the first time only.

`--no-model` only skips the *pre-fetch* — it does not disable the case report.
The model still downloads the first time someone opens that tab, mid-demo and
without a progress bar, which is exactly why the pre-fetch exists. Use it to
shorten setup on a machine that will never open the Case Report tab, not as a
way to make a slow machine faster. For that, untick **Pose estimation** in the
sidebar: it drops a second model per frame and changes nothing but the skeleton
overlay.

For non-technical hand-off, use [RUN_THIS_DEMO.md](RUN_THIS_DEMO.md) instead —
same procedure, written for someone who has never opened this folder.

---

## What you drive from the sidebar

| Step | Control | Notes |
|:--|:--|:--|
| 1 | Upload a clip, or pick a sample | Clips are **not** committed to git — see below |
| 2 | Camera placement | Five options; this is what decides INBOUND vs OUTBOUND |
| 2 | Pose estimation | On by default. Off is faster and changes nothing but the skeleton overlay |
| 3 | Zones | Drawn in the **Zone Editor** tab; the sidebar shows a running summary |
| 4 | Rule thresholds | Blocked-door, static-cart and unattended-cart timings |
| ⚙ | VLM backend | Local model by default; switch to Claude and an API-key box appears |

Then **Run Analysis**. **Cancel run** stops one mid-flight. **Recompute
analytics** re-applies zones and thresholds to the cached run with no GPU work.
**Re-run detection** throws the cache away and starts the model over.

---

## Result tabs

| Tab | What it holds |
|:--|:--|
| Zone Editor | Draw entrance/exit/aisle polygons on a frame of the clip |
| POPS | Final 0–100 risk score per cart, with the operational categories it fell into |
| Events | Chronological event log — links, releases, direction changes, alerts |
| Operational Alerts | Rule-engine outcomes: blocked doors, static carts, unattended carts, incoming empty carts. One row per incident, separate from the theft-risk event log on purpose |
| Analytics | Traffic heatmap (downloadable PNG), spike windows, dwell times, journey summaries |
| Case Report | Written narrative of the run, produced by the local Qwen3-VL model |
| Detection / Video Info / Config | Counts, clip metadata, and the exact settings the run used |
| Legend | Colour and label key for the annotated video |

The annotated MP4 and the full JSON report are downloadable from the sidebar.

---

## Two things to know before sending this folder to someone

1. **Video clips are not in the repository.** `*.mp4` is gitignored, and these
   are store recordings of identifiable people. Send a clip separately and have
   the recipient drop it into `sample_videos/`. Without one the sample dropdown
   is empty — they can still drag a file into the upload box.

2. **The model weights are Git LFS objects.** This only matters if the
   recipient *clones*. They need `git lfs install` then `git lfs pull`, or the
   `.pt` files arrive as 130-byte pointers. `run_demo.bat` detects that and says
   so in plain English rather than failing inside torch. If you are handing over
   a copied folder, LFS does not apply — but exclude `venv_gk-pops-enhanced/`
   from the copy.

Verify a target machine with `run_demo.bat --check-only` before the demo.

---

## Hardware

An NVIDIA GPU makes this comfortable but is not required. Without one, the demo
still runs — detection and the case report are both much slower. The driver is
enough; the CUDA toolkit is not needed, because the torch wheels carry their own
CUDA runtime.
