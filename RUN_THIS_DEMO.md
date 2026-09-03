# How to run the POPS demo

This is the cart push-out detection demo. You give it a video clip of a store
entrance; it marks up the video, scores each cart for how likely it is being
pushed out without payment, and writes up what it saw.

You do not need to install anything, set anything up, or know any Python. The
whole procedure is below.

---

## What you need

- A Windows PC. A machine with an NVIDIA graphics card is much faster, but the
  demo works without one.
- An internet connection, **the first time only**.
- The video clip I sent you separately.

---

## Step 1 — Put the video clip in place

Open this project folder. Inside it there is a folder called
**`sample_videos`**. Copy the `.mp4` clip I sent you into that folder.

Do this first. The clips are not stored in the project — they are recordings of
real people in a real store, so they are kept out on purpose. If you skip this
step, the demo still opens, but the list of clips to choose from is empty.

## Step 2 — Start it

In the project folder, double-click **`run_demo.bat`**.

A black console window opens and starts printing what it is doing. Leave it
open — closing it stops the demo.

**The first time, this takes 10 to 20 minutes and downloads several gigabytes.**
That is normal and it has not hung, even when a line sits there for a few
minutes. It is setting up its own private working environment and fetching the
model that writes the report. Every run after this one starts in a few seconds.

When it is ready, your browser opens by itself at **http://localhost:7860**. If
it does not, open a browser and type that address in yourself.

## Step 3 — Run an analysis

In the panel down the left-hand side:

1. Under **step 1**, pick your clip from the *"Or pick a sample clip"*
   dropdown — its heading shows how many clips it found, e.g. *(1 available)*.
   You can also drag a video file into the upload box instead.
2. Under **step 2**, choose the **camera placement** that matches the clip —
   whether the camera is outside looking at the entrance, or inside looking at
   the exit, and which side the exit is on. This matters: it is how the system
   works out which direction is *into* the store and which is *out*.
3. Leave everything else as it is.
4. Click the big blue **Run Analysis** button.

Processing takes a few minutes for a short clip. Progress shows on screen. If
you want to stop it, click **Cancel run**.

## Step 4 — Look at the results

The tabs across the top fill in as the run finishes. In order of interest:

- **POPS** — the headline. One row per cart, with its final 0–100 risk score.
  A score above 70 is a push-out alert.
- **Events** — everything that happened, in order, with timestamps: which
  person was matched to which cart, when a cart was let go, when it turned
  toward the exit.
- **Operational Alerts** — the non-theft findings: doors blocked by carts,
  carts left standing, carts left unattended, empty carts coming back in.
- **Analytics** — traffic heatmap, busy periods, how long people lingered.
- **Case Report** — a written account of the run in plain English.
- **Zone Editor** — draw the entrance, exit and aisle areas onto a frame of the
  clip if you want the alerts to be area-aware. The **Saved zone sets** panel
  at the bottom of that tab writes the polygons to a file per clip and lists
  what is on disk, with **Load** and **Delete** on every row; the newest set
  comes back on its own the next time you pick that clip, so a doorway only
  has to be drawn once. Files live in `zone_presets/` and are named
  `<clip>__<set name>__<when>.json` — the panel header shows the folder.

The marked-up video and the full data file can be downloaded from the bottom of
the left-hand panel.

---

## If something goes wrong

**The console window shows an error and closes.**
Run it again — the first attempt sometimes stops part-way through a download.
If it fails a second time, send me the last few lines from the console window.

**It is very slow.**
The PC probably has no usable NVIDIA graphics card, so everything is running on
the processor. Under **step 2** in the left-hand panel, untick **Pose
estimation** and run again. That skips a second model on every frame and is
noticeably faster; the skeleton overlay disappears from the video and nothing
else changes — scores, events, alerts and the report are all the same.

**I want to check the PC is ready without starting the demo.**
Open the project folder, right-click an empty part of the window and choose
**Open in Terminal** (on older Windows, hold **Shift** while right-clicking and
choose **Open PowerShell window here**), then type:

```
.\run_demo.bat --check-only
```

It sets everything up, tells you whether the machine is good to go, and stops
without opening anything.

**There is no clip dropdown, or it is empty.**
Step 1 was skipped, or the clip went into the wrong folder — it has to be
directly inside `sample_videos`, not in a sub-folder. The list of clips is read
once when the demo starts, so after adding the clip, close the console window
and double-click `run_demo.bat` again.

---

## Stopping it

Close the browser tab, then close the black console window. Nothing is left
running and nothing needs cleaning up.
