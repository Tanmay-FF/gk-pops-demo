# Analysing a video without opening the app

This folder can do its work two ways.

The **app** opens in your browser. You pick a video, press a button, and read
the results on screen. That is `run_demo.bat`, and [RUN_THIS_DEMO.md](RUN_THIS_DEMO.md)
explains it.

This page is about the **other** way: you type one line, it analyses one video
and writes the results to a file. No browser, nothing to click. It is the same
analysis — the same models, the same scores, the same rules. Only the way you
ask for it is different.

You would use this way when you want the numbers in a file rather than on a
screen: to send to someone, to keep alongside the footage, to feed into a
spreadsheet or another system, or to work through a stack of clips one after
another without sitting in front of the browser.

**You do not need to know anything about Python or the command line.** The first
section gets you to a working result. Everything after it is there for when you
want more.

---

## Contents

- [Start here](#start-here) — the first run, step by step
- [What you get](#what-you-get) — the three files, and what each is for
- [Zones, and why they matter](#zones-and-why-they-matter)
- [Typing the command yourself](#typing-the-command-yourself)
- [The things you are most likely to want](#the-things-you-are-most-likely-to-want)
- [Every option](#every-option)
- [When something goes wrong](#when-something-goes-wrong)
- [For whoever set this up](#for-whoever-set-this-up)

---

## Start here

### 1. Put a video in the folder

Open the `sample_videos` folder and copy your `.mp4` file into it.

### 2. Double-click `gk_pops.bat`

That is the whole procedure. A black window opens.

**The first time**, it spends 3 to 5 minutes getting the machine ready. It
needs an internet connection for this part. It is installing the software this
analysis needs into a folder of its own, inside this one — it does not change
anything else on your computer, and it does not need you to be an administrator.

You will see it work through a checklist. Green ticks are good:

```
  [1/2]  Environment
  ──────────────────────────────────────────────────────────────────────
   [✓]  torch                 2.11.0+cu128
   [✓]  ultralytics           8.4.19
   [✓]  CUDA                  NVIDIA GeForce RTX 3070 Ti Laptop GPU

  [2/2]  Checkout
  ──────────────────────────────────────────────────────────────────────
   [✓]  model weights         4 files present
```

**Every time after that**, this takes about six seconds.

> If you have already run `run_demo.bat` at some point, this step is instant —
> the app and this share the same setup, so whichever you ran first paid for it.

### 3. Answer the questions

Because you did not type anything after `gk_pops.bat`, it now asks you what to
do. There are five questions and every one of them has a suggested answer, so
you can press **Enter** five times and get a sensible result.

```
  Which video do you want to analyse?
  ──────────────────────────────────────────────────────────────────────
   ▶1  1763942423220_B8A44F5B742E-medium.mp4
                              zones have been drawn for this one
    2  1765967670150_B8A44FDCC0BF-medium.mp4
                              no zones drawn yet — see question 3
    3  Something else
                              type the path to a video yourself
   Type a number and press Enter [1]:
```

The `▶` marks the suggested answer. Type a number and press Enter, or just
press Enter to take the suggestion.

The five questions are:

| | It asks | What to say |
|---|---|---|
| 1 | Which video | The one you copied in |
| 2 | Where the camera was | See [below](#question-2-where-was-the-camera) — this one matters |
| 3 | Which areas to watch | Take the suggestion |
| 4 | What to keep | Take the suggestion — the report and the marked-up video |
| 5 | The written case report | Say no the first time. It adds several minutes. |

### 4. It shows you the command, then runs it

```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Here is your command. Next time you can type it straight in.
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     gk_pops.bat sample_videos\myclip.mp4 ^
         --camera-placement inside_facing_exit --auto-zones ^
         --json-path out\ --video-path out\

   Run it now? [Y/n]:
```

This is the point of the questions. It is not hiding the command from you — it
is showing you the one it built, so that next time you can skip the questions
entirely and type it straight in. Write it down, or take a photo of the screen.

Press Enter and it runs. A 20-second clip takes about 30 seconds.

### 5. Read the result

When it finishes you get a summary and a list of the files it made:

```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ▶  Complete in 26.5s  ·  peak POPS 75 PUSHOUT ALERT  ·  3 findings
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   out\myclip.json                                                2.1 MB
   out\myclip_annotated.mp4                                       5.8 MB
```

Everything is in the **`out`** folder, next to `gk_pops.bat`.

That middle line is the headline:

- **peak POPS 75** — the highest risk score any trolley in this clip reached,
  out of 100.
- **PUSHOUT ALERT** — what a score that high is called. The scale runs CLEAR,
  MONITORING, SUSPICIOUS, PUSHOUT ALERT.
- **3 findings** — three things the operational rules flagged: a blocked
  doorway, a trolley left standing, a trolley abandoned.

---

## What you get

### The marked-up video — `..._annotated.mp4`

Open it in any video player. It is your footage with the analysis drawn on top:
boxes around people and trolleys, a number for each one so you can follow it,
lines joining a person to the trolley they are pushing, and a score in the
corner that rises and falls as the clip plays.

**This is the one to look at first,** and the one to show somebody else. If the
analysis got something wrong, you will see it here in seconds.

### The report file — `....json`

Everything the analysis worked out, as a file. It is meant for a computer to
read rather than a person — it is long, and a 20-second clip produces about
2 MB of it.

You can still open it in Notepad or a browser if you want to look. Inside are
the per-trolley scores, every event with the second it happened, the rule
findings, and a record of exactly what settings produced it.

Send this to whoever asked for the analysis. It contains everything.

### The heat map — `..._heatmap.png` *(only if you ask)*

A single picture of where people spent their time, painted over a frame of the
video. Warm colours are busy places.

---

## Zones, and why they matter

A **zone** is an area of the picture that somebody has drawn around — the
doorway, an aisle, the area in front of the tills.

Three of the checks cannot run without them, because they are questions about a
*place*:

- Is a trolley **blocking the doorway**?
- Has a trolley been **left standing** in an aisle?
- Has a trolley been **abandoned** — the person walked off and left it?

If nobody has drawn the doorway, the analysis has no way to know where the
doorway is. It cannot guess, and it does not try — a guessed doorway would
produce confident findings about a patch of empty floor, which is worse than no
findings at all.

**So: without zones, a run will still work, and will still be useful** — you get
the trolley scores, the events, and the marked-up video. But the three checks
above report nothing, and the report file says so rather than pretending
everything was clear.

### Drawing them, once, per camera

You only ever do this once for a given camera, and you do it in the app:

1. Run `run_demo.bat`
2. Choose your video
3. Go to the **Zones** step and draw around the doorway
4. Press **Save zone set**

From then on, every run of `gk_pops.bat` on that video finds those zones by
itself. The question 3 in the walkthrough says *"zones have been drawn for this
one"* when it has found them.

A zone set belongs to the video it was drawn on. A different camera needs its
own.

---

## Typing the command yourself

Once you have seen the command, you never need the questions again.

Open the folder, hold **Shift** and right-click in the empty space, and choose
**Open PowerShell window here** or **Open command window here**. Then type:

```
.\gk_pops.bat sample_videos\myclip.mp4 --auto-zones --video-path out\
```

Press Enter.

Reading that line:

| Part | Meaning |
|---|---|
| `.\gk_pops.bat` | run this folder's analyser |
| `sample_videos\myclip.mp4` | the video — **always comes first** |
| `--auto-zones` | use the zones somebody drew for this video |
| `--video-path out\` | also make and keep the marked-up video, in the `out` folder |

Everything beginning with `--` is optional. The video is not.

> **A tip that saves a lot of typing:** you can drag a video file from Explorer
> into the black window, and it types the full path for you.

### Question 2: where was the camera?

This is the one setting that is easy to get wrong and will not tell you it is
wrong.

The analysis needs to know which direction is *out of the shop*, because a
trolley heading for the exit is the thing it is looking for. Get it backwards
and the clip will look quiet — no alarms, nothing flagged — which reads exactly
like a clip where nothing happened.

Pick the one that matches where the camera was mounted:

| Type this | When |
|---|---|
| `outside_facing_entrance` | camera outside, looking at the doors from the car park |
| `inside_facing_exit` | camera inside, looking straight at the way out |
| `inside_exit_on_right` | camera inside, exit is to the right of the picture |
| `inside_exit_on_left` | camera inside, exit is to the left of the picture |
| `inside_exit_on_both` | camera inside, between two sets of doors |

No quotation marks, no capital letters to get right, and these are the same
five words the POPS HTTP service takes — so whichever of the two you use, you
type the same thing.

The longer wording the app's dropdown shows still works if you prefer it, but
then the quotation marks are needed, because those words have spaces in them:
`--camera-placement "Inside (facing exit)"`.

If you leave it out entirely it assumes `outside_facing_entrance`.

---

## The things you are most likely to want

**Skipping the batch file entirely.** `gk_pops.py` is the thing `gk_pops.bat`
ends up calling, and it takes exactly the same arguments. It does no setup of
its own, so the environment has to be built already (run `gk_pops.bat` once) and
active in this window:

```
.\venv_gk-pops-enhanced\Scripts\activate
python .\gk_pops.py sample_videos\1763942423220_B8A44F5B742E-medium.mp4 --camera-placement "Inside (facing exit)" --auto-zones --json-path result.json --video-path annotated_video.mp4
```

That writes `result.json` and `annotated_video.mp4` into the folder you ran it
from. Without the environment active it fails on the first missing package
rather than fetching anything.

**Just analyse it, keep only the report.** The report lands next to the video,
named after it.

```
.\gk_pops.bat sample_videos\myclip.mp4
```

**Keep the marked-up video too, everything in one folder.**

```
.\gk_pops.bat sample_videos\myclip.mp4 --video-path out\ --json-path out\
```

**Use the zones somebody drew, and say where the camera was.**

```
.\gk_pops.bat sample_videos\myclip.mp4 --auto-zones ^
    --camera-placement inside_facing_exit --video-path out\
```

(The `^` at the end of a line means *this command carries on below*. You can
also just type it all on one long line.)

**Check it would work, without waiting for the analysis.** Useful when you have
typed a long command and want to know the paths are right before committing
several minutes to it.

```
.\gk_pops.bat sample_videos\myclip.mp4 --auto-zones --dry-run
```

It checks everything, tells you exactly where each file would go, and stops.

**Make the report file small.** A full report is about 2 MB for a 20-second
clip, most of which is a frame-by-frame record. If you only want the conclusions
— the scores, events and findings — this takes the same clip to about 16 KB:

```
.\gk_pops.bat sample_videos\myclip.mp4 --auto-zones --frames none --no-html
```

The analysis is identical. Every frame is still examined. This only changes how
much detail is written to the file.

**Get the written case report.** Several minutes per clip, and the first time it
downloads a large model:

```
.\gk_pops.bat sample_videos\myclip.mp4 --auto-zones --case-report out\
```

**Do several videos in a row.** Paste this into a file called `nightly.bat` in
this folder, and double-click it:

```
@echo off
cd /d "%~dp0"
for %%V in (sample_videos\*.mp4) do (
    call "%~dp0gk_pops.bat" "%%V" --auto-zones --json-path out\
)
pause
```

(`%~dp0` means *the folder this file is in*. Without it, Windows sometimes
cannot find `gk_pops.bat` even though it is sitting right next to it.)

It works through every video in the folder, one after another, and puts all the
reports in `out`. One video failing does not stop the rest — but check the
window afterwards, because it will say which.

---

## Every option

For the full list, with the exact wording of everything:

```
.\gk_pops.bat --help
```

The ones worth knowing about:

| Option | What it does |
|---|---|
| `--auto-zones` | use the zones saved for this video |
| `--camera-placement inside_facing_exit` | where the camera was — see [above](#question-2-where-was-the-camera) |
| `--json-path out\` | put the report in the `out` folder |
| `--video-path out\` | also make and keep the marked-up video (slower) |
| `--heatmap-path out\` | also keep the heat map |
| `--pose` / `--no-pose` | force the skeleton overlay on or off — by default it runs only when `--video-path` or `--case-report` will show it |
| `--case-report` | also write the AI case report (slow). Give it a path — `--case-report out\` — to choose where it lands; bare, it lands next to the video |
| `--dry-run` | check everything and stop, without analysing |
| `--frames none` | smaller report file, same analysis |
| `--force` | overwrite a report that is already there |
| `--quiet` | less on screen |
| `--verbose` | more on screen — every line the analysis prints as it works |
| `--engine-log run.log` | keep those lines in a file instead of on screen |
| `--help` | the full list |

A note on `--verbose`: the analysis narrates itself in some detail while it
runs — which cart voted which way, how long each stage took, what was cached.
None of that is needed to read a result, so it is kept out of the way by
default rather than switched off. If a run fails, the last lines of it are
printed automatically; `--verbose` shows all of it live, and `--engine-log`
writes all of it to a file.

A note on `--video-path` and `--heatmap-path`: the heat map is made on every run
whether you ask for it or not, and naming a path is what *keeps* it. The
marked-up video is different — it is **only made when `--video-path` asks for
one**, because making it is not free: a frame is encoded on every pass through
the loop and the whole thing is then decoded again to draw the rule badges in.

Leaving `--video-path` off therefore does make the run faster, and by more than
the encode alone. Pose estimation — the skeleton drawn over each person — is a
second model run on every frame, and the only thing that ever displays it is the
marked-up video or the case report's evidence images. So it follows the same
question: with `--video-path` or `--case-report` it runs, without them it does
not. Nothing else changes. Pose feeds no score, no rule and no number in the
report; a clip run both ways produces the same findings, the same events and the
same per-frame records.

On the 20-second sample clip that is 31s with the video and 21s without, on the
same machine back to back. `--pose` and `--no-pose` override the choice in
either direction if you want it settled by hand.

---

## When something goes wrong

The window tells you what happened and what to do about it. It is worth reading
— it is written for this, not for a programmer.

```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ZoneSpecError
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --zone 1: unknown kind "doorway"

  one of: analytics, wall, aisle, door, fixture
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The common ones:

**"no such file"** — the video path is wrong. Drag the file into the window
instead of typing it.

**"already exists"** — there is a report there from a previous run. Add
`--force` to replace it, or use a different `--json-path`.

**"no saved zone preset for ..."** — `--auto-zones` could not find zones for
this video, because nobody has drawn them. See
[Zones](#zones-and-why-they-matter), or drop `--auto-zones` and run without.

**"This preset was drawn on a 1920x1080 frame but the current video is
1280x720"** — the zones belong to a different-sized recording from the same
camera. They have to be redrawn for this one; stretching them would put the
doorway in the wrong place.

**"the engine could not be built"** — the model files did not arrive properly.
Whoever sent you this folder needs to send it again, or run
`git lfs install && git lfs pull` in it.

**It says a warning but carries on.** That is deliberate. A warning means the
run finished and something about it is worth knowing — most often that no zones
were loaded, so the door and trolley rules could not run. The report is still
good; just read the warning.

**Nothing at all happens when you double-click.** Windows may be blocking the
file because it came from the internet. Right-click `gk_pops.bat`, choose
**Properties**, and if there is an **Unblock** box at the bottom, tick it and
press OK.

**If you are stuck**, take a photo of the whole window and send it to whoever
gave you this folder. The window has everything needed to work out what
happened.

---

## For whoever set this up

The things a non-technical reader does not need, in one place.

**Nothing is installed system-wide.** `gk_pops.bat` looks for a usable Python in
three places, in order: an environment a previous run already built, a Python
3.12 or 3.11 already on the machine, and failing both, [uv](https://astral.sh/uv),
which fetches a private 3.12 into the user profile. The environment itself is
`venv_gk-pops-enhanced/` inside the folder. No administrator rights, no PATH
changes, nothing outside this directory and `%USERPROFILE%\.local`.

**`run_demo.bat` and `gk_pops.bat` share one environment.** `run_headless.py`
imports `ensure_environment()` and `check_weights()` from `run_demo.py` rather
than reimplementing them, so there is one setup path and whichever entry point
runs first pays for it.

**Exit codes** are passed through from `gk_pops.py`, so this can be scheduled:

| | |
|---|---|
| `0` | the run completed and the report was written |
| `2` | something about the command was wrong — nothing ran, nothing written |
| `3` | the run started and failed. A report file **is still written**, with `cli.exit` set to `"error"` and the traceback inside it |
| `130` | interrupted |

`1` is never returned deliberately, so it stays the signal that the process died
before it could report for itself.

A failure report is a tombstone rather than a result, so re-running over one
does **not** need `--force`. Fix the problem, run the same command again.

**Set `GK_NO_PAUSE=1`** to stop the window waiting for a keypress at the end.
Without it, a double-click pauses so the results can be read; with it, a
scheduled task exits cleanly.

**Run it from anywhere** by calling the batch file with its full path. Output
paths you give are resolved against the directory you ran it *from*, not the
folder the analysis lives in.

**On macOS or Linux** there is no `.bat`. Run `python run_headless.py` with the
same arguments — the batch file is only the Python-finding wrapper.

**The details** — the JSON structure, every flag, the parity guarantee against
the app — are in the "Run it without the UI" section of
[README.md](README.md#run-it-without-the-ui) and in `gk_pops.py --help`.
