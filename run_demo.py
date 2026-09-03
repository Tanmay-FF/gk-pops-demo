#!/usr/bin/env python3
"""Start the POPS demo: check the checkout, build the environment if needed,
open the UI.

    python run_demo.py
    python run_demo.py --check-only   # everything except opening the demo
    python run_demo.py --no-model     # do not pre-download the case-report model

Safe to run repeatedly. The environment is only built the first time; after
that this takes a couple of seconds to verify and then launches.

On Windows, double-clicking run_demo.bat calls this — that wrapper exists to
find (or download) a Python 3.12 first, so a machine with no Python, or the
wrong Python, still works. This file assumes it already has a usable one.

The three things that go wrong on someone else's machine, all checked here
before anything slow starts:

  * git-lfs was not installed, so the model weights are 130-byte pointer files
    and the demo would die at model load with an unreadable torch error.
  * sample_videos/ is empty, so the dropdown has nothing in it. Not fatal —
    you can drag a video into the UI — but worth saying before the browser
    opens rather than after.
  * the virtual environment is missing or half-built.
  * the local case-report model is not in the Hugging Face cache. It is a 4 GB
    download that otherwise happens the first time someone clicks Run
    Analysis, silently, while the case-report tab appears to hang. Better to
    pay for it during setup, where the wait is expected and visible.

No environment variables are required. ANTHROPIC_API_KEY is read only if you
pick a Claude backend in the UI instead of the local model, and the UI has a
box for the key anyway. The GK_* variables in app_poc_v2.py are debugging
switches that default to off.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import console_ui as ui

#: transformers still probes for TensorFlow: image_transforms.py does
#: `if is_tf_available(): import tensorflow as tf`, which fires when the
#: Qwen3-VL processor is built and prints two absl/oneDNN banners to stderr.
#: Nothing here uses TF. USE_TF=0 makes is_tf_available() False so the import
#: never happens; the other two only matter if something else pulls TF in.
#: All three have to be set before the first transformers import -- once TF
#: has initialised they are ignored. Set here as well as in app_poc_v2.py so
#: the child processes launched below -- the model fetch and the demo itself --
#: inherit them; nothing in this file imports transformers directly.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

#: Without this, this script's prints sit in a buffer while the pip and
#: setup output of the child processes goes straight out, and the log reads
#: out of order -- confusing for anyone trying to follow along. The stream's
#: encoding is console_ui's business and is already settled by the import
#: above.
sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent

#: Everything engine/config.py loads at startup. Sizes are what git-lfs gives
#: you; a pointer file is ~130 bytes, so anything under this is not a model.
MIN_WEIGHT_BYTES = 1_000_000
REQUIRED_WEIGHTS = (
    Path("weights/detection/weights/best.pt"),
    Path("weights/cart_quality/weights/best.pt"),
    Path("weights/fill_and_bag_classifier/weights/best.pt"),
    Path("weights/pose_estimation/yolo26l-pose.pt"),
)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")
URL = "http://localhost:7860"


def check_weights() -> bool:
    """git-lfs check. Without it the .pt files are text pointers, and the
    failure that produces downstream names neither git nor lfs."""
    broken = []
    for rel in REQUIRED_WEIGHTS:
        path = HERE / rel
        if not path.exists():
            broken.append((rel, "missing"))
        elif path.stat().st_size < MIN_WEIGHT_BYTES:
            broken.append((rel, f"{path.stat().st_size} bytes — a git-lfs "
                                f"pointer, not the model"))
    if not broken:
        ui.ok("model weights", f"{len(REQUIRED_WEIGHTS)} files present")
        return True

    ui.fail("model weights",
            f"{len(broken)} of {len(REQUIRED_WEIGHTS)} unusable")
    ui.problem("The model weights did not come through.", [
        *(f"  {rel}  ({why})" for rel, why in broken),
        "",
        "This repository stores its weights with Git LFS. A plain `git",
        "clone` on a machine without LFS installed silently substitutes",
        "small text pointers for the real files.",
        "",
        "To fix it, in this folder:",
        "",
        "    git lfs install",
        "    git lfs pull",
        "",
        "Install Git LFS first if that command is not found:",
        "    https://git-lfs.com",
        "",
        "Then run this again.",
    ])
    return False


def check_sample_videos() -> None:
    """Not fatal. The upload box still works."""
    folder = HERE / "sample_videos"
    clips = ([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
             if folder.is_dir() else [])
    if clips:
        ui.ok("sample clips", f"{len(clips)} in sample_videos/")
        return
    ui.warn("sample clips", "none found")
    ui.note("The dropdown will be empty. Drop an .mp4 into")
    ui.note(str(folder))
    ui.note("and restart, or drag a video into the upload box once the page "
            "opens. Clips are not in the repository — they are store "
            "recordings, so whoever sent you this sends the clip separately.")


def venv_python() -> Path:
    venv = HERE / f"venv_{HERE.name.lower()}"
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment() -> bool:
    """Build the venv if it is missing, verify it if it is not. All of the
    actual logic lives in create_virtual_env.py; this just calls it in the mode
    that is safe to run every time."""
    ui.detail("First run downloads several GB and can take 10-20 minutes.")
    ui.detail("Later runs verify in a couple of seconds.")
    started = time.monotonic()
    result = subprocess.run([sys.executable,
                             str(HERE / "create_virtual_env.py"), "--ensure"])
    if result.returncode == 0:
        ui.took(time.monotonic() - started)
    return result.returncode == 0


#: Run inside the environment. Deliberately NOT huggingface_hub's
#: snapshot_download: on Windows without Developer Mode that raises
#: "[WinError 1314] A required privilege is not held by the client" while
#: trying to symlink a blob into the snapshot directory, even for a model it
#: has just downloaded. transformers' own per-file path handles the
#: no-symlink case, and it is what the app itself uses, so warming the cache
#: this way warms exactly what Run Analysis will later read.
_FETCH_MODEL = r"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from engine.config import QWEN3_VL_MODEL_ID as MODEL
from transformers.utils import cached_file


# A directory rather than a repo id means someone put an offline copy in
# models/ -- see models/README.txt. Nothing to fetch.
if os.path.isdir(MODEL):
    print("---FETCH-OK---bundled")
    sys.exit(0)


def already_cached():
    # True only if the config AND at least one weight shard are on disk.
    # Config alone is not enough: an interrupted first run leaves exactly
    # that.
    try:
        config = cached_file(MODEL, "config.json", local_files_only=True)
    except Exception:
        return False
    if not config:
        return False
    snapshot = os.path.dirname(config)
    return any(f.endswith(".safetensors") for f in os.listdir(snapshot))


if already_cached():
    print("---FETCH-OK---cached")
    sys.exit(0)

try:
    from huggingface_hub import list_repo_files
    for name in list_repo_files(MODEL):
        if not name.endswith("/"):
            cached_file(MODEL, name)
except Exception as e:
    print("---FETCH-FAILED---" + type(e).__name__ + ": " + str(e))
    sys.exit(0)

print("---FETCH-OK---downloaded")
"""


def fetch_case_report_model() -> None:
    """Warm the Hugging Face cache. Never fatal: everything except the written
    case report works without it, and a machine behind a proxy that blocks
    huggingface.co should still get a demo."""
    py = venv_python()
    ui.detail("4 GB, downloaded once and then reused.")
    started = time.monotonic()
    out = subprocess.run([str(py), "-c", _FETCH_MODEL], capture_output=True,
                         text=True)
    if "---FETCH-OK---" in out.stdout:
        how = out.stdout.split("---FETCH-OK---")[1].strip().splitlines()[0]
        detail = {"bundled": "offline copy in models/",
                  "cached": "already downloaded",
                  "downloaded": "downloaded now"}.get(how, how)
        ui.ok("case-report model", f"ready — {detail}")
        ui.took(time.monotonic() - started)
        return
    reason = ""
    for chunk in out.stdout.split("---FETCH-FAILED---")[1:]:
        reason = chunk.strip().splitlines()[0]
    ui.warn("case-report model",
            f"not downloaded — {reason or 'unknown error'}")
    ui.note("Everything else works. The written case report will be "
            "unavailable, and the first Run Analysis will try the download "
            "again.")


def main(argv=None) -> int:
    #: Runs every check and builds the environment, then stops instead of
    #: launching. This is what you run before handing the folder to someone
    #: else: it proves their path works without leaving a server running.
    args = argv if argv is not None else sys.argv[1:]
    check_only = "--check-only" in args
    #: Skips the 4 GB warm-up. For a machine that already has the model, or
    #: one that is only ever going to use a Claude backend.
    no_model = "--no-model" in args

    # engine/config.py resolves MODEL_PATH relative to the working directory.
    os.chdir(HERE)

    #: Built before anything prints, so the [n/total] counter matches the
    #: flags actually passed instead of counting phases that get skipped.
    plan = ["Environment"]
    if not no_model:
        plan.append("Case-report model")
    plan.append("Checkout")
    if not check_only:
        plan.append("Launch")
    step = 0

    def next_phase() -> None:
        nonlocal step
        ui.phase(step + 1, len(plan), plan[step])
        step += 1

    ui.banner("GATEKEEPER SYSTEMS   POPS",
              "Pushout prevention — single-video analysis demo",
              ui.os_label())

    next_phase()
    if not ensure_environment():
        ui.problem("Setup did not finish.", [
            "The messages above say why. Nothing was left running; fix the",
            "problem and start this again.",
        ])
        return 1

    if not no_model:
        next_phase()
        fetch_case_report_model()

    next_phase()
    if not check_weights():
        return 1
    check_sample_videos()

    if check_only:
        print()
        ui.detail("Everything checks out. Run this again without "
                  "--check-only to start the demo.")
        print()
        return 0

    next_phase()
    ui.launching(URL, [
        "Your browser opens by itself. The first load takes a minute while",
        "the models come up.",
        "",
        "Leave this window open — closing it stops the demo.",
        "Press Ctrl+C here when you are finished.",
    ])

    py = venv_python()
    try:
        return subprocess.run([str(py), "app_poc_v2.py"]).returncode
    except KeyboardInterrupt:
        print()
        ui.detail("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
