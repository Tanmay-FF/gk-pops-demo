"""One-command environment setup for the POPS demo.

    python create_virtual_env.py                # auto-detect GPU, install everything
    python create_virtual_env.py --dry-run      # print the plan and the exact pip
                                                # commands, install nothing
    python create_virtual_env.py --cpu          # force the CPU build of torch
    python create_virtual_env.py --cuda cu126   # a different CUDA wheel index
    python create_virtual_env.py --venv .venv   # somewhere other than the default

Run it with any Python 3.11 or 3.12 interpreter, from anywhere:

    C:\\Python312\\python.exe D:\\path\\to\\create_virtual_env.py

It reads requirements.txt from the directory this file lives in, not from the
shell's working directory, so it does not matter where you invoke it.

WHAT YOU NEED ON THE MACHINE FIRST
----------------------------------
* Python 3.11 or 3.12. Not 3.13: numpy 1.26.4 publishes no cp313 wheel and pip
  would try to build it from source. Not 3.10: scipy 1.17.1 requires >= 3.11.
  Both pins are in requirements.txt with the reasons.
* For GPU: an NVIDIA driver, and nothing else. You do NOT need the CUDA
  toolkit. The torch wheels this script installs from download.pytorch.org
  carry their own CUDA runtime — that is the entire reason they are several
  gigabytes. An earlier version of this script downloaded and silently
  installed CUDA 12.2 system-wide, which needed admin rights and was never
  necessary.
* No ffmpeg. imageio-ffmpeg brings its own binary and engine/video_io.py:15
  uses that one, never a system install.

WHAT IT DOES NOT DO ANY MORE
----------------------------
The old version called delete_empty_folders(".") and unzip_and_delete() before
setting anything up. unzip_and_delete extracted every .zip in the working
directory and then deleted the archive — engine.zip sits in the repo root, so a
first run of the setup script destroyed a repo file. Both are gone. Setup
should not modify the checkout.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The window every pin in requirements.txt is satisfiable in. numpy 1.26.4
#: ships cp39-cp312 wheels and scipy 1.17.1 declares requires-python >= 3.11,
#: so the intersection is exactly these two. Verified against PyPI metadata.
SUPPORTED_PYTHON = ((3, 11), (3, 12))

#: The versions venv_gk-pops-enhanced runs and the whole test suite is green
#: on. Kept here rather than in requirements.txt because the CUDA builds live
#: on a separate index that `pip install -r` cannot reach — see the torch
#: section of requirements.txt.
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"

#: Default CUDA wheel index. A plain `torch==2.11.0` specifier matches the
#: index's `2.11.0+cu128` build (PEP 440: a specifier with no local version
#: matches any local version), so the same pin string works on every index and
#: only the URL has to change.
DEFAULT_CUDA_TAG = "cu128"
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"

#: Anything else and the venv is not covered by .gitignore, which is how a
#: multi-gigabyte environment ends up in `git status`.
GITIGNORED_VENVS = ("venv", ".venv", "venv_" + HERE.name.lower())


class SetupError(RuntimeError):
    """Something the user has to fix. Printed without a traceback."""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python(allow_any: bool) -> None:
    v = sys.version_info[:2]
    if v in SUPPORTED_PYTHON:
        return
    supported = " or ".join(f"{a}.{b}" for a, b in SUPPORTED_PYTHON)
    message = (
        f"this interpreter is Python {v[0]}.{v[1]} ({sys.executable}); the "
        f"pins in requirements.txt need {supported}.\n"
        f"  Python 3.13+  numpy 1.26.4 has no cp313 wheel; pip falls back to a "
        f"source build that usually fails.\n"
        f"  Python 3.10-  scipy 1.17.1 declares requires-python >= 3.11.\n"
        f"Re-run this script with a {supported} interpreter, or pass "
        f"--allow-any-python to try anyway."
    )
    if allow_any:
        print(f"[WARN] {message}\n")
        return
    raise SetupError(message)


def detect_nvidia_gpu() -> str | None:
    """Return a one-line GPU description, or None if there is no NVIDIA GPU.

    nvidia-smi ships with the driver, so its presence is the question we
    actually care about: a machine with a CUDA toolkit but no driver cannot run
    torch, and a machine with a driver and no toolkit can.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0].strip()


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def build_plan(args) -> dict:
    """Decide what to install before installing any of it, so --dry-run can
    print the same decisions the real run makes."""
    gpu = detect_nvidia_gpu()

    if args.cpu:
        tag, reason = "cpu", "--cpu was passed"
    elif args.cuda:
        tag, reason = args.cuda, f"--cuda {args.cuda} was passed"
    elif gpu:
        tag, reason = DEFAULT_CUDA_TAG, f"nvidia-smi reports: {gpu}"
    else:
        tag, reason = "cpu", "no NVIDIA GPU found (nvidia-smi did not answer)"

    venv_path = Path(args.venv) if args.venv else HERE / f"venv_{HERE.name.lower()}"
    return {
        "gpu": gpu,
        "tag": tag,
        "reason": reason,
        "index_url": TORCH_INDEX.format(tag=tag),
        "torch": f"torch=={args.torch_version}",
        "torchvision": f"torchvision=={args.torchvision_version}",
        "venv_path": venv_path.resolve(),
        "requirements": (Path(args.requirements).resolve() if args.requirements
                         else HERE / "requirements.txt"),
    }


def venv_python(venv_path: Path) -> Path:
    return (venv_path / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python"))


def activate_hint(venv_path: Path) -> str:
    if os.name == "nt":
        return (f'  cmd:        {venv_path}\\Scripts\\activate.bat\n'
                f'  PowerShell: {venv_path}\\Scripts\\Activate.ps1')
    return f"  source {venv_path}/bin/activate"


def print_plan(plan) -> None:
    print("Plan")
    print("----")
    print(f"  python        {sys.version.split()[0]}  {sys.executable}")
    print(f"  gpu           {plan['gpu'] or 'none detected'}")
    print(f"  torch build   {plan['tag']}  ({plan['reason']})")
    print(f"  torch index   {plan['index_url']}")
    print(f"  venv          {plan['venv_path']}")
    print(f"  requirements  {plan['requirements']}")
    print()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run(cmd, dry_run: bool, what: str) -> None:
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n=== {what}\n$ {printable}")
    if dry_run:
        return
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise SetupError(f"{what} failed (exit {result.returncode}). The pip "
                         f"output above says why.")


def create_venv(plan, args) -> None:
    path = plan["venv_path"]
    if path.exists():
        if not args.reuse:
            raise SetupError(
                f"{path} already exists. Delete it and re-run for a clean "
                f"build, or pass --reuse to install into it as it is.")
        print(f"\n=== Reusing the existing environment at {path}")
        return
    if path.name not in GITIGNORED_VENVS:
        print(f"[WARN] {path.name} is not one of the venv names .gitignore "
              f"covers ({', '.join(GITIGNORED_VENVS)}). It will show up in "
              f"`git status` unless you add it.")
    run([sys.executable, "-m", "venv", str(path)], args.dry_run,
        f"Creating the virtual environment at {path}")


def install(plan, args) -> None:
    py = venv_python(plan["venv_path"])
    pip = [str(py), "-m", "pip", "install"]

    run(pip + ["--upgrade", "pip", "setuptools", "wheel"], args.dry_run,
        "Upgrading pip")

    if args.skip_torch:
        print("\n=== Skipping torch (--skip-torch); requirements.txt will pull "
              "whatever plain PyPI resolves")
    else:
        # Before requirements.txt, and from its own index. ultralytics depends
        # on torch, so if requirements.txt goes first pip satisfies that from
        # plain PyPI — which on Windows is the CPU-only wheel, installed
        # without complaint. The demo then runs on the CPU and the only symptom
        # is that it is slow. Installing the right build first means the later
        # step finds the requirement already satisfied and leaves it alone.
        run(pip + [plan["torch"], plan["torchvision"],
                   "--index-url", plan["index_url"]],
            args.dry_run,
            f"Installing torch/torchvision ({plan['tag']}) from "
            f"{plan['index_url']}")

    if not plan["requirements"].exists():
        raise SetupError(f"{plan['requirements']} not found.")
    run(pip + ["-r", str(plan["requirements"])], args.dry_run,
        f"Installing {plan['requirements'].name}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

#: Run inside the new environment. Imports everything the demo imports at
#: startup, so a broken install surfaces here instead of at first inference,
#: and reports the one thing that is easy to get silently wrong.
_VERIFY = r"""
import json
report = {"errors": []}
try:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_built"] = torch.version.cuda
    if torch.cuda.is_available():
        report["device"] = torch.cuda.get_device_name(0)
except Exception as e:
    report["errors"].append(f"torch: {e}")
for name in ("torchvision", "ultralytics", "cv2", "numpy", "gradio",
             "transformers", "accelerate", "PIL", "imageio_ffmpeg"):
    try:
        m = __import__(name)
        report[name] = getattr(m, "__version__", "ok")
    except Exception as e:
        report["errors"].append(f"{name}: {e}")
try:
    import scipy
    report["scipy"] = scipy.__version__
except Exception:
    report["scipy"] = "not installed (optional)"
print("---VERIFY---" + json.dumps(report))
"""


def verify(plan, args) -> bool:
    """Report what the new environment can actually do. Returns False when the
    environment works but not the way the plan intended."""
    py = venv_python(plan["venv_path"])
    print(f"\n=== Verifying the environment\n$ {py} -c <import check>")
    if args.dry_run:
        return True

    out = subprocess.run([str(py), "-c", _VERIFY], capture_output=True,
                         text=True)
    marker = "---VERIFY---"
    if marker not in out.stdout:
        raise SetupError("the verification script did not run:\n"
                         f"{out.stdout}\n{out.stderr}")
    report = json.loads(out.stdout.split(marker, 1)[1].strip())

    print()
    for key in ("torch", "torchvision", "ultralytics", "numpy", "cv2",
                "gradio", "transformers", "accelerate", "PIL",
                "imageio_ffmpeg", "scipy"):
        if key in report:
            print(f"  {key:16} {report[key]}")

    ok = True
    if report["errors"]:
        ok = False
        print("\n[FAIL] some packages did not import:")
        for e in report["errors"]:
            print(f"  {e}")

    wanted_gpu = plan["tag"] != "cpu"
    got_gpu = report.get("cuda_available", False)
    if wanted_gpu and got_gpu:
        print(f"\n  CUDA             yes — {report.get('device')} "
              f"(torch built against CUDA {report.get('cuda_built')})")
    elif wanted_gpu and not got_gpu:
        ok = False
        print(
            "\n[FAIL] a CUDA build was requested but torch.cuda.is_available() "
            "is False.\n"
            "       The demo will run on the CPU, and the only symptom is that "
            "it is slow —\n"
            "       so this is worth fixing now rather than wondering later.\n"
            "       Usually one of:\n"
            "         * the NVIDIA driver is older than the CUDA runtime in "
            "these wheels.\n"
            "           `nvidia-smi` prints the driver version; update it, or "
            "re-run this\n"
            "           script with an older index, e.g. --cuda cu126.\n"
            "         * a CPU torch was already present and pip left it alone. "
            "Delete the\n"
            "           venv and re-run.\n"
            "       If the machine genuinely has no usable GPU, re-run with "
            "--cpu."
        )
    elif not wanted_gpu and plan["gpu"]:
        print(f"\n  CUDA             no — CPU build installed on purpose, but "
              f"this machine has a GPU\n"
              f"                   ({plan['gpu']}). Re-run without --cpu to "
              f"use it.")
    else:
        print("\n  CUDA             no — CPU build. The demo works; inference "
              "is much slower.")
    return ok


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Create the POPS demo's virtual environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --dry-run first if you want to see the plan.")
    p.add_argument("--venv", metavar="PATH",
                   help="where to create it (default: venv_<repo folder name> "
                        "next to this script)")
    p.add_argument("--requirements", metavar="PATH",
                   help="default: requirements.txt next to this script")
    p.add_argument("--cpu", action="store_true",
                   help="install the CPU build of torch even if a GPU is found")
    p.add_argument("--cuda", metavar="TAG",
                   help=f"CUDA wheel index tag, e.g. cu126 or cu121 "
                        f"(default: {DEFAULT_CUDA_TAG} when a GPU is found)")
    p.add_argument("--torch-version", default=TORCH_VERSION,
                   help=f"default: {TORCH_VERSION}")
    p.add_argument("--torchvision-version", default=TORCHVISION_VERSION,
                   help=f"default: {TORCHVISION_VERSION}")
    p.add_argument("--skip-torch", action="store_true",
                   help="do not install torch separately (it then arrives from "
                        "plain PyPI as an ultralytics dependency, CPU-only on "
                        "Windows)")
    p.add_argument("--reuse", action="store_true",
                   help="install into an existing venv instead of refusing")
    p.add_argument("--ensure", action="store_true",
                   help="if the venv already exists and verifies, do nothing "
                        "and exit 0; otherwise build it. This is the mode "
                        "a launcher wants: a second run costs a second, "
                        "not a download.")
    p.add_argument("--allow-any-python", action="store_true",
                   help="proceed on an unsupported interpreter")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and every command, change nothing")
    args = p.parse_args(argv)

    if args.cpu and args.cuda:
        print("[ERROR] --cpu and --cuda are mutually exclusive.")
        return 2

    print(f"POPS demo setup   {platform.system()} {platform.release()}\n")
    try:
        if args.ensure:
            # Deliberately before check_python: --ensure asks whether a
            # working environment exists, and if one does, the version of
            # whatever interpreter happens to be running this script does
            # not matter.
            plan = build_plan(args)
            if venv_python(plan["venv_path"]).exists():
                print(f"An environment already exists at {plan['venv_path']}.")
                if verify(plan, args):
                    print("\nNothing to do.")
                    return 0
                print("\n[WARN] that environment did not verify. Repairing "
                      "it with the same steps a fresh install uses.")
                args.reuse = True
            else:
                print("No environment yet -- building one.\n")

        check_python(args.allow_any_python)
        plan = build_plan(args)
        print_plan(plan)
        create_venv(plan, args)
        install(plan, args)
        ok = verify(plan, args)
    except SetupError as e:
        print(f"\n[ERROR] {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[ERROR] interrupted. The half-built venv is still on disk; "
              "delete it before re-running.")
        return 130

    if args.dry_run:
        print("\nDry run — nothing was installed.")
        return 0

    print(f"\nDone. Activate it with:\n{activate_hint(plan['venv_path'])}\n")
    print(f"Then:\n"
          f"  {venv_python(plan['venv_path'])} app_poc_v2.py       # the demo, "
          f"on http://localhost:7860\n"
          f"  {venv_python(plan['venv_path'])} tests/run_all.py --fast   "
          f"# 26 test files, about a minute\n")
    print(f"::venv_name::{plan['venv_path'].name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
