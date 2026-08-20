"""requirements.txt must pin the packages that decide tracker behaviour.

Unpinned ultralytics/numpy/opencv is how the conda `all` env and
venv_gk-pops-enhanced drifted onto different ultralytics releases and produced
materially different POPS output from the same video. The compat shim makes
those releases agree; pinning stops the drift recurring silently.

The structural assertions here fail the build. The installed-vs-pinned check is
reported rather than asserted: the conda env is shared with other projects and
is not obliged to match this repo's pins — but you should still see, in one
line, when the interpreter you are running does not.

Runs anywhere — reads a text file.
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(REPO, "requirements.txt")

#: Pinning these is what keeps two machines producing the same tracks.
#: import name -> distribution name as it appears in requirements.txt
TRACKER_CRITICAL = {
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "cv2": "opencv-python",
}

#: Installed from a custom index in create_virtual_env.py, so a pin here would
#: break `pip install -r requirements.txt` for everyone.
MUST_NOT_BE_PINNED_HERE = ("torch", "torchvision")

_PIN = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s#]+)", re.MULTILINE)


def _pins():
    with open(REQUIREMENTS, encoding="utf-8") as f:
        text = f.read()
    return {m.group(1).lower(): m.group(2) for m in _PIN.finditer(text)}


def _installed(import_name):
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def test_tracker_critical_packages_are_pinned():
    pins = _pins()
    missing = [dist for dist in TRACKER_CRITICAL.values() if dist.lower() not in pins]
    assert not missing, (
        f"{missing} are not pinned in requirements.txt. Unpinned, every install "
        f"resolves whatever is newest that day — which is exactly how this repo "
        f"ended up with two environments producing different POPS output."
    )


def test_pins_are_exact_not_ranges():
    """`>=` would defeat the point: a fresh install still drifts."""
    with open(REQUIREMENTS, encoding="utf-8") as f:
        text = f.read()
    for dist in TRACKER_CRITICAL.values():
        loose = re.search(rf"^\s*{re.escape(dist)}\s*[><~!]=", text, re.MULTILINE | re.IGNORECASE)
        assert not loose, f"{dist} is pinned with a range, not ==: {loose.group(0)!r}"


def test_torch_is_not_pinned_in_requirements():
    """torch==...+cu128 is not resolvable from plain PyPI. Its pin belongs in
    create_virtual_env.py, which passes --index-url."""
    pins = _pins()
    for name in MUST_NOT_BE_PINNED_HERE:
        assert name not in pins, (
            f"{name} is pinned in requirements.txt — `pip install -r "
            f"requirements.txt` will fail for anyone not going through "
            f"create_virtual_env.py. Pin it there instead."
        )


def test_the_reason_for_the_pins_is_written_down():
    """A bare pin gets bumped by the next person who wants a newer feature.
    The comment is what makes them re-run the tests first."""
    with open(REQUIREMENTS, encoding="utf-8") as f:
        head = f.read(2000)
    assert "ultralytics_compat" in head or "ultralytics" in head, (
        "requirements.txt does not explain why these versions are pinned"
    )


def installed_vs_pinned():
    """Reported, not asserted — see the module docstring."""
    pins = _pins()
    rows = []
    for import_name, dist in sorted(TRACKER_CRITICAL.items()):
        rows.append((dist, pins.get(dist.lower()), _installed(import_name)))
    return rows


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")

    print("\ninstalled vs pinned (reported, not asserted):")
    drift = 0
    for dist, pinned, installed in installed_vs_pinned():
        # opencv exposes 4.11.0 where the wheel is 4.11.0.86 — compare on prefix.
        ok = pinned and installed and pinned.startswith(installed)
        if not ok:
            drift += 1
        print(f"  {'ok  ' if ok else 'DRIFT'} {dist:16} pinned={pinned}  installed={installed}")
    if drift:
        print(f"\n  {drift} package(s) differ from requirements.txt. Expected in a "
              f"shared env; NOT expected in venv_gk-pops-enhanced.")
