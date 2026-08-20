"""Run every test file in tests/ with the current interpreter.

    python tests/run_all.py            # everything
    python tests/run_all.py --fast     # skip the ones that need weights or a clip

There is no pytest in either environment, and the files are standalone scripts
with their own runners, so this just executes each one and reports its exit
code. Use it before committing, and in BOTH environments if you have more than
one — cross-environment agreement is the property that broke and that
tests/test_golden_clip.py now protects.
"""
import os
import subprocess
import sys
import time
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

#: Need the detection weights and/or the golden clip; everything else is pure
#: logic and runs anywhere in about a second.
SLOW = {"test_golden_clip", "test_device_guard", "test_device_guard_e2e",
        "test_cancel_run_e2e"}

_SUMMARY_MARKERS = ("passed", "All green", "PASSED")


def summarise(output):
    for line in reversed(output.strip().splitlines()):
        if any(m in line for m in _SUMMARY_MARKERS):
            return line.strip()
    return ""


def main(argv):
    fast = "--fast" in argv
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    if not files:
        print("no tests found")
        return 1

    failures, skipped = [], []
    width = max(len(os.path.basename(f)) for f in files) - 3

    print(f"python {sys.version.split()[0]}  {sys.executable}")
    try:
        import ultralytics
        print(f"ultralytics {ultralytics.__version__}")
    except Exception:
        pass
    print("-" * (width + 46))

    for path in files:
        name = os.path.basename(path)[:-3]
        if fast and name in SLOW:
            skipped.append(name)
            print(f"  skip  {name:<{width}}  (--fast)")
            continue

        started = time.perf_counter()
        proc = subprocess.run([sys.executable, path], cwd=REPO,
                              capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        combined = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode != 0:
            failures.append(name)
            print(f"  FAIL  {name:<{width}}  {elapsed:5.1f}s")
            for line in combined.strip().splitlines()[-8:]:
                print(f"           {line}")
        else:
            print(f"  ok    {name:<{width}}  {elapsed:5.1f}s  {summarise(combined)}")

    print("-" * (width + 46))
    if failures:
        print(f"  {len(failures)} FAILING: {', '.join(failures)}")
    else:
        print(f"  all {len(files) - len(skipped)} test files passed"
              + (f" ({len(skipped)} skipped)" if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
