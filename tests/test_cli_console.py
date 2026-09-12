# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The headless runner's console surface: widths, plain mode, and the engine
progress strings the phase headings are derived from.

Run with:  python tests/test_cli_console.py

Two things are pinned here.

**The layout.** console_ui.WIDTH is 76 and every helper is supposed to stay
inside it. A line that overruns wraps in the terminal and the whole run reads as
ragged, which is the failure this module exists to prevent. The one deliberate
exception is a path longer than the line budget: outcome() prints those intact
rather than truncating, because a mangled path cannot be copied.

**The five progress descriptions.** `process_video()` reports progress as
`progress(..., desc="...")`, and gk_pops.py maps those five strings onto phase
headings. They are engine internals, so a rename would silently degrade the
console into one undifferentiated phase. Scanned here -- with `ast`, not
imported, since engine/tracker.py costs six seconds of ultralytics and torch --
so the rename fails a test instead.

The mapping itself (an unrecognised description still opens a phase rather than
being swallowed) is tested where the map lives, in tests/test_cli_args.py.
"""
import ast
import io
import re
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import console_ui as ui                                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER = os.path.join(REPO, "engine", "tracker.py")

#: Every desc= string gk_pops.py's progress stub maps to a phase heading, in the
#: order process_video() emits them.
ENGINE_PROGRESS_DESCS = (
    "Processing frames",
    "Computing analytics",
    "Encoding video",
    "Building tracking JSON",
    "Rendering panels",
)

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def render(fn) -> list[str]:
    """Run `fn` with stdout captured; return the lines it printed."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue().splitlines()


def raw(fn) -> str:
    """Run `fn` with stdout captured; return the text verbatim, \\r included."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


# Plain for the whole file. Colour escapes would make len() measure bytes of
# ANSI rather than columns on screen, and every assertion here is about columns.
ui.set_plain(True)

# ---------------------------------------------------------------------------
section("engine progress descriptions still exist")

with open(TRACKER, encoding="utf-8") as fh:
    tracker_tree = ast.parse(fh.read(), filename=TRACKER)

_descs = set()
for node in ast.walk(tracker_tree):
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if (kw.arg == "desc" and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)):
                _descs.add(kw.value.value)

for want in ENGINE_PROGRESS_DESCS:
    check(f'engine still emits desc="{want}"', want in _descs,
          "" if want in _descs else "renamed or removed — the phase map is stale")

# ---------------------------------------------------------------------------
section("plain mode")

check("set_plain(True) drops colour", ui.COLOR is False and ui.PLAIN is True)
check("...and zeroes every escape constant",
      all(v == "" for v in (ui.RESET, ui.BOLD, ui.DIM, ui.GREEN, ui.YELLOW,
                            ui.RED, ui.CYAN)))
ui.set_plain(False)
check("set_plain(False) cannot force colour onto a non-TTY",
      ui.COLOR is False,
      "asks _ansi_ok() again rather than assuming")
ui.set_plain(True)

check("the glyph set carries a bar glyph", "bar" in ui.G and ui.G["bar"])

# ---------------------------------------------------------------------------
section("nothing overruns WIDTH")

lines = render(lambda: (
    ui.banner("GK-POPS " + ui.G["sep"] + " headless run",
              "northgate-0142 " + ui.G["sep"] + " a-long-enough-clip-name.mp4",
              "1615c40"),
    ui.logo("headless run", "a-clip-name.mp4", "1615c40"),
    # The name that cannot share the line with the title. It must move to its
    # own line rather than be truncated or push the line past WIDTH.
    ui.logo("headless run",
            "a-really-very-long-store-camera-clip-name-that-cannot-fit.mp4",
            "1615c40"),
    # No subtitle. run_headless.py's "getting ready" header has no clip to
    # name yet, so this is the shape a bare `gk_pops.bat` actually prints, and
    # the one where the right-aligned trailer has no title+joiner in front of
    # it to be measured against.
    ui.logo("getting ready", "", ui.os_label()),
    ui.phase(1, 6, "Inputs"),
    ui.ok("video", "1280x720 " + ui.G["sep"] + " 20.0 fps " + ui.G["sep"]
          + " 401 frames " + ui.G["sep"] + " 20.1s"),
    ui.warn("blocked door", "Cart 3 " + ui.G["sep"] + " 11.2s"),
    ui.fail("zones", "PresetError: frame size 1920x1080 vs 1280x720"),
    ui.info("device", "cuda"),
    ui.note("A continuation line that runs on for long enough to need wrapping "
            "at least once, and then some more words after that."),
    ui.detail("the frame work above is finished"),
    ui.took(214.7, "0.46x realtime"),
    ui.hr(),
    ui.hr(heavy=True),
))
over = [l for l in lines if len(l) > ui.WIDTH]
check("every helper stays inside WIDTH", not over,
      f"longest {max((len(l) for l in lines), default=0)} of {ui.WIDTH}"
      if not over else f"{len(over)} over: {over[0][:60]!r}")

check("hr() draws to the same width as phase()'s rule",
      len(render(lambda: ui.hr())[0]) == ui.WIDTH)
check("hr(heavy=True) uses the double rule",
      render(lambda: ui.hr(heavy=True))[0].strip()[0] == ui.G["H"])

# ---------------------------------------------------------------------------
section("progress_bar, plain")

ui.set_plain(True)
ui.progress_end()
plain_lines = render(lambda: [ui.progress_bar(i, 658, "frames",
                                              "37.3 ms/frame")
                              for i in range(1, 659)])
check("658 frames leave ~10 lines, not 658",
      5 <= len(plain_lines) <= 14, f"{len(plain_lines)} lines")
check("...with no carriage returns in any of them",
      all("\r" not in l for l in plain_lines))
check("...the last one reaches 100%",
      plain_lines[-1].strip().endswith("37.3 ms/frame")
      and "100%" in plain_lines[-1])
check("...and none overruns WIDTH",
      all(len(l) <= ui.WIDTH for l in plain_lines))

ui.progress_end()
unknown = render(lambda: [ui.progress_bar(i, 0, "frames") for i in range(1, 301)])
check("an unknown total prints a count with no percentage",
      all("%" not in l for l in unknown) and len(unknown) <= 5,
      f"{len(unknown)} lines")

# ---------------------------------------------------------------------------
section("progress_bar, in place")

ui.set_plain(False)
ui.PLAIN = False                  # force the \r path on a non-TTY, for the test
ui.progress_end()
text = raw(lambda: [ui.progress_bar(i, 10, "frames", "37.3 ms/frame")
                    for i in range(1, 11)])
check("redraws in place with \\r and no newline", text.count("\r") == 10
      and "\n" not in text)
check("...and the bar fills as it goes",
      text.split("\r")[1].count(ui.G["bar"]) <
      text.split("\r")[-1].count(ui.G["bar"]))
check("...each redraw stays inside WIDTH + the 2-space eraser",
      all(len(frag) <= ui.WIDTH + 2 for frag in text.split("\r") if frag))

after = raw(lambda: ui.phase(2, 6, "Detection and tracking"))
check("a heading after an open bar starts with a newline",
      after.startswith("\n"),
      "otherwise it lands on top of the half-drawn bar")

ui.progress_end()
ui.progress_end()
check("progress_end() twice is harmless",
      raw(lambda: ui.progress_end()) == "")

text2 = raw(lambda: (ui.progress_bar(3, 10, "frames"), ui.ok("tracked", "9 carts")))
check("a status row after an open bar closes it first",
      "\n" in text2 and text2.splitlines()[-1].lstrip().startswith("["))
ui.progress_end()
ui.set_plain(True)

# ---------------------------------------------------------------------------
section("outcome")

import tempfile                                                # noqa: E402

tmp = tempfile.mkdtemp(prefix="pops_cli_console_")
# Run from inside it and pass bare names. %TEMP% on Windows is already ~45
# characters, and outcome() deliberately overflows rather than truncating a
# path — so an absolute temp path would test the overflow branch twice and the
# fits-on-one-line branch never.
_cwd = os.getcwd()
os.chdir(tmp)
small = "result.json"
with open(small, "wb") as fh:
    fh.write(b"x" * 5_662_310)
missing = "never_encoded.mp4"

out = render(lambda: ui.outcome(
    "Complete in 343.1s " + ui.G["sep"] + " peak POPS 75 " + ui.G["sep"]
    + " 3 findings", "cuda", [small, missing]))
check("outcome brackets the verdict with a heavy rule",
      out[1].strip()[0] == ui.G["H"] and out[3].strip()[0] == ui.G["H"])
check("...lists a real file with its size",
      any(small in l and "5.4 MB" in l for l in out),
      next((l for l in out if small in l), "")[-20:])
check("...lists a missing file anyway, and says so",
      any(missing in l and "missing" in l for l in out),
      "a bare path someone has to notice lacks a size is worse than the word")
check("...and a real file carries the clock time it was written",
      any(small in l and ":" in l.split(small)[-1] for l in out),
      "every artifact is normally seconds old; one that is not was not "
      "written by this run")
check("...and every line it emitted fits",
      all(len(l) <= ui.WIDTH for l in out),
      f"longest {max(len(l) for l in out)}")

long_path = "a" * 90 + ".json"
with open(long_path, "wb") as fh:
    fh.write(b"y" * 1024)
out2 = render(lambda: ui.outcome("Complete", "", [long_path]))
check("a path too long for the line is printed intact, not truncated",
      any(long_path in l for l in out2),
      "a mangled path cannot be copied")
check("...with its size and time moved to their own right-aligned line",
      any("1 KB" in l and long_path not in l and len(l) == ui.WIDTH
          for l in out2),
      "the meta moves rather than the path being cut")

# ---------------------------------------------------------------------------
section("one glyph, one meaning")

# The point of the vocabulary is that the left gutter can be read on its own.
# A timing is not a status, so it must not take a badge.
_gutter = render(lambda: (ui.phase(1, 5, "Inputs", rule=False),
                          ui.ok("device", "cuda"),
                          ui.info("not kept", "heatmap"),
                          ui.warn("warnings", "2"),
                          ui.fail("zones", "PresetError"),
                          ui.phase_end(23.6)))
# The badge column, not the [1/5] counter in the heading -- a badge row is
# indented one column further, which is what BADGE_W and PAD buy.
_marks = [l.split("]")[0].split("[")[-1]
          for l in _gutter if l.startswith(ui.PAD + " [")]
check("every badge is one of the four", set(_marks) <= {ui.G["ok"], ui.G["info"],
                                                        ui.G["warn"], ui.G["fail"]},
      f"saw {sorted(set(_marks))}")
check("a section's duration takes no badge",
      not any(ui.G["dot"] in l for l in _gutter),
      "took() is for the launchers; a phase closes with its rule")

# ---------------------------------------------------------------------------
section("the duration hangs off the closing rule, right-aligned")

_ends = render(lambda: (ui.phase_end(0.8), ui.phase_end(23.5),
                        ui.phase_end(196.8)))
check("each closing rule ends with its own duration",
      [l.rstrip()[-5:] for l in _ends] == ["0.8s", "23.5s", "196.8s"][0:0]
      or all(l.rstrip().endswith(t)
             for l, t in zip(_ends, ("0.8s", "23.5s", "196.8s"))))
check("...and every one of them ends at the same column",
      len({len(l) for l in _ends}) == 1 and len(_ends[0]) == ui.WIDTH,
      f"widths {sorted({len(l) for l in _ends})} of {ui.WIDTH}")
check("the rule shrinks to make room rather than overrunning",
      all(len(l) <= ui.WIDTH for l in _ends))

# ---------------------------------------------------------------------------
section("paths shorten against a root, and only against a root")

_root = r"D:\proj"
check("a path under the root loses the prefix",
      ui.shorten(_root + r"\zone_presets\a.json", _root) == r"zone_presets\a.json")
check("a path outside it keeps its own spelling",
      ui.shorten(r"C:\other\a.json", _root) == r"C:\other\a.json",
      "relpath across drives must not produce a .. chain")
_elided = ui.shorten("x" * 80 + ".json", None, 40)
check("what is left is elided in the middle, never at the ends",
      len(_elided) == 40 and _elided.startswith("x") and _elided.endswith(".json")
      and "\u2026" in _elided, repr(_elided))
check("a path that already fits is untouched",
      ui.shorten("short.json", None, 40) == "short.json")

# ---------------------------------------------------------------------------
section("a row with no value leaves no trailing spaces")

_empty = render(lambda: ui.ok("project", ""))[0]
check("no padding is emitted after an empty value", _empty == _empty.rstrip(),
      repr(_empty))

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
section("the artifact list")

_labelled = render(lambda: ui.outcome("Complete", "", [("report", small),
                                                      ("engine json", small)]))
_rows = [l for l in _labelled if small in l]
check("a label says which file is which", len(_rows) == 2
      and "REPORT" in _rows[0] and "ENGINE JSON" in _rows[1],
      "two .json files, and only one of them is this tool's document")
check("...and the sizes right-align against one edge",
      len({l.index("5.4 MB") for l in _rows}) == 1)
check("the label is separated from the location by a colon",
      all(" : " in l for l in _rows),
      "every row reads as one sentence: what it is, and where it went")
_col = render(lambda: ui.outcome("C", "", [("report generated at", small)]))
check("...and the dim separator is not counted as columns",
      all(len(l) <= ui.WIDTH for l in _col),
      f"longest {max(len(l) for l in _col)} of {ui.WIDTH}")
check("a bare path still works, with no label column",
      render(lambda: ui.outcome("Complete", "", [small]))[4].startswith(
          ui.PAD + " " + small),
      "the launchers pass paths, not pairs")

# ---------------------------------------------------------------------------
section("the findings frame")

_rows = [("[!!]", ui.RED, "SAFETY Blocked door       Cart 2 / Door-way    13.7s"),
         ("[-] ", ui.CYAN, "WATCH  Static cart        Cart 1               20.0s")]
_tbl = render(lambda: ui.table("findings", _rows, "2 total"))
check("the frame is square", len({len(l) for l in _tbl if l.strip()}) == 1,
      f"widths {sorted({len(l) for l in _tbl if l.strip()})}")
check("...and gutter-to-gutter, like the rules above it",
      len(_tbl[1]) == ui.WIDTH - len(ui.PAD) + len(ui.PAD))
check("the header carries its trailer on the right",
      _tbl[2].rstrip().endswith("2 total " + ui.G["v"]))
check("every row keeps its right-hand border",
      all(l.rstrip().endswith(ui.G["v"]) for l in _tbl[2:3] + _tbl[4:6]))

# A row wider than the frame must lose its own tail, never the border.
_over = render(lambda: ui.table("findings", [("[!!]", "", "x" * 200)], ""))
check("an over-wide row is cut, not allowed to tear the frame off",
      all(len(l) == len(_over[1]) for l in _over if l.strip())
      and _over[4].rstrip().endswith(ui.G["v"]))

# ---------------------------------------------------------------------------
section("colour never changes the column count")

_plain_widths = [len(l) for l in _tbl]
ui.set_plain(False)
ui.COLOR = True
for _name, _code in (("RESET", "0"), ("BOLD", "1"), ("DIM", "2"),
                     ("GREEN", "32"), ("YELLOW", "33"), ("RED", "31"),
                     ("CYAN", "36")):
    setattr(ui, _name, "\033[" + _code + "m")
_coloured = render(lambda: ui.table("findings", _rows, "2 total"))
_strip = lambda t: re.sub(r"\033\[[0-9;]*m", "", t)
check("padding is measured on the plain text, not the escapes",
      [len(_strip(l)) for l in _coloured] == _plain_widths,
      "an escape counted as a column tears the right border off")

check("the verdict mark takes its colour from the tone",
      ui.RED in render(lambda: ui.outcome("Found 3", "", [], tone="fail"))[2]
      and ui.GREEN in render(lambda: ui.outcome("Clean", "", [],
                                                tone="ok"))[2])
check("...and an unknown tone falls back to green rather than raising",
      ui.GREEN in render(lambda: ui.outcome("?", "", [], tone="banana"))[2])
ui.set_plain(True)

# ---------------------------------------------------------------------------
_right = render(lambda: ui.outcome("Complete in 343.1s", "cuda", []))
check("the right-hand trailer is right-aligned inside WIDTH",
      _right[2].rstrip().endswith("cuda") and len(_right[2]) <= ui.WIDTH)
_wide = render(lambda: ui.outcome("C" * (ui.WIDTH - 8), "cuda", []))
check("...and is dropped rather than pushing the line over WIDTH",
      "cuda" not in _wide[2] and len(_wide[2]) <= ui.WIDTH)

import shutil                                                  # noqa: E402
os.chdir(_cwd)
shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------
print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
