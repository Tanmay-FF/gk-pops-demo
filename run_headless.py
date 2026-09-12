#!/usr/bin/env python3
# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Prepare the machine, then analyse one video. No browser involved.

    gk_pops.bat                                   ask me questions, build the
                                                  command, offer to run it
    gk_pops.bat CLIP.mp4 --auto-zones             run exactly that
    gk_pops.bat --check-only                      build the environment, stop
    gk_pops.bat --with-case-report-model          also pre-fetch the 4 GB model

Two jobs, in this order.

**Make the machine ready.** The same environment `run_demo.bat` builds, built
the same way -- this file imports `ensure_environment()` and `check_weights()`
from run_demo.py rather than reimplementing them, so whichever of the two you
run first pays the download and the other is instant afterwards. Nothing here
is a second setup path.

**Then either run the command you typed, or help you write one.** Arguments are
passed through to gk_pops.py untouched, inside the environment. With no
arguments at all, `guided()` below asks a handful of plain questions, prints the
command it assembled, and offers to run it -- so the first thing someone learns
is what the command LOOKS like, not just that a report appeared.

This file deliberately imports nothing from `engine/`. It runs on whatever
interpreter gk_pops.bat found, which on a first run is a bare Python with no
torch in it; the analysis itself runs in the environment, as a subprocess.
That is also why the zone and camera-placement lists below are read out of the
repo's own files with `json` and `ast` rather than imported.

Exit codes are gk_pops.py's, passed straight through: 0 worked, 2 something
about the command was wrong, 3 the analysis failed, 130 interrupted. A setup
failure is 2 as well -- nothing ran.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional, Sequence

import console_ui as ui
from run_demo import (HERE, VIDEO_SUFFIXES, check_weights, ensure_environment,
                      fetch_case_report_model, venv_python)

#: Flags this file consumes itself. Everything else belongs to gk_pops.py and
#: is passed through untouched -- so a flag added there needs no change here.
OWN_FLAGS = {
    "--check-only": "build the environment and stop, without analysing anything",
    "--setup-only": "the same as --check-only",
    "--with-case-report-model": "also pre-fetch the 4 GB case-report model",
}

EXIT_OK = 0
EXIT_USAGE = 2

#: Where the guided walkthrough puts everything it is asked to keep. One folder
#: rather than "beside the clip" because the person using the walkthrough is the
#: person most helped by all of it landing in the same findable place.
GUIDED_OUT = "out"


# ---------------------------------------------------------------------------
# Reading the repo's own lists, without importing the engine
# ---------------------------------------------------------------------------
def camera_placements() -> list[str]:
    """The five camera angles, read out of engine/config.py with `ast`.

    Not imported: engine/config.py pulls in torchvision, which does not exist
    yet on a first run. Not retyped either -- a hardcoded copy here would drift
    from the dropdown and from gk_pops.py's own choices, and a placement the
    scoring layer does not recognise is silently treated as "outside".
    """
    source = (HERE / "engine" / "config.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CAMERA_PLACEMENTS"
                for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError(
        "engine/config.py no longer defines CAMERA_PLACEMENTS as a plain list. "
        "run_headless.py reads it from there so the two cannot drift.")


def camera_placement_slugs() -> dict[str, str]:
    """slug -> display string, read out of engine/config.py with `ast`.

    The same reason as camera_placements(): engine/config.py cannot be imported
    here on a first run. CAMERA_PLACEMENT_SLUGS is a plain dict literal over
    there precisely so this reader can see it.
    """
    source = (HERE / "engine" / "config.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CAMERA_PLACEMENT_SLUGS"
                for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(
        "engine/config.py no longer defines CAMERA_PLACEMENT_SLUGS as a plain "
        "dict. run_headless.py reads it from there so the two cannot drift.")


#: Plain-English labels for the five angles, for people who have never seen the
#: app. The VALUE passed to gk_pops.py is always the config string.
PLACEMENT_HELP = {
    "Outside (facing entrance)":
        "the camera is outside, looking at the doors from the car park",
    "Inside (facing exit)":
        "the camera is inside, looking straight at the way out",
    "Inside (exit on right)":
        "the camera is inside, and the way out is to the right of the picture",
    "Inside (exit on left)":
        "the camera is inside, and the way out is to the left of the picture",
    "Inside (exit on both sides)":
        "the camera is inside, between two sets of doors",
}


def clips() -> list[Path]:
    """Every video in sample_videos/, sorted. Empty when nobody has added one."""
    folder = HERE / "sample_videos"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in VIDEO_SUFFIXES)


def saved_placements() -> dict[str, str]:
    """Clip basename -> the camera placement saved with its newest zone set.

    Clips with no preset, and presets written before the placement was
    recorded, are simply absent. Absent means "nobody has said", which is what
    lets the question below fall back to the first option rather than to a
    guess.

    Read with the standard library for the same reason `clips_with_zones()` is
    -- see its docstring. `camera_placement` is a plain string in the payload,
    so `engine.zone_presets`, which cannot be imported before the environment
    exists, is not needed to find it.
    """
    folder = HERE / "zone_presets"
    if not folder.is_dir():
        return {}
    newest: dict[str, tuple[float, str]] = {}
    for path in folder.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                      # a stray file is not an error here
        video = payload.get("video")
        placement = payload.get("camera_placement")
        if not video or not placement:
            continue
        when = float(payload.get("saved_at") or 0.0)
        if when >= newest.get(str(video), (-1.0, ""))[0]:
            newest[str(video)] = (when, str(placement))
    return {video: placement for video, (_when, placement) in newest.items()}


def clips_with_zones() -> set[str]:
    """Basenames of clips that have at least one saved zone set.

    Read straight from the JSON in zone_presets/ with the standard library.
    engine/zone_presets.py is the real reader and it is not importable here --
    see the module docstring -- but the one field this needs, the recorded
    video basename, is plain JSON.
    """
    folder = HERE / "zone_presets"
    if not folder.is_dir():
        return set()
    found = set()
    for path in folder.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                      # a stray file is not an error here
        video = payload.get("video")
        if video:
            found.add(str(video))
    return found


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------
def shorten_out(path: Path) -> str:
    """`path` relative to this file when it sits under it, else as given."""
    try:
        return str(Path(path).resolve().relative_to(HERE))
    except ValueError:
        return str(path)


class Abandoned(Exception):
    """The person answering pressed Ctrl-C or closed the input."""


def _read(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise Abandoned from None


#: Keystrokes the arrow reader has to recognise, spelled as chr() rather than
#: as escapes so that nothing between here and the file on disk can reinterpret
#: them. msvcrt hands Ctrl-C back as a byte instead of raising, so these two are
#: the difference between a walkthrough somebody can leave and one they cannot.
CTRL_C = chr(3).encode("latin-1")
CTRL_D = chr(4).encode("latin-1")

#: The byte that says "an arrow key follows". Windows sends one of two: the
#: arrow pad uses 0xe0, the numeric keypad with Num Lock off uses 0x00.
ARROW_PREFIXES = (chr(0xE0).encode("latin-1"), chr(0).encode("latin-1"))

#: Cursor control, for redrawing the option list where it already is.
_ESC = chr(27)
CURSOR_UP = _ESC + "[%dA"
ERASE_LINE = _ESC + "[K"


#: Width of the gutter every option label starts after: PAD, a space, the
#: default marker, the number, two spaces. An explanation is indented to this
#: so it sits under its own option rather than under the number.
#: `ui.note()` cannot do it -- that aligns to the value column of a status
#: row, which in a numbered list puts the text halfway across the page with
#: nothing above it.
_OPTION_INDENT = len(ui.PAD) + 1 + 1 + 1 + 2

#: Printed under the options whenever the arrow keys are driving them.
#: Both ways are named: typing a number is what every transcript and
#: screenshot of this walkthrough shows, and it still works.
KEY_FOOTER = (f"{ui.PAD} {ui.DIM}Up and Down to move, Enter to choose, "
              f"or type a number.{ui.RESET}")


def _windows_key():
    """One keypress on Windows, as a name this module understands.

    Returns "up", "down", "enter", a digit character, or "" for a key with no
    meaning here. Raises Abandoned on Ctrl-C and Ctrl-D.

    That last part is the whole reason this is not three lines. msvcrt.getch
    does NOT raise KeyboardInterrupt: it hands Ctrl-C back as a byte like any
    other. Left unchecked, the arrow reader would swallow it and the
    walkthrough would become something a person cannot get out of, which is
    exactly what the typed prompt's `_read` is careful to prevent.
    """
    import msvcrt

    ch = msvcrt.getch()
    if ch in (CTRL_C, CTRL_D):
        raise Abandoned
    if ch in (b"\r", b"\n"):
        return "enter"
    # Arrow keys arrive as two reads: a marker byte, then the key. Both
    # markers are in use -- one for the arrow pad, one for the numeric keypad
    # with Num Lock off -- and reading only the first leaves the second byte
    # in the buffer to be misread as a keystroke of its own.
    if ch in ARROW_PREFIXES:
        return {b"H": "up", b"P": "down"}.get(msvcrt.getch(), "")
    if ch.isdigit():
        return ch.decode("ascii")
    return ""


#: The key reader, or None where there is not one. A module attribute rather
#: than a platform check inside `choose` so that it is a seam: the tests set it
#: to None to exercise the typed path, which is the one a piped or redirected
#: session gets and therefore the one that has to keep working.
_KEY_READER = _windows_key if sys.platform == "win32" else None


def _arrows_usable() -> bool:
    """Whether the option list can be driven with the arrow keys.

    Needs a reader, a real keyboard, and somewhere to move the cursor back to.
    `ui.COLOR` is the ANSI test: in plain mode the escape codes that repaint the
    list would be printed literally, so the typed prompt is used instead.
    """
    if _KEY_READER is None or not ui.COLOR:
        return False
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _option_lines(options: Sequence[tuple[str, ...]], selected: int
                  ) -> list[str]:
    """Every line of the option block, with `selected` marked.

    Built as a list rather than printed, because redrawing has to know exactly
    how many lines it put on screen: explanations wrap to a variable number of
    lines and only some options carry a badge, so the count cannot be derived
    from len(options).
    """
    pad = " " * _OPTION_INDENT
    lines = []
    for i, option in enumerate(options, 1):
        label, why = option[0], option[1]
        badge = option[2] if len(option) > 2 else ""
        mark = ui.G["go"] if i == selected else " "
        colour = ui.CYAN if i == selected else ui.DIM
        line = (f"{ui.PAD} {colour}{mark}{i}{ui.RESET}  "
                f"{ui.BOLD}{label}{ui.RESET}")
        if badge:
            line += (f"  {ui.DIM}{ui.G['sep']}{ui.RESET}  "
                     f"{ui.CYAN}{badge}{ui.RESET}")
        lines.append(line)
        if why:
            # Wrapped here rather than through ui.note() so it lines up under
            # the label above it.
            lines += [f"{pad}{ui.DIM}{chunk}{ui.RESET}"
                      for chunk in textwrap.wrap(
                          why, width=ui.WIDTH - _OPTION_INDENT,
                          break_long_words=False) or [""]]
    return lines


def choose(question: str, options: Sequence[tuple[str, ...]],
           default: int = 1, hint: str = "") -> int:
    """Ask a multiple-choice question. Returns a 1-based index.

    `options` is (label, explanation) or (label, explanation, badge). The
    answer is given with the arrow keys where the terminal allows it and by
    typing a number everywhere else; both accept Enter alone as `default`, so
    the whole walkthrough can still be completed by pressing Enter five times,
    which is the fastest way for someone to see what a working command looks
    like.

    The badge sits on the label's own line. It is for saying that an option is
    not merely the default but a recorded answer, and that has to be legible
    even when the two coincide, because a default marker on option 1 looks
    identical whether it was chosen for this clip or just came first.

    `hint` is a line under the heading, printed here rather than by the caller
    because `choose` prints the heading itself: a caller printing it first
    would put it above the question it belongs to.
    """
    print()
    ui.phase_free(question)
    if hint:
        # Wrapped, unlike ui.detail() -- a hint is prose written by a caller
        # who is not counting columns, and one long line is exactly the
        # overrun console_ui's WIDTH exists to prevent.
        for chunk in textwrap.wrap(hint, width=ui.WIDTH - len(ui.PAD) - 1):
            ui.detail(chunk)

    if _arrows_usable():
        return _choose_with_keys(options, default)

    for line in _option_lines(options, default):
        print(line)
    while True:
        answer = _read(f"{ui.PAD} Type a number and press Enter "
                       f"[{default}]: ")
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        ui.warn("not a choice", f"type a number from 1 to {len(options)}")


def _choose_with_keys(options: Sequence[tuple[str, ...]], default: int) -> int:
    """`choose`, driven by the arrow keys. Returns a 1-based index.

    Typing a number still works and still selects immediately, so a transcript
    of the old walkthrough, the docs, and muscle memory all keep working. Both
    ways are named on the footer for that reason.
    """
    selected = default
    height = len(_option_lines(options, selected))
    for line in _option_lines(options, selected):
        print(line)
    print(KEY_FOOTER)

    while True:
        key = _KEY_READER()
        if key == "enter":
            break
        if key == "up":
            selected = len(options) if selected == 1 else selected - 1
        elif key == "down":
            selected = 1 if selected == len(options) else selected + 1
        elif key.isdigit() and 1 <= int(key) <= len(options):
            selected = int(key)
            _repaint(options, selected, height)
            break
        else:
            continue
        _repaint(options, selected, height)

    # Off the footer and onto a fresh line, so whatever prints next does not
    # land on top of the list.
    print()
    return selected


def _repaint(options: Sequence[tuple[str, ...]], selected: int,
             height: int) -> None:
    """Redraw the option block in place, `height` + 1 lines above the cursor.

    Every line is erased before it is rewritten. Labels differ in length, so
    without the erase the tail of a longer previous line is left on screen
    beside the shorter one that replaced it.

    `height` is passed in rather than recomputed, so the cursor goes back
    exactly as far as it came even if a future option block were to change
    height between paints.
    """
    out = sys.stdout
    out.write(CURSOR_UP % (height + 1))          # past the footer too
    for line in _option_lines(options, selected):
        out.write(ERASE_LINE + line + "\n")
    out.write(ERASE_LINE + KEY_FOOTER + "\n")
    out.flush()


def confirm(question: str, default: bool = True) -> bool:
    """A yes/no question. Enter alone takes the default."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = _read(f"{ui.PAD} {question} {hint}: ").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def quote(value: str) -> str:
    """Wrap an argument in quotes when the shell would otherwise split it."""
    return f'"{value}"' if (" " in value or "(" in value) else value


def printed_command(argv: Sequence[str]) -> str:
    """The assembled command, formatted the way someone would type it.

    A flag and its value are one unit and never split across the line break --
    `--video-path` on one line and `out\\` on the next is the version somebody
    copies wrong. Continuations use cmd's `^`, since the person reading this is
    at a Windows prompt.
    """
    groups = [quote(argv[0])]
    rest = list(argv[1:])
    while rest:
        flag = rest.pop(0)
        if rest and not rest[0].startswith("-"):
            groups.append(f"{flag} {quote(rest.pop(0))}")
        else:
            groups.append(flag)

    lines, current = [], f"{ui.PAD}   gk_pops.bat"
    indent = f"{ui.PAD}       "
    for group in groups:
        if len(current) + 1 + len(group) > ui.WIDTH - 2:
            lines.append(current + " ^")
            current = indent + group
        else:
            current += " " + group
    lines.append(current)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The walkthrough
# ---------------------------------------------------------------------------
def guided() -> Optional[list[str]]:
    """Ask what to analyse and how, and return the gk_pops.py arguments.

    Returns None when the person decides not to run it, or when there is
    nothing to analyse. Raises Abandoned on Ctrl-C.

    Every question has a default that Enter accepts, and the assembled command
    is printed BEFORE anything runs -- the point of this walkthrough is that
    the second time round nobody needs it.
    """
    available = clips()
    zoned = clips_with_zones()
    placements_saved = saved_placements()

    ui.banner(f"POPS  {ui.G['sep']}  analyse a video", "", ui.os_label())
    ui.detail("A few questions, then it shows you the command and runs it.")
    ui.detail("Press Enter to take the suggested answer.")

    # --- which video ------------------------------------------------------
    # Asked until it produces a file that exists. A mistyped path used to end
    # the whole session, which meant one typo cost the walkthrough, every
    # answer already given, and the startup that got here. Ctrl-C is still the
    # way out, and it still works: `_read` raises Abandoned from inside this
    # loop exactly as it does anywhere else.
    while True:
        if available:
            options = [(p.name,
                        "zones have been drawn for this one" if p.name in zoned
                        else "no zones drawn yet; see question 3")
                       for p in available]
            options.append(("Something else",
                            "type the path to a video yourself"))
            picked = choose("Which video do you want to analyse?", options)
            if picked <= len(available):
                # Relative to this folder, not absolute: the whole point of
                # printing the command is that someone retypes it, and
                # `sample_videos\clip.mp4` is retypable where a 70-character
                # absolute path is not. gk_pops.py resolves it against the
                # directory the command was run from, which is this one.
                video = str(available[picked - 1].relative_to(HERE))
            else:
                video = _read(f"{ui.PAD} Path to the video: ").strip('"')
        else:
            ui.phase_free("Which video do you want to analyse?")
            ui.warn("sample_videos", "empty")
            ui.note("Videos are not part of this folder; they are store "
                    "recordings, so whoever sent you this sends them "
                    "separately. Drop one into sample_videos/ and it will be "
                    "listed here next time.")
            video = _read(f"{ui.PAD} Path to the video: ").strip('"')

        # is_file(), not exists(): a directory passes exists() and would be
        # carried all the way to probe_video's decode before failing, by which
        # point every other question has been answered again.
        if video and Path(video).is_file():
            break

        if not video:
            ui.warn("nothing typed", "no path, so nothing to analyse")
        elif Path(video).is_dir():
            ui.warn("that is a folder", video)
            ui.note("Name the video file inside it, not the folder itself.")
        else:
            ui.warn("not found", video)
            ui.note("Check the path and try again. Dragging the file into "
                    "this window types its path for you. Ctrl-C stops.")

        # With no clips to list there is no question to go back to, only the
        # same prompt again, and a prompt that repeats itself with no way
        # forward is worse than saying so and letting them start over.
        if not available:
            return None

    argv: list[str] = [video]

    # --- where was the camera --------------------------------------------
    placements = camera_placements()
    slugs = {display: slug
             for slug, display in camera_placement_slugs().items()}
    # Whoever drew the zones for this clip also answered this question, and
    # their answer is the better default -- they had the picture in front of
    # them. Still a question rather than a silent inheritance: the angle
    # decides which side of the door counts as leaving, so a wrong one
    # produces a quiet run rather than a failed one.
    saved = placements_saved.get(Path(video).name)
    default_pick = placements.index(saved) + 1 if saved in placements else 1
    picked = choose("Where was this camera?",
                    [(p, PLACEMENT_HELP.get(p, ""),
                      "SAVED WITH THIS VIDEO" if p == saved else "")
                     for p in placements],
                    default=default_pick,
                    hint=("Already set to the angle saved with this video's "
                          "zones. Press Enter to take it.")
                    if saved in placements else "")
    placement = placements[picked - 1]
    # Omitted only when gk_pops.py would resolve the same angle on its own, so
    # the printed command stays short without ever dropping a choice.
    #
    # That is no longer simply "option 1". gk_pops.py's fallback is now the
    # placement saved with the zones, and only the built-in default when there
    # is none -- so deliberately picking option 1 over a preset that says
    # something else is a real choice, and omitting the flag there would let
    # the preset silently overrule the person who just answered the question.
    # A clip with a saved angle always gets the flag, even when the answer
    # matches the file. The zones question has not been asked yet, so whether
    # gk_pops.py will read that preset at all is not known here -- and the
    # printed command is meant to be kept and retyped, where being explicit
    # about the angle is worth one more line.
    if placement != placements[0] or saved is not None:
        # The slug, not the label: this command is printed to be retyped, and
        # `inside_facing_exit` survives retyping where `"Inside (facing exit)"`
        # loses its quotes and becomes three arguments. gk_pops.py accepts
        # both. The question above still shows the human wording.
        argv += ["--camera-placement", slugs.get(placement, placement)]

    # --- zones ------------------------------------------------------------
    has_zones = Path(video).name in zoned
    if has_zones:
        picked = choose(
            "Which areas of the picture should be watched?",
            [("Use the zones already drawn for this video",
              "somebody marked the doorway in the app and saved it"),
             ("No zones",
              "you still get cart scores and the marked-up video, but the "
              "door and trolley rules cannot run")])
        if picked == 1:
            argv.append("--auto-zones")
    else:
        ui.phase_free("Which areas of the picture should be watched?")
        ui.warn("no zones", "nobody has marked this video up yet")
        ui.note("Without a marked doorway the blocked-door, unattended-trolley "
                "and abandoned-trolley rules cannot run. You still get cart "
                "scores, the events list and the marked-up video. To draw the "
                "zones once: run run_demo.bat, pick this video, draw the "
                "doorway, press Save zone set, then come back here.")
        if not confirm("Carry on without zones?", default=True):
            return None

    # --- what to keep -----------------------------------------------------
    # One ladder, each rung naming the files it adds. The case report used to
    # be a separate yes/no after this question, which made the real choice --
    # how much of the run survives it -- two questions wide and left the most
    # expensive output looking like an afterthought rather than the top of the
    # ladder.
    #
    # The heat map rides with the annotated video rather than having a rung of
    # its own. The engine draws it on every run whatever is asked for, so
    # --heatmap-path only decides whether the PNG is kept; charging a separate
    # question for a file that has already been produced buys nothing.
    picked = choose(
        "What should it keep?",
        [("JSON report only",
          "one .json file holding everything the run found: every frame, "
          "every cart, every event, every score"),
         ("JSON report and the annotated video",
          "adds the MP4 with boxes, trails and scores drawn on it, plus the "
          "heat map of where people spent their time"),
         ("JSON report, annotated video and the AI case report",
          "adds a written account of what a model saw in the frames the run "
          "captured, as a page you can open in a browser. Several minutes "
          "per video on top of everything above")],
        default=2)
    argv += ["--json-path", GUIDED_OUT + os.sep]
    if picked >= 2:
        argv += ["--video-path", GUIDED_OUT + os.sep,
                 "--heatmap-path", GUIDED_OUT + os.sep]
    if picked >= 3:
        # The flag takes the path. Bare --case-report derives its name
        # beside the CLIP, which in guided mode is the footage folder -- the one
        # place the walkthrough has just promised nothing lands ("Everything it
        # keeps goes into the out\ folder", below).
        argv += ["--case-report", GUIDED_OUT + os.sep]

    # --- outputs from an earlier clip in this same session -----------------
    # gk_pops.py refuses to clobber an existing output without --force, which
    # is right: a report someone is reading should not vanish under a rerun.
    # Inside a loop, though, demoing the same clip twice is the normal thing to
    # do, and the refusal arrives after the questions are answered and reads as
    # the walkthrough being broken. Asked here instead, before the command is
    # printed, so --force is visible in the command rather than applied behind
    # it.
    #
    # The JSON only. Guided mode always asks for it, so it is the one output
    # that is always on the disk after a run; the MP4 and the PNG beside it are
    # covered by the same --force.
    existing = Path(GUIDED_OUT) / (Path(video).stem + ".json")
    if existing.exists():
        print()
        ui.warn("already analysed", shorten_out(existing))
        if confirm("Overwrite what that run produced?", default=True):
            argv.append("--force")
        else:
            ui.note("Nothing was overwritten. Pick a different video, or move "
                    f"the {GUIDED_OUT}\ folder aside first.")
            return None

    # --- show it, then offer to run it ------------------------------------
    print()
    ui.hr(heavy=True)
    print(f"{ui.PAD} {ui.BOLD}Here is your command.{ui.RESET} "
          f"{ui.DIM}Next time you can type it straight in.{ui.RESET}")
    ui.hr(heavy=True)
    print()
    print(printed_command(argv))
    print()
    ui.detail(f"Everything it keeps goes into the {GUIDED_OUT}\\ folder, "
              f"beside this file.")
    print()

    return argv if confirm("Run it now?", default=True) else None


# ---------------------------------------------------------------------------
# Setup and dispatch
# ---------------------------------------------------------------------------
def prepare(with_model: bool) -> bool:
    """Build or verify the environment and the weights. False if it cannot."""
    plan = ["Environment"] + (["Case-report model"] if with_model else []) \
        + ["Checkout"]
    step = 0

    def next_phase() -> None:
        nonlocal step
        ui.phase(step + 1, len(plan), plan[step])
        step += 1

    next_phase()
    if not ensure_environment():
        ui.problem("Setup did not finish.", [
            "The messages above say why. Nothing was left running; fix the",
            "problem and start this again.",
        ])
        return False

    if with_model:
        next_phase()
        fetch_case_report_model()

    next_phase()
    return check_weights()


def analyse(argv: Sequence[str]) -> int:
    """Run gk_pops.py inside the environment. Returns its exit code."""
    python = venv_python()
    if not python.exists():
        ui.problem("The environment is not there.", [
            f"Expected an interpreter at {python}",
            "",
            "Run this file again (it builds one), or run run_demo.bat once.",
        ])
        return EXIT_USAGE
    print()
    return subprocess.run([str(python), str(HERE / "gk_pops.py"), *argv],
                          cwd=HERE).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Prepare the machine, then run or build the analysis command."""
    args = list(sys.argv[1:] if argv is None else argv)

    # Consumed here; everything else belongs to gk_pops.py.
    with_model = "--with-case-report-model" in args
    setup_only = bool({"--check-only", "--setup-only"} & set(args))
    passthrough = [a for a in args if a not in OWN_FLAGS]

    os.chdir(HERE)

    # --help is gk_pops.py's, and it needs the environment to print it -- the
    # choices come from engine.config. So it goes through setup like any other
    # command rather than short-circuiting here.
    if not passthrough and not setup_only:
        ui.logo("getting ready", "", ui.os_label())
        ui.detail("First run downloads several GB. Later runs take seconds.")

    if not prepare(with_model):
        return EXIT_USAGE

    if setup_only:
        print()
        ui.detail("Everything checks out. Run this again with a video to")
        ui.detail("analyse one, or with nothing after it to be walked through.")
        print()
        return EXIT_OK

    if passthrough:
        return analyse(passthrough)

    # No arguments: walk them through building one.
    if not sys.stdin.isatty():
        # Piped or scheduled. There is nobody to answer questions, and hanging
        # on input() would look like a hang rather than a prompt.
        menu()
        return EXIT_USAGE
    # One session, as many clips as they want. Everything above this loop --
    # the wordmark, prepare()'s download, the isatty check -- is per-machine
    # and stays outside it: reprinting the setup phases between clips would
    # read as the tool having restarted.
    #
    # Only a run that actually happened comes back here. Ctrl-C during the
    # questions, and answering no to "Run it now?", both end the session as
    # they always did -- someone who just declined a run is not asking to be
    # walked through the same questions again.
    ran = 0
    worst = EXIT_OK
    while True:
        try:
            chosen = guided()
        except Abandoned:
            return _farewell(ran, worst, "Stopped.")
        if chosen is None:
            return _farewell(ran, worst,
                             "Nothing more was run." if ran
                             else "Nothing was analysed.")

        code = analyse(chosen)
        ran += 1
        # The FIRST failure, not the last run: four clips of which one failed
        # is not a clean session, and gk_pops.bat hands this code straight back
        # to whatever called it.
        if code != EXIT_OK and worst == EXIT_OK:
            worst = code

        print()
        ui.hr()
        try:
            if not confirm("Analyse another video?", default=True):
                return _farewell(ran, worst)
        except Abandoned:
            return _farewell(ran, worst)


def _farewell(ran: int, code: int, lead: str = "") -> int:
    """Close the guided session, saying what it did. Returns `code` unchanged.

    The count is here because the loop makes "nothing was analysed" a claim
    that can be wrong -- after three clips it is the opposite of what happened.
    """
    print()
    if ran:
        clips = "1 video" if ran == 1 else f"{ran} videos"
        ui.detail(f"{lead} {clips} analysed this session; everything they kept "
                  f"is in the {GUIDED_OUT}\ folder.".strip())
    else:
        # The lead carries "nothing was analysed" on the paths where that is
        # true; saying it again here is how this printed it twice.
        ui.detail(f"{lead} Run this again when you are ready.".strip())
    print()
    return code


def menu() -> None:
    """The copy-and-paste list, for when there is nobody to ask questions of."""
    print()
    ui.phase_free("What do you want to do?")
    for title, command in (
            ("Analyse one video, simplest possible",
             "gk_pops.bat sample_videos\\myclip.mp4"),
            ("...and keep the marked-up video too",
             "gk_pops.bat sample_videos\\myclip.mp4 --video-path out\\"),
            ("...using the zones somebody drew in the app",
             "gk_pops.bat sample_videos\\myclip.mp4 --auto-zones"),
            ("Prepare this machine and stop",
             "gk_pops.bat --check-only"),
            ("See every option there is",
             "gk_pops.bat --help")):
        print()
        ui.detail(title)
        print(f"{ui.PAD}   {ui.CYAN}{command}{ui.RESET}")
    print()
    ui.detail("Full instructions, in plain language: RUN_WITHOUT_THE_APP.md")
    print()


if __name__ == "__main__":
    sys.exit(main())
