# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The guided session: one walkthrough per clip, as many clips as they want.

Run with:  python tests/test_cli_session.py

`gk_pops.bat` with no arguments walks somebody through building a command and
then runs it. It used to end there, which meant demoing a second clip cost a
second startup. `run_headless.main()` now loops, and a loop has more ways to be
wrong than a straight line does:

  * it must not become one the user cannot leave -- declining a run, and
    closing stdin, both have to end the session;
  * the closing line makes a claim about how many videos were analysed, and
    "nothing was analysed" is the opposite of the truth after three of them;
  * the exit code is handed straight back by gk_pops.bat, so one failure
    inside a four-clip session cannot be hidden by the runs that follow it;
  * everything that is per-machine rather than per-clip -- the download, the
    weight check -- has to stay outside the loop.

`analyse()` is replaced throughout, so nothing here loads a model: what is
under test is the walkthrough's control flow, not the pipeline.
"""
import io
import os
import sys
import builtins
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_headless as rh                                      # noqa: E402

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (_PASS if ok else _FAIL).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
#: Answers keyed by a substring of the prompt rather than by position, so a
#: question added to the walkthrough does not silently shift every later answer
#: onto the wrong prompt. Anything unmatched takes the question's own default,
#: which is the path a person pressing Enter through the whole thing follows.
def drive(script: dict, codes: list[int], stop_after: int | None = None):
    """Run `main([])` against canned answers. Returns (exit code, runs, lines).

    `codes` is the exit status of each stubbed run, in order. `stop_after`
    raises EOFError once that many questions have been asked, which is what a
    closed stdin or a Ctrl-C looks like from inside `_read`.
    """
    seq, runs, asked = iter(codes), [], [0]

    def responder(prompt: str = "") -> str:
        asked[0] += 1
        if stop_after is not None and asked[0] > stop_after:
            raise EOFError
        low = prompt.lower()
        for key, answers in script.items():
            if key in low and answers:
                return answers.pop(0)
        return ""

    rh.analyse = lambda argv: (runs.append(list(argv)), next(seq))[1]
    builtins.input = responder
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = rh.main([])
    lines = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
    return code, runs, lines


def closing(lines: list[str]) -> str:
    return lines[-1] if lines else ""


_real_analyse = rh.analyse
_real_prepare = rh.prepare
_real_stdin = rh.sys.stdin

# A terminal, and a machine that is already set up. Neither is what this file
# is testing, and prepare() would otherwise try to build an environment.
rh.sys.stdin = type("_Tty", (), {"isatty": staticmethod(lambda: True)})()
rh.prepare = lambda with_model: True

# The typed path, pinned. isatty() alone does not choose it -- the stub above
# says True -- so without this every question in this file would sit waiting
# for a keypress that a redirected harness can never deliver. The arrow path
# gets its own section at the bottom, with a reader it can actually drive.
_real_reader = rh._KEY_READER
rh._KEY_READER = None

#: Press Enter through the numbered questions, say yes to running and to
#: overwriting. Individual cases override what they need.
YES = {"run it now": ["y"] * 9,
       "overwrite": ["y"] * 9}


def answers(**over) -> dict:
    merged = {k: list(v) for k, v in YES.items()}
    merged.update({k.replace("_", " "): list(v) for k, v in over.items()})
    return merged


try:
    # -----------------------------------------------------------------------
    section("the loop runs once per clip, and stops when asked")

    code, runs, lines = drive(answers(analyse_another=["n"]), [0])
    check("one run, then declining another ends the session",
          code == 0 and len(runs) == 1)
    check("...and the closing line counts it",
          "1 video analysed this session" in closing(lines),
          closing(lines))

    code, runs, lines = drive(answers(analyse_another=["y", "y", "n"]),
                              [0, 0, 0])
    check("saying yes walks through the next clip",
          len(runs) == 3, f"{len(runs)} runs")
    check("...and the count is pluralised",
          "3 videos analysed this session" in closing(lines),
          closing(lines))

    # -----------------------------------------------------------------------
    section("every way out of the loop is a way out")

    code, runs, lines = drive(answers(run_it_now=["n"]), [])
    check("declining the very first run ends the session",
          code == 0 and not runs,
          "someone who just said no is not asking to be asked again")
    check("...and does not claim anything was analysed",
          "Nothing was analysed." in closing(lines)
          and "analysed this session" not in closing(lines),
          closing(lines))

    # 9 questions is partway through the SECOND clip's walkthrough: the first
    # clip has been run by then, so this also pins the count on the way out.
    code, runs, lines = drive(answers(analyse_another=["y"]), [0],
                              stop_after=9)
    check("a closed stdin mid-walkthrough ends the session",
          code == 0 and len(runs) == 1,
          "_read turns EOF into Abandoned; a loop must not outlive its input")
    check("...and still says what the session did",
          "Stopped." in closing(lines)
          and "1 video analysed this session" in closing(lines),
          closing(lines))

    # -----------------------------------------------------------------------
    section("the exit code survives the clips that follow it")

    code, runs, _ = drive(answers(analyse_another=["y", "n"]), [3, 0])
    check("a failure followed by a clean run still exits non-zero",
          code == 3 and len(runs) == 2,
          "gk_pops.bat hands this straight back to whatever called it")
    code, runs, _ = drive(answers(analyse_another=["y", "n"]), [0, 4])
    check("a failure after a clean run is reported too", code == 4)
    code, _, _ = drive(answers(analyse_another=["y", "y", "n"]), [0, 5, 6])
    check("the FIRST failure is the one reported, not the last",
          code == 5, "5, not 6")

    # -----------------------------------------------------------------------
    section("outputs from an earlier clip in the same session")

    # The loop makes rerunning one clip normal, and gk_pops.py refuses to
    # clobber an output without --force. Asked before the command is printed,
    # so --force is visible in it rather than applied behind it.
    import tempfile                                            # noqa: E402
    from pathlib import Path                                   # noqa: E402

    clips = rh.clips()
    if not clips:
        check("SKIP: no clips in sample_videos to rerun", True)
    else:
        stem = Path(clips[0]).stem
        out = Path(rh.HERE) / rh.GUIDED_OUT
        out.mkdir(parents=True, exist_ok=True)
        stamp = out / f"{stem}.json"
        had = stamp.exists()
        if not had:
            stamp.write_text("{}", encoding="utf-8")
        try:
            _c, runs, _l = drive(answers(analyse_another=["n"]), [0])
            check("an existing report puts --force in the command",
                  bool(runs) and "--force" in runs[0],
                  "the refusal would otherwise arrive after every question "
                  "was answered and read as the walkthrough being broken")

            _c, runs, lines = drive(
                answers(overwrite=["n"], analyse_another=["n"]), [])
            check("...and declining it runs nothing",
                  not runs)
            check("...saying so rather than failing silently",
                  any("Nothing was overwritten" in ln for ln in lines))
        finally:
            if not had:
                stamp.unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    section("what the run keeps is one ladder, not two questions")

    def keep(rung: int) -> list[str]:
        """The command built when "What should it keep?" is answered `rung`.

        By intercepting `choose` rather than counting questions: every
        numbered question shares one prompt string ("Type a number..."), so a
        positional answer lands on whichever question happens to be third,
        and the zones question is only asked for a clip that has zones.
        """
        real = rh.choose

        def picky(question, options, default=1, hint=""):
            return rung if "keep" in question.lower() else default

        rh.choose = picky
        try:
            _c, runs, _l = drive(answers(analyse_another=["n"]), [0])
        finally:
            rh.choose = real
        return runs[0] if runs else []

    tier1, tier2, tier3 = keep(1), keep(2), keep(3)

    check("rung 1 keeps the JSON and nothing else",
          "--json-path" in tier1
          and not {"--video-path", "--heatmap-path", "--case-report"}
          & set(tier1))
    check("rung 2 adds the annotated video",
          "--video-path" in tier2 and "--case-report" not in tier2)
    check("...and the heat map with it",
          "--heatmap-path" in tier2,
          "the engine draws it on every run regardless, so keeping it costs "
          "nothing and does not deserve a question of its own")
    check("rung 3 adds the case report on top of both",
          {"--json-path", "--video-path", "--case-report"} <= set(tier3))
    # A subset check over the flag alone passed while the report was still
    # landing beside the CLIP -- which in guided mode is the footage folder,
    # the one place the walkthrough promises nothing lands. So the path is
    # asserted too, and asserted as the value that FOLLOWS the flag.
    check("...and points it at the same out\ folder as everything else",
          "--case-report" in tier3
          and tier3[tier3.index("--case-report") + 1]
          == rh.GUIDED_OUT + os.sep,
          "the walkthrough promises everything it keeps lands in out\\")

    # It was a separate yes/no after this question, which made one decision --
    # how much of the run survives it -- two questions wide.
    _c, _r, lines = drive(answers(analyse_another=["n"]), [0])
    check("the case report is no longer asked about separately",
          not any("Do you want the written case report" in ln
                  for ln in lines))
    check("...and its cost is still stated where it is now chosen",
          any("Several minutes per" in ln for ln in lines),
          "the warning was the point of the question it replaced")

    # -----------------------------------------------------------------------
    section("the question a saved angle is answered with")

    import console_ui as ui                                    # noqa: E402

    places = rh.camera_placements()

    def render_choose(**kw) -> list[str]:
        builtins.input = lambda prompt="": ""
        buf = io.StringIO()
        with redirect_stdout(buf):
            rh.choose("Where was this camera?", **kw)
        return buf.getvalue().splitlines()

    opts = [(p, "why " + p, "SAVED WITH THIS VIDEO" if p == places[0] else "")
            for p in places]
    shown = render_choose(options=opts, default=1,
                          hint="Already set to the angle saved with this "
                               "video's zones. Press Enter to take it.")
    marked = [ln for ln in shown if "SAVED WITH THIS VIDEO" in ln]
    check("the saved angle is named on its own line, not just defaulted",
          len(marked) == 1 and places[0] in marked[0],
          "a default marker on option 1 looks the same whether it was chosen "
          "for this clip or merely came first")
    check("...and it is the one Enter takes",
          ui.G["go"] in marked[0])

    # The bug this replaced: explanations went through ui.note(), which aligns
    # to a status row's value column and left them floating mid-page under
    # nothing.
    body = [ln for ln in shown if ln.strip().startswith("why ")]
    check("an explanation sits under the option it explains",
          bool(body) and all(len(ln) - len(ln.lstrip()) == rh._OPTION_INDENT
                             for ln in body),
          f"indent {rh._OPTION_INDENT}")

    check("nothing the question prints overruns WIDTH",
          all(len(ln) <= ui.WIDTH for ln in shown),
          max((len(ln) for ln in shown), default=0))

    plain = render_choose(options=[(p, "why " + p, "") for p in places],
                          default=1)
    check("a clip with no saved angle gets no badge and no hint",
          not any("SAVED" in ln for ln in plain)
          and not any("press Enter to take it" in ln for ln in plain))

    # -----------------------------------------------------------------------
    section("a path that is not a video re-asks instead of ending the run")

    from pathlib import Path as _P                             # noqa: E402

    def pick_path(typed: list[str]):
        """Answer the video question with "Something else" and these paths.

        Returns (times the video question was asked, whether a later
        question was reached, the printed output).
        """
        paths = iter(typed)
        asked = [0]
        seen: list[str] = []
        real = rh.clips()

        def responder(prompt: str = "") -> str:
            low = prompt.lower()
            if "path to the video" in low:
                return next(paths, str(real[0]) if real else "")
            if "run it now" in low:
                return "n"
            return ""

        # By question text, not by counting prompts: every numbered question
        # shares one prompt string, so a counter would tick on the camera and
        # zone questions too.
        original = rh.choose

        def picky(question, options, default=1, hint=""):
            seen.append(question)
            if "which video" not in question.lower():
                return default
            asked[0] += 1
            # "Something else" is the last option; once the bad paths are
            # spent, take the first real clip.
            return len(options) if asked[0] <= len(typed) else 1

        rh.choose = picky
        builtins.input = responder
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                try:
                    rh.guided()
                except rh.Abandoned:
                    pass
        finally:
            rh.choose = original
        text = buf.getvalue()
        # `choose` is patched, so the camera heading is never printed; that a
        # later question was ASKED is the signal the video question let go.
        got_past = any("camera" in q.lower() for q in seen)
        return asked[0], got_past, text

    if not rh.clips():
        check("SKIP: no clips in sample_videos", True)
    else:
        asked, got_past, text = pick_path(["hretr", "also-nonsense.mp4"])
        check("two bad paths ask the question three times",
              asked == 3, f"asked {asked}")
        check("...and the walkthrough carries on from there",
              got_past,
              "one typo used to cost the session, every answer already "
              "given, and the startup that got there")
        check("...having said what was wrong each time",
              text.count("NOT FOUND") == 2)

        # exists() is true for a directory, so this used to be carried all the
        # way to probe_video's decode before failing.
        _asked, _got, text = pick_path(["sample_videos"])
        check("a folder is refused as a video, by name",
              "THAT IS A FOLDER" in text,
              "exists() passes a directory; is_file() is the check")
        check("...and that is a different message from a missing file",
              "NOT FOUND" not in text)

    # -----------------------------------------------------------------------
    section("arrow keys pick the same answers typing a number does")

    def with_keys(keys: list[str], options, default: int = 1) -> int:
        """Run one `choose` against canned keystrokes on the arrow path."""
        pressed = iter(keys)
        rh._KEY_READER = lambda: next(pressed)
        rh._arrows_usable = lambda: True
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                return rh.choose("Pick one", options, default=default)
        finally:
            rh._KEY_READER = None
            rh._arrows_usable = _real_usable

    _real_usable = rh._arrows_usable
    three = [("A", "first"), ("B", "second"), ("C", "third")]

    check("Enter alone takes the default",
          with_keys(["enter"], three, default=2) == 2)
    check("Down moves forward", with_keys(["down", "enter"], three) == 2)
    check("Up moves back",
          with_keys(["down", "down", "up", "enter"], three) == 2)
    check("Down off the end wraps to the top",
          with_keys(["down", "down", "down", "enter"], three) == 1,
          "a list you can fall off the bottom of is a list you have to count")
    check("Up off the top wraps to the end",
          with_keys(["up", "enter"], three) == 3)
    check("typing a number still chooses it outright",
          with_keys(["3"], three) == 3,
          "every transcript and screenshot of this walkthrough shows a "
          "number being typed, and those have to keep working")
    check("a number out of range is ignored, not obeyed",
          with_keys(["9", "enter"], three) == 1)
    check("a key with no meaning here is ignored",
          with_keys(["", "enter"], three) == 1)

    # msvcrt.getch() returns Ctrl-C as a byte instead of raising, so a reader
    # that does not check for it turns the walkthrough into something nobody
    # can leave. `_windows_key` checks; this pins that the rest of the loop
    # lets the exception through rather than catching it.
    def ctrl_c():
        raise rh.Abandoned

    rh._KEY_READER = ctrl_c
    rh._arrows_usable = lambda: True
    try:
        raised = False
        try:
            with redirect_stdout(io.StringIO()):
                rh.choose("Pick one", three)
        except rh.Abandoned:
            raised = True
        check("Ctrl-C during an arrow selection ends the session",
              raised, "a loop must not be able to swallow it")
    finally:
        rh._KEY_READER = None
        rh._arrows_usable = _real_usable

    check("a redirected or piped session never takes the arrow path",
          not rh._arrows_usable(),
          "repainting needs ANSI and a keyboard; without both, numbers")

finally:
    rh._KEY_READER = _real_reader
    rh.analyse = _real_analyse
    rh.prepare = _real_prepare
    rh.sys.stdin = _real_stdin

# ---------------------------------------------------------------------------
print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
