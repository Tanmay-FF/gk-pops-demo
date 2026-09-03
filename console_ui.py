"""Console formatting for the two launcher scripts.

Stdlib only, on purpose. create_virtual_env.py is the thing that *builds* the
virtual environment, so it runs on whatever bare interpreter run_demo.bat
found — rich and colorama are not there yet, and never will be for that
process. Everything here is ANSI escapes and str.

Two things degrade rather than fail:

  * colour, if stdout is not a terminal, NO_COLOR is set, TERM=dumb, or
    Windows refuses to turn on virtual-terminal processing. A redirected log
    then contains no escape sequences at all, which is what you want when
    someone mails you setup.log.
  * the box-drawing and status glyphs, if the stream cannot encode them.
    This is a backstop rather than a common path: the reconfigure below
    normally gets the stream to UTF-8 first, and the probe then passes. It
    fires when the reconfigure could not happen at all -- a stdout that is
    not a TextIOWrapper, e.g. under pytest's capture -- and turns the box
    into +---+ rather than raising UnicodeEncodeError from a cp437 or cp1252
    stream.

Nothing in here parses. The sentinel lines the launchers pass between
processes (---VERIFY---, ---FETCH-OK---, ::venv_name::) must stay unstyled
plain prints; a colour code inside one breaks the match silently.
"""
import os
import platform
import sys
import textwrap

#: 76, not 80. Windows consoles wrap at 80 and a line that exactly fills the
#: width leaves a blank line behind it in some terminals.
WIDTH = 76

#: Everything is indented by this. The left gutter is what makes the phases
#: read as nested under their heading without drawing a box around output we
#: do not control (pip's, and the child processes').
PAD = "  "

#: Before anything is probed or printed. On a stock Windows console
#: sys.stdout.encoding follows the code page -- cp437, or cp1252 with output
#: redirected to a file -- and printing an em dash there raises
#: UnicodeEncodeError. UTF-8 is what the launchers' text is written in, so ask
#: for it once, here, rather than in each of the three entry points that
#: import this module. errors="replace" so a stream that cannot be
#: reconfigured degrades to a question mark instead of a traceback.
#:
#: PYTHONIOENCODING covers the child processes: run_demo.py starts
#: create_virtual_env.py and app_poc_v2.py, and they inherit os.environ.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    #: Not a TextIOWrapper -- pytest's capture, or a stream someone replaced.
    #: The glyph probe below reads whatever encoding it does have and falls
    #: back to ASCII if that cannot hold the box characters.
    pass


def _ansi_ok() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    # Windows 10 1511+ has the sequences but not always enabled on the
    # handle. Turning them on is a no-op where they already are.
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(kernel32.SetConsoleMode(
            handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


COLOR = _ansi_ok()


def _c(code: str) -> str:
    return f"\033[{code}m" if COLOR else ""


RESET = _c("0")
BOLD = _c("1")
DIM = _c("2")
GREEN = _c("32")
YELLOW = _c("33")
RED = _c("31")
CYAN = _c("36")


def _can_encode(text: str) -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_FANCY = {"h": "─", "H": "━", "v": "│", "tl": "┌", "tr": "┐", "bl": "└",
          "br": "┘", "ok": "✓", "warn": "!", "fail": "✗", "info": "·",
          "dot": "•", "go": "▶", "sep": "·"}
_PLAIN = {"h": "-", "H": "=", "v": "|", "tl": "+", "tr": "+", "bl": "+",
          "br": "+", "ok": "+", "warn": "!", "fail": "x", "info": "-",
          "dot": "*", "go": ">", "sep": "-"}

G = _FANCY if _can_encode("".join(_FANCY.values())) else _PLAIN

#: Width of the label column in a status row. Wide enough for the longest
#: label either launcher prints ("virtual environment", "case-report model")
#: so the values line up into a single column down the whole run.
LABEL_W = 22

#: Badges are padded to the widest mark so the label column starts at the
#: same place for every row whichever glyph set is in use. Every mark in both
#: sets is one character, so this is 1 -- the padding exists so a longer mark
#: added later does not shift the column.
MARK_W = max(len(G[k]) for k in ("ok", "warn", "fail", "info", "dot"))

#: What a rendered badge occupies on screen: "[", the mark, "]".
BADGE_W = MARK_W + 2


def os_label() -> str:
    """`Windows 11 build 22631`, or `Linux 6.8.0-51-generic`.

    platform.release() is not usable on its own here. Windows 11 kept the
    10.0 kernel version, so it reports "10" -- and so does the registry's
    ProductName. The build number is the only thing that separates them:
    22000 is the first Windows 11 build. The product-type check keeps a
    Server release off that rewrite (Server 2025 is build 26100, and it is
    not Windows 11).
    """
    system = platform.system()
    if system != "Windows":
        return f"{system} {platform.release()}".strip()

    release = platform.release()
    build = 0
    workstation = True
    try:
        version = sys.getwindowsversion()
        build = version.build
        # 1 is VER_NT_WORKSTATION; 2 and 3 are the domain-controller and
        # server types.
        workstation = getattr(version, "product_type", 1) == 1
    except AttributeError:
        # Not CPython on Windows, or a build without getwindowsversion.
        # platform.version() is "10.0.22631" on the same machine.
        parts = platform.version().split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            build = int(parts[2])

    if release == "10" and workstation and build >= 22000:
        release = "11"

    label = f"Windows {release}"
    return f"{label} build {build}" if build else label

def banner(title: str, subtitle: str = "", right: str = "") -> None:
    """The one title block. Fixed text only — never a path; the frame has a
    right edge and a long path would blow through it."""
    inner = WIDTH - 4
    print()
    print(f"{PAD}{DIM}{G['tl']}{G['h'] * inner}{G['tr']}{RESET}")

    left = f"{BOLD}{title}{RESET}"
    visible = len(title)
    if right:
        gap = inner - 2 - visible - len(right)
        if gap < 1:
            gap = 1
        body = f"{left}{' ' * gap}{DIM}{right}{RESET}"
    else:
        body = left + " " * (inner - 2 - visible)
    print(f"{PAD}{DIM}{G['v']}{RESET} {body} {DIM}{G['v']}{RESET}")

    if subtitle:
        print(f"{PAD}{DIM}{G['v']}{RESET} {DIM}{subtitle}{RESET}"
              f"{' ' * (inner - 2 - len(subtitle))} {DIM}{G['v']}{RESET}")
    print(f"{PAD}{DIM}{G['bl']}{G['h'] * inner}{G['br']}{RESET}")


def phase(index: int, total: int, title: str) -> None:
    """`[2/4]  Case-report model` plus an underline. Numbered because the
    count is the part that reads as a system working through a list rather
    than a script printing whatever occurs to it."""
    print()
    print(f"{PAD}{CYAN}[{index}/{total}]{RESET}  {BOLD}{title}{RESET}")
    print(f"{PAD}{DIM}{G['h'] * (WIDTH - len(PAD))}{RESET}")


def phase_free(title: str) -> None:
    """A phase heading without the [n/total] counter, for the app process --
    it does not know how many steps the launcher had."""
    print()
    print(f"{PAD}{BOLD}{title}{RESET}")
    print(f"{PAD}{DIM}{G['h'] * (WIDTH - len(PAD))}{RESET}")


def _badge(colour: str, mark: str) -> str:
    """`[x]` -- brackets dim, the mark itself in the status colour, so the
    badge column reads as a column rather than as stray punctuation."""
    return (f"{DIM}[{RESET}{colour}{mark.ljust(MARK_W)}{RESET}"
            f"{DIM}]{RESET}")


def _row(colour: str, mark: str, label: str, value: str) -> None:
    print(f"{PAD} {_badge(colour, mark)}  {label.ljust(LABEL_W)}{value}")


def ok(label: str, value: str = "") -> None:
    _row(GREEN, G["ok"], label, value)


def warn(label: str, value: str = "") -> None:
    _row(YELLOW, G["warn"], label, f"{YELLOW}{value}{RESET}" if value else "")


def fail(label: str, value: str = "") -> None:
    _row(RED, G["fail"], label, f"{RED}{value}{RESET}" if value else "")


def info(label: str, value: str = "") -> None:
    _row(DIM, G["info"], label, value)


def note(text: str) -> None:
    """A continuation line under a status row: no badge, aligned with the
    value column so a two-line explanation still reads as one entry. Wrapped,
    except for a single long word -- a path has no break points and mangling
    it into two lines makes it uncopyable."""
    indent = f"{PAD} {' ' * BADGE_W}  {' ' * LABEL_W}"
    for chunk in textwrap.wrap(text, width=WIDTH - len(indent),
                               break_long_words=False,
                               break_on_hyphens=False) or [""]:
        print(f"{indent}{DIM}{chunk}{RESET}")


def detail(text: str) -> None:
    """A line under a phase heading that is prose, not a status row."""
    print(f"{PAD} {DIM}{text}{RESET}")


def command(text: str) -> None:
    """Echo the command a phase is about to run. Dim, because it is there for
    the person debugging a failed setup, not for the person watching a good
    one."""
    print(f"{PAD} {DIM}$ {text}{RESET}")


def took(seconds: float) -> None:
    print(f"{PAD} {_badge(DIM, G['dot'])}  {DIM}done in "
          f"{seconds:.1f}s{RESET}")


def problem(heading: str, body_lines) -> None:
    """The block that replaces a wall of prints when something is actually
    wrong. Rules above and below only, no side edges -- the lines inside carry
    paths and commands, and a right edge would either cut them or force them
    to wrap somewhere unhelpful.

    Body lines are wrapped, because half of them come from a SetupError
    message written as prose and the rest are hand-broken. Their own leading
    indent is preserved on the continuation, so a bullet stays a bullet.
    """
    print()
    print(f"{PAD}{RED}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    print(f"{PAD}{RED}{BOLD}{heading}{RESET}")
    print(f"{PAD}{RED}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    for text in body_lines:
        if not text.strip():
            print()
            continue
        lead = " " * (len(text) - len(text.lstrip()))
        for chunk in textwrap.wrap(text, width=WIDTH - len(PAD),
                                   subsequent_indent=lead + "  ",
                                   break_long_words=False,
                                   break_on_hyphens=False):
            print(f"{PAD}{chunk}")
    print(f"{PAD}{RED}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    print()


def launching(url: str, lines) -> None:
    """The last thing on screen before the child process takes over the
    stream."""
    print()
    print(f"{PAD}{DIM}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    print(f"{PAD} {GREEN}{G['go']}{RESET}  {BOLD}Starting the demo{RESET}"
          f"  {DIM}{G['sep']}{RESET}  {CYAN}{url}{RESET}")
    print(f"{PAD}{DIM}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    for text in lines:
        print(f"{PAD} {DIM}{text}{RESET}" if text else "")
    print()
