"""Console formatting for the launcher scripts and the headless runner.

Stdlib only, on purpose. create_virtual_env.py is the thing that *builds* the
virtual environment, so it runs on whatever bare interpreter run_demo.bat
found: rich and colorama are not there yet, and never will be for that
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

Four consumers now: run_demo.py, create_virtual_env.py, console_noise.py and
gk_pops.py. The last is why set_plain(), progress_bar(), hr(), outcome() and
set_stream()
exist -- a headless run has a frame counter, a closing artifact list, and an
engine that prints over both of them, and putting the answers here rather than
in gk_pops.py is what stops the launcher and the runner drifting into two
house styles.

Nothing here writes to sys.stdout directly. set_stream()/stream() choose where
it goes, defaulting to whatever sys.stdout is AT THE MOMENT OF THE CALL; see
the note above _STREAM for why that matters.
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


# ---------------------------------------------------------------------------
# Where this module writes
#
# Every printer below goes through _print()/_out() rather than through the
# builtin print, for one reason: gk_pops.py replaces sys.stdout with a sink for
# the whole span in which the engine runs, so the engine's own [VOTE]/[PERF]/
# [CACHE] chatter cannot land in the middle of a redrawn progress bar. The
# phases and the bar are printed FROM INSIDE that span, by the progress
# callback, and they have to keep reaching the real terminal.
#
# None means "resolve sys.stdout at call time", and that is the default on
# purpose: contextlib.redirect_stdout is how tests/test_cli_console.py captures
# this module, and a stream pinned at import would leave it holding a handle
# pytest has already replaced -- the tests would then pass against nothing.
# ---------------------------------------------------------------------------
_STREAM = None


def _out():
    """The stream to write to right now."""
    return sys.stdout if _STREAM is None else _STREAM


def set_stream(stream) -> None:
    """Pin the output stream, or pass None to follow sys.stdout again."""
    global _STREAM
    _STREAM = stream


def stream(target):
    """`with ui.stream(sys.stderr): ...` -- redirect this module for a block.

    Restores the PREVIOUS ui stream rather than resetting to None, because the
    nesting is real: gk_pops.py pins the terminal for the engine span, and a
    warning printed from inside that span opens a stderr block within it.
    Resetting to None there would drop the warning into the sink.
    """
    import contextlib

    @contextlib.contextmanager
    def _swap():
        global _STREAM
        saved = _STREAM
        _STREAM = target
        try:
            yield
        finally:
            _STREAM = saved
    return _swap()


def _print(text: str = "") -> None:
    _out().write(text + "\n")


def _cap(text: str) -> str:
    """Upper-case a LABEL.

    Only the naming text goes through this -- the banner title, a phase
    heading, a status row's label, a problem heading. Never a value: paths are
    case-sensitive on every filesystem that is not Windows, model names and
    device strings are written the way their libraries spell them, and a
    sentence in note() or detail() is prose rather than a label.

    Applied here rather than at the ~60 call sites so the house style is one
    decision, and so run_demo.py and create_virtual_env.py inherit it without
    being touched.

    A unit stuck to a number is put back: `44.5s` is a duration and `44.5S` is
    a typo, and the same goes for the `x` in `1280x720`. The rule is narrow on
    purpose -- a single letter, directly after a digit, with no letter after
    it -- so it cannot reach into a word.
    """
    import re
    return re.sub(r"(?<=[0-9])([A-Z])(?![A-Za-z])",
                  lambda m: m.group(1).lower(), str(text).upper())


def _ansi_ok() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not getattr(_out(), 'isatty', lambda: False)():
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

#: Plain mode: no colour, and no carriage returns -- progress_bar() prints
#: occasional whole lines instead of redrawing one in place. For log files, CI
#: and `tee`, where a redrawn line is a wall of ^M and a colour code is noise.
#:
#: Starts equal to "colour was not available anyway", so a redirected launcher
#: log is already plain with nobody having asked. gk_pops.py sets it explicitly
#: from --log.
PLAIN = not COLOR


def set_plain(plain: bool = True) -> None:
    """Turn plain mode on or off after import.

    COLOR and the seven escape constants below it are bound once, at import,
    from _ansi_ok(). That is right for the launchers -- they have no flag --
    and wrong for a CLI, where --log plain and --quiet arrive after this module
    is already loaded. Rebinding them here is the only way a later decision can
    reach code that already captured the globals.

    set_plain(False) asks for colour rather than forcing it: _ansi_ok() is
    consulted again, so a non-TTY or NO_COLOR still wins. Nothing can make a
    pipe accept escape sequences.
    """
    global PLAIN, COLOR, RESET, BOLD, DIM, GREEN, YELLOW, RED, CYAN
    PLAIN = bool(plain)
    COLOR = False if PLAIN else _ansi_ok()
    RESET = _c("0")
    BOLD = _c("1")
    DIM = _c("2")
    GREEN = _c("32")
    YELLOW = _c("33")
    RED = _c("31")
    CYAN = _c("36")


def shorten(path, root=None, width: int = 0) -> str:
    """A path as short as it can be while still naming the same file.

    Two steps, in order. A path under `root` is printed relative to it, on the
    understanding that the caller has printed `root` once already -- 45 columns
    of repeated prefix on every row pushes the part that differs off the right
    edge. Whatever is left is then middle-elided to `width`.

    This deliberately reverses the rule outcome() documents for the artifact
    list, where a path is never truncated because copying it is the only thing
    anyone does with it. A status row is not that: it is read, not copied, and
    the envelope's `artifacts` block keeps every path absolute and whole. The
    middle is what goes, never the ends -- the drive and the filename are the
    two parts a reader identifies a file by.

    Args:
        path: The path, or anything str() names a path with.
        root: A directory to print relative to, or None.
        width: Longest result, or 0 for no limit.
    """
    import os as _os

    text = str(path)
    if root:
        try:
            rel = _os.path.relpath(text, str(root))
        except ValueError:
            #: Different drives on Windows -- relpath raises rather than
            #: returning a .. chain that would not resolve.
            rel = text
        if not rel.startswith(".." + _os.sep) and rel != "..":
            text = rel
    if width and len(text) > width:
        keep = width - 1
        tail = keep * 2 // 3
        text = text[:keep - tail] + "\u2026" + text[len(text) - tail:]
    return text


def _can_encode(text: str) -> bool:
    encoding = getattr(_out(), "encoding", None) or "ascii"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


#: One glyph, one meaning, so the left gutter can be scanned on its own:
#:
#:     ok    ready, or produced
#:     info  off, or skipped
#:     warn  a warning
#:     fail  an error
#:
#: Nothing else may take a badge. A timing is not a status and no longer has
#: one -- phase_end() hangs it off the closing rule instead. "dot" stays
#: defined because MARK_W measures it, and dropping it would silently narrow
#: the badge column for every row.
_FANCY = {"h": "─", "H": "━", "v": "│", "tl": "┌", "tr": "┐", "bl": "└",
          "br": "┘", "lt": "├", "rt": "┤",
          "ok": "✓", "warn": "!", "fail": "✗", "info": "·",
          "dot": "•", "go": "▶", "sep": "·", "bar": "█"}
_PLAIN = {"h": "-", "H": "=", "v": "|", "tl": "+", "tr": "+", "bl": "+",
          "br": "+", "lt": "+", "rt": "+",
          "ok": "+", "warn": "!", "fail": "x", "info": "-",
          "dot": "*", "go": ">", "sep": "-", "bar": "#"}

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

# ---------------------------------------------------------------------------
# In-place progress
#
# The one piece of state in this module. Every other function here is a pure
# print, but a bar drawn with \r leaves the cursor mid-line, and the next
# heading or status row would then land on top of it. _close_bar() is called at
# the top of each printer below; it is a no-op unless a bar is actually open,
# so the two launchers -- which never draw one -- are unaffected.
# ---------------------------------------------------------------------------
#: True between a progress_bar() that used \r and the newline that closes it.
_bar_open = False

#: Last whole percent announced in plain mode, so a 658-frame run prints ten
#: lines instead of 658. Reset by progress_end().
_bar_last_pct = -1

#: Plain mode prints a line when the percentage crosses a multiple of this.
PLAIN_PROGRESS_STEP = 10


def _close_bar() -> None:
    """End an open \\r line so the next print starts at column 0."""
    global _bar_open
    if _bar_open:
        _out().write("\n")
        _out().flush()
        _bar_open = False


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

#: The wordmark, as five lines of ASCII. Deliberately not box-drawing
#: characters: this is the first thing on screen, it is what a photo of the
#: window shows, and it has to survive a cp437 console, a redirected log and a
#: paste into an email. Every glyph here is in ASCII, so none of that can
#: mangle it -- there is no plain-mode fallback because none is needed.
#:
#: Trailing spaces are stripped; a line's own leading spaces are the art.
LOGO = r"""
   ____ _  __   ____   ___  ____  ____
  / ___| |/ /  |  _ \ / _ \|  _ \/ ___|
 | |  _| ' /   | |_) | | | | |_) \___ \
 | |_| | . \   |  __/| |_| |  __/ ___) |
  \____|_|\_\  |_|    \___/|_|   |____/
""".strip("\n").split("\n")


def logo(title: str = "", subtitle: str = "", right: str = "") -> None:
    """The wordmark, then one line saying what this run is.

    The alternative -- banner() -- puts a frame around its text, which is why
    it refuses a path: a right edge and a filename are not compatible at 76
    columns. There is no right edge here, so the clip name goes on the line
    under the wordmark at full length and the short SHA sits at the far right
    of it.

    Args:
        title: What kind of run this is, e.g. "headless run". Upper-cased.
        subtitle: Free text after the title, typically the clip name. Printed
            as given -- it is a filename, and filenames are case-sensitive.
        right: A right-aligned trailer, e.g. the checkout's short SHA. Dropped
            rather than pushing the line past WIDTH.
    """
    _close_bar()
    _print()
    for line in LOGO:
        _print(f"{PAD}{CYAN}{line}{RESET}")
    _print()

    title = _cap(title)
    joiner = f"  {G['sep']}  "
    # One line while the name fits beside the title, which is the common case;
    # its own line when it does not. Never truncated -- the subtitle is a
    # filename, and half a filename tells you less than a second line costs.
    inline = (subtitle and title
              and len(PAD) + len(title) + len(joiner) + len(subtitle)
              + (2 + len(right) if right else 0) <= WIDTH)

    head = title
    line = f"{PAD}{BOLD}{title}{RESET}"
    if subtitle and (inline or not title):
        head = f"{title}{joiner}{subtitle}" if title else subtitle
        line += f"{DIM}{joiner}{RESET}{subtitle}" if title else subtitle
    visible = len(PAD) + len(head)
    if right and visible + 2 + len(right) <= WIDTH:
        line += " " * (WIDTH - visible - len(right)) + f"{DIM}{right}{RESET}"
    _print(line)
    if subtitle and not inline and title:
        _print(f"{PAD}{subtitle}")
    _print(f"{PAD}{DIM}{G['h'] * (WIDTH - len(PAD))}{RESET}")


def banner(title: str, subtitle: str = "", right: str = "") -> None:
    """The one title block. Fixed text only, never a path; the frame has a
    right edge and a long path would blow through it."""
    _close_bar()
    inner = WIDTH - 4
    _print()
    _print(f"{PAD}{DIM}{G['tl']}{G['h'] * inner}{G['tr']}{RESET}")

    title = _cap(title)
    left = f"{BOLD}{title}{RESET}"
    visible = len(title)
    if right:
        gap = inner - 2 - visible - len(right)
        if gap < 1:
            gap = 1
        body = f"{left}{' ' * gap}{DIM}{right}{RESET}"
    else:
        body = left + " " * (inner - 2 - visible)
    _print(f"{PAD}{DIM}{G['v']}{RESET} {body} {DIM}{G['v']}{RESET}")

    if subtitle:
        _print(f"{PAD}{DIM}{G['v']}{RESET} {DIM}{subtitle}{RESET}"
              f"{' ' * (inner - 2 - len(subtitle))} {DIM}{G['v']}{RESET}")
    _print(f"{PAD}{DIM}{G['bl']}{G['h'] * inner}{G['br']}{RESET}")


#: Width the duration on a closing rule is padded to, so the digits of every
#: section line up into a column that can be compared down the page rather
#: than read one number at a time. Wide enough for "999.9s".
PHASE_TIME_W = 6


def phase(index: int, total: int, title: str, rule: bool = True) -> None:
    """`[2/4]  Case-report model` plus an underline. Numbered because the
    count is the part that reads as a system working through a list rather
    than a script printing whatever occurs to it.

    `rule=False` leaves the underline out, for a caller that will close the
    section with phase_end() instead -- see there for why the rule moved.
    """
    _close_bar()
    _print()
    _print(f"{PAD}{CYAN}[{index}/{total}]{RESET}  {BOLD}{_cap(title)}{RESET}")
    if rule:
        _print(f"{PAD}{DIM}{G['h'] * (WIDTH - len(PAD))}{RESET}")


def phase_end(seconds: float) -> None:
    """Close a section with its measured duration on the right of the rule.

        ──────────────────────────────────────────────────────────  23.6s

    The rule moves from under the heading to under the section because that is
    where it can carry the timing. A separate "done in 23.6s" line is a whole
    line of chrome for one number, and five of them in a run is five lines
    that say nothing about what the run found.

    The header itself would be the other place to put it, and cannot be: it is
    printed before the work starts, and rewriting it afterwards means counting
    physical lines, which a terminal narrower than WIDTH silently makes wrong.
    """
    _close_bar()
    text = f"{seconds:.1f}s".rjust(PHASE_TIME_W)
    rule = G["h"] * (WIDTH - len(PAD) - 1 - len(text))
    _print(f"{PAD}{DIM}{rule} {text}{RESET}")


def phase_free(title: str) -> None:
    """A phase heading without the [n/total] counter, for the app process --
    it does not know how many steps the launcher had."""
    _close_bar()
    _print()
    _print(f"{PAD}{BOLD}{_cap(title)}{RESET}")
    _print(f"{PAD}{DIM}{G['h'] * (WIDTH - len(PAD))}{RESET}")


def _badge(colour: str, mark: str) -> str:
    """`[x]` -- brackets dim, the mark itself in the status colour, so the
    badge column reads as a column rather than as stray punctuation."""
    return (f"{DIM}[{RESET}{colour}{mark.ljust(MARK_W)}{RESET}"
            f"{DIM}]{RESET}")


def _row(colour: str, mark: str, label: str, value: str) -> None:
    _close_bar()
    _print(f"{PAD} {_badge(colour, mark)}  "
           f"{_cap(label).ljust(LABEL_W) if value else _cap(label)}{value}")


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
    _close_bar()
    indent = f"{PAD} {' ' * BADGE_W}  {' ' * LABEL_W}"
    for chunk in textwrap.wrap(text, width=WIDTH - len(indent),
                               break_long_words=False,
                               break_on_hyphens=False) or [""]:
        _print(f"{indent}{DIM}{chunk}{RESET}")


def detail(text: str) -> None:
    """A line under a phase heading that is prose, not a status row."""
    _close_bar()
    _print(f"{PAD} {DIM}{text}{RESET}")


def command(text: str) -> None:
    """Echo the command a phase is about to run. Dim, because it is there for
    the person debugging a failed setup, not for the person watching a good
    one."""
    _close_bar()
    _print(f"{PAD} {DIM}$ {text}{RESET}")


def took(seconds: float, extra: str = "") -> None:
    _close_bar()
    tail = f"  {DIM}{G['sep']}{RESET}  {DIM}{extra}{RESET}" if extra else ""
    _print(f"{PAD} {_badge(DIM, G['dot'])}  {DIM}DONE IN "
           f"{seconds:.1f}s{RESET}{tail}")


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def progress_bar(done: int, total: int, desc: str = "",
                 extra: str = "") -> None:
    """One frame counter, redrawn in place.

        frames  658/658  100%  ########################  37.3 ms/frame

    In plain mode there is no bar and no carriage return: a whole line is
    printed only when the percentage crosses a PLAIN_PROGRESS_STEP boundary,
    so a 658-frame run leaves ten lines in a log file rather than 658.

    `total <= 0` means the length is unknown -- the count is printed without a
    percentage or a bar rather than dividing by zero.

    Args:
        done: Items finished so far, 1-based at the point of the call.
        total: Items expected, or 0/negative when not known.
        desc: The short noun for what is being counted, e.g. "frames".
        extra: A right-hand trailer, e.g. a rate. Dropped first when the line
            is too narrow for it.
    """
    global _bar_open, _bar_last_pct

    pct = int(done * 100 / total) if total > 0 else -1

    if PLAIN:
        if pct < 0:
            step = done % 100 == 0
        else:
            step = (pct // PLAIN_PROGRESS_STEP
                    != max(_bar_last_pct, 0) // PLAIN_PROGRESS_STEP
                    or pct >= 100 and _bar_last_pct < 100)
            step = step and pct != _bar_last_pct
        if not step:
            return
        _bar_last_pct = pct
        head = f"{PAD} {desc}  {done}/{total}" if total > 0 else f"{PAD} {desc}  {done}"
        body = f"  {pct}%" if pct >= 0 else ""
        tail = f"  {extra}" if extra else ""
        _print(f"{head}{body}{tail}"[:WIDTH])
        return

    head = (f"{PAD} {desc}  {done}/{total}  {pct:3d}%  " if total > 0
            else f"{PAD} {desc}  {done}  ")
    tail = f"  {extra}" if extra else ""
    room = WIDTH - len(head) - len(tail)
    if room < 8:                       # no space for a bar; drop the trailer
        tail, room = "", WIDTH - len(head)
    bar = ""
    if total > 0 and room >= 8:
        filled = int(room * done / total)
        bar = G["bar"] * filled + " " * (room - filled)
    line = f"{head}{CYAN}{bar}{RESET}{DIM}{tail}{RESET}"
    # Pad to the previous line's width so a shorter redraw cannot leave the
    # tail of the longer one behind it.
    _out().write("\r" + line + " " * 2)
    _out().flush()
    _bar_open = True


def progress_end() -> None:
    """Close an open bar and forget the plain-mode position."""
    global _bar_last_pct
    _close_bar()
    _bar_last_pct = -1


# ---------------------------------------------------------------------------
# Rules and closing blocks
# ---------------------------------------------------------------------------
def hr(heavy: bool = False) -> None:
    """The rule phase() draws under a heading, on its own.

    `heavy` is the double rule that brackets outcome(); problem() draws the
    same one in red.
    """
    _close_bar()
    glyph = G["H"] if heavy else G["h"]
    _print(f"{PAD}{DIM}{glyph * (WIDTH - len(PAD))}{RESET}")


def _fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1 << 20:
        return f"{n_bytes / (1 << 20):.1f} MB"
    if n_bytes >= 1 << 10:
        return f"{n_bytes / (1 << 10):.0f} KB"
    return f"{n_bytes} B"


#: Verdict tones for outcome(). The name, not the escape, because the escapes
#: are rebound by set_plain() and a caller holding one would keep a stale
#: colour -- or a live colour after colour was turned off.
TONES = {"ok": "GREEN", "warn": "YELLOW", "fail": "RED"}


def outcome(title: str, right: str = "", files=(), root=None,
            tone: str = "ok") -> None:
    """The closing block: heavy rule, one-line verdict, heavy rule, artifacts.

    Args:
        title: The verdict, already assembled -- "Complete in 343.1s · peak
            POPS 75 · 3 findings". Not wrapped; keep it inside WIDTH.
        tone: "ok", "warn" or "fail" -- what colour the verdict mark takes. A
            run that found something must not look the same as one that found
            nothing when all you did was glance at it.
        right: A right-aligned trailer on the verdict line. Dropped when the
            title leaves no room rather than pushing the line over WIDTH.
        files: What to list underneath. Either a bare path, or a
            `(label, path)` pair -- the label saying what the file IS, which
            an extension does not ("engine json" and "json" are both .json,
            and only one of them is the document this tool wrote).

    Each row carries the file's size and the clock time it was last written,
    both read from disk rather than from what the run believes it did. The
    time is the part that earns its column: every file here is normally
    seconds old, so one that is not is a file this run did NOT write -- an
    artifact left behind by an earlier run, or one the engine's tidy-up took
    before the path could be kept. A size alone cannot say that.

    Only the clock, not the date. The full timestamps are in the envelope
    under `cli.started_at` / `cli.finished_at`; this is a glance, and a date
    on every row would cost eight columns the path wants.

    A path that does not exist is listed anyway, with `missing` where its size
    would be -- saying a file is absent is more useful than omitting the row.

    `root` shortens each path for display -- relative to root where it lies
    under it, middle-elided where it does not. Size and time are read from the
    REAL path first; shortening before the stat would look up a filename with
    an ellipsis in it. Without a root the paths print whole, which is what the
    launchers want.
    """
    _close_bar()
    inner = WIDTH - len(PAD)
    _print()
    _print(f"{PAD}{DIM}{G['H'] * inner}{RESET}")
    title = _cap(title)
    colour = globals().get(TONES.get(tone, "GREEN"), GREEN)
    line = f"{PAD} {colour}{G['go']}{RESET}  {BOLD}{title}{RESET}"
    visible = len(PAD) + 1 + len(G["go"]) + 2 + len(title)
    if right and visible + 2 + len(right) <= WIDTH:
        line += " " * (WIDTH - visible - len(right)) + f"{DIM}{right}{RESET}"
    _print(line)
    _print(f"{PAD}{DIM}{G['H'] * inner}{RESET}")

    entries = [(e if isinstance(e, (tuple, list)) else ("", e))
               for e in files if e and (not isinstance(e, (tuple, list)) or e[1])]
    if entries:
        _artifact_rows(entries, root)


#: "999.9 MB" and "missing" both fit; the column is sized once so every size
#: in the list right-aligns against the same edge.
SIZE_W = 8
#: "16:04:12".
WHEN_W = 8


def _artifact_rows(entries, root=None) -> None:
    """The `(label, path)` rows under a verdict, in four aligned columns.

    The label column is only as wide as the labels actually present, so a run
    that kept one file does not indent it past a column of blanks.
    """
    import time as _time

    label_w = max(len(_cap(label)) for label, _ in entries)
    #: PAD, the leading space, " : " after the label, and the two-space gaps
    #: before the size and the time.
    fixed = len(PAD) + 1 + label_w + (3 if label_w else 0) + 2 + SIZE_W + 2
    path_w = WIDTH - fixed - WHEN_W

    for label, path in entries:
        try:
            stat = os.stat(str(path))
            size = _fmt_size(stat.st_size)
            when = _time.strftime("%H:%M:%S", _time.localtime(stat.st_mtime))
        except OSError:
            size, when = "missing", ""
        # Only shortened with a root. A caller that did not opt in keeps the
        # old promise: the path prints whole, however long it is.
        text = shorten(path, root, path_w) if root else str(path)
        head = (f"{PAD} {_cap(label).ljust(label_w)}{DIM} : {RESET}"
                if label_w else f"{PAD} ")
        meta = f"{size.rjust(SIZE_W)}  {when.rjust(WHEN_W)}".rstrip()
        # Measured, not len(head): the separator is dim, and counting its
        # escape bytes as columns would push every row over WIDTH.
        head_w = fixed - 2 - SIZE_W - 2
        if head_w + len(text) <= WIDTH - 2 - len(meta):
            _print(f"{head}{text.ljust(path_w)}  {DIM}{meta}{RESET}".rstrip())
        else:
            # A whole path that cannot share its line. Never truncated here --
            # only a caller that passed a root asked for that -- so the meta
            # goes underneath, right-aligned to the same edge it would have
            # had. A mangled path cannot be copied.
            _print(f"{head}{text}")
            if meta:
                _print(f"{PAD}{' ' * (WIDTH - len(PAD) - len(meta))}"
                       f"{DIM}{meta}{RESET}")


#: How wide a boxed table is. The full gutter-to-gutter width, so it lines up
#: with the phase rules above and below it.
TABLE_W = WIDTH - len(PAD)


def table(title: str, rows, right: str = "") -> None:
    """A framed block: a titled header, a divider, then one line per row.

    For the findings, which are the payload of the whole run and had been
    reduced to a count. A frame is worth its four extra lines exactly once in
    a run, on the thing someone is reading the run to see.

    Args:
        title: The header's left side. Upper-cased, like every other label.
        rows: `(mark, colour, text)` per row. `mark` is a short badge printed
            in `colour` -- "[!!]", "[!]", "[-]" -- and `text` is the rest of
            the line, ALREADY laid out into its columns by the caller. The
            caller owns the column widths because it is the only side that
            knows what the fields mean; this owns the frame.
        right: A right-aligned trailer in the header, e.g. "3 total".

    Padding is measured on the plain text and the colour is wrapped around it
    afterwards. Measuring a string that already carries escapes would count
    the escape bytes as columns and tear the right-hand border off.
    """
    _close_bar()
    inner = TABLE_W - 2
    _print()
    _print(f"{PAD}{DIM}{G['tl']}{G['h'] * inner}{G['tr']}{RESET}")

    head = _cap(title)
    gap = inner - 2 - len(head) - len(right)
    if gap < 1:
        gap = 1
    _print(f"{PAD}{DIM}{G['v']}{RESET} {BOLD}{head}{RESET}{' ' * gap}"
           f"{DIM}{right}{RESET} {DIM}{G['v']}{RESET}")
    _print(f"{PAD}{DIM}{G['lt']}{G['h'] * inner}{G['rt']}{RESET}")

    for mark, colour, text in rows:
        pad = inner - 2 - len(mark) - 1 - len(text)
        if pad < 0:
            text, pad = text[:len(text) + pad], 0
        _print(f"{PAD}{DIM}{G['v']}{RESET} {colour}{mark}{RESET} {text}"
               f"{' ' * pad} {DIM}{G['v']}{RESET}")
    _print(f"{PAD}{DIM}{G['bl']}{G['h'] * inner}{G['br']}{RESET}")


def problem(heading: str, body_lines) -> None:
    """The block that replaces a wall of prints when something is actually
    wrong. Rules above and below only, no side edges -- the lines inside carry
    paths and commands, and a right edge would either cut them or force them
    to wrap somewhere unhelpful.

    Body lines are wrapped, because half of them come from a SetupError
    message written as prose and the rest are hand-broken. Their own leading
    indent is preserved on the continuation, so a bullet stays a bullet.
    """
    _close_bar()
    _print()
    _print(f"{PAD}{RED}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    _print(f"{PAD}{RED}{BOLD}{_cap(heading)}{RESET}")
    _print(f"{PAD}{RED}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    for text in body_lines:
        if not text.strip():
            _print()
            continue
        lead = " " * (len(text) - len(text.lstrip()))
        for chunk in textwrap.wrap(text, width=WIDTH - len(PAD),
                                   subsequent_indent=lead + "  ",
                                   break_long_words=False,
                                   break_on_hyphens=False):
            _print(f"{PAD}{chunk}")
    _print(f"{PAD}{RED}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    _print()


def launching(url: str, lines) -> None:
    """The last thing on screen before the child process takes over the
    stream."""
    _close_bar()
    _print()
    _print(f"{PAD}{DIM}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    _print(f"{PAD} {GREEN}{G['go']}{RESET}  {BOLD}STARTING THE DEMO{RESET}"
          f"  {DIM}{G['sep']}{RESET}  {CYAN}{url}{RESET}")
    _print(f"{PAD}{DIM}{G['H'] * (WIDTH - len(PAD))}{RESET}")
    for text in lines:
        _print(f"{PAD} {DIM}{text}{RESET}" if text else "")
    _print()
