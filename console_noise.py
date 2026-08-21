"""Silence third-party console noise that is not actionable here.

Sibling to console_ui.py, which formats OUR output. This one suppresses other
people's — the two things that flood the demo console on Windows and say
nothing a reader can act on. Both are third-party defects, so the only options
are to filter them or to read past them during a live demo.

Deliberately narrow. Anything that could be a real signal from our own code, or
a warning someone could actually fix, stays visible: a broad
`warnings.simplefilter("ignore")` or a bare `except Exception` around the event
loop would also hide the next genuine bug.

Called once from app_poc_v2.py before the server starts. Nothing is discarded:
anything suppressed here that was not a plain client disconnect is appended in
full to temp/server_errors.log, and the console keeps one line pointing at it.


1. ConnectionResetError: [WinError 10054]
----------------------------------------
    Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
    ...
      File "...\\asyncio\\proactor_events.py", line 165, in _call_connection_lost
        self._sock.shutdown(socket.SHUT_RDWR)
    ConnectionResetError: [WinError 10054] An existing connection was forcibly
    closed by the remote host

A CPython bug, not ours. `_call_connection_lost` shuts the socket down inside a
bare `finally:` with no `except`:

    finally:
        if hasattr(self._sock, 'shutdown') and self._sock.fileno() != -1:
            self._sock.shutdown(socket.SHUT_RDWR)   # <-- unguarded
        self._sock.close()
        self._sock = None
        server = self._server
        ...

When the browser has already dropped the connection — a refresh, a closed tab,
a navigation away mid-run — `shutdown()` raises WinError 10054. It is an
OSError, so a single `except OSError: pass` there would have swallowed it;
CPython simply does not have one. Because the exception escapes a `finally`,
the FOUR statements after it never run, so each occurrence also leaks the
socket and never calls `server._detach()` — and a server whose connection count
never drops can hang on a graceful shutdown instead of exiting.

So this does not just mute the message: it finishes the teardown CPython
skipped. Only ConnectionResetError is swallowed; every other failure from that
callback still surfaces.


2. StarletteDeprecationWarning: HTTP_422_UNPROCESSABLE_ENTITY
-------------------------------------------------------------
    gradio\\routes.py:1404: StarletteDeprecationWarning:
    'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. Use
    'HTTP_422_UNPROCESSABLE_CONTENT' instead.

Raised by gradio's own route handler against its own starlette dependency.
Nothing in this repo references that constant, so there is no version of this
codebase in which the warning is actionable — it goes away when gradio updates.

It repeats rather than printing once because `StarletteDeprecationWarning`
subclasses UserWarning (starlette/exceptions.py), and because gradio uses
`warnings.catch_warnings()` in several helpers (image_utils, processing_utils,
networking). Entering one mutates the global filter state, which invalidates
every module's `__warningregistry__`, so the once-per-location de-duplication
resets and the same line prints again on the next request.

Filtered by CATEGORY, not by muting UserWarning at large: a real UserWarning
from numpy, torch or our own code still prints.
"""
import os
import sys
import traceback
import warnings

_applied = False

#: Where a suppressed traceback goes. Nothing is discarded -- the console stays
#: clean, the detail stays on disk. temp/ is already gitignored and already
#: where the engine writes its own artefacts.
ERROR_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "temp", "server_errors.log")

#: Winsock codes that all mean "the browser went away": WSAECONNABORTED,
#: WSAECONNRESET, WSAESHUTDOWN. Matched as well as the exception classes
#: because asyncio sometimes surfaces these as a plain OSError.
_DISCONNECT_WINERRORS = frozenset({10053, 10054, 10058})


def _is_client_disconnect(exc: BaseException | None) -> bool:
    """Whether this is a browser closing a connection rather than a bug.

    ConnectionError is the base of ConnectionResetError,
    ConnectionAbortedError, BrokenPipeError and ConnectionRefusedError, so one
    check covers the whole family that a refreshed tab or a closed window can
    produce.
    """
    if exc is None:
        return False
    if isinstance(exc, ConnectionError):
        return True
    return (isinstance(exc, OSError)
            and getattr(exc, "winerror", None) in _DISCONNECT_WINERRORS)


def _record(context_message: str, exc: BaseException | None) -> str | None:
    """Append a full traceback to ERROR_LOG. Returns the path, or None if the
    log itself could not be written -- which must never take the demo down."""
    try:
        os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
        with open(ERROR_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'=' * 70}\n{context_message}\n")
            if exc is not None:
                fh.write("".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__)))
        return ERROR_LOG
    except Exception:
        return None


def _quieten_starlette_deprecations() -> bool:
    """Filter the gradio-internal starlette deprecation. True if installed."""
    try:
        from starlette.exceptions import StarletteDeprecationWarning
    except Exception:
        # Starlette moved the class, or is not installed. Fall back to matching
        # the message, so this keeps working rather than silently doing nothing.
        warnings.filterwarnings(
            "ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*is deprecated.*")
        return True
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
    return True


def _quieten_proactor_connection_reset() -> bool:
    """Complete the teardown CPython's `_call_connection_lost` skips when the
    peer is already gone, and stop it logging. True if the patch was installed.

    Idempotent via a marker attribute: uvicorn reloads and repeated imports must
    not stack wrappers.
    """
    if sys.platform != "win32":
        return False
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport as _T
    except Exception:
        return False

    original = getattr(_T, "_call_connection_lost", None)
    if original is None or getattr(original, "_pops_patched", False):
        return False

    def _call_connection_lost(self, exc):
        try:
            original(self, exc)
        except ConnectionResetError:
            # The peer vanished before we could shut the socket down. CPython
            # raised out of its own `finally`, so nothing after the shutdown()
            # ran; do it here rather than leaking the socket and leaving the
            # server's connection count high.
            sock = getattr(self, "_sock", None)
            self._sock = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            server = getattr(self, "_server", None)
            self._server = None
            if server is not None:
                try:
                    server._detach()
                except Exception:
                    # Private API. A future asyncio may rename it; failing to
                    # detach is not worth taking the connection down for.
                    pass
            self._called_connection_lost = True

    _call_connection_lost._pops_patched = True
    _T._call_connection_lost = _call_connection_lost
    return True


def _install_event_loop_guard() -> bool:
    """Stop asyncio printing raw tracebacks for anything that happens on the
    server's event loop. True if the guard was installed.

    Patching one transport method was not a net. asyncio reports every
    unhandled callback failure through `default_exception_handler`, and a
    dropped browser connection can surface from several of them --
    `_call_connection_lost`, `_loop_reading`, a write that lost its peer -- each
    printing its own "Exception in callback ..." wall. This is the one funnel
    they all pass through, so it is the right place to filter.

    Two behaviours, and the split is the point:

      * a client disconnect is dropped silently, because it is not a fault and
        there is nothing to do about it;
      * anything else prints ONE line and goes to ERROR_LOG in full.

    Not a blanket swallow. A real failure still announces itself, still names
    its type, and its traceback is still recoverable -- it just does not shove
    thirty lines of stack between the reader and the demo. That distinction is
    what stops this from hiding the next genuine bug.

    Patched on BaseEventLoop rather than installed on one loop: uvicorn builds
    its own loop inside `demo.launch()`, well after this runs, and does not set
    a custom handler. Doing it on the class covers whatever loop it makes,
    without touching the deprecated event-loop-policy machinery.
    """
    try:
        from asyncio.base_events import BaseEventLoop
    except Exception:
        return False

    original = getattr(BaseEventLoop, "default_exception_handler", None)
    if original is None or getattr(original, "_pops_patched", False):
        return False

    def default_exception_handler(self, context):
        exc = context.get("exception")
        if _is_client_disconnect(exc):
            return
        message = context.get("message") or "unhandled event-loop error"
        where = _record(f"event loop: {message}", exc)
        label = type(exc).__name__ if exc is not None else "no exception"
        line = f"[SERVER] {message} ({label})"
        if where:
            line += f" - traceback in {os.path.relpath(where)}"
        # stdout, not stderr, and flushed. Our own [PERF]/[CANCEL] lines go to
        # stdout; sending this one to stderr made it jump ahead of them in the
        # console because the two streams buffer differently, so a message
        # about frame 300 could appear above the banner.
        print(line, flush=True)

    default_exception_handler._pops_patched = True
    BaseEventLoop.default_exception_handler = default_exception_handler
    return True


def report_fatal(exc: BaseException) -> None:
    """Print a tidy block for an exception that ended the process, and put the
    traceback in ERROR_LOG.

    For the top-level `demo.launch()` guard. A stack trace is the correct output
    for a developer and the wrong output for someone being shown the product:
    it reads as a crash even when the cause is "port 7860 is already in use".
    """
    where = _record(f"fatal: {type(exc).__name__}: {exc}", exc)
    body = [f"{type(exc).__name__}: {exc}", ""]
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (48, 98, 10048):
        body.append("Port 7860 is already in use - another copy of the demo is "
                    "probably still running. Close it and try again.")
        body.append("")
    if where:
        body.append(f"Full details: {where}")
    try:
        import console_ui as _ui
        _ui.problem("The demo could not start", body)
    except Exception:
        print("\n  The demo could not start\n")
        for line in body:
            print(f"  {line}")
        print()


def apply() -> list[str]:
    """Install every filter. Returns what was silenced, for logging. Idempotent."""
    global _applied
    if _applied:
        return []
    silenced = []
    if _install_event_loop_guard():
        silenced.append("raw event-loop tracebacks")
    if _quieten_proactor_connection_reset():
        silenced.append("WinError 10054 connection-reset tracebacks")
    if _quieten_starlette_deprecations():
        silenced.append("gradio's starlette deprecation warnings")
    _applied = True
    return silenced
