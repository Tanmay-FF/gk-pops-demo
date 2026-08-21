"""The third-party console noise filters, and their limits.

Two things flood the demo console on Windows and neither is actionable from
this repo:

  * `ConnectionResetError: [WinError 10054]` out of asyncio's
    `_call_connection_lost`, which is a CPython bug — the `sock.shutdown()` at
    proactor_events.py:165 sits in a bare `finally` with no `except`, so when
    the browser has already dropped the connection the error escapes AND skips
    the four cleanup statements after it;
  * gradio's own `StarletteDeprecationWarning` about
    `HTTP_422_UNPROCESSABLE_ENTITY`, repeated on almost every request.

What these tests really guard is the LIMIT of the muting. A filter that is one
step too broad hides the next real bug, and the failure mode is silence — you
would never notice. So each "is it quiet" test is paired with a "does a real
signal still get through" test.

CPU-only, no server, no GPU.
"""
import sys
import os
import asyncio
import contextlib
import io as _io
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import console_noise

ON_WINDOWS = sys.platform == "win32"


class Skip(Exception):
    """Raised by a test that does not apply to this platform."""


# ---------------------------------------------------------------------------
# The asyncio patch
# ---------------------------------------------------------------------------

class _Sock:
    """Enough socket for `_call_connection_lost`. `shutdown` raises whatever it
    is handed, which is how the real one behaves once the peer is gone."""

    def __init__(self, raises=None):
        self.raises = raises
        self.closed = False

    def fileno(self):
        return 3                      # anything but -1: the real code checks

    def shutdown(self, how):
        if self.raises is not None:
            raise self.raises

    def close(self):
        self.closed = True


class _Protocol:
    def __init__(self):
        self.lost_with = "not called"

    def connection_lost(self, exc):
        self.lost_with = exc


class _Server:
    def __init__(self):
        self.detached = False

    def _detach(self):
        self.detached = True


class _Transport:
    """Duck-types just the attributes `_call_connection_lost` touches."""

    def __init__(self, sock):
        self._called_connection_lost = False
        self._protocol = _Protocol()
        self._sock = sock
        self._server = _Server()


def _call(transport, exc=None):
    from asyncio.proactor_events import _ProactorBasePipeTransport as T
    return T._call_connection_lost(transport, exc)


def test_a_dropped_connection_no_longer_raises():
    if not ON_WINDOWS:
        raise Skip("the proactor transport is Windows-only")
    console_noise.apply()
    t = _Transport(_Sock(raises=ConnectionResetError(10054, "forcibly closed")))
    _call(t)          # must not raise — that is the whole point


def test_the_teardown_cpython_skips_is_completed():
    """Muting is not enough. CPython raises out of its own `finally`, so the
    socket is never closed and the server is never detached — a leak per
    aborted connection, and a server whose connection count never drops can
    hang on shutdown instead of exiting."""
    if not ON_WINDOWS:
        raise Skip("the proactor transport is Windows-only")
    console_noise.apply()
    sock = _Sock(raises=ConnectionResetError(10054, "forcibly closed"))
    t = _Transport(sock)
    server = t._server
    _call(t)

    assert sock.closed, "the socket was leaked"
    assert t._sock is None, "the transport still references its socket"
    assert server.detached, "the server was never detached (shutdown can hang)"
    assert t._server is None
    assert t._called_connection_lost is True, (
        "the transport still thinks connection_lost has not run, so a second "
        "call would repeat the whole teardown"
    )


def test_the_protocol_is_still_told_the_connection_was_lost():
    """The patch wraps the original, so everything the original did before the
    shutdown must still happen."""
    if not ON_WINDOWS:
        raise Skip("the proactor transport is Windows-only")
    console_noise.apply()
    t = _Transport(_Sock(raises=ConnectionResetError(10054, "gone")))
    sentinel = OSError("why the connection ended")
    _call(t, sentinel)
    assert t._protocol.lost_with is sentinel, (
        "the protocol was not notified; the patch swallowed more than the "
        "shutdown failure"
    )


def test_a_different_failure_still_propagates():
    """The limit. Only ConnectionResetError is swallowed — anything else out of
    that callback is a real problem and must stay loud."""
    if not ON_WINDOWS:
        raise Skip("the proactor transport is Windows-only")
    console_noise.apply()
    t = _Transport(_Sock(raises=RuntimeError("something genuinely wrong")))
    try:
        _call(t)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "a non-ConnectionResetError was swallowed; the filter is too broad"
        )


def test_a_clean_close_is_untouched():
    """The normal path — peer still there, shutdown succeeds — must behave
    exactly as CPython does."""
    if not ON_WINDOWS:
        raise Skip("the proactor transport is Windows-only")
    console_noise.apply()
    sock = _Sock()
    t = _Transport(sock)
    _call(t)
    assert sock.closed and t._sock is None and t._called_connection_lost is True


def test_the_patch_is_not_stacked_by_repeated_calls():
    """uvicorn reloads and repeated imports must not wrap the wrapper."""
    console_noise.apply()
    from asyncio.proactor_events import _ProactorBasePipeTransport as T
    first = T._call_connection_lost
    console_noise._quieten_proactor_connection_reset()
    console_noise._quieten_proactor_connection_reset()
    assert T._call_connection_lost is first, "the patch was applied twice"


# ---------------------------------------------------------------------------
# The warnings filter
# ---------------------------------------------------------------------------

def test_the_gradio_starlette_deprecation_is_filtered():
    console_noise.apply()
    try:
        from starlette.exceptions import StarletteDeprecationWarning
    except Exception:
        raise Skip("starlette is not installed")
    with warnings.catch_warnings(record=True) as seen:
        # "always" first, so this proves our filter is doing the work rather
        # than the default once-per-location de-duplication.
        warnings.simplefilter("always")
        console_noise._quieten_starlette_deprecations()
        warnings.warn("'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.",
                      StarletteDeprecationWarning)
    assert not seen, f"still printing: {[str(w.message) for w in seen]}"


def test_a_real_user_warning_still_prints():
    """The limit. StarletteDeprecationWarning subclasses UserWarning, so the
    lazy fix — muting UserWarning — would also hide numpy, torch and our own
    warnings."""
    console_noise.apply()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        console_noise._quieten_starlette_deprecations()
        warnings.warn("a real problem worth reading", UserWarning)
    assert len(seen) == 1, f"a genuine UserWarning was swallowed: {seen}"


def test_apply_is_idempotent_and_reports_what_it_silenced():
    console_noise._applied = False
    first = console_noise.apply()
    assert first, "apply() reported silencing nothing at all"
    assert console_noise.apply() == [], "apply() ran twice"


# ---------------------------------------------------------------------------
# The event-loop guard
# ---------------------------------------------------------------------------
# asyncio funnels every unhandled callback failure through
# default_exception_handler, so filtering there covers the whole family a
# dropped browser can produce -- _call_connection_lost, _loop_reading, a write
# that lost its peer -- rather than the one method a single patch reaches.

def _handled(exc, message="Exception in callback"):
    """Run one context through the guard. Returns what reached the console."""
    console_noise.apply()
    loop = asyncio.new_event_loop()
    out = _io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            loop.call_exception_handler({"message": message, "exception": exc})
    finally:
        loop.close()
    return out.getvalue()


def _winerror(code):
    exc = OSError(22, "shutdown")
    exc.winerror = code
    return exc


def test_every_flavour_of_dropped_connection_is_silent():
    """A refreshed tab, a closed window and a navigation away do not all raise
    the same class, so matching only ConnectionResetError would still leak
    tracebacks onto the demo console."""
    for exc in (ConnectionResetError(10054, "forcibly closed"),
                ConnectionAbortedError(10053, "aborted"),
                BrokenPipeError(),
                _winerror(10053), _winerror(10054), _winerror(10058)):
        printed = _handled(exc)
        assert printed == "", (
            f"{type(exc).__name__} still printed: {printed!r}"
        )


def test_a_real_error_prints_exactly_one_line():
    """The limit. A genuine bug must still announce itself -- just not with
    thirty lines of stack in front of whoever is watching the demo."""
    try:
        raise ValueError("something genuinely broken")
    except ValueError as exc:
        printed = _handled(exc, "Task exception was never retrieved")
    lines = [ln for ln in printed.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected one line, got {len(lines)}: {printed!r}"
    assert "ValueError" in lines[0], (
        f"the line does not name the exception type: {lines[0]!r}"
    )


def test_a_suppressed_traceback_is_still_recoverable():
    """Quiet is not the same as lost. The console gets a pointer; the file gets
    the stack."""
    console_noise.apply()
    with contextlib.suppress(OSError):
        os.remove(console_noise.ERROR_LOG)
    try:
        raise KeyError("a-distinctive-key")
    except KeyError as exc:
        printed = _handled(exc, "Task exception was never retrieved")
    assert os.path.exists(console_noise.ERROR_LOG), "nothing was written to disk"
    with open(console_noise.ERROR_LOG, encoding="utf-8") as fh:
        body = fh.read()
    assert "a-distinctive-key" in body, "the traceback did not reach the log"
    assert "Traceback" in body, "only the message was kept, not the stack"
    assert "server_errors.log" in printed, (
        "the console line does not say where the detail went"
    )


def test_a_missing_exception_does_not_break_the_handler():
    """asyncio calls this with contexts that carry a message and no exception
    (a bad file descriptor, a socket error reported by the selector). The guard
    must not raise from inside the error path."""
    printed = _handled(None, "socket.send() raised exception")
    assert "no exception" in printed or printed.strip(), printed


def test_the_guard_is_not_stacked_by_repeated_calls():
    console_noise.apply()
    from asyncio.base_events import BaseEventLoop
    first = BaseEventLoop.default_exception_handler
    console_noise._install_event_loop_guard()
    console_noise._install_event_loop_guard()
    assert BaseEventLoop.default_exception_handler is first, "guard applied twice"


# ---------------------------------------------------------------------------
# The fatal-startup block
# ---------------------------------------------------------------------------

def test_a_startup_failure_prints_no_traceback():
    """What the manager sees if port 7860 is taken. A stack trace is the right
    output for a developer and the wrong one for someone being shown the
    product -- it reads as a crash when the cause is mundane."""
    out = _io.StringIO()
    with contextlib.redirect_stdout(out):
        console_noise.report_fatal(
            OSError(10048, "Only one usage of each socket address is "
                           "normally permitted"))
    printed = out.getvalue()
    assert "Traceback" not in printed, f"a raw traceback reached the console:\n{printed}"
    assert "could not start" in printed
    assert "Port 7860" in printed, (
        "the address-in-use case should say so in words rather than leave the "
        "reader to decode errno 10048"
    )
    assert "server_errors.log" in printed


def test_a_startup_failure_still_records_the_detail():
    console_noise.apply()
    with contextlib.suppress(OSError):
        os.remove(console_noise.ERROR_LOG)
    with contextlib.redirect_stdout(_io.StringIO()):
        console_noise.report_fatal(RuntimeError("a-distinctive-startup-fault"))
    with open(console_noise.ERROR_LOG, encoding="utf-8") as fh:
        assert "a-distinctive-startup-fault" in fh.read()


def test_report_fatal_survives_an_unwritable_log():
    """The tidy block must still print if the log cannot be opened -- the whole
    point is that nothing here can take the demo down."""
    console_noise.apply()
    real = console_noise.ERROR_LOG
    console_noise.ERROR_LOG = os.path.join(os.devnull, "nope", "server.log")
    try:
        out = _io.StringIO()
        with contextlib.redirect_stdout(out):
            console_noise.report_fatal(RuntimeError("boom"))
        assert "could not start" in out.getvalue()
    finally:
        console_noise.ERROR_LOG = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for t in tests:
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
        else:
            passed += 1
            print(f"PASS {t.__name__}")
    print(f"\n{passed} passed, {skipped} skipped")
