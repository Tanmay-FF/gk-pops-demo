"""Cooperative cancellation for a run and its chained case report.

Gradio's own `cancels=` cannot stop work that has already started: its docstring
is explicit that "functions that are currently running will be allowed to
finish". It drops queued jobs and clears the client's spinner, and that is all.
So a Cancel button that actually stops a frame loop or a local-VLM pass needs the
work itself to look at a flag, which is what this module is.

Why the sequence numbers, rather than one bare `threading.Event`:

A run and its case report are separate Gradio events (`app_poc_v2.py:1508` and
`:1567`), so at any moment there can be a report generating for run N while run
N+1 is already in its frame loop — or waiting on the engine lock. With a single
flag, cancelling run N+1 would also discard run N's pending report, which is a
silent wrong-output bug of exactly the kind `docs/device_drift_fix_plan.md`
section 9.4.2 exists to prevent. Scoping the cancel to a sequence keeps "cancel
the newest work" from reaching backwards into older work that nobody asked to
stop.

The token is engine-wide and its methods are cheap: `is_cancelled()` is an
`Event.is_set()` plus an int compare, which is why the frame loop can afford to
call it on every frame.
"""
import threading


class RunCancelled(Exception):
    """Raised out of a run (or a case report) that the user cancelled.

    Carries no state beyond its message: the caller's job is to paint a cleared
    dashboard and say so, not to salvage a partial result. Lives in its own
    module because both `tracker` and `vlm_analyzer` raise it, and `tracker`
    already imports `vlm_analyzer` — putting it in either would be a dependency
    to regret later.
    """


class CancelToken:
    """One cancel signal, scoped by a monotonic sequence per unit of work.

    Usage, from the two entry points that own a unit of work:

        seq = token.begin()                 # a run starts (or is queued)
        ...
        token.raise_if_cancelled(seq)       # at any checkpoint
        if token.is_cancelled(seq): ...     # or test it without raising

    and from the Cancel button's handler, on another thread:

        token.request()                     # cancels the newest work
    """

    def __init__(self):
        self._lock = threading.Lock()
        #: Set while a cancel is outstanding. An Event rather than a bool so a
        #: future caller can wait on it without a busy loop.
        self._event = threading.Event()
        self._seq = 0
        #: The sequence a cancel targets: work at this sequence or newer stops,
        #: anything older is left alone.
        self._cancelled_at = None

    def begin(self) -> int:
        """Claim the next sequence for a unit of work about to start.

        Call this BEFORE any waiting the unit does — before acquiring the
        engine lock, not after. A run that is blocked behind a case report has
        already begun as far as the user is concerned, and pressing Cancel
        during that wait has to reach it. Claiming the sequence first is what
        makes that work: `request()` then targets this run, and the checkpoint
        after the lock is acquired sees it.

        Also drops a cancel that no longer refers to anything. A press with
        nothing in flight targets the last sequence begun, so the next `begin()`
        moves past it and the cancel self-clears rather than killing the next
        run.
        """
        with self._lock:
            self._seq += 1
            if self._cancelled_at is not None and self._cancelled_at < self._seq:
                self._cancelled_at = None
                self._event.clear()
            return self._seq

    def request(self) -> int | None:
        """Cancel the newest unit of work. Returns the sequence it targets.

        Never blocks and never touches the engine lock — the whole point is that
        it runs on a Gradio worker thread while the run it is cancelling holds
        everything else.

        Returns None when nothing has ever begun, so the caller can tell "there
        was nothing to cancel" from "asked to stop run 4".
        """
        with self._lock:
            if self._seq == 0:
                return None
            self._cancelled_at = self._seq
            self._event.set()
            return self._seq

    def is_cancelled(self, seq: int | None) -> bool:
        """Whether work at `seq` should stop. False for `seq` None (untracked
        work) and for work older than the cancel."""
        if seq is None or not self._event.is_set():
            return False
        with self._lock:
            return self._cancelled_at is not None and seq >= self._cancelled_at

    def raise_if_cancelled(self, seq: int | None, where: str = ""):
        if self.is_cancelled(seq):
            raise RunCancelled(f"cancelled by the user{' ' + where if where else ''}")

    def clear(self):
        """Forget any outstanding cancel. For tests and for a clean reset; the
        normal path clears itself through `begin()`."""
        with self._lock:
            self._cancelled_at = None
            self._event.clear()
