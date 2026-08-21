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

A new run does not wait for whatever is in flight -- it supersedes it. See
`supersede()` for why that needs its own predicate rather than a `request()`
before a `begin()`.

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


class RunSuperseded(RunCancelled):
    """Raised out of work that a NEWER run replaced, rather than work the user
    cancelled.

    The distinction is not cosmetic, it decides what gets painted. A user Cancel
    should say "cancelled" in the panel. A supersede must write NOTHING: the run
    that replaced it has already flushed those same components (case_report_html
    is in FLUSH_OUTPUT_NAMES), so returning any value here paints the corpse of
    the old run over the fresh panels of the new one -- a wrong-output bug of
    exactly the kind the sequence numbers exist to prevent, just reaching
    forwards instead of backwards.
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
        #: The sequence a Cancel PRESS targets: work at this sequence or newer
        #: stops, anything older is left alone. A floor.
        self._cancelled_at = None
        #: The newest sequence a SUPERSEDE cancelled: work at this sequence or
        #: OLDER stops. A ceiling, and the mirror image of `_cancelled_at`.
        #:
        #: Two predicates rather than one because the two triggers point in
        #: opposite directions. A press means "stop the newest thing"; a new run
        #: means "stop everything already in flight, but not me". Reusing the
        #: floor for both would cancel the new run along with what it replaced.
        #:
        #: Monotonic and never cleared: sequences only increase, so `seq <=
        #: ceiling` is automatically false for every future unit of work. That
        #: also makes it durable -- superseded work sees it on its next poll no
        #: matter how much later that is, which a flag something else clears
        #: cannot promise.
        self._cancelled_through = None

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
                # Only when no supersede is outstanding. Clearing the event
                # while superseded work is still unwinding would send its next
                # is_cancelled() down the fast path to False, and it would never
                # stop.
                if self._cancelled_through is None:
                    self._event.clear()
            return self._seq

    def supersede(self) -> tuple[int | None, int]:
        """Cancel everything already in flight, then claim the next sequence for
        the work replacing it. Returns (cancelled_through, new_seq).

        This is what a second Run press does: the user asked for a different
        video, so a case report or frame loop still running for the previous one
        is no longer wanted and should stop now rather than be waited out.

        Atomic on purpose. `request()` followed by `begin()` cannot express this
        -- `begin()` clears the cancel as stale the moment it bumps the sequence,
        so the work being cancelled would very likely never observe the flag at
        all (a local VLM polls between generated tokens, tens of milliseconds
        apart, while those two calls are microseconds apart). Doing both under
        one lock, with a ceiling the new sequence sits above, removes the race
        instead of narrowing it.

        `cancelled_through` is None when nothing had begun, so the caller can
        tell "replaced run 4" from "there was nothing to replace".
        """
        with self._lock:
            superseded = self._seq if self._seq else None
            if superseded is not None:
                self._cancelled_through = superseded
                self._event.set()
            self._seq += 1
            # A press that no longer refers to anything self-clears, exactly as
            # in begin(). The event stays set while the ceiling stands.
            if self._cancelled_at is not None and self._cancelled_at < self._seq:
                self._cancelled_at = None
            return superseded, self._seq

    def was_superseded(self, seq: int | None) -> bool:
        """Whether `seq` was stopped by a NEWER RUN rather than by the Cancel
        button. Decides whether the caller paints "cancelled" or writes nothing
        -- see RunSuperseded."""
        if seq is None:
            return False
        with self._lock:
            return (self._cancelled_through is not None
                    and seq <= self._cancelled_through)

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
            if (self._cancelled_through is not None
                    and seq <= self._cancelled_through):
                return True
            return self._cancelled_at is not None and seq >= self._cancelled_at

    def raise_if_cancelled(self, seq: int | None, where: str = ""):
        if not self.is_cancelled(seq):
            return
        tail = f" {where}" if where else ""
        if self.was_superseded(seq):
            raise RunSuperseded(f"superseded by a newer run{tail}")
        raise RunCancelled(f"cancelled by the user{tail}")

    def clear(self):
        """Forget any outstanding cancel. For tests and for a clean reset; the
        normal path clears itself through `begin()`."""
        with self._lock:
            self._cancelled_at = None
            self._cancelled_through = None
            self._event.clear()
