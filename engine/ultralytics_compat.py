# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Keeps ByteTrack's low-confidence second association alive across ultralytics
versions.

THE BUG
-------
ultralytics 8.4.40 started applying `fuse_score` to the SECOND association in
`BYTETracker.update`:

    dists = matching.iou_distance(r_tracked_stracks, detections_second)
    if self.args.fuse_score:                       # <-- added in 8.4.40
        dists = matching.fuse_score(dists, detections_second)
    matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)

`fuse_score` returns `cost = 1 - IoU * score`, and that call site gates at
`cost <= 0.5`, so a match needs `IoU * score >= 0.5`. But the second pool is by
construction `[track_low_thresh, track_high_thresh)` — 0.1 to 0.3 in
botsort_retail.yaml. Even at a perfect IoU of 1.0 the best achievable cost is
`1 - 0.3 = 0.7`, which never passes. The low-confidence pass is not degraded on
8.4.40, it is dead: measured 0 matches where 8.4.19 makes 23.

The two rules cancel: the second pass exists to rescue low-confidence
detections, and fusing score then penalises them for being low-confidence.

WHY IT MATTERS HERE
-------------------
That pass is what keeps a partially-occluded person or cart on the same track ID
instead of dropping it to Lost. On
1764092528600_B8A44F40EFB5-medium-OUTSIDE.mp4 it costs the person track on 84 of
415 frames, including a 74-frame unbroken run (342-415) covering the whole exit
— which turns a linked OUTBOUND cart into an UNLINKED EXIT and changes the POPS
verdict.

This is upstream behaviour, not a consequence of our tuning: stock
`botsort.yaml`/`bytetrack.yaml` ship `track_high_thresh: 0.25` with
`fuse_score: True`, giving an even worse floor of 0.75. Raising
`track_high_thresh` cannot work around it — you would need it above 0.5 AND
near-perfect IoU, which defeats the point of a recovery pass.

WHY NOT JUST `fuse_score: false`
--------------------------------
Because that switch is not per-pass. Turning it off also drops fusion from the
FIRST association, where our thresholds were tuned. Measured on the same clip
that costs 197 of 351 linked frames (351 -> 154) as cart association degrades.

THE FIX
-------
Restore the 8.4.19 split, which the yaml has no way to express — fuse in the
first association, never in the second:

  * hold `args.fuse_score` at False, so `update()`'s second association never
    fuses;
  * re-apply the configured value for the duration of `get_dists`, which IS the
    first association.

Delivered as SUBCLASSES registered into ultralytics' `TRACKER_MAP` rather than
as monkeypatched methods on their classes. Two reasons: nothing upstream is
mutated, and if a future ultralytics changes the `get_dists` signature this
breaks loudly at call time instead of silently mismatching.

Applied UNCONDITIONALLY, on every version. On releases whose second association
never fused (8.4.19 and earlier) `BOTSORT.get_dists` is the only place a BOTSORT
run reads `args.fuse_score` at all, so the swap is behaviourally inert there —
verified at 0 differing frames of 415 against a stock 8.4.19 run. Deliberately
NOT gated on a version check or a source sniff: a gate that stops recognising a
refactored upstream would silently disarm the fix, which is the exact failure
mode this module exists to prevent. `describe()` reports what was detected, for
logging only.

The behavioural guarantee is pinned by tests/test_bytetrack_low_confidence_recovery.py,
which asserts the outcome (a track survives a low-confidence dip) rather than
any of this implementation.
"""
import inspect

from ultralytics.trackers import track as _track
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from .config import NESTED_DUP_CONTAIN_MIN
from .detection_dedup import mask_for_results

#: Substring identifying the regressed call site in BYTETracker.update.
_SECOND_ASSOC_MARKER = "fuse_score(dists, detections_second)"

_installed = False


def _remember_original(args):
    """Stash the configured fuse_score on the ARGS, not just on the tracker.

    `on_predict_start` builds one merged config and one tracker per
    `dataset.bs`, all sharing that object. Reading `args.fuse_score` directly
    would make every tracker after the first see the False its predecessor
    wrote, and silently lose first-association fusion. `model.track()` forces
    batch=1 today, so this is hardening rather than a live bug.
    """
    if not hasattr(args, "_fuse_score_original"):
        args._fuse_score_original = bool(getattr(args, "fuse_score", False))
    return args._fuse_score_original


class _FirstAssociationFusionOnly:
    """Mixin: fuse detection score into the first association only.

    Ordered before the tracker base class so `__init__`/`get_dists` resolve
    here first.
    """

    def __init__(self, args, frame_rate: int = 30):
        self._fuse_first_association = _remember_original(args)
        args.fuse_score = False
        super().__init__(args, frame_rate)

    def get_dists(self, tracks, detections):
        prev = self.args.fuse_score
        self.args.fuse_score = self._fuse_first_association
        try:
            return super().get_dists(tracks, detections)
        finally:
            # Restore even on exception: leaking True here would hand the very
            # next second association the fusion this class exists to remove.
            self.args.fuse_score = prev


class _DropNestedDuplicates:
    """Mixin: collapse nested same-class detections before tracking them.

    Filters `update`'s input rather than post-processing its output, so no
    second track is ever created for a duplicate — see
    engine/detection_dedup.py for why that matters and why IoU NMS cannot do
    this. Applied ahead of update()'s high/low confidence split so a pair that
    straddles the split is still caught.
    """

    def update(self, results, img=None, feats=None):
        keep = mask_for_results(results, NESTED_DUP_CONTAIN_MIN)
        if not keep.all():
            results = results[keep]
            # feats is parallel to results; upstream indexes them together.
            if feats is not None and len(feats):
                feats = feats[keep]
        return super().update(results, img, feats)


class RetailBOTSORT(_DropNestedDuplicates, _FirstAssociationFusionOnly, BOTSORT):
    """BOTSORT that keeps the low-confidence pass alive and refuses to track one
    object twice."""


class RetailBYTETracker(_DropNestedDuplicates, _FirstAssociationFusionOnly, BYTETracker):
    """Same two guarantees for plain ByteTrack, in case the yaml selects it."""


#: The two predict-time events ultralytics' `register_tracker()` hooks. Both are
#: `functools.partial` wrappers around functions in `ultralytics.trackers.track`.
TRACKING_CALLBACK_EVENTS = ("on_predict_start", "on_predict_postprocess_end")


def _is_tracking_callback(fn) -> bool:
    """Is `fn` one of ultralytics' registered tracking callbacks?

    They are registered as `functools.partial(on_predict_*, persist=...)`, so the
    partial's wrapped function is what identifies them. Matching on the module
    rather than on the name keeps this from catching a same-named callback from
    another ultralytics subsystem (`utils.callbacks.base`/`hub` both define
    `on_predict_start`).
    """
    func = getattr(fn, "func", None)
    if func is None:
        return False
    return getattr(func, "__module__", "").endswith("trackers.track")


def tracking_callback_counts(model) -> dict:
    """How many tracking callbacks are registered per predict event."""
    callbacks = getattr(model, "callbacks", None) or {}
    return {
        event: sum(1 for fn in (callbacks.get(event) or [])
                   if _is_tracking_callback(fn))
        for event in TRACKING_CALLBACK_EVENTS
    }


def dedupe_tracking_callbacks(model) -> int:
    """Collapse duplicate tracking callbacks to one per event. Returns how many
    were dropped.

    WHY THIS EXISTS
    ---------------
    `Model.track()` calls `register_tracker()` whenever it finds no predictor,
    and that APPENDS its two callbacks instead of replacing them. There is still
    only ONE tracker instance, so every extra copy calls `tracker.update()` again
    on the same frame — each pass seeing only the boxes the previous one kept.
    The tracker's Kalman predict and its internal frame counter therefore advance
    N times per real frame while the detections it sees shrink: track
    confirmation and `track_buffer` are effectively divided by N, weak detections
    at door distance never survive to be confirmed, and new tracks stop being
    created part-way through a clip. What the operator sees is "the carts and
    people stopped being detected".

    `Model._apply()` nulls `predictor` on every `.to()`, so any device move can
    trigger the re-registration. engine/tracker.py parks and re-attaches the
    predictor across the local-VLM offload precisely to avoid it, but that
    contract has to hold across a Gradio event chain where a case report and the
    next run can overlap — and a contract that must hold everywhere is worth
    also enforcing at the one place that can see it broken. This is that place:
    it makes the invariant "exactly one tracking callback per event" true no
    matter which path lost the predictor.

    The LAST registration is the one kept. The callbacks are stateless with
    respect to the tracker — they read `predictor.trackers` when they run — so
    which copy survives cannot matter; keeping the newest just means the
    surviving partial is the one whose `persist` matches the live call.
    """
    callbacks = getattr(model, "callbacks", None) or {}
    dropped = 0
    for event in TRACKING_CALLBACK_EVENTS:
        registered = callbacks.get(event)
        if not registered:
            continue
        keep_index = None
        for i, fn in enumerate(registered):
            if _is_tracking_callback(fn):
                keep_index = i
        if keep_index is None:
            continue
        surviving = [fn for i, fn in enumerate(registered)
                     if i == keep_index or not _is_tracking_callback(fn)]
        dropped += len(registered) - len(surviving)
        registered[:] = surviving
    return dropped


def describe() -> str:
    """One-line diagnostic: which upstream shape is installed. Logging only —
    nothing branches on this."""
    try:
        import ultralytics
        version = ultralytics.__version__
    except Exception:
        version = "unknown"
    try:
        affected = _SECOND_ASSOC_MARKER in inspect.getsource(BYTETracker.update)
        shape = "fuses the second association" if affected else "second association unfused"
    except (OSError, TypeError):
        shape = "second association unreadable (compiled or stripped)"
    return f"ultralytics {version}: {shape}"


def install() -> bool:
    """Point ultralytics' tracker registry at our subclasses.

    `on_predict_start` asserts `tracker_type in {"bytetrack", "botsort"}`, so
    the registry entries have to keep those names — botsort_retail.yaml stays
    valid for stock ultralytics too. Idempotent; returns True the first time.
    """
    global _installed
    if _installed:
        return False
    _track.TRACKER_MAP["botsort"] = RetailBOTSORT
    _track.TRACKER_MAP["bytetrack"] = RetailBYTETracker
    _installed = True
    return True


#: Kept as an alias so callers reading older notes still work.
apply = install
