# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Compatibility shim for ultralytics >= 8.4.40 — restores ByteTrack's
low-confidence second association.

8.4.40 started applying `fuse_score` to the SECOND association in
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

Setting `fuse_score: false` in the yaml is NOT a fix: it also drops fusion from
the FIRST association, where our thresholds were tuned. Measured on the same
clip that costs 197 of 351 linked frames (351 -> 154) because cart association
degrades.

So restore the 8.4.19 split exactly — fuse in the first association, never in
the second — which the yaml has no way to express:

  * flip `args.fuse_score` off so `update()`'s second association stays unfused;
  * re-apply the original value for the duration of `BOTSORT.get_dists`, which
    is the first association.

Idempotent, and a no-op on versions whose second association never fused.
"""
import inspect

from ultralytics.trackers import bot_sort as _bot_sort

_SECOND_ASSOC_MARKER = "fuse_score(dists, detections_second)"

_applied = False


def second_association_fuses_score() -> bool:
    """True if the installed BYTETracker.update fuses score into the second
    (low-confidence) association. Source-sniffed rather than version-compared so
    an upstream fix silently turns this shim off again."""
    try:
        src = inspect.getsource(_bot_sort.BYTETracker.update)
    except (OSError, TypeError):
        return False
    return _SECOND_ASSOC_MARKER in src


def apply() -> bool:
    """Patch BOTSORT if needed. Returns True if the patch was installed."""
    global _applied
    if _applied or not second_association_fuses_score():
        return False

    orig_init = _bot_sort.BOTSORT.__init__
    orig_get_dists = _bot_sort.BOTSORT.get_dists

    def __init__(self, args, frame_rate: int = 30):
        orig_init(self, args, frame_rate)
        # Remember what the yaml asked for, then take it away from update().
        #
        # Stash the original ON THE ARGS, not just on self: on_predict_start
        # builds one cfg and constructs one tracker per dataset.bs, all sharing
        # that object. Reading args.fuse_score directly would make every
        # tracker after the first see the False this line just wrote and
        # silently lose first-association fusion. model.track() forces batch=1
        # today, so this is hardening, not a live bug.
        if not hasattr(args, "_fuse_score_original"):
            args._fuse_score_original = bool(getattr(args, "fuse_score", False))
        self._fuse_first_association = args._fuse_score_original
        args.fuse_score = False

    def get_dists(self, tracks, detections):
        # ...and hand it back for the first association only.
        prev = self.args.fuse_score
        self.args.fuse_score = getattr(self, "_fuse_first_association", prev)
        try:
            return orig_get_dists(self, tracks, detections)
        finally:
            self.args.fuse_score = prev

    _bot_sort.BOTSORT.__init__ = __init__
    _bot_sort.BOTSORT.get_dists = get_dists
    _applied = True
    return True
