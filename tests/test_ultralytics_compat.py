"""ByteTrack second-association compatibility shim.

ultralytics 8.4.40 applies `fuse_score` to the low-confidence SECOND
association, whose cost is `1 - IoU*score` gated at 0.5. With
botsort_retail.yaml's `track_high_thresh: 0.3` the second pool is [0.1, 0.3),
so the best achievable cost is 0.7 and no match can ever pass — the pass is
dead, not merely degraded.

These tests pin the shim's two guarantees:
  * the first association still fuses (that is where the thresholds were tuned);
  * the second association never does.

Runs on any machine — no GPU, no weights, no video.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ultralytics.trackers import bot_sort
from ultralytics.trackers.utils import matching

from engine import ultralytics_compat

#: False on an ultralytics whose second association never fused — the shim is a
#: no-op there and the behavioural tests have nothing to assert against.
AFFECTED = ultralytics_compat.second_association_fuses_score()


class Skip(Exception):
    """Raised by a test that does not apply to the installed ultralytics."""


class Args:
    """Minimal stand-in for the merged tracker config."""
    def __init__(self, **kw):
        self.tracker_type = "botsort"
        self.track_high_thresh = 0.3
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.4
        self.track_buffer = 120
        self.match_thresh = 0.7
        self.fuse_score = True
        self.gmc_method = "sparseOptFlow"
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.with_reid = False
        self.__dict__.update(kw)


class Det:
    """Only `.score` is read by fuse_score."""
    def __init__(self, score):
        self.score = score


def test_second_association_cost_floor_exceeds_the_gate():
    """The arithmetic the shim exists for: with a [0.1, 0.3) pool, fusing score
    into the second association puts every cost above the 0.5 gate, even at a
    perfect IoU. Version-independent — this is why the pass cannot be rescued
    by retuning track_high_thresh."""
    perfect_iou_cost = np.zeros((1, 3))          # IoU == 1.0 everywhere
    dets = [Det(0.11), Det(0.2), Det(0.299)]     # the whole legal low pool
    fused = matching.fuse_score(perfect_iou_cost, dets)
    assert fused.min() > 0.5, (
        "second-association fusion should be unsatisfiable for the low pool; "
        f"got min cost {fused.min()}"
    )


def test_apply_is_idempotent():
    first = ultralytics_compat.apply()
    assert ultralytics_compat.apply() is False, "apply() must not patch twice"
    if not AFFECTED:
        assert first is False, "must be a no-op where the second pass never fused"


def test_shim_disarms_update_but_keeps_first_association_fusion():
    if not AFFECTED:
        raise Skip("installed ultralytics does not fuse the second association")
    ultralytics_compat.apply()
    trk = bot_sort.BOTSORT(args=Args(fuse_score=True), frame_rate=30)

    # What BYTETracker.update sees for the second association: off.
    assert trk.args.fuse_score is False
    # What get_dists will restore for the first association: on.
    assert trk._fuse_first_association is True

    # get_dists must hand the flag back exactly as it found it, or the very
    # next second association inherits the fusion this shim removed.
    trk.get_dists([], [])
    assert trk.args.fuse_score is False, "get_dists leaked the first-association flag"


def test_first_association_actually_fuses():
    """Not just the flag — get_dists must produce score-fused costs."""
    if not AFFECTED:
        raise Skip("installed ultralytics does not fuse the second association")
    ultralytics_compat.apply()
    trk = bot_sort.BOTSORT(args=Args(fuse_score=True), frame_rate=30)

    seen = {}
    real_fuse = matching.fuse_score

    def spy(cost, dets):
        seen["called"] = True
        return real_fuse(cost, dets)

    matching.fuse_score = spy
    try:
        # One track, one detection, boxes identical so IoU is 1.0.
        box = np.array([10.0, 10.0, 50.0, 90.0])

        class Obj:
            score = 0.6
            angle = None          # iou_distance branches on this
            def __init__(self):
                self.xyxy = box
                self.tlwh = np.array([10.0, 10.0, 40.0, 80.0])

        trk.get_dists([Obj()], [Obj()])
    finally:
        matching.fuse_score = real_fuse

    assert seen.get("called"), "first association no longer fuses detection score"


def test_shared_args_across_trackers_keep_first_association_fusion():
    """on_predict_start builds ONE cfg and one tracker per dataset.bs. The
    second tracker must not read the False the first one wrote."""
    if not AFFECTED:
        raise Skip("installed ultralytics does not fuse the second association")
    ultralytics_compat.apply()
    shared = Args(fuse_score=True)
    first = bot_sort.BOTSORT(args=shared, frame_rate=30)
    second = bot_sort.BOTSORT(args=shared, frame_rate=30)
    assert first._fuse_first_association is True
    assert second._fuse_first_association is True, (
        "second tracker lost first-association fusion to the shared args object"
    )


def test_fuse_score_false_in_yaml_is_still_honoured():
    """A user who genuinely wants no fusion anywhere must still get that."""
    if not AFFECTED:
        raise Skip("installed ultralytics does not fuse the second association")
    ultralytics_compat.apply()
    trk = bot_sort.BOTSORT(args=Args(fuse_score=False), frame_rate=30)
    assert trk.args.fuse_score is False
    assert trk._fuse_first_association is False


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
    import ultralytics
    print(f"\n{passed} passed, {skipped} skipped "
          f"(ultralytics={ultralytics.__version__}, affected={AFFECTED})")
