"""Unit tests for engine/ultralytics_compat.py — the wiring, not the behaviour.

The behavioural guarantee (a track survives a low-confidence dip) is asserted
by tests/test_bytetrack_low_confidence_recovery.py, deliberately without
reference to this module, so it keeps working if the fix is ever re-implemented
a different way. What is pinned HERE is only what this module promises:

  * the registry ultralytics reads is pointed at our subclasses;
  * the registry keys stay the two names on_predict_start will accept;
  * installation is idempotent and unconditional;
  * `fuse_score: false` in the yaml is still honoured.

Runs anywhere — no GPU, no weights, no video.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ultralytics.trackers import bot_sort, byte_tracker
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.trackers.utils import matching

from engine import ultralytics_compat


class Args:
    """The merged tracker config, as ultralytics hands it to a tracker."""

    def __init__(self, **kw):
        self.tracker_type = "botsort"
        self.track_high_thresh = 0.3
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.4
        self.track_buffer = 120
        self.match_thresh = 0.7
        self.fuse_score = True
        self.gmc_method = "none"
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.with_reid = False
        self.__dict__.update(kw)


class Det:
    """Only `.score` is read by fuse_score."""

    def __init__(self, score):
        self.score = score


def test_second_association_cost_floor_exceeds_the_gate():
    """The arithmetic this module exists for: with a [0.1, 0.3) pool, fusing
    score into the second association puts every cost above its 0.5 gate, even
    at a perfect IoU. Version-independent — this is why the pass cannot be
    rescued by retuning track_high_thresh."""
    perfect_iou_cost = np.zeros((1, 3))          # IoU == 1.0 everywhere
    dets = [Det(0.11), Det(0.2), Det(0.299)]     # the whole legal low pool
    fused = matching.fuse_score(perfect_iou_cost, dets)
    assert fused.min() > 0.5, (
        "second-association fusion should be unsatisfiable for the low pool; "
        f"got min cost {fused.min()}"
    )


def test_install_points_the_registry_at_our_subclasses():
    ultralytics_compat.install()
    assert TRACKER_MAP["botsort"] is ultralytics_compat.RetailBOTSORT
    assert TRACKER_MAP["bytetrack"] is ultralytics_compat.RetailBYTETracker


def test_registry_keys_stay_the_names_upstream_accepts():
    """trackers/track.py asserts `tracker_type in {"bytetrack", "botsort"}`, so
    the fix cannot be shipped under a new tracker_type — botsort_retail.yaml has
    to stay valid for stock ultralytics too."""
    ultralytics_compat.install()
    assert {"bytetrack", "botsort"} <= set(TRACKER_MAP)


def test_subclasses_still_are_the_upstream_trackers():
    """Anything upstream that isinstance-checks a tracker must keep working."""
    assert issubclass(ultralytics_compat.RetailBOTSORT, bot_sort.BOTSORT)
    assert issubclass(ultralytics_compat.RetailBYTETracker, byte_tracker.BYTETracker)


def test_install_is_idempotent():
    ultralytics_compat.install()
    assert ultralytics_compat.install() is False, "install() must not run twice"
    # ...and the registry is not disturbed by the second call.
    assert TRACKER_MAP["botsort"] is ultralytics_compat.RetailBOTSORT


def test_install_is_unconditional():
    """Deliberately NOT gated on a version check or source sniff: a gate that
    stopped recognising a refactored upstream would silently disarm the fix."""
    import inspect
    src = inspect.getsource(ultralytics_compat.install)
    assert "second_association" not in src, (
        "install() must not branch on the detected upstream shape"
    )


def test_describe_reports_without_branching():
    text = ultralytics_compat.describe()
    assert "ultralytics" in text and text.strip()


def test_configured_fuse_score_is_remembered_and_withheld():
    ultralytics_compat.install()
    trk = ultralytics_compat.RetailBOTSORT(args=Args(fuse_score=True), frame_rate=30)
    # What update()'s second association sees: off.
    assert trk.args.fuse_score is False
    # What get_dists will restore for the first association: on.
    assert trk._fuse_first_association is True


def test_fuse_score_false_in_yaml_is_still_honoured():
    """A user who genuinely wants no fusion anywhere must still get that."""
    ultralytics_compat.install()
    trk = ultralytics_compat.RetailBOTSORT(args=Args(fuse_score=False), frame_rate=30)
    assert trk.args.fuse_score is False
    assert trk._fuse_first_association is False


def test_get_dists_restores_the_flag_even_when_it_raises():
    """A leaked True would hand the very next second association exactly the
    fusion this module removes."""
    ultralytics_compat.install()
    trk = ultralytics_compat.RetailBOTSORT(args=Args(fuse_score=True), frame_rate=30)

    real = matching.iou_distance

    def boom(a, b):
        raise RuntimeError("upstream blew up")

    matching.iou_distance = boom
    try:
        trk.get_dists([], [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the injected failure to propagate")
    finally:
        matching.iou_distance = real

    assert trk.args.fuse_score is False, "get_dists leaked the flag on the error path"


def test_shared_args_across_trackers_keep_first_association_fusion():
    """on_predict_start builds ONE cfg and one tracker per dataset.bs. The
    second tracker must not read the False the first one wrote."""
    ultralytics_compat.install()
    shared = Args(fuse_score=True)
    first = ultralytics_compat.RetailBOTSORT(args=shared, frame_rate=30)
    second = ultralytics_compat.RetailBOTSORT(args=shared, frame_rate=30)
    assert first._fuse_first_association is True
    assert second._fuse_first_association is True, (
        "second tracker lost first-association fusion to the shared args object"
    )


# ---------------------------------------------------------------------------
# Duplicate tracking callbacks. Model.track() registers its two callbacks
# whenever it finds no predictor and APPENDS them, so any path that loses the
# predictor (every .to() nulls it) leaves a second copy running the tracker a
# second time on every frame. Reproduced on the real pipeline: run 2 of one
# process fell from 4 carts / 15 people to 2 carts / 7 people.
# ---------------------------------------------------------------------------
class _FakeModel:
    """Just the `callbacks` dict ultralytics keeps on a Model."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks is not None else {}


def _tracking_cb(persist=True):
    """A stand-in for what register_tracker() appends."""
    import functools
    import ultralytics.trackers.track as track_mod
    return functools.partial(track_mod.on_predict_postprocess_end, persist=persist)


def _foreign_cb():
    """A same-named callback from another ultralytics subsystem, which the
    dedupe must leave alone — utils.callbacks.base and .hub both define one."""
    from ultralytics.utils.callbacks import base
    return base.on_predict_postprocess_end


def test_a_single_registration_is_left_alone():
    model = _FakeModel({"on_predict_postprocess_end": [_foreign_cb(), _tracking_cb()]})
    assert ultralytics_compat.dedupe_tracking_callbacks(model) == 0
    assert len(model.callbacks["on_predict_postprocess_end"]) == 2


def test_duplicates_collapse_to_one_per_event():
    model = _FakeModel({
        "on_predict_start": [_tracking_cb(), _tracking_cb(), _tracking_cb()],
        "on_predict_postprocess_end": [_foreign_cb(), _tracking_cb(), _tracking_cb()],
    })
    dropped = ultralytics_compat.dedupe_tracking_callbacks(model)
    assert dropped == 3, dropped
    assert ultralytics_compat.tracking_callback_counts(model) == {
        "on_predict_start": 1, "on_predict_postprocess_end": 1}


def test_the_foreign_callback_survives_and_keeps_its_place():
    foreign = _foreign_cb()
    model = _FakeModel({"on_predict_postprocess_end": [foreign, _tracking_cb(), _tracking_cb()]})
    ultralytics_compat.dedupe_tracking_callbacks(model)
    assert model.callbacks["on_predict_postprocess_end"][0] is foreign


def test_the_last_registration_is_the_one_kept():
    first, last = _tracking_cb(persist=False), _tracking_cb(persist=True)
    model = _FakeModel({"on_predict_postprocess_end": [first, last]})
    ultralytics_compat.dedupe_tracking_callbacks(model)
    assert model.callbacks["on_predict_postprocess_end"] == [last]


def test_the_list_is_mutated_in_place():
    # ultralytics hands the same list object to the predictor, so replacing the
    # dict value would leave the predictor calling the old list.
    registered = [_tracking_cb(), _tracking_cb()]
    model = _FakeModel({"on_predict_postprocess_end": registered})
    ultralytics_compat.dedupe_tracking_callbacks(model)
    assert model.callbacks["on_predict_postprocess_end"] is registered
    assert len(registered) == 1


def test_no_callbacks_at_all_is_not_an_error():
    assert ultralytics_compat.dedupe_tracking_callbacks(_FakeModel()) == 0
    assert ultralytics_compat.dedupe_tracking_callbacks(_FakeModel({})) == 0
    assert ultralytics_compat.dedupe_tracking_callbacks(object()) == 0
    assert ultralytics_compat.tracking_callback_counts(_FakeModel()) == {
        "on_predict_start": 0, "on_predict_postprocess_end": 0}


def test_dedupe_is_idempotent():
    model = _FakeModel({"on_predict_postprocess_end": [_tracking_cb(), _tracking_cb()]})
    assert ultralytics_compat.dedupe_tracking_callbacks(model) == 1
    assert ultralytics_compat.dedupe_tracking_callbacks(model) == 0


def test_the_frame_loop_checks_the_invariant():
    # Structural: the check has to sit in the frame loop, right after the
    # .track() call that does the re-registration. A dedupe that only ran at
    # engine construction, or only at run start, would never see it — and one
    # that only ran on the first frame would miss a predictor nulled mid-run by
    # the previous run's case report.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "engine", "tracker.py"),
        encoding="utf-8").read()
    assert src.count("dedupe_tracking_callbacks(") >= 2, (
        "expected the run-start sweep AND the in-loop check")
    after_track = src.split("results = self.model.track(", 1)[1]
    check_at = after_track.find("dedupe_tracking_callbacks(")
    assert check_at != -1, "no dedupe after the .track() call that appends"
    # Before the loop's next iteration, i.e. inside the same frame's body.
    next_frame = after_track.find("for _frame_no in range")
    assert next_frame == -1 or check_at < next_frame
    assert "_frame_no == 0" not in after_track[:check_at], (
        "the check is gated on the first frame; a mid-run offload re-registers "
        "after it has already passed")


if __name__ == "__main__":
    import ultralytics

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed "
          f"(ultralytics={ultralytics.__version__}; {ultralytics_compat.describe()})")
