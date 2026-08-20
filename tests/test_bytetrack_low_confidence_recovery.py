"""ByteTrack must keep a track alive through a low-confidence dip.

This is the behavioural tripwire for the whole class of bug found in
engine/ultralytics_compat.py: it asserts the OUTCOME (a track survives when its
detection drops into the low pool), not any particular implementation. It does
not care how the fix is injected, which upstream version is installed, or
whether the fuse_score call site still looks the way it did — so an upstream
refactor that silently defeats the fix fails HERE, loudly, instead of quietly
shifting POPS numbers on the next run.

Why this matters: a person walking out behind their cart is partly occluded, so
the detector's confidence dips. If the tracker cannot re-match a low-confidence
detection, the person track dies mid-exit, the person/cart link breaks, and a
normal customer leaving is scored as an UNLINKED EXIT. Measured on
1764092528600_B8A44F40EFB5-medium-OUTSIDE.mp4 that cost the person track on 84
of 415 frames, including 74 unbroken frames covering the entire exit.

No video, no weights, no GPU — synthetic detections driven straight through
BOTSORT.update(). Runs anywhere, in about a second.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Importing the engine installs whatever tracker fix is currently in force.
# Deliberately imported for its side effect: these tests must exercise the
# tracker exactly as a real run configures it, not a hand-built variant.
import engine.tracker  # noqa: F401

from ultralytics.trackers import bot_sort  # noqa: F401  (reloaded by the negative control)
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.trackers.utils import matching


def _tracker_cls():
    """The tracker class a real run actually gets — resolved through
    ultralytics' registry, exactly as trackers/track.py:on_predict_start does,
    so these tests follow the fix wherever it is injected."""
    return TRACKER_MAP["botsort"]


#: Index of track_id inside STrack.result -> [x1, y1, x2, y2, track_id, ...]
_TRACK_ID = 4

#: Mirrors engine/botsort_retail.yaml. The second (low-confidence) pool is
#: everything scoring above LOW and below HIGH.
HIGH = 0.3
LOW = 0.1
NEW = 0.4


class Args:
    """The merged tracker config, as ultralytics hands it to BOTSORT."""

    def __init__(self, **kw):
        self.tracker_type = "botsort"
        self.track_high_thresh = HIGH
        self.track_low_thresh = LOW
        self.new_track_thresh = NEW
        self.track_buffer = 120
        self.match_thresh = 0.7
        self.fuse_score = True
        # "none" keeps camera-motion compensation out of the picture: it needs
        # a real image and would make this test depend on optical flow.
        self.gmc_method = "none"
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.with_reid = False
        self.__dict__.update(kw)


class Dets:
    """Duck-types the ultralytics Boxes object BYTETracker.update consumes:
    `.conf`, `.cls`, `.xywh`, `len()`, and boolean-mask indexing."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Dets(self.xywh[mask], self.conf[mask], self.cls[mask])


def _one_box(step, conf):
    """A single box drifting 4 px right per frame — slow enough that
    consecutive frames overlap heavily, like a person walking on CCTV."""
    return Dets([[300.0 + 4 * step, 400.0, 80.0, 200.0]], [conf], [0.0])


def _ids(rows):
    return [int(r[_TRACK_ID]) for r in rows]


def _run(confidences, args=None):
    """Drive one synthetic object through the tracker, one confidence per
    frame. Returns the list of track IDs reported on each frame."""
    trk = _tracker_cls()(args=args or Args(), frame_rate=30)
    seen = []
    for step, conf in enumerate(confidences):
        rows = trk.update(_one_box(step, conf))
        seen.append(_ids(rows) if len(rows) else [])
    return seen


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------

def test_track_survives_a_low_confidence_dip():
    """Six confident frames, then four in the low pool. The ID must persist.

    This is the assertion that fails if ByteTrack's second association is ever
    disabled again."""
    confident = [0.9] * 6
    dip = [0.2] * 4                     # LOW < 0.2 < HIGH: second pool only
    seen = _run(confident + dip)

    warm = seen[5]
    assert warm, "track never activated on confident detections"
    tid = warm[0]

    for i, frame in enumerate(seen[6:], start=6):
        assert frame == [tid], (
            f"frame {i}: expected the track to survive the low-confidence dip "
            f"as ID {tid}, got {frame}. ByteTrack's second association is not "
            f"recovering low-confidence detections — see "
            f"engine/ultralytics_compat.py."
        )


def test_low_confidence_alone_never_starts_a_track():
    """The other half of the contract. Recovering low-confidence detections
    must not also mean trusting them to create tracks, or every flicker of
    detector noise becomes a phantom cart."""
    seen = _run([0.2] * 8)
    assert all(f == [] for f in seen), (
        f"low-confidence detections started a track on their own: {seen}"
    )


def test_below_track_low_thresh_is_discarded_entirely():
    """Detections under track_low_thresh are not even in the second pool."""
    seen = _run([0.9] * 6 + [0.05] * 4)
    assert seen[5], "track never activated"
    assert all(f == [] for f in seen[6:]), (
        f"a detection below track_low_thresh ({LOW}) was still tracked: {seen[6:]}"
    )


def test_dip_then_recover_keeps_the_same_id():
    """The end-to-end shape of the real failure: confident, occluded, confident
    again. A tracker that drops the track during the dip hands back a NEW id on
    recovery — which is exactly what breaks the person/cart link."""
    seen = _run([0.9] * 6 + [0.2] * 5 + [0.9] * 4)
    before = seen[5]
    after = seen[-1]
    assert before and after, f"track missing entirely: {seen}"
    assert before[0] == after[0], (
        f"ID changed across the occlusion: {before[0]} -> {after[0]}. The "
        f"person/cart link cannot survive an ID change."
    )


def test_a_long_dip_still_recovers_within_track_buffer():
    """track_buffer is 120 frames in botsort_retail.yaml — deliberately high
    because carts sit still behind people. A 30-frame dip is well inside it."""
    seen = _run([0.9] * 6 + [0.2] * 30 + [0.9] * 2)
    assert seen[5] and seen[-1], f"track missing entirely: {seen}"
    assert seen[5][0] == seen[-1][0], (
        f"ID changed across a 30-frame dip: {seen[5][0]} -> {seen[-1][0]}"
    )


def test_fuse_score_still_applies_to_the_first_association():
    """The fix must not be 'turn fusion off everywhere'. Score fusion in the
    FIRST association is what the thresholds in botsort_retail.yaml were tuned
    against; dropping it cost 197 of 351 linked frames when measured."""
    trk = _tracker_cls()(args=Args(fuse_score=True), frame_rate=30)
    trk.update(_one_box(0, 0.9))        # one confident track to match against

    seen = {}
    real = matching.fuse_score

    def spy(cost, dets):
        seen["called"] = True
        return real(cost, dets)

    matching.fuse_score = spy
    try:
        trk.update(_one_box(1, 0.9))    # confident -> first association
    finally:
        matching.fuse_score = real

    assert seen.get("called"), (
        "the first association no longer fuses detection score — the tuned "
        "thresholds in engine/botsort_retail.yaml no longer mean what they did"
    )


def test_fuse_score_never_applies_to_the_second_association():
    """The exact regression, asserted directly: whatever runs for the low pool
    must not multiply cost by detection score, because that pool's scores are
    capped at track_high_thresh and can never clear the 0.5 gate."""
    trk = _tracker_cls()(args=Args(fuse_score=True), frame_rate=30)
    for step in range(6):
        trk.update(_one_box(step, 0.9))

    calls = []
    real = matching.fuse_score

    def spy(cost, dets):
        calls.append([float(d.score) for d in dets])
        return real(cost, dets)

    matching.fuse_score = spy
    try:
        trk.update(_one_box(6, 0.2))    # low pool only
    finally:
        matching.fuse_score = real

    for scores in calls:
        assert not scores or min(scores) >= HIGH, (
            f"fuse_score was applied to detections scoring {scores}, which is "
            f"the low pool. Those can never clear the 0.5 gate — the second "
            f"association is disabled."
        )


def test_yaml_can_still_turn_fusion_off_entirely():
    """A deliberate `fuse_score: false` must still mean no fusion anywhere.
    The fix restores upstream's old split; it must not take the switch away."""
    trk = _tracker_cls()(args=Args(fuse_score=False), frame_rate=30)
    trk.update(_one_box(0, 0.9))

    called = []
    real = matching.fuse_score

    def spy(cost, dets):
        called.append(1)
        return real(cost, dets)

    matching.fuse_score = spy
    try:
        trk.update(_one_box(1, 0.9))
    finally:
        matching.fuse_score = real

    assert not called, "fuse_score: false was overridden by the compat shim"


def test_two_trackers_sharing_one_config_both_behave():
    """ultralytics builds ONE merged config and one tracker per batch element.
    A fix that mutates that shared object must not leave the second tracker
    behaving differently from the first."""
    shared = Args(fuse_score=True)
    cls = _tracker_cls()
    a = cls(args=shared, frame_rate=30)
    b = cls(args=shared, frame_rate=30)
    seen_a, seen_b = [], []
    for step, conf in enumerate([0.9] * 6 + [0.2] * 4):
        seen_a.append(_ids(a.update(_one_box(step, conf))))
        seen_b.append(_ids(b.update(_one_box(step, conf))))
    assert seen_a[-1], f"first tracker lost the track: {seen_a}"
    assert seen_b[-1], f"second tracker lost the track: {seen_b}"


def test_plain_bytetrack_gets_the_same_guarantee():
    """botsort_retail.yaml selects botsort, so the bytetrack path is not what
    this project runs — but the registry patches both, and a fix that only
    half-applied would be a trap for whoever switches tracker_type later."""
    trk = TRACKER_MAP["bytetrack"](args=Args(tracker_type="bytetrack"), frame_rate=30)
    seen = []
    for step, conf in enumerate([0.9] * 6 + [0.2] * 4):
        rows = trk.update(_one_box(step, conf))
        seen.append(_ids(rows) if len(rows) else [])
    assert seen[5], "track never activated on confident detections"
    assert seen[-1] == seen[5], (
        f"plain ByteTrack lost the track across the low-confidence dip: {seen}"
    )


if __name__ == "__main__":
    import ultralytics

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed (ultralytics={ultralytics.__version__})")
