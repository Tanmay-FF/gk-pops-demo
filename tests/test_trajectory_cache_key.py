"""The trajectory cache key must cover the environment, not just the video.

L2 lives in a shared %TEMP%/pops_traj_cache directory, so every interpreter on
the machine hits the same files. Keying on the video alone meant a venv run and
a conda run traded trajectories — which is precisely how a real cross-version
difference (see engine/ultralytics_compat.py) could be masked instead of seen.

Spurious misses cost a re-run. Spurious hits corrupt the answer. These tests
pin that trade-off in the safe direction.

Runs anywhere — no GPU, no weights, no video.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import trajectory_cache as tc
from engine.analytics_models import TrajectoryBundle


def _tmp_video(name="clip.mp4", body=b"not really a video"):
    path = os.path.join(tempfile.mkdtemp(prefix="pops_key_test_"), name)
    with open(path, "wb") as f:
        f.write(body)
    return path


def _bundle(key):
    """TrajectoryBundle has several required fields; only video_key matters here."""
    return TrajectoryBundle(video_key=key, video_path="", width=1, height=1,
                            fps=30.0, total_frames=1)


def _with_fingerprint(value, fn):
    """Run fn() with the module's cached fingerprint forced to `value`."""
    saved = tc._ENV_FINGERPRINT
    tc._ENV_FINGERPRINT = value
    try:
        return fn()
    finally:
        tc._ENV_FINGERPRINT = saved


def test_fingerprint_names_the_things_that_change_trajectories():
    fp = tc.environment_fingerprint(refresh=True)
    for field in ("ultralytics=", "tracker_cfg=", "weights="):
        assert field in fp, f"fingerprint is missing {field!r}: {fp}"


def test_fingerprint_is_cached_between_calls():
    first = tc.environment_fingerprint(refresh=True)
    assert tc.environment_fingerprint() == first


def test_key_changes_when_the_environment_changes():
    """The actual bug: two interpreters, same video, same key."""
    video = _tmp_video()
    a = _with_fingerprint("ultralytics=8.4.19|x", lambda: tc.make_video_key(video))
    b = _with_fingerprint("ultralytics=8.4.40|x", lambda: tc.make_video_key(video))
    assert a != b, (
        "two ultralytics versions produced the same cache key — one env's "
        "trajectories will be served to the other"
    )


def test_key_is_stable_for_the_same_video_and_environment():
    """Stability matters as much as separation: an unstable key would make the
    cache useless and silently re-run detection on every zone redraw."""
    video = _tmp_video()
    fn = lambda: tc.make_video_key(video)  # noqa: E731
    assert _with_fingerprint("fixed", fn) == _with_fingerprint("fixed", fn)


def test_key_changes_when_the_video_changes():
    """The original contract must survive: different content -> different key."""
    a = _tmp_video(body=b"clip one")
    b = _tmp_video(body=b"clip two, and a different length")
    fn_a = lambda: tc.make_video_key(a)  # noqa: E731
    fn_b = lambda: tc.make_video_key(b)  # noqa: E731
    assert _with_fingerprint("fixed", fn_a) != _with_fingerprint("fixed", fn_b)


def test_missing_video_still_yields_a_key():
    """The engine calls this before it has checked the path exists. It must not
    raise — a crash here would take the whole run down."""
    key = tc.make_video_key(os.path.join(tempfile.gettempdir(), "no_such_clip.mp4"))
    assert isinstance(key, str) and len(key) == 16


def test_fingerprint_survives_unreadable_config_and_weights():
    """Same reasoning: a fingerprint that throws is worse than a wide one.
    A missing tracker yaml or missing weights must degrade, not raise."""
    from engine import config

    saved_fp = tc._ENV_FINGERPRINT
    saved_model, saved_tracker = config.MODEL_PATH, config.TRACKER_CONFIG
    config.MODEL_PATH = os.path.join(tempfile.gettempdir(), "no_such_weights.pt")
    config.TRACKER_CONFIG = os.path.join(tempfile.gettempdir(), "no_such_tracker.yaml")
    try:
        fp = tc.environment_fingerprint(refresh=True)
        assert isinstance(fp, str) and fp, "fingerprint collapsed to nothing"
        assert "missing" in fp, f"missing weights should be visible in {fp}"
        assert "unreadable" in fp, f"missing tracker cfg should be visible in {fp}"
        # and it must still produce a usable key
        assert len(tc.make_video_key(_tmp_video())) == 16
    finally:
        config.MODEL_PATH, config.TRACKER_CONFIG = saved_model, saved_tracker
        tc._ENV_FINGERPRINT = saved_fp


def test_entries_from_a_foreign_environment_are_simply_not_found():
    """Old pickles do not need deleting — they just stop being addressable."""
    cache_dir = tempfile.mkdtemp(prefix="pops_key_cache_")
    cache = tc.TrajectoryCache(cache_dir=cache_dir)
    video = _tmp_video()

    old_key = _with_fingerprint("ultralytics=8.4.19|x", lambda: tc.make_video_key(video))
    cache.put(_bundle(old_key))

    new_key = _with_fingerprint("ultralytics=8.4.40|x", lambda: tc.make_video_key(video))
    cache._mem.clear()                      # force the L2 path
    assert cache.get(new_key) is None, (
        "a bundle written under a different environment was served anyway"
    )
    cache._mem.clear()
    assert cache.get(old_key) is not None, "the original entry should still resolve"


def test_clear_still_reaps_entries_from_every_environment():
    cache_dir = tempfile.mkdtemp(prefix="pops_key_cache_")
    cache = tc.TrajectoryCache(cache_dir=cache_dir)
    cache.put(_bundle("aaaaaaaaaaaaaaaa"))
    cache.put(_bundle("bbbbbbbbbbbbbbbb"))
    cache.clear()
    leftover = list(cache.cache_dir.glob("*.pkl"))
    assert not leftover, f"clear() left {leftover} behind"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")
    print(f"fingerprint: {tc.environment_fingerprint(refresh=True)}")
