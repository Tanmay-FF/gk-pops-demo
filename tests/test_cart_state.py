"""The 3-class detector's per-cart empty / non-empty read (engine/cart_state.py)
and how TrackingEngine routes a read through it.

Pins: the mean-since-last-read rule and its threshold edge, that unusable
samples (None / NaN / non-numeric / a person's NaN) never reach a cart's
mean, that an EMPTY read never runs a classifier, that a NON-EMPTY read only
ever reaches classify_loaded(), that an unreadable NON-EMPTY crop is NOT
recorded (the user's choice, docs/three_class_detector_plan.md Q2), and that
a read with no detector evidence falls back to classify() instead of being
lost.

Runs anywhere — no GPU, no weights, no video.
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import cart_state as CS  # noqa: E402


def test_mean_since_last_read_decides():
    r = CS.decide_cart_state([0.9, 0.8, 0.1], threshold=0.5)
    assert r.state == CS.NON_EMPTY and r.n_samples == 3
    assert math.isclose(r.p_non_empty, 0.6)
    assert CS.decide_cart_state([0.1, 0.2], 0.5).state == CS.EMPTY


def test_threshold_is_inclusive():
    assert CS.decide_cart_state([0.5], 0.5).state == CS.NON_EMPTY
    assert CS.decide_cart_state([0.4999], 0.5).state == CS.EMPTY


def test_unusable_samples_are_skipped_not_averaged():
    r = CS.decide_cart_state([None, float("nan"), "x", float("inf"), 0.9], 0.5)
    assert r.n_samples == 1 and r.state == CS.NON_EMPTY


def test_values_are_clamped_to_probability_range():
    r = CS.decide_cart_state([1.7, -0.3], 0.5)
    assert math.isclose(r.p_non_empty, 0.5)


def test_no_usable_sample_is_unknown():
    for samples in ([], [None], [float("nan")]):
        r = CS.decide_cart_state(samples, 0.5)
        assert r.state == CS.UNKNOWN and r.n_samples == 0 and math.isnan(r.p_non_empty)


def test_bad_threshold_is_refused():
    for t in (0.0, -1, 1.5):
        try:
            CS.decide_cart_state([0.5], t)
        except ValueError:
            continue
        raise AssertionError(f"threshold {t} accepted")


def test_only_confident_boxes_vote():
    assert CS.usable_sample(0.9, 0.30, 0.25) == 0.9
    assert CS.usable_sample(0.9, 0.25, 0.25) == 0.9, "the threshold is inclusive"
    assert CS.usable_sample(0.9, 0.12, 0.25) is None, "a weak box tracked but must not vote"
    for p, c in ((float("nan"), 0.9), (0.9, float("nan")), (None, 0.9), (0.9, None), ("x", 0.9)):
        assert CS.usable_sample(p, c, 0.25) is None, (p, c)
    assert CS.usable_sample(1.4, 0.9, 0.25) == 1.0, "probabilities are clamped"


def test_detector_empty_result_is_shaped_like_a_classifier_result():
    r = CS.detector_empty_result(0.2)
    for key in ("is_valid", "quality", "fill", "bag", "quality_conf", "fill_conf", "bag_conf"):
        assert key in r, key
    assert r["fill"] == "empty" and r["bag"] == "not_applicable"
    assert r["quality"] == "valid_cart" and r["is_valid"] is True
    assert math.isclose(r["fill_conf"], 0.8) and r["source"] == "detector"
    try:
        CS.detector_empty_result(float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN accepted")


def test_accumulator_drains_per_read_and_bounds_memory():
    acc = CS.CartStateAccumulator(max_samples=3)
    for p in (0.1, 0.2, 0.3, 0.4, float("nan"), None):
        acc.add(7, p)
    assert acc.take(7) == [0.2, 0.3, 0.4], "newest samples must win, unusable dropped"
    assert acc.take(7) == [], "a read must drain the buffer"
    acc.count_read(7, CS.READ_EMPTY)
    acc.count_read(7, CS.READ_EMPTY)
    assert acc.read_counts(7) == {CS.READ_EMPTY: 2}
    assert acc.read_counts(99) == {}


# ---------------------------------------------------------------------------
# Routing in TrackingEngine._classify_cart_read
# ---------------------------------------------------------------------------
class FakeClassifier:
    def __init__(self, loaded_result):
        self.calls = []
        self._loaded = loaded_result

    def classify(self, frame, bbox):
        self.calls.append("classify")
        return {"is_valid": True, "quality": "valid_cart", "fill": "full", "bag": "bagged",
                "quality_conf": 0.9, "fill_conf": 0.9, "bag_conf": 0.9}

    def classify_loaded(self, frame, bbox):
        self.calls.append("classify_loaded")
        return dict(self._loaded)


VALID_LOADED = {"is_valid": True, "quality": "valid_cart", "fill": "partial", "bag": "unbagged",
                "quality_conf": 0.8, "fill_conf": 0.7, "bag_conf": 0.6}
UNREADABLE = {"is_valid": False, "quality": "unclear", "fill": "non-applicable",
              "bag": "non-applicable", "quality_conf": 0.9, "fill_conf": 0.0, "bag_conf": 0.0}


def _engine(loaded_result=VALID_LOADED, uses_state=True):
    import engine.tracker as T

    fake = types.SimpleNamespace(
        _uses_cart_state=uses_state,
        _cart_state=CS.CartStateAccumulator(),
        _classifier=FakeClassifier(loaded_result),
    )
    return fake, (lambda cd: T.TrackingEngine._classify_cart_read(fake, cd, None, (0, 0, 10, 10)))


def test_two_class_backend_reads_exactly_as_before():
    fake, read = _engine(uses_state=False)
    fake._cart_state.add(1, 0.0)  # must be ignored
    assert read(1)["fill"] == "full"
    assert fake._classifier.calls == ["classify"]


def test_empty_read_runs_no_classifier():
    fake, read = _engine()
    for p in (0.1, 0.2, 0.0):
        fake._cart_state.add(1, p)
    r = read(1)
    assert r["fill"] == "empty" and r["source"] == "detector"
    assert fake._classifier.calls == [], fake._classifier.calls
    assert fake._cart_state.read_counts(1) == {CS.READ_EMPTY: 1}


def test_non_empty_read_only_reaches_classify_loaded():
    fake, read = _engine()
    fake._cart_state.add(1, 0.9)
    r = read(1)
    assert fake._classifier.calls == ["classify_loaded"]
    assert r["fill"] == "partial" and r["detector_state"] == CS.NON_EMPTY
    assert math.isclose(r["p_non_empty"], 0.9)
    assert fake._cart_state.read_counts(1) == {CS.READ_NON_EMPTY: 1}


def test_unreadable_non_empty_read_is_returned_unreadable_and_counted():
    """The caller adds only quality == valid_cart reads to the history, so an
    unreadable result is how 'ignore this read' is expressed."""
    fake, read = _engine(loaded_result=UNREADABLE)
    fake._cart_state.add(1, 0.95)
    r = read(1)
    assert r["quality"] != "valid_cart"
    assert fake._cart_state.read_counts(1) == {CS.READ_QUALITY_FAIL: 1}


def test_read_without_detector_evidence_falls_back_to_classify():
    fake, read = _engine()
    r = read(1)
    assert fake._classifier.calls == ["classify"]
    assert r["detector_state"] == CS.UNKNOWN
    assert fake._cart_state.read_counts(1) == {CS.READ_NO_STATE: 1}


def test_each_read_uses_only_frames_since_the_previous_one():
    fake, read = _engine()
    for p in (0.9, 0.9, 0.9):
        fake._cart_state.add(1, p)
    assert read(1)["detector_state"] == CS.NON_EMPTY
    for p in (0.1, 0.1):
        fake._cart_state.add(1, p)
    assert read(1)["fill"] == "empty", "old non-empty frames leaked into the next read"


def test_returned_results_are_fresh_dicts():
    fake, read = _engine()
    fake._cart_state.add(1, 0.9)
    r = read(1)
    r["fill"] = "mutated"
    fake._cart_state.add(1, 0.9)
    assert read(1)["fill"] == "partial"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")
