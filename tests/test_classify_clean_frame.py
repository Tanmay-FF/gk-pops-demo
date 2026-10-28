"""The cart classifiers must see the decoded frame, not the annotated one.

engine/tracker.py draws box outlines, label banners and centroid trails onto
the frame it writes to the output video, and those drawings used to land in
every cart crop the quality / fill / bag models classified. Measured on
held-out clips, that alone made the quality model reject 49% of crops instead
of 29%, and made the fill model read empty carts "partial".

This runs the real engine on the primary golden clip, keeps an untouched copy
of every decoded frame, and requires every frame handed to the classifier to
be pixel-identical to it. A crop of the painted frame always differs (each
cart's own box outline sits on its crop border), so any regression fails here.

Run on the original 2-class detector and on every 3-class one, whose
NON-EMPTY reads reach the classifier through classify_loaded() instead.

Needs the detection weights and the golden clip; skips otherwise (a 3-class
run also needs CUDA and that backend's weights).
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from test_golden_clip import Skip, load_baseline, find_clip, run_pipeline  # noqa: E402
from test_three_class_e2e import _require_environment  # noqa: E402


def _check_classifiers_see_decoded_frames(settings):
    """Run the engine with `settings`; return {entry point: number of calls}.
    Fails if any call got a frame other than the one cap.read() decoded."""
    clip = find_clip(load_baseline())

    import engine.tracker as T
    from engine.classifier import CartClassifier

    decoded = {"frame": None}
    real_open = T.open_video

    class KeepDecoded:
        """Wraps the engine's capture and remembers each frame as decoded."""

        def __init__(self, cap):
            self._cap = cap

        def read(self):
            ok, im = self._cap.read()
            decoded["frame"] = im.copy() if ok else None
            return ok, im

        def __getattr__(self, name):
            return getattr(self._cap, name)

    def open_video_spy(*a, **k):
        cap, *rest = real_open(*a, **k)
        return (KeepDecoded(cap), *rest)

    calls, mismatches = {}, []
    # Every classifier entry point the engine may use on this branch.
    entry_points = [n for n in ("classify", "classify_loaded") if hasattr(CartClassifier, n)]
    originals = {n: getattr(CartClassifier, n) for n in entry_points}

    def make_spy(name):
        real = originals[name]

        def spy(self, frame, bbox, *a, **k):
            calls[name] = calls.get(name, 0) + 1
            ref = decoded["frame"]
            if ref is None or frame.shape != ref.shape or not np.array_equal(frame, ref):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                diff = 0 if ref is None else int((frame[y1:y2, x1:x2] != ref[y1:y2, x1:x2]).any(axis=2).sum())
                mismatches.append(f"{name} got a frame that differs from the decoded one "
                                  f"({diff} changed pixels inside box {[x1, y1, x2, y2]})")
            return real(self, frame, bbox, *a, **k)
        return spy

    T.open_video = open_video_spy
    for n in entry_points:
        setattr(CartClassifier, n, make_spy(n))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(clip, settings)
    finally:
        T.open_video = real_open
        for n, f in originals.items():
            setattr(CartClassifier, n, f)

    total = sum(calls.values())
    assert total > 0, "the classifier was never called on the golden clip"
    assert not mismatches, (f"{len(mismatches)} of {total} classifier calls saw an annotated "
                            f"frame:\n  " + "\n  ".join(mismatches[:5]))
    return calls


def test_classifiers_get_the_unannotated_frame():
    calls = _check_classifiers_see_decoded_frames(load_baseline()["settings"])
    print(f"  {sum(calls.values())} classifier calls, every one on the decoded frame")


def test_three_class_reads_get_the_unannotated_frame():
    from engine import config as C

    ran, skipped = [], []
    for backend in C.THREE_CLASS_BACKENDS:
        try:
            _, settings = _require_environment(backend)
        except Skip as why:
            skipped.append(f"{backend}: {why}")
            continue
        calls = _check_classifiers_see_decoded_frames(settings)
        # The golden clip's pushout cart is loaded: if no read reached
        # classify_loaded(), this run proved nothing about that path.
        assert calls.get("classify_loaded", 0) > 0, f"{backend}: no NON-EMPTY read: {calls}"
        ran.append(f"{backend} {calls}")
    if not ran:
        raise Skip("; ".join(skipped))
    print(f"  every call on the decoded frame: {ran}" + (f"; skipped: {skipped}" if skipped else ""))


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
