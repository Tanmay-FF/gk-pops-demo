# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Per-cart empty / non-empty decision from the 3-class detector.

The 3-class backend reports P(non-empty) for every cart box on every frame
(engine/detection_backends.py). A classification READ happens every
CLASSIFY_EVERY_N_FRAMES; at each read the cart's state is the MEAN of its
P(non-empty) over every frame since its previous read, so every frame's
detection contributes, not just the read frame's.

    EMPTY      -> the read is recorded as ("empty", "not_applicable") straight
                  from the detector; quality / fill / bag models are not run.
    NON-EMPTY  -> the crop goes through the quality gate and a fill model that
                  can only answer a loaded class (CartClassifier.classify_loaded).
    UNKNOWN    -> no usable detector value (should not happen for a cart the
                  3-class backend detected); the caller falls back to the
                  original classify() so a read is never silently lost.

Pure, model-free code: unit-tested in tests/test_cart_state.py.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Hashable, Iterable

EMPTY = "empty"
NON_EMPTY = "non_empty"
UNKNOWN = "unknown"

#: Read outcomes counted per cart for the tracking JSON (pops_summary).
READ_EMPTY = "empty"
READ_NON_EMPTY = "non_empty"
READ_QUALITY_FAIL = "non_empty_quality_fail"
READ_NO_STATE = "no_detector_state"

#: Upper bound on per-frame samples held for one cart between two reads. Reads
#: drain the buffer every CLASSIFY_EVERY_N_FRAMES frames, so this only bounds
#: memory when reads stop (no quality model loaded); the newest samples win.
MAX_SAMPLES_PER_CART = 512


@dataclass(frozen=True)
class CartStateRead:
    state: str            # EMPTY | NON_EMPTY | UNKNOWN
    p_non_empty: float    # mean over the samples used; NaN when UNKNOWN
    n_samples: int


def _finite_probability(value) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    return min(1.0, max(0.0, p))


def usable_sample(p_non_empty, det_conf, min_conf: float) -> float | None:
    """One tracked cart box's vote for its next read, or None if it must not vote.

    The tracker is fed boxes down to a low confidence (as Model.track() does)
    so tracks survive weak frames, but only a box detected at `min_conf` or
    above may say empty / non-empty: a barely-detected cart's class guess is
    noise. None also for an unusable P(non-empty) or confidence.
    """
    p = _finite_probability(p_non_empty)
    try:
        c = float(det_conf)
    except (TypeError, ValueError):
        return None
    if p is None or not math.isfinite(c) or c < float(min_conf):
        return None
    return p


def decide_cart_state(samples: Iterable, threshold: float) -> CartStateRead:
    """Mean P(non-empty) over `samples` against `threshold`.

    Non-numeric, None and NaN samples are skipped (a person box's NaN must
    never reach a cart's mean); values are clamped to [0, 1]. No usable
    sample -> UNKNOWN.
    """
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    vals = [p for p in (_finite_probability(s) for s in samples) if p is not None]
    if not vals:
        return CartStateRead(UNKNOWN, float("nan"), 0)
    p = sum(vals) / len(vals)
    return CartStateRead(NON_EMPTY if p >= threshold else EMPTY, p, len(vals))


def detector_empty_result(p_non_empty: float) -> dict:
    """The classification record for a read the detector called EMPTY.

    Shaped like CartClassifier.classify()'s output so every consumer (vote,
    frame JSON, POPS kill switch) reads it unchanged. `quality` is
    valid_cart because the cart WAS read — by the detector; `quality_conf`
    is 0.0 because the quality model did not run (`source` says so).
    """
    p = _finite_probability(p_non_empty)
    if p is None:
        raise ValueError(f"p_non_empty must be a finite number, got {p_non_empty!r}")
    return {
        "is_valid": True, "quality": "valid_cart",
        "fill": "empty", "bag": "not_applicable",
        "quality_conf": 0.0, "fill_conf": 1.0 - p, "bag_conf": 1.0,
        "source": "detector", "detector_state": EMPTY, "p_non_empty": p,
    }


class CartStateAccumulator:
    """Per-cart P(non-empty) samples between reads, and per-cart read counts."""

    def __init__(self, max_samples: int = MAX_SAMPLES_PER_CART):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._max = max_samples
        self._samples: dict[Hashable, deque] = {}
        self._reads: dict[Hashable, Counter] = {}

    def add(self, cart_id: Hashable, p_non_empty) -> None:
        """Record one frame's value. Unusable values are dropped here, so a
        buffer never holds anything decide_cart_state() would skip."""
        p = _finite_probability(p_non_empty)
        if p is None:
            return
        buf = self._samples.get(cart_id)
        if buf is None:
            buf = self._samples[cart_id] = deque(maxlen=self._max)
        buf.append(p)

    def take(self, cart_id: Hashable) -> list[float]:
        """Drain and return the cart's samples since its previous read."""
        buf = self._samples.pop(cart_id, None)
        return list(buf) if buf else []

    def count_read(self, cart_id: Hashable, outcome: str) -> None:
        self._reads.setdefault(cart_id, Counter())[outcome] += 1

    def read_counts(self, cart_id: Hashable) -> dict[str, int]:
        return dict(self._reads.get(cart_id, {}))
