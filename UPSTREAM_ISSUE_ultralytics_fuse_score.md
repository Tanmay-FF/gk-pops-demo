# Upstream bug report — draft, not yet filed

Ready to paste into https://github.com/ultralytics/ultralytics/issues/new.
Deliberately not filed automatically; review it, then post it under your own
account.

Local workaround already in this repo: `engine/ultralytics_compat.py`.

---

## Title

`fuse_score` makes ByteTrack's second association mathematically impossible to satisfy

## Description

Since 8.4.40, `BYTETracker.update()` applies `fuse_score` to the **second**
(low-confidence) association. With the shipped default configs this makes that
association unsatisfiable for every possible input, so the low-confidence
recovery pass — the main thing ByteTrack adds over SORT — is silently disabled.

This is not a degradation at the margins. No match can ever pass, at any IoU.

### The arithmetic

`trackers/byte_tracker.py`, in `update()`:

```python
detections_second = self.init_track(results_second, feats_second)
r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
dists = matching.iou_distance(r_tracked_stracks, detections_second)
if self.args.fuse_score:                                  # added in 8.4.40
    dists = matching.fuse_score(dists, detections_second)
matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
```

`matching.fuse_score` returns `cost = 1 - IoU * score`, and this call site gates
at `cost <= 0.5`, so a match requires `IoU * score >= 0.5`.

But `detections_second` is by construction the pool
`track_low_thresh <= score < track_high_thresh`. Its scores are therefore capped
below `track_high_thresh`. With the shipped defaults in
`cfg/trackers/botsort.yaml` and `cfg/trackers/bytetrack.yaml`
(`track_high_thresh: 0.25`, `fuse_score: True`), the best achievable cost even at
a perfect `IoU = 1.0` is:

```
cost = 1 - (1.0 * 0.25) = 0.75  >  0.5
```

The two rules contradict each other. The second pass exists specifically to
rescue low-confidence detections, and `fuse_score` then penalises those
detections precisely for being low-confidence.

Because the gate is 0.5 and the pool is bounded by `track_high_thresh`, the pass
can only ever match when `track_high_thresh > 0.5` **and** IoU is near-perfect,
which defeats the purpose of a recovery pass. Raising `track_high_thresh` is not
a viable workaround.

### Reproduction

No weights, no video, no GPU. Synthetic detections driven straight through
`BOTSORT.update()`, using ultralytics' own default thresholds:

```python
"""Minimal repro: BYTETracker's second association cannot match anything when
fuse_score is enabled.

    pip install ultralytics && python repro.py
"""
import numpy as np
import ultralytics
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching


class Args:
    tracker_type = "botsort"
    track_high_thresh = 0.25      # ultralytics default (cfg/trackers/botsort.yaml)
    track_low_thresh = 0.1
    new_track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    fuse_score = True             # ultralytics default
    gmc_method = "none"
    proximity_thresh = 0.5
    appearance_thresh = 0.25
    with_reid = False


class Dets:
    """Stands in for Results.boxes: .conf/.cls/.xywh, len(), mask indexing."""
    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return Dets(self.xywh[m], self.conf[m], self.cls[m])


def box(step, conf):
    # One object drifting 4 px/frame, so consecutive frames overlap heavily.
    return Dets([[300.0 + 4 * step, 400.0, 80.0, 200.0]], [conf], [0.0])


# Count what the second association is actually asked to do.
stats = {"calls": 0, "pairs": 0, "matches": 0, "min_cost": float("inf")}
_real = matching.linear_assignment


def spy(cost, thresh, use_lap=True):
    out = _real(cost, thresh, use_lap)
    if abs(thresh - 0.5) < 1e-9:          # the second-association call site
        stats["calls"] += 1
        stats["pairs"] += int(cost.size)
        stats["matches"] += int(len(out[0]))
        if cost.size:
            stats["min_cost"] = min(stats["min_cost"], float(cost.min()))
    return out


import ultralytics.trackers.byte_tracker as bt
bt.matching.linear_assignment = spy

trk = BOTSORT(args=Args(), frame_rate=30)
seen = []
# 6 confident frames to establish a track, then 4 in the low pool.
for step, conf in enumerate([0.9] * 6 + [0.2] * 4):
    rows = trk.update(box(step, conf))
    seen.append([int(r[4]) for r in rows] if len(rows) else [])

print(f"ultralytics {ultralytics.__version__}")
print(f"track IDs per frame: {seen}")
print(f"second association: {stats['matches']} matches over {stats['pairs']} "
      f"candidate pairs in {stats['calls']} calls; min cost seen "
      f"{stats['min_cost']:.3f} against a 0.5 gate")
```

### Actual (8.4.40)

```
ultralytics 8.4.40
track IDs per frame: [[1], [1], [1], [1], [1], [1], [], [], [], []]
second association: 0 matches over 1 candidate pairs in 10 calls; min cost seen 0.805 against a 0.5 gate
```

The detection at 0.2 overlaps the previous box almost exactly, and the track is
still dropped. Note the minimum cost ever observed, 0.805, never comes within
0.3 of the 0.5 gate.

### Expected (8.4.19, same script)

```
ultralytics 8.4.19
track IDs per frame: [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
second association: 4 matches over 4 candidate pairs in 10 calls; min cost seen 0.013 against a 0.5 gate
```

### Real-world impact

Measured on a retail CCTV clip (415 frames, 1280x720, YOLO detector at
`imgsz=640`), comparing 8.4.19 against 8.4.40 with everything else held
constant — same weights, same video, byte-identical decoded frames and
letterboxed input tensors, detector scores agreeing to the fourth decimal:

| | person tracked on | linked frames |
|---|---|---|
| 8.4.19 | 368 / 415 | 351 |
| 8.4.40 | 286 / 415 | 351 |

84 frames lose the person track, including a 74-frame unbroken run covering the
subject's entire walk out of frame — exactly the partial-occlusion case the
second association exists to handle.

On a second clip, 268 of 420 frames (63.8%) had a different set of track IDs
between the two versions. Aggregate counters hid it completely: both versions
reported 16 distinct IDs and a 103-frame median track life. Only a per-frame
ID-set comparison showed the divergence.

### Note on `fuse_score: false`

Turning the option off is not a workaround, because it is not per-pass: it also
removes score fusion from the **first** association, which changes association
behaviour that existing configs are tuned around. On the clip above that took
linked frames from 351 down to 154.

### Suggested fix

Do not fuse detection score into the second association — that is, restore the
pre-8.4.40 behaviour at that call site:

```python
dists = matching.iou_distance(r_tracked_stracks, detections_second)
matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
```

If score fusion there is wanted deliberately, it needs its own gate derived from
`track_high_thresh` rather than the fixed 0.5, plus a separate config key so it
can be controlled independently of the first association.

### Environment

- ultralytics 8.4.40 (bug) and 8.4.19 (correct)
- Python 3.12.9, Windows 11
- Reproduces without torch CUDA, weights, or video — pure `numpy` + the tracker
