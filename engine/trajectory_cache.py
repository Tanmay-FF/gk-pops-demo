# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Two-level trajectory cache so YOLO+BoTSORT runs at most once per (video file,
mtime, size) tuple.  Lets the user re-draw zones and re-aggregate analytics
in <1s without re-running detection.

L1: in-memory LRU on the TrackingEngine instance.
L2: pickle on disk under %TEMP%/pops_traj_cache/.

Pickle is fine here — local-only, single-user demo.

The key also covers the ENVIRONMENT that produced the trajectories, not just
the video. L2 lives in a shared %TEMP% directory, so two interpreters on the
same machine — a venv and a conda env, say — hit the same directory. Keying on
the video alone meant whichever ran first served its trajectories to the other,
which silently masked exactly the kind of cross-environment divergence that
engine/ultralytics_compat.py documents. A spurious MISS only costs a re-run; a
spurious HIT corrupts the answer, so the fingerprint deliberately errs wide.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .analytics_models import TrajectoryBundle


CACHE_DIR_NAME = "pops_traj_cache"
DEFAULT_L1_CAPACITY = 4


#: Cached so the fingerprint is computed once per process, not per call.
_ENV_FINGERPRINT: Optional[str] = None


def _file_stamp(path: str) -> str:
    """`size:mtime_ns` for a file, or a marker when it cannot be read. Never
    raises — a fingerprint that throws would take the whole run with it."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "missing"


def environment_fingerprint(refresh: bool = False) -> str:
    """Everything outside the video file that can change the trajectories.

    Deliberately wide: the tracker library version, the resolved tracker config
    (its CONTENTS, so editing thresholds in place invalidates), and the identity
    of the detection weights. Anything that turns out to matter and is missing
    here shows up as a stale cache hit, so prefer over-invalidating.
    """
    global _ENV_FINGERPRINT
    if _ENV_FINGERPRINT is not None and not refresh:
        return _ENV_FINGERPRINT

    try:
        import ultralytics
        version = ultralytics.__version__
    except Exception:
        version = "unknown"

    # Imported lazily: config pulls in torchvision, and this module is imported
    # by tooling that has no reason to pay for that.
    try:
        from .config import MODEL_PATH, TRACKER_CONFIG
    except Exception:
        MODEL_PATH = TRACKER_CONFIG = ""

    try:
        with open(TRACKER_CONFIG, "rb") as f:
            tracker_cfg = hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        tracker_cfg = f"unreadable:{TRACKER_CONFIG}"

    parts = (
        f"ultralytics={version}",
        f"tracker_cfg={tracker_cfg}",
        f"weights={_file_stamp(MODEL_PATH)}",
    )
    _ENV_FINGERPRINT = "|".join(parts)
    return _ENV_FINGERPRINT


def make_video_key(path: str) -> str:
    """sha1(abs_path + mtime_ns + size + environment fingerprint).
    16 hex chars → cheap collision-safe key.

    Changing the ultralytics version, the tracker yaml or the detection weights
    changes the key, so cached trajectories are never reused across a change
    that would have produced different ones. Entries keyed under a previous
    fingerprint simply stop being found; `clear()` still reaps them.
    """
    p = os.path.abspath(path)
    try:
        st = os.stat(p)
        payload = f"{p}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        payload = p
    payload = f"{payload}|{environment_fingerprint()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class TrajectoryCache:
    """LRU in-memory cache backed by a pickle directory."""

    def __init__(self, capacity: int = DEFAULT_L1_CAPACITY,
                 cache_dir: Optional[str] = None):
        self.capacity = capacity
        self._mem: OrderedDict[str, TrajectoryBundle] = OrderedDict()
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), CACHE_DIR_NAME)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, video_key: str) -> Optional[TrajectoryBundle]:
        if video_key in self._mem:
            self._mem.move_to_end(video_key)
            print(f"[CACHE] hit (L1) key={video_key}")
            return self._mem[video_key]

        disk = self._disk_path(video_key)
        if disk.exists():
            try:
                with open(disk, "rb") as f:
                    bundle = pickle.load(f)
                if not isinstance(bundle, TrajectoryBundle):
                    raise TypeError(f"pickle is not a TrajectoryBundle: {type(bundle)}")
                self._promote(video_key, bundle)
                print(f"[CACHE] hit (L2) key={video_key} path={disk}")
                return bundle
            except (pickle.UnpicklingError, AttributeError, ModuleNotFoundError,
                    EOFError, TypeError) as e:
                print(f"[CACHE] stale pickle for {video_key} ({e}); invalidating")
                self.invalidate(video_key)
        print(f"[CACHE] miss key={video_key}")
        return None

    def put(self, bundle: TrajectoryBundle) -> None:
        key = bundle.video_key
        self._promote(key, bundle)
        try:
            with open(self._disk_path(key), "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[CACHE] wrote L2 key={key}")
        except OSError as e:
            print(f"[CACHE] L2 write failed key={key}: {e} (L1 still warm)")

    def invalidate(self, video_key: str) -> None:
        self._mem.pop(video_key, None)
        try:
            self._disk_path(video_key).unlink()
        except FileNotFoundError:
            pass

    def clear(self) -> None:
        self._mem.clear()
        for p in self.cache_dir.glob("*.pkl"):
            try:
                p.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _disk_path(self, video_key: str) -> Path:
        return self.cache_dir / f"{video_key}.pkl"

    def _promote(self, key: str, bundle: TrajectoryBundle) -> None:
        if key in self._mem:
            self._mem.move_to_end(key)
            self._mem[key] = bundle
            return
        self._mem[key] = bundle
        while len(self._mem) > self.capacity:
            evicted_key, _ = self._mem.popitem(last=False)
            print(f"[CACHE] evicted L1 key={evicted_key}")
