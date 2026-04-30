# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Two-level trajectory cache so YOLO+BoTSORT runs at most once per (video file,
mtime, size) tuple.  Lets the user re-draw zones and re-aggregate analytics
in <1s without re-running detection.

L1: in-memory LRU on the TrackingEngine instance.
L2: pickle on disk under %TEMP%/pops_traj_cache/.

Pickle is fine here — local-only, single-user demo.
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


def make_video_key(path: str) -> str:
    """sha1(abs_path + mtime_ns + size). 16 hex chars → cheap collision-safe key."""
    p = os.path.abspath(path)
    try:
        st = os.stat(p)
        payload = f"{p}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")
    except OSError:
        payload = p.encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


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
