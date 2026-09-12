# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
On-disk zone presets: save the polygons drawn for a clip, load them back the
next time that clip is selected, delete the ones that are no longer wanted.

Gradio-free for the same reason `zone_editor.py` is - the UI layer only
shuttles paths and `Zone` lists between `gr.State` and these functions, so the
file format and the reload rules are unit-testable on their own.

One JSON file per preset, named ``<video stem>__<label>__<when>.json`` -
the clip, what the set is called, and the local time it was created, so the
folder is readable without opening anything. That also means Save never
overwrites: saving the same name twice keeps both, newest first in the list.
A file records
the frame size it was drawn against because `Zone.polygon` is in source-video
pixel coordinates: reloading a 1080p polygon set onto a 720p re-encode of the
same clip would put every zone in the wrong place, which is worse than
refusing, so a size mismatch is reported instead of silently rescaled.

A file also records the camera placement the zones were drawn under, when the
caller knew it. Zones and placement are one answer to one question - where the
door is in this picture, and which side of it the camera is on - and splitting
them across two places is how a clip ends up analysed with the right doorway
and the wrong direction of travel. The field is optional because presets
written before it existed are still valid; `None` there means "not recorded"
and is never silently read as a placement.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import config
from .analytics_models import Zone, ZoneKind
from .zone_editor import LAYOUT_KINDS, coerce_applies_to, make_zone

#: Bumped when the payload below gains or loses a field that changes how a
#: file must be read. Files written by an older build stay loadable as long as
#: the fields this reader needs are present; anything higher than this was
#: written by a newer build and is refused rather than half-understood.
PRESET_SCHEMA_VERSION = 1

PRESET_SUFFIX = ".json"

#: Separates the three parts of a filename - clip stem, label, creation
#: timestamp. Two underscores because a single one is legal inside all three.
_SEP = "__"

#: Creation time in the filename. Sorts lexicographically in date order, which
#: is what makes a plain `ls` of the folder read chronologically.
_TS_FMT = "%Y%m%d-%H%M%S"

#: Labels and video stems both become filenames, so both are reduced to this
#: whitelist. Free text from a textbox otherwise reaches the filesystem, where
#: "../" and ":" mean something.
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

MAX_LABEL_LEN = 48

#: Stems Windows refuses to create a file for, whatever the extension.
_RESERVED_STEMS = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

DEFAULT_LABEL = "default"


class PresetError(Exception):
    """Anything the caller should show the user instead of a traceback."""


@dataclasses.dataclass(frozen=True)
class PresetInfo:
    """One saved file, as the UI needs to describe it."""
    label: str
    path: str
    n_zones: int
    saved_at: float                  # unix seconds, when it was created
    frame_w: int
    frame_h: int
    video: str                       # basename of the clip it was drawn on
    #: Where the camera was, as one of `config.CAMERA_PLACEMENTS`, or None for
    #: a file written before placement was recorded. None means "not recorded",
    #: never "outside" - see `camera_placement_of()` for why that distinction
    #: has to survive all the way to the caller.
    camera_placement: Optional[str] = None

    @property
    def when(self) -> str:
        return time.strftime("%d %b %Y, %H:%M", time.localtime(self.saved_at))

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def display(self) -> str:
        plural = "" if self.n_zones == 1 else "s"
        return f"{self.label} - {self.n_zones} zone{plural} - {self.when}"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def _sanitize(text: str) -> str:
    clean = _SAFE_CHARS.sub("-", (text or "").strip()).strip("-._")
    return clean[:MAX_LABEL_LEN]


def sanitize_label(label: str) -> str:
    """Filesystem-safe preset label, or raise PresetError with a reason.

    Rejecting rather than silently rewriting: quietly renaming what the user
    typed files the preset under a name they never chose, and then they look
    for it under the one they did.
    """
    clean = _sanitize(label)
    if not clean:
        raise PresetError(
            "Preset name must contain letters, digits, '.', '-' or '_'.")
    if clean.lower() in _RESERVED_STEMS:
        raise PresetError(f'"{clean}" is a reserved filename on Windows.')
    if _SEP in clean:
        raise PresetError(f"Preset name cannot contain '{_SEP}'.")
    return clean


def video_stem(video_path: str) -> str:
    """Stable per-clip filename half.

    NOT `trajectory_cache.make_video_key()`. That key folds in mtime and an
    environment fingerprint so a cached trajectory is never reused across a
    change that would have produced a different one - exactly the wrong
    property here, where an ultralytics bump or a re-copied file must not
    orphan hand-drawn polygons. The clip's own name is what a person means by
    "the zones for this video".
    """
    stem = Path(str(video_path or "")).stem
    return _sanitize(stem) or "video"


def resolve_placement(value: Optional[str]) -> Optional[str]:
    """One of `config.CAMERA_PLACEMENTS`, or raise PresetError saying so.

    `None` in, `None` out: "not recorded" is a legitimate state for a preset
    written before this field existed, and it has to stay distinguishable from
    a recorded placement all the way to the caller.

    Anything else is resolved through `config.resolve_camera_placement()`, so
    both the display string and the command line's slug are accepted and the
    display string is what comes back. A value that resolves to nothing raises
    rather than falling back to the default, for the reason spelled out at
    `config.CAMERA_PLACEMENTS`: the INBOUND kill switch treats an unrecognised
    placement as "outside", so guessing here would score the clip from the
    wrong side of the door and produce a quiet run rather than a failed one.
    """
    if value is None:
        return None
    resolved = config.resolve_camera_placement(str(value))
    if resolved is None:
        raise PresetError(
            f'"{value}" is not a camera placement. Expected one of: '
            + ", ".join(config.CAMERA_PLACEMENT_SLUGS))
    return resolved


def preset_dir(directory: Optional[str] = None) -> Path:
    return Path(directory or config.ZONE_PRESET_DIR)


def timestamp_slug(when: Optional[float] = None) -> str:
    return time.strftime(_TS_FMT, time.localtime(when if when else time.time()))


def preset_path(video_path: str, label: str, when: Optional[float] = None,
                directory: Optional[str] = None) -> Path:
    """``<clip>__<label>__<when>.json`` in the preset folder.

    The creation time is part of the name, so two saves under the same label
    are two files rather than one overwriting the other - a zone set is hand
    work, and losing the previous one to a re-save with the same name is not
    a trade the demo needs to make.
    """
    name = (f"{video_stem(video_path)}{_SEP}{sanitize_label(label)}"
            f"{_SEP}{timestamp_slug(when)}{PRESET_SUFFIX}")
    return preset_dir(directory) / name


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def _free_path(path: Path) -> Path:
    """`path`, or the first "-2", "-3" ... variant of it that does not exist.

    The filename carries the creation time to the second, so two saves under
    the same name inside one second would otherwise be one file - the second
    click silently overwriting the first. A zone set is hand work; a suffix is
    cheaper than losing one.
    """
    if not path.exists():
        return path
    for n in range(2, 100):
        cand = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not cand.exists():
            return cand
    raise PresetError("Too many zone sets saved under that name this second.")



def _zone_to_dict(z: Zone) -> dict:
    return {
        "name": z.name,
        # tolist() rather than the array itself: json cannot encode numpy
        # scalars, and int() per coordinate keeps the file readable as plain
        # [x, y] pairs.
        "polygon": [[int(x), int(y)] for x, y in np.asarray(z.polygon).tolist()],
        "applies_to": z.applies_to,
        "kind": getattr(z, "kind", "analytics"),
        # Written for readability only - the loader re-derives it from `kind`,
        # the same way retype_zone() does, so a file hand-edited to a new kind
        # cannot leave the overlay drawing a door in an analytics colour.
        "color": list(z.color),
    }


def save_preset(video_path: str, label: str, zones: Iterable[Zone],
                frame_shape: Optional[tuple] = None,
                directory: Optional[str] = None,
                camera_placement: Optional[str] = None) -> Path:
    """Write `zones` as a new preset named `label` for `video_path`.

    Args:
        camera_placement: Where the camera was, as a display string or a slug
            from `config.CAMERA_PLACEMENT_SLUGS`. Resolved to the display
            string before it is written, so the file never holds a spelling
            the scoring layer would not recognise. Omitted from the payload
            entirely when None, which keeps a file written without it
            byte-identical to one written by the previous build.

    Raises:
        PresetError: No zones, an unusable label, or a `camera_placement` that
            is not one of the five. Refusing is the whole point: the INBOUND
            kill switch reads an unknown placement as "outside", so a bad value
            would not fail, it would score the clip from the wrong side of the
            door and look quiet rather than wrong.
    """
    zones = list(zones or [])
    if not zones:
        raise PresetError("Nothing to save - draw at least one zone first.")

    created_at = time.time()
    path = _free_path(preset_path(video_path, label, when=created_at,
                                  directory=directory))
    h = w = 0
    if frame_shape is not None and len(frame_shape) >= 2:
        h, w = int(frame_shape[0]), int(frame_shape[1])

    payload = {
        "schema_version": PRESET_SCHEMA_VERSION,
        "label": sanitize_label(label),
        "video": os.path.basename(str(video_path or "")),
        "frame_w": w,
        "frame_h": h,
        "saved_at": created_at,
        "zones": [_zone_to_dict(z) for z in zones],
    }
    if camera_placement is not None:
        payload["camera_placement"] = resolve_placement(camera_placement)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-replace: a crash mid-write would otherwise leave a truncated
    # file that reads as "this clip's zones are corrupt".
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def _read_payload(path) -> dict:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PresetError(f"Preset file is gone: {p.name}")
    except (OSError, json.JSONDecodeError) as e:
        raise PresetError(f"Could not read {p.name}: {e}")
    if not isinstance(payload, dict) or not isinstance(payload.get("zones"), list):
        raise PresetError(f"{p.name} is not a zone preset file.")
    version = int(payload.get("schema_version") or 1)
    if version > PRESET_SCHEMA_VERSION:
        raise PresetError(
            f"{p.name} was written by a newer build (schema v{version}).")
    return payload


def list_presets(video_path: Optional[str] = None,
                 directory: Optional[str] = None) -> list[PresetInfo]:
    """Presets for one clip, newest first. `video_path=None` lists every
    preset in the folder."""
    folder = preset_dir(directory)
    if not folder.is_dir():
        return []
    if video_path:
        pattern = f"{video_stem(video_path)}{_SEP}*{PRESET_SUFFIX}"
        wanted_video = os.path.basename(str(video_path))
    else:
        pattern = f"*{PRESET_SUFFIX}"
        wanted_video = None

    out: list[PresetInfo] = []
    for p in sorted(folder.glob(pattern)):
        try:
            info = read_preset_info(p)
        except PresetError:
            continue                       # a stray file is not an error here
        # Two different clips can sanitize to the same stem, so the recorded
        # basename is what actually decides whether a file belongs to this one.
        if wanted_video and info.video and info.video != wanted_video:
            continue
        out.append(info)
    out.sort(key=lambda i: i.saved_at, reverse=True)
    return out


def read_preset_info(path) -> PresetInfo:
    """Describe one preset file, without rebuilding its polygons.

    The same `PresetInfo` `list_presets` yields - it is built here, so the two
    cannot describe the same file differently - but reachable for a path that
    did not come out of a listing. `--zones SOMEFILE.json` is the case: it
    names a file directly, so nothing has checked which clip it was drawn on.

    Raises:
        PresetError: The file will not read or is not a preset.
    """
    p = Path(path)
    payload = _read_payload(p)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    # The label in the payload wins; the filename's middle part is the
    # fallback for a file written before the label was recorded.
    parts = p.stem.split(_SEP)
    return PresetInfo(
        label=str(payload.get("label")
                  or (parts[1] if len(parts) > 1 else p.stem)),
        path=str(p),
        n_zones=len(payload["zones"]),
        saved_at=float(payload.get("saved_at") or mtime),
        frame_w=int(payload.get("frame_w") or 0),
        frame_h=int(payload.get("frame_h") or 0),
        video=str(payload.get("video") or ""),
        # Hand-edited, or written by a build whose placement list has since
        # changed: dropped to None rather than raising. Describing a file is a
        # browse, and one unreadable field should grey out a row rather than
        # make the whole folder unlistable. Everything that ACTS on the value
        # goes through `camera_placement_of()`, which does raise.
        camera_placement=_placement_or_none(payload),
    )


def _placement_or_none(payload: dict) -> Optional[str]:
    """The payload's placement, or None if it is absent or unrecognisable."""
    try:
        return resolve_placement(payload.get("camera_placement"))
    except PresetError:
        return None


def camera_placement_of(path) -> Optional[str]:
    """The camera placement recorded in one preset file, or None.

    Separate from `load_preset` on purpose. `load_preset` returns
    `(zones, notes)` and four callers unpack exactly that pair, so widening it
    to carry a third value would break every one of them to serve a caller
    that usually wants the placement WITHOUT rebuilding the polygons -
    `gk_pops.py` resolving `--camera-placement` before it has a frame to check
    the zones against, for one.

    Raises:
        PresetError: The file will not read, or its placement is not one of
            the five. Unlike the listing path this refuses rather than
            shrugging: a caller asking this question is about to score a clip
            with the answer.
    """
    return resolve_placement(_read_payload(path).get("camera_placement"))


def load_preset(path, frame_shape: Optional[tuple] = None,
                max_zones: Optional[int] = None) -> tuple[list[Zone], list[str]]:
    """Rebuild the saved polygons as live `Zone`s. Returns (zones, notes),
    where each note is a message worth putting in front of the user.

    Every zone goes back through `zone_editor.make_zone()` rather than being
    handed straight to `Zone(...)`:

      * `polygon` and `color` come out of JSON as lists, while the overlay and
        the rule engine both expect an int32 ndarray and a BGR tuple;
      * `color` is derived from `kind`, so it is recomputed, not trusted;
      * a fresh `zone_id` is minted. `zone_editor.find_zone()` matches the
        FIRST id it sees, so restoring saved ids would let the same preset
        loaded twice give two zones one identity, and every rename, retype and
        remove would then hit whichever came first.
    """
    payload = _read_payload(path)
    notes: list[str] = []

    if frame_shape is not None and len(frame_shape) >= 2:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        saved_w = int(payload.get("frame_w") or 0)
        saved_h = int(payload.get("frame_h") or 0)
        if saved_w and saved_h and (saved_w, saved_h) != (w, h):
            # Refused, not rescaled. Zone coordinates are absolute pixels; a
            # doorway polygon stretched onto a different frame size lands off
            # the doorway, and a rule that then reports nothing looks like a
            # working configuration.
            raise PresetError(
                f"This preset was drawn on a {saved_w}x{saved_h} frame but the "
                f"current video is {w}x{h}. Redraw the zones for this clip.")

    raw = list(payload["zones"])
    if max_zones is not None and len(raw) > max_zones:
        # The zone manager is a fixed pool of rows. Zones past the last row
        # would still draw and still feed the rule engine, with no way to
        # rename or remove them - so they are dropped, loudly.
        notes.append(f"Preset holds {len(raw)} zones; loaded the first "
                     f"{max_zones} (the editor shows {max_zones}).")
        raw = raw[:max_zones]

    zones: list[Zone] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            notes.append(f"Skipped entry {idx + 1}: not a zone.")
            continue
        pts = item.get("polygon") or []
        if len(pts) < 3:
            label = item.get("name") or f"entry {idx + 1}"
            notes.append(f'Skipped "{label}": fewer than 3 vertices.')
            continue
        kind: ZoneKind = item.get("kind") or "analytics"
        if kind != "analytics" and kind not in LAYOUT_KINDS:
            notes.append(f'Unknown zone type "{kind}" - loaded as analytics.')
            kind = "analytics"
        applies_to = coerce_applies_to(kind, item.get("applies_to") or "person")
        zones.append(make_zone(
            str(item.get("name") or ""),
            [(int(p[0]), int(p[1])) for p in pts],
            applies_to, idx, kind=kind,
        ))

    if not zones:
        raise PresetError(f"{Path(path).name} holds no usable zones.")
    return zones, notes


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------
def delete_preset(path) -> bool:
    """Remove one preset file. False when it was already gone."""
    p = Path(path)
    if p.suffix != PRESET_SUFFIX:
        # The UI only ever passes paths that came out of list_presets(), so a
        # non-preset path here is a bug, not a user action - refuse rather than
        # unlink whatever it points at.
        raise PresetError(f"Refusing to delete a non-preset file: {p.name}")
    if not p.is_file():
        return False
    p.unlink()
    return True
