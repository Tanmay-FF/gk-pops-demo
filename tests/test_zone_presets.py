# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Saved zone presets — file round-trip, rehydration types, and the refusals.

Run with:  python tests/test_zone_presets.py

Deliberately stdlib only, no pytest — matches the rest of the repo. Writes
into a temp directory, never into ZONE_PRESET_DIR.

The types matter as much as the coordinates. A preset comes back out of JSON as
plain lists, while `render_zone_overlay` calls `polygon.astype()` and cv2 wants
a BGR tuple for `color`, so a loader that handed those lists straight to
`Zone(...)` would pass a round-trip check on values and still blow up on the
first redraw.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine.analytics_models import Zone
from engine.zone_editor import make_zone
from engine import zone_presets as zp

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


def raises(fn, needle=""):
    """True when `fn` raises PresetError, optionally carrying `needle`."""
    try:
        fn()
    except zp.PresetError as e:
        return needle in str(e)
    except Exception:
        return False
    return False


VIDEO = "1763916270090_B8A44F9DC9A1-medium.mp4"
FRAME_SHAPE = (720, 1280, 3)                 # (h, w, c) — a 1280x720 clip

DIR  = tempfile.mkdtemp(prefix="pops_zone_presets_")
LDIR = tempfile.mkdtemp(prefix="pops_zone_list_")
SDIR = tempfile.mkdtemp(prefix="pops_zone_stem_")


def zones():
    return [
        make_zone("Doorway", [(10, 10), (200, 10), (200, 300), (10, 300)],
                  "both", 0, kind="door"),
        make_zone("Checkout", [(400, 100), (600, 100), (600, 400)],
                  "cart", 1, kind="analytics"),
    ]


def edited(path, mutate):
    """Rewrite a saved preset with `mutate(payload)` applied — stands in for a
    hand-edited file and for one written by an older build."""
    payload = json.loads(open(path, encoding="utf-8").read())
    mutate(payload)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload))
    return path


try:
    # -----------------------------------------------------------------------
    section("round trip")
    zp.save_preset(VIDEO, "default", zones(), FRAME_SHAPE, directory=DIR)
    infos = zp.list_presets(VIDEO, directory=DIR)
    check("one preset listed for the clip it was saved against", len(infos) == 1,
          f"got {len(infos)}")
    loaded, notes = zp.load_preset(infos[0].path, frame_shape=FRAME_SHAPE)
    check("no complaints on a clean file", notes == [], f"got {notes}")
    check("names and order survive",
          [z.name for z in loaded] == ["Doorway", "Checkout"])
    check("kinds survive", [z.kind for z in loaded] == ["door", "analytics"])
    check("applies_to survives", [z.applies_to for z in loaded] == ["both", "cart"])
    check("every entry is a Zone", all(isinstance(z, Zone) for z in loaded))
    # The types the overlay and the rule engine depend on, not just the values.
    check("polygon comes back as an int32 ndarray",
          all(isinstance(z.polygon, np.ndarray) and z.polygon.dtype == np.int32
              for z in loaded),
          f"got {[type(z.polygon).__name__ for z in loaded]}")
    check("colour comes back as a tuple, not a list",
          all(isinstance(z.color, tuple) for z in loaded))
    check("vertices are unchanged",
          all(np.array_equal(a.polygon, b.polygon)
              for a, b in zip(loaded, zones())))
    check("colours match what the editor would have drawn",
          [z.color for z in loaded] == [z.color for z in zones()])

    # -----------------------------------------------------------------------
    section("derived state is re-derived, not trusted")
    # A door left on "person" matches zero cart tracks, so the blocked-door
    # rule over it reports nothing while the zone looks perfectly correct on
    # screen. coerce_applies_to() has to run on load as well as on draw.
    p = edited(zp.save_preset(VIDEO, "doors", [zones()[0]], FRAME_SHAPE,
                              directory=DIR),
               lambda d: d["zones"][0].__setitem__("applies_to", "person"))
    (door,), _ = zp.load_preset(p, frame_shape=FRAME_SHAPE)
    check('a door saved as "person" loads as "both"',
          door.kind == "door" and door.applies_to == "both",
          f"got {door.kind}/{door.applies_to}")

    p = edited(zp.save_preset(VIDEO, "recolour", zones(), FRAME_SHAPE,
                              directory=DIR),
               lambda d: d["zones"][0].__setitem__("color", [1, 2, 3]))
    recoloured, _ = zp.load_preset(p, frame_shape=FRAME_SHAPE)
    check("colour is recomputed from kind, not read from the file",
          recoloured[0].color != (1, 2, 3), f"got {recoloured[0].color}")

    p = edited(zp.save_preset(VIDEO, "unknown-kind", zones(), FRAME_SHAPE,
                              directory=DIR),
               lambda d: d["zones"][0].__setitem__("kind", "escalator"))
    odd, notes = zp.load_preset(p, frame_shape=FRAME_SHAPE)
    check("an unrecognised kind falls back to analytics and says so",
          odd[0].kind == "analytics" and any("escalator" in n for n in notes),
          f"got {odd[0].kind} / {notes}")

    # zone_editor.find_zone() matches the FIRST id it sees, so two zones
    # sharing one id would send every rename, retype and remove to whichever
    # came first.
    path = zp.save_preset(VIDEO, "twice", zones(), FRAME_SHAPE, directory=DIR)
    first, _ = zp.load_preset(path, frame_shape=FRAME_SHAPE)
    second, _ = zp.load_preset(path, frame_shape=FRAME_SHAPE)
    ids = [z.zone_id for z in first] + [z.zone_id for z in second]
    check("loading the same preset twice mints new zone ids",
          len(set(ids)) == len(ids), f"got {ids}")

    # -----------------------------------------------------------------------
    section("refusals")
    sized = zp.save_preset(VIDEO, "sized", zones(), FRAME_SHAPE, directory=DIR)
    check("a different frame size is refused, not rescaled",
          raises(lambda: zp.load_preset(sized, frame_shape=(1080, 1920, 3)),
                 "1280x720"))
    same, _ = zp.load_preset(sized, frame_shape=FRAME_SHAPE)
    check("...and the same size still loads", len(same) == 2)
    nocheck, _ = zp.load_preset(sized)
    check("...as does a load with no frame to check against", len(nocheck) == 2)

    check("saving zero zones is an error",
          raises(lambda: zp.save_preset(VIDEO, "empty", [], FRAME_SHAPE,
                                        directory=DIR)))

    newer = edited(zp.save_preset(VIDEO, "future", zones(), FRAME_SHAPE,
                                  directory=DIR),
                   lambda d: d.__setitem__("schema_version",
                                           zp.PRESET_SCHEMA_VERSION + 1))
    check("a file from a newer build is refused rather than half-read",
          raises(lambda: zp.load_preset(newer), "newer build"))

    broken = os.path.join(DIR, f"{zp.video_stem(VIDEO)}__broken.json")
    with open(broken, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    check("a corrupt file is skipped by the listing rather than breaking it",
          all("broken" not in i.path
              for i in zp.list_presets(VIDEO, directory=DIR)))
    check("...and reported when loaded directly",
          raises(lambda: zp.load_preset(broken)))

    degenerate = edited(zp.save_preset(VIDEO, "degenerate", zones(),
                                       FRAME_SHAPE, directory=DIR),
                        lambda d: d["zones"][0].__setitem__("polygon",
                                                            [[1, 1], [2, 2]]))
    kept, notes = zp.load_preset(degenerate, frame_shape=FRAME_SHAPE)
    check("a 2-vertex zone is dropped with a note, the rest still load",
          [z.name for z in kept] == ["Checkout"]
          and any("fewer than 3 vertices" in n for n in notes),
          f"got {[z.name for z in kept]} / {notes}")

    # -----------------------------------------------------------------------
    section("the fixed row pool")
    many = [make_zone(f"Z{i}", [(i, 0), (i + 10, 0), (i + 10, 10)],
                      "person", i) for i in range(5)]
    p = zp.save_preset(VIDEO, "many", many, FRAME_SHAPE, directory=DIR)
    clamped, notes = zp.load_preset(p, frame_shape=FRAME_SHAPE, max_zones=3)
    # Zones past the last manager row would still draw and still feed the rule
    # engine, with no way to rename or remove them.
    check("more zones than the editor has rows are dropped, loudly",
          len(clamped) == 3 and any("loaded the first 3" in n for n in notes),
          f"got {len(clamped)} / {notes}")

    # -----------------------------------------------------------------------
    section("listing is per clip")
    zp.save_preset(VIDEO, "older", zones(), FRAME_SHAPE, directory=LDIR)
    time.sleep(1.05)                    # a different second in the filename
    zp.save_preset(VIDEO, "newer", zones(), FRAME_SHAPE, directory=LDIR)
    zp.save_preset("other_clip.mp4", "theirs", zones(), FRAME_SHAPE,
                   directory=LDIR)
    labels = [i.label for i in zp.list_presets(VIDEO, directory=LDIR)]
    check("newest first, and only this clip's", labels == ["newer", "older"],
          f"got {labels}")
    check("no video path lists the whole folder",
          len(zp.list_presets(directory=LDIR)) == 3)
    check("a clip with nothing saved lists nothing",
          zp.list_presets("never_saved.mp4", directory=LDIR) == [])
    check("a missing folder lists nothing rather than raising",
          zp.list_presets(VIDEO, directory=os.path.join(LDIR, "nope")) == [])

    # Two clips in different folders can share a basename stem, so the
    # filename alone cannot decide ownership — the recorded basename does.
    zp.save_preset(r"C:\a\clip.mp4", "a", zones(), FRAME_SHAPE, directory=SDIR)
    check("a same-stem clip does not borrow another clip's preset",
          zp.list_presets(r"C:\b\clip.mov", directory=SDIR) == [])
    check("...while the clip it was saved for still finds it",
          len(zp.list_presets(r"C:\a\clip.mp4", directory=SDIR)) == 1)
    check("saved presets survive an mtime change on the clip",
          # make_video_key() folds in mtime and an environment fingerprint;
          # video_stem() deliberately does not, or every ultralytics bump
          # would orphan hand-drawn polygons.
          zp.video_stem(r"C:\a\clip.mp4") == zp.video_stem(r"D:\else\clip.mp4"))

    # -----------------------------------------------------------------------
    section("labels reaching the filesystem")
    for bad in ("", "   ", "..", "///", "nul", "LPT1", "a__b"):
        check(f'label {bad!r} is rejected',
              raises(lambda b=bad: zp.sanitize_label(b)))
    for messy in ("doorway v2!", "../../etc/passwd", r"..\..\weights", "a/b:c"):
        p = zp.preset_path(VIDEO, messy, directory=DIR)
        check(f"label {messy!r} stays inside the preset folder",
              os.path.dirname(str(p)) == DIR
              and p.name.endswith(".json")
              and not (set(p.name) & set(" !:/\\")),
              f"got {p}")

    # -----------------------------------------------------------------------
    section("filename carries clip, set name and creation time")
    NDIR = tempfile.mkdtemp(prefix="pops_zone_name_")
    try:
        when = time.time()
        p = zp.save_preset(VIDEO, "doorway-v2", zones(), FRAME_SHAPE,
                           directory=NDIR)
        parts = p.stem.split("__")
        check("three parts, separated by __", len(parts) == 3, f"got {p.name}")
        check("part 1 is the clip", parts[0] == zp.video_stem(VIDEO),
              f"got {parts[0]}")
        check("part 2 is the set name", parts[1] == "doorway-v2",
              f"got {parts[1]}")
        check("part 3 is a YYYYmmdd-HHMMSS stamp",
              bool(re.fullmatch(r"\d{8}-\d{6}", parts[2])), f"got {parts[2]}")
        check("the stamp is the creation time",
              parts[2] == zp.timestamp_slug(when)
              or parts[2] == zp.timestamp_slug(when + 1),
              f"got {parts[2]} for {zp.timestamp_slug(when)}")
        check("...and it matches the saved_at the listing reports",
              zp.timestamp_slug(zp.list_presets(VIDEO, directory=NDIR)[0].saved_at)
              == parts[2])

        # Two saves inside one second would otherwise be one file, with the
        # second click silently overwriting the first.
        a = zp.save_preset(VIDEO, "same", zones(), FRAME_SHAPE, directory=NDIR)
        b = zp.save_preset(VIDEO, "same", zones(), FRAME_SHAPE, directory=NDIR)
        check("a same-second re-save is a second file, not an overwrite",
              a != b and a.is_file() and b.is_file(), f"{a.name} / {b.name}")
        check("...and both are listed",
              len([i for i in zp.list_presets(VIDEO, directory=NDIR)
                   if i.label == "same"]) == 2)
    finally:
        shutil.rmtree(NDIR, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("delete")
    gone = zp.save_preset(VIDEO, "gone", zones(), FRAME_SHAPE, directory=DIR)
    check("delete removes the file", zp.delete_preset(gone) is True
          and not os.path.exists(gone))
    check("deleting it again reports False rather than raising",
          zp.delete_preset(gone) is False)
    check("...and it is out of the listing",
          all(i.label != "gone" for i in zp.list_presets(VIDEO, directory=DIR)))

    victim = os.path.join(DIR, "weights.pt")
    with open(victim, "w", encoding="utf-8") as fh:
        fh.write("not a preset")
    check("delete refuses a path that is not a preset file",
          raises(lambda: zp.delete_preset(victim)) and os.path.exists(victim))
finally:
    for d in (DIR, LDIR, SDIR):
        shutil.rmtree(d, ignore_errors=True)

print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
