"""End-to-end golden regression: one real clip, frame-by-frame, against a
committed baseline.

The unit tripwires (test_bytetrack_low_confidence_recovery.py) pin one specific
guarantee with synthetic detections. This pins the WHOLE pipeline — detector,
tracker, classifier, linker, POPS scoring — on real pixels, so any dependency
change that shifts the output shows up as a diff instead of as a quietly
different answer on the next demo.

That is the point: this file is not about the fuse_score bug. It is the net
that catches the NEXT one, whatever it turns out to be.

WHAT IS COMMITTED
-----------------
Only `tests/fixtures/golden/baseline.json` — per-frame counts plus the
provenance of the run that produced them. No imagery: the clip is real store
CCTV with identifiable people in it and is deliberately NOT committed.

RUNNING IT
----------
The clip is located, in order, from:
  1. the POPS_GOLDEN_CLIP environment variable
  2. the absolute path recorded in baseline.json
  3. tests/fixtures/golden/clip.mp4

If none of those exist, or the file found does not match the sha256 recorded in
the baseline, the test SKIPS with a message saying which. It never silently
passes on the wrong input.

Needs the detection weights and takes ~30s. Regenerate the baseline with
  python tests/make_golden_baseline.py <clip.mp4>
after any DELIBERATE behaviour change, and review the diff before committing it.
"""
import sys
import os
import json
import glob
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO, "tests", "fixtures", "golden")
BASELINE = os.path.join(GOLDEN_DIR, "baseline.json")
FALLBACK_CLIP = os.path.join(GOLDEN_DIR, "clip.mp4")

#: Frames allowed to differ before the test fails. Zero on purpose — two
#: environments four ultralytics releases and two CUDA versions apart matched
#: exactly once the tracker semantics agreed, so any drift here is real. Raise
#: it only with a measurement that justifies the number.
#:
#: Note compare() also prepends whole-run "distinct carts/persons" lines, which
#: are not frame rows; a nonzero tolerance would absorb them too. Keep it at 0
#: unless you have a reason, and re-read the diff rather than raising it.
ALLOWED_FRAME_DIFFS = int(os.environ.get("POPS_GOLDEN_TOLERANCE", "0"))


class Skip(Exception):
    """Raised when the golden clip is unavailable on this machine."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_baselines():
    """Every committed baseline, primary first.

    More than one clip is pinned now: `baseline.json` is the original, and
    `baseline_<case>.json` files pin footage that exists to hold a specific
    behaviour — `baseline_outside.json` is the merchandise-removal clip that
    docs/missed_pushout_fix_plan.md is about, and it is the one that would
    silently stop covering owner linking if a later change quietly broke it.

    Returned as (name, path) so failures can name which case drifted.
    """
    found = []
    if os.path.exists(BASELINE):
        found.append(("baseline", BASELINE))
    for path in sorted(glob.glob(os.path.join(GOLDEN_DIR, "baseline_*.json"))):
        found.append((os.path.basename(path)[:-5], path))
    return found


def load_baseline(path=None):
    """Load one baseline. Defaults to the primary, for callers that want any
    real clip to compare against (tests/test_device_guard_e2e.py)."""
    path = path or BASELINE
    if not os.path.exists(path):
        raise Skip(f"no baseline at {path} — run tests/make_golden_baseline.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_clip(baseline):
    """Return the clip path, or raise Skip explaining exactly why not.

    POPS_GOLDEN_CLIP is tried first but validated by sha256 like every other
    candidate, so with several baselines committed it simply does not match the
    ones it was not pointed at — those fall through to their recorded path.
    """
    name = baseline.get("clip", {}).get("name") or ""
    candidates = [
        (os.environ.get("POPS_GOLDEN_CLIP"), "POPS_GOLDEN_CLIP"),
        (baseline.get("clip", {}).get("path"), "the path recorded in the baseline"),
        (os.path.join(GOLDEN_DIR, name) if name else None,
         f"tests/fixtures/golden/{name}"),
        (FALLBACK_CLIP, "tests/fixtures/golden/clip.mp4"),
    ]
    tried = []
    for path, where in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            tried.append(f"{where}: not found ({path})")
            continue
        want = baseline.get("clip", {}).get("sha256")
        got = sha256(path)
        if want and got != want:
            tried.append(f"{where}: WRONG CLIP (sha256 {got[:12]} != {want[:12]})")
            continue
        return path
    raise Skip("golden clip unavailable — " + "; ".join(tried or ["nothing to try"]))


#: Per-cart fields lifted out of the run's own `pops_summary` for the decisions
#: block. Every one of them is a DECISION, not a measurement — no classifier
#: confidence floats, which are GPU-nondeterministic and would make the baseline
#: environment-specific (see the parity notes in the repo).
_DECISION_CART_FIELDS = ("max_score", "peak_event", "owner", "abandoned",
                         "merch_removed", "peak_frame")
#: Same for a logged event row.
_DECISION_EVENT_FIELDS = ("frame", "cart_id", "event", "pops_score", "fill",
                          "bag", "direction", "speed_status", "linked",
                          "abandoned")


def decisions_from_json(json_str):
    """The run's scoring decisions, as a comparable dict.

    Per-frame counts and display IDs catch detection and tracking drift, but
    they are completely blind to the layer where the bug in
    docs/missed_pushout_fix_plan.md lived: a cart can be detected on every
    frame, tracked under one stable ID, and still be linked to the wrong person,
    never marked abandoned, and scored 45 instead of 75. None of that moves a
    single count. This is the part of the baseline that sees it.
    """
    doc = json.loads(json_str)
    carts = {}
    for cd, info in sorted((doc.get("pops_summary") or {}).items()):
        carts[cd] = {k: info.get(k) for k in _DECISION_CART_FIELDS}
    events = [
        {k: ev.get(k) for k in _DECISION_EVENT_FIELDS}
        for ev in (doc.get("events") or [])
    ]
    return {
        "carts": carts,
        "events": events,
        "summary": {k: (doc.get("summary") or {}).get(k) for k in
                    ("total_events", "high_priority", "medium_priority",
                     "total_links_established")},
    }


def run_pipeline(clip, settings, engine_obj=None, capture_decisions=False):
    """Drive the real engine and capture per-frame counts AND display IDs.

    With `capture_decisions`, returns `(frames, decisions)` instead of just
    `frames` — see decisions_from_json(). Off by default so existing callers
    (tests/test_device_guard_e2e.py, which only cares about frame parity) are
    unaffected.

    Pass `engine_obj` to reuse an existing TrackingEngine — that is how
    tests/test_device_guard_e2e.py runs the same clip twice in one process and
    compares run 2 against the same baseline.

    draw_hud is the narrowest place that sees the final person/cart/link counts
    for a frame, after every filter the pipeline applies. draw_bbox is where a
    display ID is first committed to the frame.

    Recording the IDs and not just the counts is deliberate. The first version
    of this baseline stored counts only, and was therefore completely blind to
    one physical cart being tracked as Cart 1 AND Cart 3 — the counts were
    "right", the identities were not. Any future ID churn now shows up here.
    """
    os.chdir(REPO)
    import engine.tracker as T

    captured = {}
    pending = {"cart": set(), "person": set()}
    real_draw_hud = T.draw_hud
    real_draw_bbox = T.draw_bbox

    def bbox_spy(im0, box, disp, c, names, colors, *args, **kwargs):
        label = names[int(c)]
        if label in pending:
            pending[label].add(int(disp))
        return real_draw_bbox(im0, box, disp, c, names, colors, *args, **kwargs)

    def spy(im0, person_count, cart_count, link_count, frame_idx, total_frames, w,
            *args, **kwargs):
        captured[int(frame_idx)] = {
            "counts": [int(person_count), int(cart_count), int(link_count)],
            "carts": sorted(pending["cart"]),
            "persons": sorted(pending["person"]),
        }
        pending["cart"] = set()
        pending["person"] = set()
        return real_draw_hud(im0, person_count, cart_count, link_count,
                             frame_idx, total_frames, w, *args, **kwargs)

    T.draw_hud = spy
    T.draw_bbox = bbox_spy
    try:
        if engine_obj is None:
            engine_obj = T.TrackingEngine()
        result = engine_obj.process_video(
            clip,
            camera_placement=settings["camera_placement"],
            vlm_backend="Claude (API)", vlm_api_key="",
            zones=settings.get("zones") or [],
            defer_case_report=True,
            enable_pose=settings.get("enable_pose", True),
            progress=lambda *a, **k: None,
        )
    finally:
        T.draw_hud = real_draw_hud
        T.draw_bbox = real_draw_bbox
    if capture_decisions:
        # process_video returns (out_path, json_path, json_str, ...) — the
        # third slot is the whole tracking document as text, which is where
        # every finalised decision lives after reconciliation.
        return captured, decisions_from_json(result[2])
    return captured


def compare(actual, baseline):
    """Return a list of human-readable differences, most useful first."""
    expected = {int(k): v for k, v in baseline["frames"].items()}
    diffs = []

    if len(actual) != len(expected):
        diffs.append(
            f"frame COUNT changed: {len(expected)} -> {len(actual)}. The decoder "
            f"is returning a different number of frames (opencv/ffmpeg change?)."
        )

    labels = ("persons", "carts", "links")
    for frame in sorted(set(expected) | set(actual)):
        want, got = expected.get(frame), actual.get(frame)
        if want == got:
            continue
        if want is None or got is None:
            diffs.append(f"frame {frame}: {'missing from run' if got is None else 'unexpected'}")
            continue
        changed = [
            f"{labels[i]} {want['counts'][i]}->{got['counts'][i]}"
            for i in range(3) if want["counts"][i] != got["counts"][i]
        ]
        for key in ("carts", "persons"):
            if want.get(key) != got.get(key):
                changed.append(f"{key} IDs {want.get(key)}->{got.get(key)}")
        diffs.append(f"frame {frame}: " + ", ".join(changed))

    # An identity churn that keeps per-frame counts constant would otherwise
    # slip through: one cart handed off between two IDs looks like one cart on
    # every single frame.
    def distinct(d, key):
        return sorted({i for v in d.values() for i in v.get(key, [])})

    for key in ("carts", "persons"):
        w, g = distinct(expected, key), distinct(actual, key)
        if w != g:
            diffs.insert(0, f"distinct {key} across the run: {w} -> {g}")
    return diffs


def compare_decisions(actual, baseline):
    """Differences in the scoring layer, most consequential first.

    Ordered deliberately: the event list and the per-cart peak event come first
    because those are what an operator sees, and a change there is a change in
    what the demo claims. Ownership and abandonment follow, because they are the
    inputs that explain it.
    """
    expected = baseline.get("decisions")
    if not expected:
        return []
    diffs = []

    def rowkey(ev):
        return (ev.get("cart_id"), ev.get("event"))

    want_events = {rowkey(e): e for e in expected.get("events", [])}
    got_events = {rowkey(e): e for e in actual.get("events", [])}
    for key in sorted(set(want_events) | set(got_events), key=lambda k: (str(k[0]), str(k[1]))):
        w, g = want_events.get(key), got_events.get(key)
        if w == g:
            continue
        if w is None:
            diffs.append(f"NEW event: cart {key[0]} {key[1]} {g}")
        elif g is None:
            diffs.append(f"LOST event: cart {key[0]} {key[1]} (was {w})")
        else:
            changed = [f"{k} {w.get(k)}->{g.get(k)}" for k in w if w.get(k) != g.get(k)]
            diffs.append(f"event cart {key[0]} {key[1]}: " + ", ".join(changed))

    want_carts = expected.get("carts", {})
    got_carts = actual.get("carts", {})
    for cd in sorted(set(want_carts) | set(got_carts)):
        w, g = want_carts.get(cd), got_carts.get(cd)
        if w == g:
            continue
        if w is None or g is None:
            diffs.append(f"cart {cd}: {'not scored any more' if g is None else 'newly scored'}")
            continue
        changed = [f"{k} {w.get(k)!r}->{g.get(k)!r}" for k in w if w.get(k) != g.get(k)]
        if changed:
            diffs.append(f"cart {cd}: " + ", ".join(changed))

    w, g = expected.get("summary", {}), actual.get("summary", {})
    changed = [f"{k} {w.get(k)}->{g.get(k)}" for k in w if w.get(k) != g.get(k)]
    if changed:
        diffs.insert(0, "run summary: " + ", ".join(changed))
    return diffs


def test_golden_clips_match_their_baselines():
    """Every committed baseline, each against its own clip.

    A case whose clip is missing on this machine skips, and skipping one does not
    skip the rest — that is the point of naming them. The test only skips
    entirely when no baseline had a usable clip.
    """
    cases = discover_baselines()
    if not cases:
        raise Skip("no baselines committed — run tests/make_golden_baseline.py")

    failures, unavailable = [], []
    for name, path in cases:
        baseline = load_baseline(path)
        try:
            clip = find_clip(baseline)
        except Skip as why:
            unavailable.append(f"{name}: {why}")
            continue
        actual, decided = run_pipeline(clip, baseline["settings"],
                                       capture_decisions=True)
        diffs = compare(actual, baseline)
        if len(diffs) > ALLOWED_FRAME_DIFFS:
            head = "\n  ".join(diffs[:25])
            more = f"\n  ... and {len(diffs) - 25} more" if len(diffs) > 25 else ""
            failures.append(
                f"[{name}] drifted on {len(diffs)} frame(s):\n  {head}{more}\n"
                f"  recorded with: {baseline.get('provenance')}\n"
                f"  regenerate: python tests/make_golden_baseline.py {clip}"
            )
        # Reported separately from frame drift on purpose. Frames moving means
        # detection or tracking changed; decisions moving with frames identical
        # means the scoring layer changed, and those are different bugs with
        # different owners.
        dec_diffs = compare_decisions(decided, baseline)
        if dec_diffs:
            failures.append(
                f"[{name}] SCORING decisions changed on identical frames:\n  "
                + "\n  ".join(dec_diffs[:25])
            )

    if failures:
        raise AssertionError("\n\n".join(failures))
    if unavailable and len(unavailable) == len(cases):
        raise Skip("; ".join(unavailable))


def test_baselines_record_enough_to_reproduce_them():
    """A baseline you cannot attribute is a baseline you cannot trust."""
    cases = discover_baselines()
    if not cases:
        raise Skip("no baselines committed — run tests/make_golden_baseline.py")
    for name, path in cases:
        baseline = load_baseline(path)
        for key in ("frames", "settings", "clip", "provenance", "decisions"):
            assert key in baseline, f"{name}.json is missing {key!r}"
        assert baseline["clip"].get("sha256"), f"{name} does not identify its clip"
        assert baseline["frames"], f"{name} has no frames"
        sample = next(iter(baseline["frames"].values()))
        for key in ("counts", "carts", "persons"):
            assert key in sample, (
                f"{name} frames are missing {key!r} — counts alone cannot detect "
                f"one object being tracked under two display IDs"
            )
        for key in ("carts", "events", "summary"):
            assert key in baseline["decisions"], (
                f"{name} decisions are missing {key!r} — counts and IDs alone "
                f"cannot detect a cart linked to the wrong person, never marked "
                f"abandoned, and scored a tier too low"
            )
        for field in ("ultralytics", "tracker_config_sha", "weights"):
            assert field in baseline["provenance"], (
                f"{name} provenance is missing {field!r}")


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
    if skipped:
        print("\nTo enable the end-to-end check, point POPS_GOLDEN_CLIP at the clip\n"
              "recorded in one of tests/fixtures/golden/baseline*.json, or put the\n"
              "clips back at the paths those files record.")
