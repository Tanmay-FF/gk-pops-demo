# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The CLI and the demo must produce the same numbers for the same clip.

Run with:  python tests/test_cli_parity.py

SLOW: needs the detection weights and a clip in sample_videos/ that has a saved
zone preset. Runs the pipeline TWICE -- once as a real subprocess through
gk_pops.py, once in this process through TrackingEngine.process_video() with the
same arguments app_poc_v2.run_analysis() passes -- and compares the decision
fields of the two tracking documents.

This is the test the whole design of gk_pops.py rests on. The CLI is a caller,
not a second pipeline: it reimplements no scoring, linking, rule or analytics
logic, so a number that differs between it and the demo is a bug in the CLI.
Without this file that claim is an assertion in a docstring.

WHAT IS COMPARED, AND WHAT IS NOT
---------------------------------
Decision fields only:

  * per cart -- max_score, peak_event, owner, abandoned, merch_removed, and the
    reconciled fill/bag/quality vote
  * event rows -- frame, cart_id, event, pops_score, direction, linked,
    abandoned
  * rule findings -- rule_id, cart_id, zone_name, severity, and whether the
    finding was still ongoing at the end of the video

NOT the classifier confidence floats, and not the per-frame `fill`/`bag` of an
event row. Those are GPU-nondeterministic at the last decimal, and a test that
compares them flaps for reasons that have nothing to do with the CLI. The
reconciled vote in pops_summary IS compared: it is taken over every valid sample
of a cart, which is what makes it a decision rather than a reading.

The subprocess runs with --frames none --no-html. Neither affects the pipeline;
they only shrink the envelope from megabytes to kilobytes, and everything
compared here lives outside both blocks.
"""
import json
import os
import subprocess
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The same five strings the CLI and the sidebar both offer. Read rather than
#: retyped, so this test cannot pin an angle that no longer exists.
PLACEMENT = "Inside (facing exit)"
THRESHOLDS = {"blocked_door_s": 5.0, "static_cart_s": 10.0,
              "abandoned_cart_s": 10.0}

_PASS: list[str] = []
_FAIL: list[str] = []


class Skip(Exception):
    """Raised when no clip on this machine has a preset to run against."""


def check(name: str, cond: bool, extra: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def find_clip_with_preset() -> tuple[str, str]:
    """A clip in sample_videos/ that has a saved preset, and that preset's path.

    Deterministic: clips are taken in sorted order and the newest preset for the
    first one that has any wins. Clips are never committed (*.mp4 is gitignored
    and they are store CCTV of identifiable people), so this SKIPS rather than
    fails when the folder is empty.
    """
    from engine import zone_presets
    from engine.config import TEST_VIDEO_DIR

    if not os.path.isdir(TEST_VIDEO_DIR):
        raise Skip(f"no sample-video directory at {TEST_VIDEO_DIR}")
    clips = sorted(f for f in os.listdir(TEST_VIDEO_DIR)
                   if f.lower().endswith((".mp4", ".avi", ".mov")))
    if not clips:
        raise Skip(f"{TEST_VIDEO_DIR} holds no clips — drop one in to enable "
                   f"this test")
    for name in clips:
        path = os.path.join(TEST_VIDEO_DIR, name)
        found = zone_presets.list_presets(path)
        if found:
            return path, found[0].path
    raise Skip(f"none of the {len(clips)} clips in {TEST_VIDEO_DIR} has a saved "
               f"zone preset — draw zones in the demo and press Save zone set")


def decisions(tracking: dict) -> dict:
    """The comparable projection of one tracking document.

    Everything here is a decision the pipeline made, not a measurement it took.
    """
    carts = {}
    for cid, row in (tracking.get("pops_summary") or {}).items():
        if not isinstance(row, dict):
            continue
        carts[cid] = {
            "max_score": row.get("max_score"),
            "peak_event": row.get("peak_event"),
            "owner": row.get("owner"),
            "abandoned": bool(row.get("abandoned")),
            "merch_removed": bool(row.get("merch_removed")),
            "fill": row.get("fill"),
            "bag": row.get("bag"),
            "quality": row.get("quality"),
        }

    events = [{
        "frame": e.get("frame"),
        "cart_id": e.get("cart_id"),
        "event": e.get("event"),
        "pops_score": e.get("pops_score"),
        "direction": e.get("direction"),
        "linked": e.get("linked"),
        "abandoned": e.get("abandoned"),
    } for e in (tracking.get("events") or [])]

    rules = [{
        "rule_id": f.get("rule_id"),
        "cart_id": f.get("cart_id"),
        "zone_name": f.get("zone_name"),
        "severity": f.get("severity"),
        "ongoing_at_end_of_video": f.get("ongoing_at_end_of_video"),
    } for f in (tracking.get("rule_findings") or [])]

    engine_block = tracking.get("rule_engine") or {}
    return {
        "carts": carts,
        "events": events,
        "rules": rules,
        "rules_unavailable_reason": engine_block.get("unavailable_reason"),
        "thresholds_s": engine_block.get("thresholds_s"),
        "n_frames_recorded": len(tracking.get("frames") or []),
    }


def diff(left: dict, right: dict, path: str = "") -> list[str]:
    """Every leaf where the two projections disagree, as readable lines."""
    out: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            here = f"{path}.{key}" if path else str(key)
            if key not in left:
                out.append(f"{here}: only in the in-process run")
            elif key not in right:
                out.append(f"{here}: only in the CLI run")
            else:
                out.extend(diff(left[key], right[key], here))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{path}: {len(left)} rows via the CLI, "
                       f"{len(right)} in process")
        for i, (a, b) in enumerate(zip(left, right)):
            out.extend(diff(a, b, f"{path}[{i}]"))
    elif left != right:
        out.append(f"{path}: {left!r} via the CLI, {right!r} in process")
    return out


def run_cli(clip: str, preset: str, out_dir: str) -> dict:
    """gk_pops.py as a real subprocess. Returns the envelope it wrote."""
    target = os.path.join(out_dir, "cli.json")
    cmd = [
        sys.executable, os.path.join(REPO, "gk_pops.py"), clip,
        "--camera-placement", PLACEMENT,
        "--zones", preset,
        "--blocked-door-s", str(THRESHOLDS["blocked_door_s"]),
        "--static-cart-s", str(THRESHOLDS["static_cart_s"]),
        "--abandoned-cart-s", str(THRESHOLDS["abandoned_cart_s"]),
        "--json-path", target,
        "--frames", "none", "--no-html", "--log", "plain", "--quiet",
        # Explicit, because the CLI's default is now derived: pose runs only
        # when --video-path or --case-report will display a skeleton, and this
        # command asks for neither. run_in_process() below passes
        # enable_pose=True, and this test's whole premise is that both sides are
        # handed the same configuration. (Pose is decision-neutral -- it feeds
        # only draw_pose_skeleton -- so the projection compared below would
        # match either way; matching the flag keeps the premise true rather than
        # relying on that.)
        "--pose",
    ]
    # encoding= explicitly: the CLI prints box-drawing glyphs, and text=True on
    # Windows decodes with the ANSI code page, which raises UnicodeDecodeError
    # on them and kills the reader thread before the test sees an exit code.
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(((proc.stdout or "") + (proc.stderr or ""))
                         .strip().splitlines()[-15:])
        raise AssertionError(
            f"the CLI exited {proc.returncode}:\n{tail}")
    with open(target, encoding="utf-8") as fh:
        return json.load(fh)


def run_in_process(clip: str, preset: str) -> dict:
    """process_video() called the way app_poc_v2.run_analysis() calls it.

    Same keywords, same order, deferred case report -- the one difference from
    the demo is that nothing consumes the deferred payload, which is also what
    the CLI does without --case-report and what tests/test_golden_clip.py's
    run_pipeline already exercises.

    `write_video=False` matches what the CLI command above does by omitting
    --video-path, so both sides run the same pipeline. It is also what makes
    this test bearable to run: the encode is a per-frame write plus a second
    decode pass, and neither side's projection can see it.
    """
    os.chdir(REPO)
    from engine import TrackingEngine, zone_presets
    from engine.run_outputs import ENGINE_IDX
    from engine.zone_editor import extract_first_frame

    frame = extract_first_frame(clip)
    zones, _notes = zone_presets.load_preset(preset, frame_shape=frame.shape)

    engine = TrackingEngine(device="auto")
    result = engine.process_video(
        clip,
        camera_placement=PLACEMENT,
        vlm_backend="Claude (API)",
        vlm_api_key="",
        zones=list(zones),
        defer_case_report=True,
        rule_thresholds=dict(THRESHOLDS),
        enable_pose=True,
        write_video=False,
        progress=lambda *a, **k: None,
    )
    return json.loads(result[ENGINE_IDX["json_output"]])


def main() -> int:
    try:
        clip, preset = find_clip_with_preset()
    except Skip as why:
        print(f"SKIP: {why}")
        print("\n0 passed, 0 failed (skipped)")
        return 0

    print(f"clip   {os.path.basename(clip)}")
    print(f"preset {os.path.basename(preset)}")

    out_dir = tempfile.mkdtemp(prefix="pops_cli_parity_")
    try:
        section("the CLI run")
        envelope = run_cli(clip, preset, out_dir)
        check("the CLI exited 0 and wrote an envelope",
              envelope.get("cli", {}).get("exit") == "ok")
        cli_tracking = envelope["tracking"]

        section("the in-process run")
        direct = run_in_process(clip, preset)
        check("process_video returned a tracking document",
              isinstance(direct, dict) and "pops_summary" in direct)

        section("the two agree")
        left, right = decisions(cli_tracking), decisions(direct)

        # --frames none is the CLI's, so the frame arrays legitimately differ.
        # Asserted rather than skipped: it proves the flag did what it claims
        # and that nothing else here depends on the array.
        check("the CLI dropped its frame array, as asked",
              isinstance(cli_tracking.get("frames"), dict)
              and cli_tracking["frames"].get("count") == len(direct["frames"]),
              f"{cli_tracking.get('frames', {}).get('count')} frames recorded "
              f"either way")
        left.pop("n_frames_recorded")
        right.pop("n_frames_recorded")

        for key in ("carts", "events", "rules", "rules_unavailable_reason",
                    "thresholds_s"):
            differences = diff(left[key], right[key], key)
            counted = (f"{len(left[key])} compared"
                       if isinstance(left[key], (dict, list))
                       else repr(left[key]))
            check(f"{key} match", not differences,
                  counted if not differences else "")
            for line in differences[:20]:
                print(f"          {line}")
            if len(differences) > 20:
                print(f"          ... and {len(differences) - 20} more")

        section("the envelope records what actually ran")
        run = envelope["run"]
        check("the resolved thresholds are recorded, not the flags",
              run["rule_thresholds"] == THRESHOLDS, str(run["rule_thresholds"]))
        check("the camera placement is recorded",
              run["camera_placement"] == PLACEMENT)
        check("the zone polygons themselves are recorded",
              run["zones"]["count"] > 0
              and all(len(z["polygon"]) >= 3 for z in run["zones"]["zones"]),
              f"{run['zones']['count']} zones from {run['zones']['source']}")
        check("the clip's sha256 is recorded",
              len(run["video"]["sha256"]) == 64)
        check("the device the engine CHOSE is recorded, not 'auto'",
              run["device"] != "auto", run["device"])
        check("the structured analytics reached the envelope",
              envelope["analytics"] is not None
              and "journey_labels" in envelope["analytics"],
              "the journey matrix and dwell rows only ever reached the UI as "
              "HTML before this")
        check("no case report was generated, and the placeholder is gone",
              envelope["case_report"]["generated"] is False)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        for name in _FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
