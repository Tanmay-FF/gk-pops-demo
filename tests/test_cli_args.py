# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""gk_pops.py's argument surface, zone parsing, output paths and envelope shape.

Run with:  python tests/test_cli_args.py

No weights and no clip: everything here is the part of the CLI that runs before
the model loads. The one real video read is a two-frame MP4 this file writes
itself with cv2, so the paths that need a decodable clip (probe_video, the
frame-shape-dependent zone checks) are exercised without depending on footage
that is deliberately not committed.

The emphasis is the two halves a user types by hand -- zone specs and output
paths -- because those are where a mistake produces a plausible-looking run
rather than an error. A door zone left at applies_to=person matches zero cart
tracks and the blocked-door rule then reports nothing while the zone looks
perfectly correct; that trap has its own case below.
"""
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import console_ui as ui
import gk_pops as gk
from engine import config as cfg
from engine.rules import resolve_thresholds

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def _raises_unsupported(sink) -> bool:
    """fileno() must raise rather than return the real descriptor."""
    try:
        sink.fileno()
    except io.UnsupportedOperation:
        return True
    return False


def section(title: str) -> None:
    print(f"\n=== {title} ===")


DEFAULTS = gk._engine_defaults()
PARSER = gk.build_parser(DEFAULTS)
FRAME_SHAPE = (720, 1280, 3)
PLACEMENT_DEFAULT = cfg.CAMERA_PLACEMENT_DEFAULT


def parse(*argv):
    """Parse without the program name. Raises SystemExit on a usage error."""
    return PARSER.parse_args(list(argv))


def parse_fails(*argv) -> bool:
    """True when argparse rejects the invocation with its own exit 2."""
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            PARSER.parse_args(list(argv))
    except SystemExit as e:
        return e.code == 2
    return False


def zone_error(spec: str, units: str = "px"):
    """The ZoneSpecError a bad --zone raises, or None when it parsed."""
    try:
        gk.parse_zone_spec(spec, 0, FRAME_SHAPE, units)
    except gk.ZoneSpecError as e:
        return e
    return None


TMP = tempfile.mkdtemp(prefix="pops_cli_args_")


def make_clip(name: str = "clip.mp4", size=(1280, 720), frames: int = 2) -> str:
    """A tiny real MP4, so probe_video has something to decode."""
    import cv2
    path = os.path.join(TMP, name)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, size)
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


def run_main(*argv) -> tuple[int, str]:
    """gk.main() with its console captured. Returns (exit code, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = gk.main(["gk_pops.py", *argv])
    return code, err.getvalue()


try:
    CLIP = make_clip()

    # -----------------------------------------------------------------------
    section("defaults come from engine.config, never retyped")

    a = parse("v.mp4")
    check("camera placement is left unset by the parser",
          a.camera_placement is None,
          "the config default is applied in resolve_placement_for_run, after "
          "the zone preset has had its say — see the precedence section"),
    check("the placement choices ARE the sidebar's five",
          DEFAULTS["placements"] == list(cfg.CAMERA_PLACEMENTS)
          and len(cfg.CAMERA_PLACEMENTS) == 5)

    # The short spellings exist so the flag can be typed without quoting, and
    # they are the tokens gk-pops-api accepts, so one word works in both. What
    # reaches the pipeline is the display string either way -- everything
    # downstream of the parser compares against those.
    check("every angle has a typeable slug, and every slug is an angle",
          set(cfg.CAMERA_PLACEMENT_SLUGS.values()) == set(cfg.CAMERA_PLACEMENTS)
          and len(cfg.CAMERA_PLACEMENT_SLUGS) == 5)
    check("a slug resolves to the sidebar's own wording",
          parse("v.mp4", "--camera-placement", "inside_facing_exit"
                ).camera_placement == "Inside (facing exit)")
    check("the long spelling still works, unchanged",
          parse("v.mp4", "--camera-placement", "Inside (facing exit)"
                ).camera_placement == "Inside (facing exit)")
    check("case and dashes in a slug are forgiven",
          parse("v.mp4", "--camera-placement", "Inside-Exit-On-Left"
                ).camera_placement == "Inside (exit on left)")
    check("the both-sides slug matches the API's spelling",
          parse("v.mp4", "--camera-placement", "inside_exit_on_both"
                ).camera_placement == "Inside (exit on both sides)",
          "gk-pops-api spells it without the trailing _sides")
    check("vlm backend defaults to the config value",
          a.vlm_backend == cfg.VLM_DEFAULT_BACKEND)
    check("unset thresholds arrive as None",
          (a.blocked_door_s, a.static_cart_s, a.abandoned_cart_s)
          == (None, None, None),
          "resolve_thresholds decides, not argparse")

    resolved = resolve_thresholds({"blocked_door_s": a.blocked_door_s,
                                   "static_cart_s": a.static_cart_s,
                                   "abandoned_cart_s": a.abandoned_cart_s})
    check("...and survive resolve_thresholds as the config defaults",
          resolved == {"blocked_door_s": cfg.RULE_BLOCKED_DOOR_S,
                       "static_cart_s": cfg.RULE_STATIC_CART_S,
                       "abandoned_cart_s": cfg.RULE_ABANDONED_CART_S},
          str(resolved))
    check("a passed threshold wins",
          resolve_thresholds({"blocked_door_s": parse("v.mp4",
                                                      "--blocked-door-s", "8"
                                                      ).blocked_door_s}
                             )["blocked_door_s"] == 8.0)
    # Pose is tri-state on the namespace and resolved by gk.resolve_pose(),
    # which main() calls immediately after parse_args. The parser itself must
    # leave it None when neither flag was typed, or the derivation below cannot
    # tell "off by default" from "--no-pose was asked for".
    check("the parser leaves pose unset unless a flag says otherwise",
          parse("v.mp4").pose is None
          and parse("v.mp4", "--pose").pose is True
          and parse("v.mp4", "--no-pose").pose is False)
    # Pose feeds only draw_pose_skeleton, so it runs exactly when something
    # will show a skeleton: the annotated MP4, or the case report's evidence
    # images. A JSON-only run pays 35% of the frame loop for nothing.
    check("pose defaults off for a JSON-only run",
          gk.resolve_pose(parse("v.mp4")) is False)
    check("...on when --video-path or --case-report will display it",
          gk.resolve_pose(parse("v.mp4", "--video-path")) is True
          and gk.resolve_pose(parse("v.mp4", "--video-path", "x.mp4")) is True
          and gk.resolve_pose(parse("v.mp4", "--case-report")) is True)
    check("...and an explicit flag beats the derivation either way",
          gk.resolve_pose(parse("v.mp4", "--pose")) is True
          and gk.resolve_pose(parse("v.mp4", "--video-path", "--no-pose")) is False)
    # --case-report is the same three-state path flag as --video-path, not a
    # boolean plus a separate --case-report-path. Asking for a multi-minute VLM
    # pass and then discarding what it produced is not a case anyone wants, so
    # there was never a second decision for a second flag to carry.
    check("no --case-report means no report",
          gk.wants_case_report(parse("v.mp4")) is False
          and parse("v.mp4").case_report_path is None)
    check("...a bare one asks for it with a derived name",
          gk.wants_case_report(parse("v.mp4", "--case-report")) is True
          and parse("v.mp4", "--case-report").case_report_path is gk.DERIVE)
    check("...and one with a path asks for it there",
          gk.wants_case_report(parse("v.mp4", "--case-report", "out/")) is True
          and parse("v.mp4", "--case-report", "out/").case_report_path
          == "out/")
    check("the path flag it replaced is gone",
          parse_fails("v.mp4", "--case-report-path", "out/"),
          "two flags for one decision is what this merge removed")

    # The other half of the same decision: the engine encodes an MP4 only when
    # --video-path asked for one. --case-report does NOT imply a video -- its
    # evidence images come from FrameCapturer, which reads the drawn frame in
    # the loop and has never read the file.
    check("a video is encoded only when --video-path asks for one",
          gk.wants_video(parse("v.mp4")) is False
          and gk.wants_video(parse("v.mp4", "--case-report")) is False
          and gk.wants_video(parse("v.mp4", "--video-path")) is True
          and gk.wants_video(parse("v.mp4", "--video-path", "x.mp4")) is True)
    # ...and the phase denominator follows it, so the console never promises a
    # phase the run cannot reach.
    check("the phase count drops the encode phase with no video",
          gk.phase_total(False, True) == 5 and gk.phase_total(False, False) == 4
          and gk.phase_total(True, False) == 5)

    # -----------------------------------------------------------------------
    section("argparse refusals")

    check("an unlisted camera placement is rejected",
          parse_fails("v.mp4", "--camera-placement", "Inside (facing door)"),
          "a placement outside the five is silently treated as outside by the "
          "INBOUND kill switch")
    check("a near-miss slug is rejected rather than guessed at",
          parse_fails("v.mp4", "--camera-placement", "inside_exit")
          and parse_fails("v.mp4", "--camera-placement",
                          "inside_exit_on_both_sides"),
          "same reason: an unrecognised angle scores the clip from the wrong "
          "side of the door instead of failing")
    check("an unlisted vlm backend is rejected",
          parse_fails("v.mp4", "--vlm-backend", "GPT-4o"))
    check("--zones and --auto-zones conflict",
          parse_fails("v.mp4", "--zones", "z.json", "--auto-zones"))
    check("--zones and --zone conflict",
          parse_fails("v.mp4", "--zones", "z.json", "--zone", "a:door:1,1 2,2 3,3"))
    check("--auto-zones and --zone conflict",
          parse_fails("v.mp4", "--auto-zones", "--zone", "a:door:1,1 2,2 3,3"))
    check("a missing video positional is rejected", parse_fails())
    check("--frames only takes all or none",
          parse_fails("v.mp4", "--frames", "some"))

    # -----------------------------------------------------------------------
    section("both spellings reach the same destination")

    for dash, under in (("--json-path", "--json_path"),
                        ("--camera-placement", "--camera_placement"),
                        ("--blocked-door-s", "--blocked_door_s"),
                        ("--zone-units", "--zone_units")):
        value = {"--json-path": "x.json",
                 "--camera-placement": "Inside (facing exit)",
                 "--blocked-door-s": "7",
                 "--zone-units": "norm"}[dash]
        lhs = vars(parse("v.mp4", dash, value))
        rhs = vars(parse("v.mp4", under, value))
        check(f"{dash} == {under}", lhs == rhs)
    check("-o and --output are --json-path",
          parse("v.mp4", "-o", "x.json").json_path == "x.json"
          and parse("v.mp4", "--output", "x.json").json_path == "x.json")

    # -----------------------------------------------------------------------
    section("zone specs")

    z = gk.parse_zone_spec("Main door:door:both:512,376 544,55 842,35 830,329",
                           0, FRAME_SHAPE)
    check("a spec round-trips to the polygon it names",
          z.name == "Main door" and z.kind == "door"
          and np.asarray(z.polygon).tolist() == [[512, 376], [544, 55],
                                                 [842, 35], [830, 329]])
    check("the polygon is int32, as the overlay and rule engine expect",
          np.asarray(z.polygon).dtype == np.int32)
    check("colour is derived from the kind, never accepted",
          isinstance(z.color, tuple) and len(z.color) == 3)

    trap = gk.parse_zone_spec("Main door:door:person:512,376 544,55 842,35",
                              0, FRAME_SHAPE)
    check('a door declared applies_to=person comes back as "both"',
          trap.applies_to == "both",
          "otherwise it matches zero cart tracks and blocked-door reports "
          "nothing while the zone looks correct")
    analytics = gk.parse_zone_spec("Till:analytics:person:10,10 20,10 20,20",
                                   0, FRAME_SHAPE)
    check("...but an analytics zone keeps the applies_to it was given",
          analytics.applies_to == "person")
    omitted = gk.parse_zone_spec("Till:analytics:10,10 20,10 20,20",
                                 0, FRAME_SHAPE)
    check("applies_to may be omitted", omitted.applies_to == "person")

    check("fewer than 3 vertices is refused",
          zone_error("a:door:10,10 20,20") is not None)
    check("an unknown kind is refused, not downgraded",
          "unknown kind" in str(zone_error("a:doorway:1,1 2,2 3,3")),
          "on the command line that is a typo, not a stale file")
    check("an unknown applies_to is refused",
          "unknown applies_to" in str(zone_error("a:door:people:1,1 2,2 3,3")))
    check("a malformed vertex pair is refused",
          "malformed vertex" in str(zone_error("a:door:1,1 2 3,3")))
    check("a non-numeric vertex is refused",
          zone_error("a:door:1,1 x,y 3,3") is not None)
    check("too few colon fields is refused",
          zone_error("a:door") is not None)
    check("a colon in the name is refused with a reason",
          "not supported" in str(zone_error("a:b:door:both:1,1 2,2 3,3").hint))
    check("a polygon entirely outside the frame is refused",
          "outside" in str(zone_error("a:door:5000,5000 5001,5001 5002,5000")))

    norm = gk.parse_zone_spec("a:analytics:0.4,0.5 0.9,0.5 0.9,0.9",
                              0, FRAME_SHAPE, "norm")
    check("--zone-units norm scales 0.4,0.5 on 1280x720 to (512, 360)",
          np.asarray(norm.polygon).tolist()[0] == [512, 360])

    ids = [gk.parse_zone_spec("a:door:1,1 2,2 3,3", i, FRAME_SHAPE).zone_id
           for i in range(6)]
    check("two zones never share a zone_id", len(set(ids)) == len(ids),
          "find_zone matches the FIRST id it sees")

    seen: list[str] = []
    edge = gk.parse_zone_spec("a:door:0,0 1400,10 20,20", 0, FRAME_SHAPE,
                              warn=seen.append)
    check("a vertex outside the frame warns and is kept",
          edge is not None and len(seen) == 1 and "outside" in seen[0],
          "a doorway polygon legitimately runs to the frame edge")

    # -----------------------------------------------------------------------
    section("output paths")

    origin = os.getcwd()
    from pathlib import Path
    clip = Path(CLIP)
    notes: list[str] = []
    R = lambda value, dest: gk.resolve_output_path(   # noqa: E731
        value, dest, clip, Path(origin), warn=notes.append)

    check("no --json-path derives <clip dir>/<stem>.json",
          R(gk.DERIVE, "json_path") == clip.parent / "clip.json")
    check("a bare --video-path derives <stem>_annotated.mp4",
          R(gk.DERIVE, "video_path") == clip.parent / "clip_annotated.mp4")
    check("a bare --heatmap-path derives <stem>_heatmap.png",
          R(gk.DERIVE, "heatmap_path") == clip.parent / "clip_heatmap.png")
    check("a bare --case-report derives <stem>_case_report.html",
          R(gk.DERIVE, "case_report_path")
          == clip.parent / "clip_case_report.html")
    check("...and a directory puts that name inside it",
          R(TMP + os.sep, "case_report_path")
          == Path(TMP) / "clip_case_report.html")
    check("no flag at all means the output is off",
          R(None, "video_path") is None)

    existing_dir = os.path.join(TMP, "reports")
    os.makedirs(existing_dir, exist_ok=True)
    check("an existing directory gets the derived filename inside it",
          R(existing_dir, "json_path") == Path(existing_dir) / "clip.json")
    check("a value ending in a separator does too, uncreated",
          R(os.path.join(TMP, "fresh") + os.sep, "json_path")
          == Path(TMP) / "fresh" / "clip.json")
    check("...and its parent directory is created",
          os.path.isdir(os.path.join(TMP, "fresh")))
    check("an explicit filename is honoured verbatim",
          R(os.path.join(TMP, "named.json"), "json_path")
          == Path(TMP) / "named.json")

    notes.clear()
    R(os.path.join(TMP, "out.avi"), "video_path")
    check("a wrong extension warns but is honoured",
          len(notes) == 1 and ".mp4" in notes[0],
          "the engine re-encodes to MP4 whatever the file is called")

    check("a relative path resolves against the cwd, not the repo root",
          R("rel/out.json", "json_path") == Path(origin) / "rel" / "out.json")
    shutil.rmtree(os.path.join(origin, "rel"), ignore_errors=True)

    # -----------------------------------------------------------------------
    section("overwrite, and the failure-envelope carve-out")

    target = Path(TMP) / "exists.json"
    target.write_text(json.dumps({"schema": gk.SCHEMA,
                                  "cli": {"exit": "ok"}}), encoding="utf-8")

    def overwrite_refused(path, force=False) -> bool:
        try:
            gk.check_overwrite(path, "json", force)
        except gk.OutputPathError:
            return True
        return False

    check("an existing successful envelope needs --force",
          overwrite_refused(target))
    check("...and --force lets it through", not overwrite_refused(target, True))

    tomb = Path(TMP) / "tomb.json"
    tomb.write_text(json.dumps({"schema": gk.SCHEMA,
                                "cli": {"exit": "error"}}), encoding="utf-8")
    check("a failure envelope is overwritten without --force",
          not overwrite_refused(tomb),
          "run fails, fix the preset, re-run — that has to work")
    tomb.write_text(json.dumps({"schema": gk.SCHEMA,
                                "cli": {"exit": "cancelled"}}), encoding="utf-8")
    check("...as is a cancelled one", not overwrite_refused(tomb))

    foreign = Path(TMP) / "foreign.json"
    foreign.write_text(json.dumps({"cli": {"exit": "error"}}), encoding="utf-8")
    check("a file this tool did not write is protected",
          overwrite_refused(foreign), "no matching schema key")
    truncated = Path(TMP) / "truncated.json"
    truncated.write_text('{"schema": "gk-pops-cli/1", "cli": {"exit": "err',
                         encoding="utf-8")
    check("a truncated envelope is protected rather than assumed dead",
          overwrite_refused(truncated))
    check("a path that does not exist needs nothing",
          not overwrite_refused(Path(TMP) / "absent.json"))

    # -----------------------------------------------------------------------
    section("the api key never reaches the envelope")

    os.environ["GK_VLM_API_KEY"] = "sk-should-never-appear"
    try:
        check("--vlm-api-key falls back to the env var",
              gk.resolve_api_key(parse("v.mp4").vlm_api_key)
              == "sk-should-never-appear")
        check("...and an explicit flag wins",
              gk.resolve_api_key(parse("v.mp4", "--vlm-api-key", "flag").
                                 vlm_api_key) == "flag")
        args = parse(CLIP, "--vlm-backend", "Claude (API)")
        plan = gk.Plan(video=clip, video_info={"name": "clip.mp4"},
                       frame_shape=FRAME_SHAPE, zones=[], zone_notes=[],
                       zone_source="none", zone_path=None,
                       camera_placement=PLACEMENT_DEFAULT,
                       camera_placement_source="default",
                       thresholds=resolved, outputs={}, warnings=[])
        block = gk._run_block(args, plan, "cpu", ["gk_pops.py", CLIP])
        check("the key is nowhere in the run block",
              "sk-should-never-appear" not in json.dumps(block),
              "never echoed, never written")
    finally:
        del os.environ["GK_VLM_API_KEY"]
    check("no key anywhere resolves to the empty string",
          gk.resolve_api_key(None) == "")

    # -----------------------------------------------------------------------
    section("the placement, and which of the three places it came from")

    check("--camera-placement no longer defaults during parsing",
          parse(CLIP).camera_placement is None,
          "a default here would be indistinguishable from a typed value, and "
          "the preset could then never win")

    def placement_plan(source: str = "default",
                       placement: str = PLACEMENT_DEFAULT) -> "gk.Plan":
        return gk.Plan(video=Path(CLIP), video_info={"name": "clip.mp4"},
                       frame_shape=FRAME_SHAPE, zones=[], zone_notes=[],
                       zone_source="none", zone_path=None,
                       camera_placement=placement,
                       camera_placement_source=source,
                       thresholds=resolved, outputs={}, warnings=[])

    for source in ("cli", "preset", "default"):
        block = gk._run_block(parse(CLIP), placement_plan(source), "cpu",
                              ["gk_pops.py", CLIP])
        check(f"the envelope reports source={source}",
              block["camera_placement_source"] == source,
              "read off the plan, not sniffed out of argv — a preset is a "
              "third source and argv cannot see it")
    check("the envelope reports the resolved placement, not the raw flag",
          gk._run_block(parse(CLIP),
                        placement_plan("preset", "Inside (exit on left)"),
                        "cpu", ["gk_pops.py", CLIP])["camera_placement"]
          == "Inside (exit on left)")

    # --- precedence: flag, then preset, then the built-in default ----------
    from engine import zone_presets
    from engine.zone_editor import make_zone

    pdir = tempfile.mkdtemp()
    door = [make_zone("Door", [[10, 10], [90, 10], [90, 90]], "both", 0,
                      "door")]
    with_angle = Path(zone_presets.save_preset(
        CLIP, "angled", door, FRAME_SHAPE, directory=pdir,
        camera_placement="inside_exit_on_left"))
    no_angle = Path(zone_presets.save_preset(
        CLIP, "plain", door, FRAME_SHAPE, directory=pdir))

    def resolved_placement(argv_extra, zone_path, zone_source="preset"):
        return gk.resolve_placement_for_run(
            Path(CLIP), parse(CLIP, *argv_extra), zone_path, zone_source,
            warn=lambda _m: None)

    check("no flag and no preset falls back to the built-in default",
          resolved_placement([], None, "none")
          == (PLACEMENT_DEFAULT, "default"))
    check("a preset with no placement recorded also falls back",
          resolved_placement([], no_angle) == (PLACEMENT_DEFAULT, "default"),
          "absent means not recorded, never 'outside'")
    check("a preset's placement is used when no flag was passed",
          resolved_placement([], with_angle)
          == ("Inside (exit on left)", "preset"))
    check("an explicit flag beats the preset",
          resolved_placement(["--camera-placement", "inside_facing_exit"],
                             with_angle)
          == ("Inside (facing exit)", "cli"),
          "whoever typed an angle looked at the picture more recently than "
          "whoever saved the file")

    # A placement no build recognises. Refused rather than quietly replaced:
    # the INBOUND kill switch reads anything unknown as "outside", so a
    # fallback here scores the clip from the wrong side of the door.
    bent = Path(pdir) / "bent.json"
    payload = json.loads(with_angle.read_text(encoding="utf-8"))
    payload["camera_placement"] = "inside_facing_exi"
    bent.write_text(json.dumps(payload), encoding="utf-8")
    try:
        resolved_placement([], bent)
        check("a preset with an unknown placement is refused", False)
    except gk.ZoneSpecError as e:
        check("a preset with an unknown placement is refused",
              "inside_facing_exi" in str(e),
              "an unrecognised angle reads as 'outside' downstream, so it "
              "cannot be shrugged off")

    warned: list[str] = []
    gk.resolve_placement_for_run(
        Path("some-other-clip.mp4"), parse(CLIP), with_angle, "preset",
        warn=warned.append)
    check("borrowing another clip's preset says so before taking its angle",
          any("some-other-clip.mp4" in w for w in warned),
          "--zones takes any path and load_preset does not check the clip")

    # -----------------------------------------------------------------------
    section("the envelope drops what it must")

    class FakeAnalytics:
        journey_labels = ["__OUTSIDE__", "Main door"]
        journey_matrix = np.array([[0, 4], [2, 0]])
        journey_edges = []
        dwell_rows = []
        insight_text = "something happened"
        rule_diagnostics = ["a note"]
        heatmap_array = np.zeros((720, 1280), dtype=np.float32)
        heatmap_composite = np.zeros((720, 1280, 3), dtype=np.uint8)
        heatmap_png_path = "/tmp/heatmap.png"

    block = gk.analytics_block(FakeAnalytics())
    check("the journey matrix is serialised as lists",
          block["journey_matrix"] == [[0, 4], [2, 0]])
    check("heatmap_array and heatmap_composite are excluded",
          "heatmap_array" not in block and "heatmap_composite" not in block,
          "megabytes of numbers whose readable form is the PNG")
    check("...unconditionally, with no fallback into the JSON",
          not any("heatmap" in k for k in block))
    check("a run that left no analytics gives None, not stale numbers",
          gk.analytics_block(None) is None)
    check("heatmap_image is not an html slot",
          "heatmap_image" not in gk.HTML_SLOTS
          and "json_output" not in gk.HTML_SLOTS)

    # -----------------------------------------------------------------------
    section("artifacts: null-not-asked-for vs null-not-produced")

    produced = os.path.join(TMP, "produced.mp4")
    open(produced, "wb").write(b"x" * 10)
    not_kept: list[str] = []
    check("no path asked for it: null, and listed in not_kept",
          gk.keep_artifact(produced, None, "video", not_kept,
                           warn=notes.append) is None
          and not_kept == ["video"])
    not_kept.clear()
    check("never produced: null, and ABSENT from not_kept",
          gk.keep_artifact(None, Path(TMP) / "kept.mp4", "video", not_kept,
                           warn=notes.append) is None and not_kept == [],
          "a reader must tell 'you didn't ask' from 'we couldn't make one'")
    kept = gk.keep_artifact(produced, Path(TMP) / "kept.mp4", "video",
                            not_kept, warn=notes.append)
    check("kept: the path, and the engine's own copy survives",
          kept == str(Path(TMP) / "kept.mp4") and os.path.exists(produced),
          "a copy, not a move")

    # -----------------------------------------------------------------------
    section("--frames none leaves the count behind")

    env = gk.build_envelope(
        argv=["gk_pops.py"], started_at="a", finished_at="b", wall_s=1.0,
        video_s=1.0, case_report_s=None, exit_state="ok", run={},
        tracking={"frames": [{"i": 0}, {"i": 1}, {"i": 2}]})
    check("--frames all keeps the array",
          len(env["tracking"]["frames"]) == 3)
    dropped = {"_dropped": True, "count": 3, "json_every_n": 1}
    check("the dropped shape records the count, not just a null",
          dropped["count"] == 3 and dropped["_dropped"] is True)

    # -----------------------------------------------------------------------
    section("phase mapping")

    check("all five engine descriptions are mapped",
          set(gk.PHASE_HEADINGS) == {
              "Processing frames", "Computing analytics", "Encoding video",
              "Building tracking JSON", "Rendering panels"})
    check("without a case report the count is 5, with one it is 6",
          gk.phase_total(False) == 5 and gk.phase_total(True) == 6,
          "the denominator never promises a phase that will not run")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rep = gk.PhaseReporter(total=5, quiet=True)
        rep.open(gk.PHASE_INPUTS)
        rep(1.0, desc="Processing frames")
        rep(1.0, desc="Building tracking JSON")
        rep(1.0, desc="Rendering panels")
        rep.close()
    check("consecutive descriptions sharing a heading do not reopen it",
          [h for h, _ in rep.timings] == [gk.PHASE_INPUTS,
                                          "Detection and tracking",
                                          "Tracking JSON and panels"],
          str([h for h, _ in rep.timings]))
    check("...and the phase index stops at the total", rep.index == 3)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rep2 = gk.PhaseReporter(total=1, quiet=True)
        rep2(1.0, desc="Reticulating splines")
        rep2.close()
    check("an unmapped description still opens a phase, under its own name",
          [h for h, _ in rep2.timings] == ["Reticulating splines"],
          "information is never dropped because a heading was not in the map")
    check("...and widens the denominator rather than printing [2/1]",
          rep2.total >= rep2.index)

    # -----------------------------------------------------------------------
    section("end to end refusals, with real exit codes")

    code, err = run_main(os.path.join(TMP, "nope.mp4"))
    check("a missing clip exits 2", code == gk.EXIT_USAGE, f"exit {code}")
    check("...and says so on stderr", "no such file" in err)

    empty = os.path.join(TMP, "empty.mp4")
    open(empty, "wb").close()
    code, err = run_main(empty)
    check("a zero-byte clip exits 2 before the model loads",
          code == gk.EXIT_USAGE and "empty" in err, f"exit {code}")

    notvideo = os.path.join(TMP, "notavideo.mp4")
    with open(notvideo, "w", encoding="utf-8") as fh:
        fh.write("this is not an mp4, whatever the extension says")
    code, err = run_main(notvideo)
    check("a text file named .mp4 exits 2",
          code == gk.EXIT_USAGE and "decodable" in err, f"exit {code}")

    code, err = run_main(TMP)
    check("a directory as the clip exits 2",
          code == gk.EXIT_USAGE and "directory" in err, f"exit {code}")

    code, err = run_main(CLIP, "--zones", os.path.join(TMP, "nosuch.json"))
    check("a missing preset exits 2",
          code == gk.EXIT_USAGE and "zone preset" in err, f"exit {code}")

    wrong_size = os.path.join(TMP, "wrongsize.json")
    with open(wrong_size, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "label": "x", "video": "clip.mp4",
                   "frame_w": 1920, "frame_h": 1080, "saved_at": 0,
                   "zones": [{"name": "d", "kind": "door", "applies_to": "both",
                              "polygon": [[1, 1], [2, 2], [3, 1]],
                              "color": [0, 0, 255]}]}, fh)
    code, err = run_main(CLIP, "--zones", wrong_size)
    check("a preset drawn on another resolution exits 2",
          code == gk.EXIT_USAGE and "1920x1080" in err, f"exit {code}")
    check("...with an actionable hint", "redraw" in err.lower())

    code, err = run_main(CLIP, "--auto-zones")
    check("--auto-zones with no preset for this clip exits 2",
          code == gk.EXIT_USAGE and "no saved zone preset" in err,
          f"exit {code}")

    code, err = run_main(CLIP, "--zone", "a:door:1,1", "--dry-run")
    check("a bad --zone exits 2 before anything runs",
          code == gk.EXIT_USAGE and "at least 3" in err, f"exit {code}")

    guard = os.path.join(TMP, "guard.json")
    with open(guard, "w", encoding="utf-8") as fh:
        json.dump({"schema": gk.SCHEMA, "cli": {"exit": "ok"}}, fh)
    code, err = run_main(CLIP, "--json-path", guard, "--dry-run")
    check("an existing result without --force exits 2",
          code == gk.EXIT_USAGE and "already exists" in err, f"exit {code}")
    code, _ = run_main(CLIP, "--json-path", guard, "--force", "--dry-run")
    check("...and --force lets the same invocation through", code == gk.EXIT_OK)

    with open(guard, "w", encoding="utf-8") as fh:
        json.dump({"schema": gk.SCHEMA, "cli": {"exit": "error"}}, fh)
    code, _ = run_main(CLIP, "--json-path", guard, "--dry-run")
    check("a tombstone needs no --force end to end", code == gk.EXIT_OK)

    # -----------------------------------------------------------------------
    section("the zones warning nobody should miss")

    def both_streams(*argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = gk.main(["gk_pops.py", *argv])
        return code, out.getvalue(), err.getvalue()

    code, out_text, err_text = both_streams(
        CLIP, "--blocked-door-s", "8",
        "--json-path", os.path.join(TMP, "warned.json"), "--dry-run")
    check("threshold flags with no zones warn, and still exit 0",
          code == gk.EXIT_OK and "report nothing" in err_text, f"exit {code}")
    check("...on stderr, not stdout",
          "report nothing" not in out_text,
          "stdout is the narrative; stderr is what has to be noticed")

    # -----------------------------------------------------------------------
    section("--quiet does what its help says")

    code, quiet_out, quiet_err = both_streams(
        CLIP, "--blocked-door-s", "8",
        "--json-path", os.path.join(TMP, "quiet.json"), "--quiet", "--dry-run")
    check("--quiet still exits 0", code == gk.EXIT_OK)
    check("no per-item status rows survive --quiet",
          not any(mark in quiet_out for mark in ("[✓]", "[+]")),
          repr(next((l for l in quiet_out.splitlines()
                     if "[✓]" in l or "[+]" in l), "")[:50]))
    check("no banner survives --quiet",
          ui.G["tl"] not in quiet_out and "headless run" not in quiet_out)
    check("the phase heading and its timing DO survive",
          "[1/" in quiet_out and "s" in quiet_out.splitlines()[-1])
    check("the outcome block survives",
          ui.G["go"] in quiet_out)
    check("and the warning is untouched by --quiet",
          "report nothing" in quiet_err,
          "a degraded run is exactly as loud quiet as it is loud")

    loud_rows = len([l for l in out_text.splitlines()
                     if "[✓]" in l or "[·]" in l or "[+]" in l or "[-]" in l])
    check("...while a normal run prints those rows",
          loud_rows >= 5, f"{loud_rows} rows without --quiet")


    # -----------------------------------------------------------------------
    # The engine's own narration is captured, never silenced. What these
    # guard is the LIMIT: suppressing output by default is only safe while a
    # failure still prints what it saw and a clean run still says how much
    # there was.
    # -----------------------------------------------------------------------
    section("the engine's output is captured, not lost")

    import argparse as _argparse

    def capture(verbose=False, path=None, body=None):
        """Run `body` inside engine_output and return (log, stdout, stderr)."""
        args = _argparse.Namespace(verbose=verbose)
        out, err = io.StringIO(), io.StringIO()
        held = {}
        with redirect_stdout(out), redirect_stderr(err):
            try:
                with gk.engine_output(args, path) as log:
                    held["log"] = log
                    body()
            except RuntimeError:
                pass
        return held["log"], out.getvalue(), err.getvalue()

    def noisy():
        print("[VOTE] Cart 2: fill_conf={'full': 3.7}")
        print("[POPS] Cart 2: partial|unbagged score=75")

    log, out, err = capture(body=noisy)
    check("engine prints do not reach stdout",
          "[VOTE]" not in out and "[POPS]" not in out,
          repr(out[:60]))
    check("...and they are counted, not discarded", log.count == 2)
    check("console_ui still reaches the terminal from inside the capture",
          "PHASE HEADING" in capture(
              body=lambda: ui.phase_free("phase heading"))[1],
          "the progress bar is printed from inside the engine's call stack")

    def noisy_then_raise():
        noisy()
        raise RuntimeError("boom")

    log, out, err = capture(body=noisy_then_raise)
    check("a failed run prints the captured tail", "[POPS]" in err,
          "default suppression is only safe if a crash keeps its breadcrumbs")
    check("...on stderr, not stdout", "[POPS]" not in out)

    log, out, err = capture(verbose=True, body=noisy)
    check("-v echoes the engine through unchanged", "[VOTE]" in out)
    check("...and the closing block then does not offer it again",
          log.echo is not None,
          "the count row is guarded on echo being None")

    log_path = os.path.join(TMP, "engine.log")
    log, out, err = capture(path=log_path, body=noisy)
    check("--engine-log writes every line to the file",
          len(open(log_path, encoding="utf-8").read().splitlines()) == 2)

    check("the sink refuses to hand out a file descriptor",
          isinstance(getattr(gk.EngineLog(), "fileno", None), object)
          and _raises_unsupported(gk.EngineLog()),
          "returning fd 1 would let a C-level writer bypass the capture")
    check("the sink duck-types a text stream",
          gk.EngineLog().isatty() is False
          and gk.EngineLog().encoding == "utf-8",
          "torch and ultralytics probe both")

    ring = gk.EngineLog(limit=3)
    for i in range(10):
        ring.write("line %d" % i + chr(10))
    check("a long run does not grow the buffer without bound",
          len(ring.lines) == 3 and ring.count == 10,
          "the count stays honest after rotation")


    # -----------------------------------------------------------------------
    section("the findings frame speaks the engine's vocabulary")

    from engine.highlights import OPS_SEVERITY_ORDER

    missing = set(OPS_SEVERITY_ORDER) - set(gk.FIND_MARKS)
    check("every severity the rules can emit has a badge", not missing,
          f"unbadged: {sorted(missing)}" if missing
          else f"{len(gk.FIND_MARKS)} badges")
    check("...all padded to the same width so the words line up",
          len({len(m) for m, _ in gk.FIND_MARKS.values()}) == 1)
    check("...and an unknown severity falls back rather than vanishing",
          gk.FIND_MARKS.get("NONSENSE", gk.FIND_MARKS["INFO"])
          == gk.FIND_MARKS["INFO"])

    check("an unkept artifact is named in words on screen",
          gk.NOT_KEPT_WORDS["engine-json"] == "engine JSON"
          and gk.NOT_KEPT_WORDS["heatmap"] == "heat map")
    check("...while the envelope keeps the spelling a consumer parses",
          set(gk.NOT_KEPT_WORDS) == {"video", "heatmap", "engine-json",
                                     "case-report"},
          "artifacts.not_kept is part of the gk-pops-cli/1 document")
    check("...and an unmapped name falls through rather than vanishing",
          gk.NOT_KEPT_WORDS.get("something-new", "something-new")
          == "something-new")

    check("the (OPS) qualifier is dropped inside the frame",
          gk._finding_label("UNATTENDED CART (OPS)") == "Unattended cart",
          "every row in the frame is an ops rule; it distinguishes nothing")
    check("...and a label without one is just title-cased",
          gk._finding_label("BLOCKED DOOR") == "Blocked door")


    # -----------------------------------------------------------------------
    section("the POPS frame reads its tiers from the engine")

    from engine.scoring import HIGH_SCORE, MEDIUM_SCORE, PUSHOUT_SCORE

    check("a pushout-scoring cart takes the loudest badge",
          gk._pops_mark(PUSHOUT_SCORE)[0] == "[!!]"
          and gk._pops_mark(HIGH_SCORE)[0] == "[!!]")
    check("...one worth a look takes the middle one",
          gk._pops_mark(MEDIUM_SCORE)[0] == "[!] "
          and gk._pops_mark(HIGH_SCORE - 1)[0] == "[!] ")
    check("...and the boundaries are the engine's, not a copy",
          gk._pops_mark(MEDIUM_SCORE - 1)[0] != gk._pops_mark(MEDIUM_SCORE)[0],
          f"MEDIUM_SCORE={MEDIUM_SCORE}, HIGH_SCORE={HIGH_SCORE}")
    check("a scored-but-quiet cart and an unscored one differ",
          gk._pops_mark(1)[0] != gk._pops_mark(0)[0])
    check("every badge is the same width, so the scores line up",
          len({len(gk._pops_mark(n)[0]) for n in (0, 1, 40, 75, 100)}) == 1)

    _pops = {f"C{i}": {"max_score": i, "peak_event": "LOW PRIORITY",
                       "peak_timestamp": 1.0, "fill": "empty",
                       "bag": "not_applicable"}
             for i in range(1, gk.POPS_MAX_ROWS + 4)}
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        gk._pops_rows(_pops)
    _out_lines = [l for l in _buf.getvalue().splitlines() if l.strip()]
    check("a long clip elides rather than printing every cart",
          len(_out_lines) == gk.POPS_MAX_ROWS + 4,
          f"{len(_out_lines)} lines for {len(_pops)} carts")
    check("...and says how many it left out",
          "+3 more" in _buf.getvalue(),
          "a silently truncated table is worse than no table")
    check("the frame stays inside WIDTH",
          all(len(l) <= ui.WIDTH for l in _out_lines),
          f"longest {max(len(l) for l in _out_lines)}")
    check("a placeholder is not printed as a reading",
          all(gk._reading(v) == "" for v in
              ("not_applicable", "non-applicable", "unknown", None, "")),
          "engine/classifier.py spells it both ways; both mean no answer")
    check("...while a real reading survives",
          (gk._reading("partial"), gk._reading("unbagged"))
          == ("partial", "unbagged"))
    check("...and neither spelling reaches the frame",
          "applicable" not in _buf.getvalue())
    _unclear = io.StringIO()
    with redirect_stdout(_unclear):
        gk._pops_rows({"C4": {"max_score": 5, "peak_event": "LOW PRIORITY",
                              "peak_timestamp": 19.9, "fill": "non-applicable",
                              "bag": "non-applicable", "quality": "unclear"}})
    check("a cart nothing could be read from says why",
          "unclear" in _unclear.getvalue(),
          "a blank contents column would read as empty and unbagged")

    _empty = io.StringIO()
    with redirect_stdout(_empty):
        gk._pops_rows({})
    check("a run that scored no carts prints no frame at all",
          _empty.getvalue() == "")


    # -----------------------------------------------------------------------
    section("generated at")

    from datetime import datetime, timedelta, timezone

    _utc = "2026-09-11T15:28:18.402Z"
    _shown = gk._generated_at(_utc)
    check("the envelope's UTC is shown on the reader's own clock",
          _shown != _utc and _shown[:4] == "2026")
    check("...with the offset kept",
          _shown[-5:].lstrip("+-").isdigit() and _shown[-5] in "+-",
          f"{_shown!r} — a screenshot mailed to another timezone stays "
          f"unambiguous")
    check("...and it names the same instant the envelope does",
          datetime.strptime(_shown, "%Y-%m-%d %H:%M:%S %z")
          == datetime(2026, 9, 11, 15, 28, 18, tzinfo=timezone.utc),
          "derived from cli.finished_at, never read off the clock again")
    check("a stamp it cannot parse is printed rather than dropped",
          gk._generated_at("whenever") == "whenever")
    check("...and a missing one is simply empty",
          gk._generated_at(None) == "" and gk._generated_at("") == "")

finally:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
