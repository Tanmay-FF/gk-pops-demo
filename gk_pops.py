#!/usr/bin/env python3
# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Run the POPS pipeline on one clip with no browser and no server.

    python gk_pops.py CLIP.mp4 --zones ZONES.json

Writes one JSON document containing everything the run produced -- the engine's
own tracking document verbatim, the structured analytics that previously only
reached the UI as HTML, the rendered panels, and a record of what actually ran.

    # everything derived from the clip's name, JSON only
    python gk_pops.py sample_videos/northgate.mp4 --auto-zones

    # the invocation this tool exists for
    python gk_pops.py sample_videos/northgate.mp4 \\
        --camera-placement inside_facing_exit \\
        --zones zone_presets/northgate__default__20260903-151029.json \\
        --blocked-door-s 5 --static-cart-s 10 --abandoned-cart-s 10 \\
        --json-path out/northgate.json --video-path out/

    # declare the zones inline, with no preset file
    python gk_pops.py clip.mp4 \\
        --zone "Main door:door:both:512,376 544,55 842,35 830,329" \\
        --save-zones northgate

THE THRESHOLD FLAGS DO NOTHING WITHOUT ZONES. The rule engine reads door-kind
zones for the blocked-door family and aisle/analytics for static-cart; with no
monitored zones it reports nothing and says why. The run still exits 0, so this
is called out here, warned about on stderr, and recorded in the envelope under
`tracking.rule_engine.unavailable_reason`.

Only the JSON is written by default. The annotated MP4 and the heat-map PNG are
produced by every run regardless -- naming a path with --video-path or
--heatmap-path is what KEEPS one, not what makes it. Unkept files are left to
the engine's own tidy-up and listed in the envelope's `artifacts.not_kept`.

The case report is off by default: --case-report turns on the VLM pass, which
is multiple minutes per clip with the local backend, and it both generates the
report and saves it. It takes an optional path like the other outputs do --
bare for `<clip stem>_case_report.html`, or a file or directory to place it.

The document is complete by default. --frames none replaces the per-frame array
with its count and --no-html drops the rendered panels; together they take a
20-second clip from about 2 MB to about 16 KB. Neither changes what the pipeline
does -- every frame is still processed and still logged either way.

`--help` is the full flag list; the examples above are the shapes worth
remembering. PROJECT_README.md's "Run it without the UI" is the same
material in prose.

Streams: stdout carries the run's narrative -- the phase headings, the status
rows and the closing block. stderr carries everything an operator has to
NOTICE: zone notes, warnings, the rules-unavailable reason, and errors. --quiet
drops the status rows from stdout and nothing at all from stderr, so a degraded
run is exactly as loud quiet as it is loud. The result JSON is always a file and
never stdout, so neither stream has to stay parseable.

The engine's own narration -- [VOTE], [POPS], [PERF], [CACHE], the per-cart
classification histories -- is CAPTURED, not silenced, and not removed from
engine/. It is the wrong altitude for a result and it prints during the frame
loop, on top of the progress bar. -v echoes it through unchanged, --engine-log
PATH writes it to a file, a failed run prints its last lines either way, and a
clean run reports how many lines there were. See EngineLog.

Exit codes:

    0    the run completed and the JSON was written
    2    bad usage -- nothing ran and nothing was written
    3    the run started and failed; an envelope was still written, with
         cli.exit = "error" and the traceback in cli.error
    130  interrupted (Ctrl-C, or the engine's own cancellation)

1 is deliberately never returned, so it stays the signal that the process died
before this file's own handler could run.

This is a CALLER, not a second pipeline. It invokes TrackingEngine.process_video()
with the same arguments app_poc_v2.run_analysis() passes it, and reimplements no
scoring, linking, rule or analytics logic. A number that differs between this
and the demo for the same clip and parameters is a bug here, and
tests/test_cli_parity.py is what catches it.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import console_ui as ui

#: engine/config.py holds repo-root-relative Windows paths for the detection
#: weights and the tracker yaml, so the engine must be constructed with this as
#: the working directory. run() chdir()s here AFTER every user-supplied path has
#: been made absolute -- see _absolute().
REPO_ROOT = Path(__file__).resolve().parent

#: Name of the envelope format. Bumped when a consumer would have to change.
SCHEMA = "gk-pops-cli/1"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 3
EXIT_INTERRUPTED = 130

#: `--video-path` with no value. A distinct object rather than True or "", so a
#: user who literally types `--video-path <derive>` cannot reach this branch.
DERIVE = object()

#: Suffix and extension for each derived output name, keyed by the argparse
#: destination it belongs to.
DERIVED = {
    "json_path":        ("",              ".json"),
    "video_path":       ("_annotated",    ".mp4"),
    "heatmap_path":     ("_heatmap",      ".png"),
    "case_report_path": ("_case_report",  ".html"),
    "engine_json_path": ("_tracking",     ".json"),
}

#: Environment variable consulted for --vlm-api-key, so a key never has to
#: appear in shell history. Never echoed and never written to the envelope.
API_KEY_ENV = "GK_VLM_API_KEY"


# ---------------------------------------------------------------------------
# Failures
#
# Every one of these carries the exit code it means, so main() maps an
# exception to a status without a chain of isinstance checks, and a new failure
# class cannot be added without deciding what it costs the caller.
# ---------------------------------------------------------------------------
class CliError(Exception):
    """Anything this tool reports to the user instead of a traceback.

    `hint` is the actionable next step, printed under the message. Optional
    because some failures genuinely have no next step beyond reading the
    message.
    """

    exit_code = EXIT_USAGE

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class UsageError(CliError):
    """Bad invocation. Nothing ran, nothing was written. Exit 2."""


class InputError(UsageError):
    """The clip is missing, unreadable, or not decodable video."""


class ZoneSpecError(UsageError):
    """A --zone spec, a preset, or the combination of zone flags is wrong."""


class OutputPathError(UsageError):
    """An output path is unwritable, or exists and --force was not given."""


class PipelineFailed(CliError):
    """The run started and raised. An envelope is still written. Exit 3."""

    exit_code = EXIT_FAILED


class RunInterrupted(CliError):
    """Ctrl-C, or the engine's own RunCancelled / RunSuperseded. Exit 130."""

    exit_code = EXIT_INTERRUPTED


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _absolute(value: str, origin: Path) -> Path:
    """A user-supplied path, made absolute against the ORIGINAL cwd.

    Every path this tool is given has to be resolved before run() chdir()s to
    the repo root, or `--json-path out.json` silently lands in the checkout
    rather than where it was typed.
    """
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (origin / path)


def _looks_like_dir(value: str, path: Path) -> bool:
    """True when the value names a directory rather than a file.

    Either it already exists as one, or it ends in a separator -- "put it in
    here" written by someone who has not created the directory yet.
    """
    return path.is_dir() or str(value).rstrip().endswith(("/", "\\", os.sep))


def derived_name(video: Path, dest: str) -> str:
    """`<clip stem><suffix><ext>` for one output, e.g. `northgate_annotated.mp4`."""
    suffix, ext = DERIVED[dest]
    return f"{video.stem}{suffix}{ext}"


def resolve_output_path(value: Any, dest: str, video: Path, origin: Path,
                        *, warn) -> Optional[Path]:
    """Where one output goes, or None when it was not asked for.

    Args:
        value: The argparse value -- None (off), DERIVE (bare flag), or a path.
        dest: The argparse destination, used for the derived name and extension.
        video: The clip, already absolute. Derived defaults sit beside it.
        origin: The cwd the user typed the command in.
        warn: Called with one string per non-fatal problem.

    Returns:
        An absolute path, or None when this output was not requested.

    Raises:
        OutputPathError: The parent directory cannot be created.

    A directory is accepted anywhere a path is: an existing directory, or a
    value ending in a separator, gets the derived filename inside it. Extensions
    are checked and never changed -- the engine re-encodes to MP4 whatever the
    file is called, and this only copies, so a warning is the honest response.
    """
    if value is None:
        return None

    _, ext = DERIVED[dest]
    if value is DERIVE:
        target = video.parent / derived_name(video, dest)
    else:
        target = _absolute(value, origin)
        if _looks_like_dir(value, target):
            target = target / derived_name(video, dest)
        elif target.suffix.lower() != ext:
            warn(f"{dest.replace('_', '-')}: {target.name} does not end in "
                 f"{ext}; the file is written as {ext[1:].upper()} anyway "
                 f"(nothing here transcodes).")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # A derived default lands next to the clip, and a read-only footage
        # share is a normal thing to point this at. Falling back beats losing a
        # finished run -- but only for a path nobody typed.
        if value is DERIVE or dest == "json_path":
            fallback = origin / target.name
            warn(f"{target.parent} is not writable ({e.strerror or e}); "
                 f"writing {target.name} to the current directory instead.")
            try:
                fallback.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e2:
                raise OutputPathError(
                    f"neither {target.parent} nor {fallback.parent} is "
                    f"writable: {e2}",
                    "name a writable location with --json-path.") from e2
            return fallback
        raise OutputPathError(
            f"cannot create {target.parent}: {e.strerror or e}",
            "check the path and the permissions on it.") from e
    return target


def is_tombstone(path: Path) -> bool:
    """True when `path` is a failure envelope this tool wrote.

    A run that failed leaves an envelope with `cli.exit` of "error" or
    "cancelled". That is a tombstone, not a result, and overwriting it must not
    need --force: otherwise the obvious sequence -- run fails, fix the preset,
    re-run -- answers "output exists" instead of producing a result.

    Anything else, including a file this tool did not write, is protected.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        # Includes the truncated file a Ctrl-C mid-write would leave. Not
        # readable as an envelope, so not provably ours -- protect it.
        return False
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return False
    return payload.get("cli", {}).get("exit") in ("error", "cancelled")


def check_overwrite(path: Optional[Path], label: str, force: bool) -> None:
    """Refuse to clobber an existing output unless --force, or it is a tombstone.

    Raises:
        OutputPathError: The file exists, --force was not given, and it is not a
            failure envelope.
    """
    if path is None or not path.exists() or force:
        return
    if path.suffix.lower() == ".json" and is_tombstone(path):
        return
    raise OutputPathError(
        f"{label} already exists: {path}",
        "pass --force to overwrite it, or name a different path.")


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
#: Anything engine/ printed before the run's EngineLog existed. Normally empty
#: now -- see `_config()` for why nothing in engine/ is touched until the run
#: itself -- but replayed into the capture when it opens, so a line that does
#: arrive early is not lost.
_PREAMBLE: list[str] = []

#: engine/config.py, loaded WITHOUT its package. Cached: see `_config()`.
_CONFIG = None


def _config():
    """engine/config.py, loaded as a standalone module.

    Not `from engine.config import ...`, and that is the whole point. Importing
    a submodule runs the package first, and `engine/__init__.py` pulls in the
    tracker, the analytics builders and the zone editor -- torch, torchvision
    and ultralytics behind them. Measured on this machine: 4.8 s, later 2.7 s
    once gradio came out of engine/tracker.py. All of it to read eight plain
    constants for the parser's `choices` and help strings.

    That cost landed BEFORE the banner. build_parser() is called on the first
    line of main(), so nothing at all appeared on screen for three seconds --
    no title, no clip name, and no way to tell a slow start from a hung one.
    Loading the file directly is 0.012 s, so the banner is now immediate and
    torch loads later, during preflight, with the header already printed.

    This is the real module, not a copy and not a re-implementation:
    `resolve_camera_placement` comes back with it, so the slug normalisation
    rules stay in exactly one place. engine/config.py can be loaded this way
    because it has no relative imports and, since CLS_TRANSFORM became lazy,
    no heavy ones either -- the plain `os`/`pathlib` constants it always was.

    The run later imports `engine.config` properly, so two module objects for
    one file exist in the process. Harmless here and only here: this file holds
    constants and two pure functions over them, with no mutable state for the
    two copies to disagree about. Put state in engine/config.py and this stops
    being true.
    """
    global _CONFIG
    if _CONFIG is None:
        import importlib.util
        path = REPO_ROOT / "engine" / "config.py"
        spec = importlib.util.spec_from_file_location("_gk_pops_config", path)
        if spec is None or spec.loader is None:          # pragma: no cover
            raise CliError(f"engine/config.py could not be read from {path}",
                           "the CLI must be run from inside the repo.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CONFIG = module
    return _CONFIG


def _engine_defaults() -> dict:
    """The config values the parser's help text and choices come from.

    Every default below is read, never retyped: the flag table in
    docs/headless_cli_implementation_plan.md is documentation, not a second
    source of truth.
    """
    cfg = _config()
    return {
        "placements": list(cfg.CAMERA_PLACEMENTS),
        "placement_slugs": dict(cfg.CAMERA_PLACEMENT_SLUGS),
        "placement": cfg.CAMERA_PLACEMENT_DEFAULT,
        "blocked_door_s": cfg.RULE_BLOCKED_DOOR_S,
        "static_cart_s": cfg.RULE_STATIC_CART_S,
        "abandoned_cart_s": cfg.RULE_ABANDONED_CART_S,
        "vlm_backends": list(cfg.VLM_BACKENDS),
        "vlm_backend": cfg.VLM_DEFAULT_BACKEND,
    }


def _slug_for(placement: str) -> str:
    """The typeable spelling of a display string, for help text and messages."""
    for slug, display in _config().CAMERA_PLACEMENT_SLUGS.items():
        if display == placement:
            return slug
    return placement


def _placement(value: str) -> str:
    """`--camera-placement` in either spelling, always returning the long one.

    The short form exists because the long one has spaces and brackets in it,
    so typing it means quoting it, and the tokens accepted here are the ones
    gk-pops-api takes on its own `camera_placement` field. What the pipeline
    receives is unchanged either way: engine.config resolves the slug back to
    the display string the scoring layer compares against.

    Raises:
        argparse.ArgumentTypeError: `value` is not one of the five in either
            spelling. Refused rather than passed through, because the INBOUND
            kill switch reads an unknown placement as "outside" and the run
            then looks quiet instead of wrong.
    """
    # _config(), not engine.config: this runs during parse_args, so importing
    # the package here would put the three-second engine load back in front of
    # the banner for every run that passes --camera-placement.
    cfg = _config()
    resolved = cfg.resolve_camera_placement(value)
    if resolved is None:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a camera placement. Use one of: "
            + ", ".join(cfg.CAMERA_PLACEMENT_SLUGS)
            + " -- or the sidebar's own wording in quotes.")
    return resolved


class _PathAction(argparse.Action):
    """`--video-path` with no value means "yes, and pick the name".

    argparse's nargs="?" gives a bare flag its `const`, and a missing flag its
    `default`, which is exactly the three-state switch these outputs need:
    absent (off), bare (on, derived name), or a path (on, that name).
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, DERIVE if values is None else values)


def build_parser(defaults: Optional[dict] = None) -> argparse.ArgumentParser:
    """The whole argument surface, in one testable place.

    Args:
        defaults: The engine.config values, from `_engine_defaults()`. Injected
            so a test can build a parser without importing torch, and so the
            production path reads config exactly once.

    Returns:
        The parser. Nothing is validated here beyond argparse's own choices and
        types -- the checks that need the clip open live in `preflight()`.
    """
    d = defaults if defaults is not None else _engine_defaults()

    p = argparse.ArgumentParser(
        prog="gk_pops.py",
        description="Run the POPS pipeline on one clip, headless, and write "
                    "everything it produced to one JSON document.",
        epilog="The threshold flags do nothing without zones: with no monitored "
               "zone the rule engine reports nothing and says why. Exit codes: "
               "0 ok, 2 bad usage, 3 the run failed (an envelope is still "
               "written), 130 interrupted.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="the clip to process")

    out = p.add_argument_group(
        "output",
        "A directory is accepted anywhere a path is. Naming a path is what "
        "keeps that output; the MP4 and the PNG are produced either way.")
    out.add_argument("--json-path", "--json_path", "-o", "--output",
                     dest="json_path", metavar="PATH",
                     help="where the result JSON goes "
                          "(default: <clip dir>/<clip stem>.json)")
    out.add_argument("--video-path", "--video_path", dest="video_path",
                     nargs="?", const=None, action=_PathAction, metavar="PATH",
                     help="keep the annotated MP4; bare uses "
                          "<clip stem>_annotated.mp4")
    out.add_argument("--heatmap-path", "--heatmap_path", dest="heatmap_path",
                     nargs="?", const=None, action=_PathAction, metavar="PATH",
                     help="keep the heat-map PNG; bare uses "
                          "<clip stem>_heatmap.png")
    out.add_argument("--engine-json-path", "--engine_json_path",
                     dest="engine_json_path", nargs="?", const=None,
                     action=_PathAction, metavar="PATH",
                     help="keep the engine's own raw <stem>_tracking.json; the "
                          "envelope's `tracking` block is the same document")
    out.add_argument("--frames", choices=("all", "none"), default="all",
                     help="`none` drops tracking.frames from the envelope and "
                          "records the count in its place. Every frame is "
                          "still processed and still logged; this only "
                          "controls what the envelope carries. (default: all)")
    out.add_argument("--no-html", "--no_html", dest="no_html",
                     action="store_true",
                     help="omit the `html` block of rendered panels")
    out.add_argument("--indent", type=int, default=2, metavar="N",
                     help="JSON indent; 0 for one line (default: 2)")
    out.add_argument("--force", action="store_true",
                     help="overwrite existing output files. Not needed to "
                          "overwrite a failure envelope.")

    pipe = p.add_argument_group("pipeline")
    # `default=None`, not the config default. A zone preset can now carry the
    # placement it was drawn under, and a default filled in here would be
    # indistinguishable from a value the user typed -- the preset could then
    # never win, or would override an explicit flag. The fallback is applied in
    # `resolve_placement_for_run()` once the preset has been read.
    pipe.add_argument("--camera-placement", "--camera_placement",
                      dest="camera_placement", type=_placement,
                      choices=d["placements"],
                      default=None, metavar="ANGLE",
                      help="one of: " + "; ".join(d["placement_slugs"])
                           + "  -- or the sidebar's own wording, quoted, e.g. "
                             f"\"{d['placement']}\"  (default: the placement "
                             f"saved with the zones, else "
                             f"{_slug_for(d['placement'])})")
    pipe.add_argument("--blocked-door-s", "--blocked_door_s",
                      dest="blocked_door_s", type=float, metavar="S",
                      help=f"blocked-door fuse (default: {d['blocked_door_s']})")
    pipe.add_argument("--static-cart-s", "--static_cart_s",
                      dest="static_cart_s", type=float, metavar="S",
                      help=f"static-cart fuse (default: {d['static_cart_s']})")
    pipe.add_argument("--abandoned-cart-s", "--abandoned_cart_s",
                      dest="abandoned_cart_s", type=float, metavar="S",
                      help="abandoned-cart fuse "
                           f"(default: {d['abandoned_cart_s']})")
    pipe.add_argument("--pose", action=argparse.BooleanOptionalAction,
                      default=None,
                      help="pose estimation. The skeleton overlay is its only "
                           "consumer, so by default it runs exactly when "
                           "something will show it: --video-path or "
                           "--case-report. --pose forces it on, --no-pose off. "
                           "(default: on with --video-path or --case-report, "
                           "otherwise off)")
    pipe.add_argument("--device", choices=("auto", "cuda", "cpu"),
                      default="auto", help="(default: auto)")

    zones = p.add_argument_group(
        "zones",
        "Without zones the threshold flags above do nothing. --zones, "
        "--auto-zones and --zone are mutually exclusive sources.")
    source = zones.add_mutually_exclusive_group()
    source.add_argument("--zones", metavar="PATH",
                        help="a zone preset JSON, as written by the demo's "
                             "Save zone set button")
    source.add_argument("--auto-zones", "--auto_zones", dest="auto_zones",
                        action="store_true",
                        help="use the newest preset saved for this clip")
    source.add_argument("--zone", action="append", metavar="SPEC",
                        help="an inline zone, repeatable: "
                             "\"name:kind:applies_to:x,y x,y x,y\". kind is one "
                             "of analytics, door, aisle, fixture, wall; "
                             "applies_to is person, cart or both and may be "
                             "omitted.")
    zones.add_argument("--zone-units", "--zone_units", dest="zone_units",
                       choices=("px", "norm"), default="px",
                       help="`norm` reads --zone vertices as 0..1 fractions of "
                            "the frame (default: px)")
    zones.add_argument("--save-zones", "--save_zones", dest="save_zones",
                       metavar="LABEL",
                       help="write the assembled zones to zone_presets/ so the "
                            "next run can use --auto-zones")

    vlm = p.add_argument_group(
        "case report (the VLM pass)",
        "Off by default: the local backend is a multi-minute GPU pass per clip.")
    vlm.add_argument("--case-report", "--case_report",
                     dest="case_report_path", nargs="?", const=None,
                     action=_PathAction, metavar="PATH",
                     help="generate AND save the case report; bare uses "
                          "<clip stem>_case_report.html, or name a file or a "
                          "directory to put it somewhere else")
    vlm.add_argument("--vlm-backend", "--vlm_backend", dest="vlm_backend",
                     choices=d["vlm_backends"], default=d["vlm_backend"],
                     metavar="BACKEND",
                     help="one of: " + "; ".join(d["vlm_backends"])
                          + f"  (default: {d['vlm_backend']})")
    vlm.add_argument("--vlm-api-key", "--vlm_api_key", dest="vlm_api_key",
                     default=None, metavar="KEY",
                     help=f"defaults to ${API_KEY_ENV}. Never echoed and never "
                          f"written to the envelope.")

    con = p.add_argument_group("console")
    con.add_argument("--quiet", "-q", action="store_true",
                     help="drop the banner, the progress bar and the per-item "
                          "status rows. Phase boundaries, the case-report "
                          "phase lines, the outcome, and everything on stderr "
                          "still print.")
    con.add_argument("--verbose", "-v", action="store_true",
                     help="echo the resolved arguments before the run, the "
                          "engine's own [VOTE]/[POPS]/[PERF] narration as it "
                          "prints, and the full traceback on a failure")
    con.add_argument("--engine-log", "--engine_log", dest="engine_log",
                     default=None, metavar="PATH",
                     help="write the engine's own narration to this file. It "
                          "is captured either way; this keeps it. A failed "
                          "run prints its last lines regardless.")
    con.add_argument("--log", choices=("pretty", "plain"), default=None,
                     help="`plain` has no carriage returns, no bars and no "
                          "colour, for log files and CI. Auto-selected when "
                          "stdout is not a terminal.")
    con.add_argument("--dry-run", "--dry_run", dest="dry_run",
                     action="store_true",
                     help="validate everything, print the resolved plan, and "
                          "run nothing")
    return p


def resolve_api_key(value: Optional[str]) -> str:
    """--vlm-api-key, falling back to the environment.

    Returned as "" rather than None when neither is set: process_video passes it
    straight through to the backend, and the local models ignore it.
    """
    if value:
        return value
    return os.environ.get(API_KEY_ENV, "") or ""


def wants_video(args: argparse.Namespace) -> bool:
    """Whether this run has any consumer for the annotated MP4.

    Only --video-path. The engine used to encode one unconditionally -- an XVID
    frame written per loop iteration, then the whole AVI decoded again and piped
    through H.264 -- and `keep_artifact` then deleted it and listed "annotated
    video" under `not_kept`. Measured on the 401-frame sample clip that is 1.6 s
    of in-loop writing plus a 2.1 s encode pass, 10% of the run, for a file with
    no reader.

    The case report does NOT count. Its evidence images come from
    FrameCapturer, which captures the drawn `im0` inside the frame loop; it has
    never read the MP4.
    """
    return args.video_path is not None


def wants_case_report(args: argparse.Namespace) -> bool:
    """Whether this run generates the case report.

    One flag asks for the report and says where it goes, the way --video-path
    does: asking for a multi-minute VLM pass and then discarding what it
    produced is not a case anyone wants, so there is nothing to separate. The
    three states are argparse's own -- absent is off, bare is on with a derived
    name, a value is on with that name.
    """
    return args.case_report_path is not None


def resolve_pose(args: argparse.Namespace) -> bool:
    """--pose, defaulted to whatever will actually display a skeleton.

    Pose is the second-largest cost in the pipeline -- 12.0 s of a 34.4 s frame
    loop on the sample clip, 35% -- and it reaches exactly one consumer,
    `draw_pose_skeleton`. Nothing in POPS, linking, the rule engine, the
    analytics or the JSON reads a keypoint, which is measurable: the same clip
    run with and without pose returns byte-identical `frames`, `events` and
    `pops_summary`.

    So the default is "on when something will show it". Two things do: the
    annotated MP4, and the case report, whose evidence images are the drawn
    frames. A JSON-only run gets none of that value and pays all of the cost.

    An explicit --pose or --no-pose still wins -- argparse leaves the flag None
    only when neither was typed.
    """
    if args.pose is not None:
        return bool(args.pose)
    return wants_video(args) or wants_case_report(args)


# ---------------------------------------------------------------------------
# Zones
#
# Three sources, none of them automatic. engine/scene_detector.py exists and
# runs MobileSAM on frame 0, so "just detect the door" looks available -- it is
# not: it returns bounding boxes labelled "Area 1".."Area 6" by mask area, with
# no semantics. A door polygon in the wrong place does not produce a quiet run,
# it produces confident blocked-door findings about a patch of floor, in a
# document that carries no hint the geometry was invented. Zone geometry stays
# operator-supplied.
# ---------------------------------------------------------------------------
#: The kinds Zone accepts. Duplicated from the ZoneKind Literal because a
#: Literal is not iterable at runtime; asserted against it in tests.
ZONE_KINDS = ("analytics", "wall", "aisle", "door", "fixture")
ZONE_APPLIES_TO = ("person", "cart", "both")


def parse_zone_spec(spec: str, index: int, frame_shape: tuple,
                    units: str = "px", *, warn=None):
    """One `--zone` value as a live `Zone`.

    Args:
        spec: `name:kind:applies_to:x,y x,y ...`. `applies_to` may be omitted.
        index: Position among the zones being built; feeds the default name and
            the colour ramp.
        frame_shape: The clip's `(h, w, ...)`, for bounds checks and for
            scaling normalized vertices.
        units: "px" for source-frame pixels, "norm" for 0..1 fractions.
        warn: Called with one string per non-fatal problem, e.g. a vertex
            outside the frame.

    Returns:
        A `Zone` built by `zone_editor.make_zone()`.

    Raises:
        ZoneSpecError: Anything malformed. On the command line these are typos,
            not stale files, so they refuse rather than degrade -- which is the
            one place this deliberately differs from `load_preset`, where an
            unknown kind is downgraded to `analytics` with a note.

    Never constructs `Zone(...)` directly. `make_zone()` mints the uuid that
    `find_zone()` matches on and derives the colour from the kind, and
    `coerce_applies_to()` forces "both" for every layout kind -- a door left at
    the "person" default matches ZERO cart tracks, so the blocked-door rule
    reports nothing while the zone looks perfectly correct. Someone typing
    `applies_to` by hand is more exposed to that than the canvas, not less.
    """
    from engine.zone_editor import coerce_applies_to, make_zone

    def fail(message: str, hint: str = "") -> "ZoneSpecError":
        return ZoneSpecError(f"--zone {index + 1}: {message}", hint)

    parts = str(spec).split(":")
    if len(parts) == 3:
        name, kind, vertex_text = parts
        applies_to = None
    elif len(parts) == 4:
        name, kind, applies_to, vertex_text = parts
    elif len(parts) > 4:
        raise fail(
            f"{len(parts)} colon-separated fields, expected 3 or 4",
            "a ':' in the zone name is not supported; the fields are "
            "name:kind:applies_to:vertices.")
    else:
        raise fail(
            f"{len(parts)} colon-separated field(s), expected 3 or 4",
            'like "Main door:door:both:512,376 544,55 842,35".')

    kind = kind.strip().lower()
    if kind not in ZONE_KINDS:
        raise fail(f'unknown kind "{kind}"',
                   "one of: " + ", ".join(ZONE_KINDS))
    if applies_to is not None:
        applies_to = applies_to.strip().lower()
        if applies_to not in ZONE_APPLIES_TO:
            raise fail(f'unknown applies_to "{applies_to}"',
                       "one of: " + ", ".join(ZONE_APPLIES_TO))

    points: list[tuple[int, int]] = []
    h, w = int(frame_shape[0]), int(frame_shape[1])
    for token in vertex_text.split():
        pair = token.split(",")
        if len(pair) != 2:
            raise fail(f'malformed vertex "{token}"', "vertices are x,y pairs "
                       "separated by spaces.")
        try:
            x, y = float(pair[0]), float(pair[1])
        except ValueError:
            raise fail(f'vertex "{token}" is not a pair of numbers') from None
        if units == "norm":
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) and warn:
                warn(f"--zone {index + 1}: vertex {token} is outside 0..1 "
                     f"with --zone-units norm.")
            x, y = x * w, y * h
        points.append((int(round(x)), int(round(y))))

    if len(points) < 3:
        raise fail(f"{len(points)} vertices, at least 3 are needed")

    outside = [p for p in points if not (0 <= p[0] < w and 0 <= p[1] < h)]
    if len(outside) == len(points):
        raise fail(
            f"every vertex is outside the {w}x{h} frame",
            "check --zone-units: px expects source-frame pixels, norm expects "
            "0..1 fractions.")
    if outside and warn:
        # A doorway polygon legitimately runs to the frame edge, so this is a
        # note, not a refusal.
        warn(f"--zone {index + 1}: {len(outside)} of {len(points)} vertices "
             f"fall outside the {w}x{h} frame.")

    return make_zone(name.strip(), points,
                     coerce_applies_to(kind, applies_to or "person"),
                     index, kind=kind)


def resolve_zones(video: Path, args: argparse.Namespace, frame_shape: tuple,
                  origin: Path, *, warn) -> tuple[list, list[str], str,
                                                  Optional[Path]]:
    """The zones for this run, and where they came from.

    Returns:
        `(zones, notes, source, path)`. `source` is one of "preset",
        "auto-preset", "inline" or "none"; `path` is the preset file when there
        was one. `notes` records what was NOT loaded -- dropped zones, unknown
        kinds, coerced applies_to -- and reaches both stderr and the envelope.

    Raises:
        ZoneSpecError: A preset that will not load, an --auto-zones with no
            preset for this clip, or a malformed --zone.

    `frame_shape` is passed to `load_preset` on purpose: it is what makes a
    preset drawn on a different resolution refuse outright instead of putting a
    doorway polygon somewhere that is not the doorway.
    """
    from engine import zone_presets

    if args.zones:
        path = _absolute(args.zones, origin)
        if not path.exists():
            raise ZoneSpecError(f"no such zone preset: {path}")
        try:
            zones, notes = zone_presets.load_preset(path, frame_shape=frame_shape)
        except zone_presets.PresetError as e:
            raise ZoneSpecError(str(e),
                                "redraw the zones for this clip in the demo, "
                                "or declare them with --zone.") from e
        return zones, notes, "preset", path

    if args.auto_zones:
        folder = zone_presets.preset_dir()
        found = zone_presets.list_presets(str(video))
        if not found:
            raise ZoneSpecError(
                f"no saved zone preset for {video.name} in {folder}",
                "draw zones in the demo and press Save zone set, or pass "
                "--zone / --zones.")
        if len(found) > 1:
            # Silently picking among several is how a run gets scored against
            # last week's door.
            warn(f"{len(found)} presets for this clip; using the newest, "
                 f'"{found[0].label}" from {found[0].when}.')
        try:
            zones, notes = zone_presets.load_preset(found[0].path,
                                                    frame_shape=frame_shape)
        except zone_presets.PresetError as e:
            raise ZoneSpecError(str(e)) from e
        return zones, notes, "auto-preset", Path(found[0].path)

    if args.zone:
        zones = [parse_zone_spec(spec, i, frame_shape, args.zone_units,
                                 warn=warn)
                 for i, spec in enumerate(args.zone)]
        return zones, [], "inline", None

    return [], [], "none", None


def resolve_placement_for_run(video: Path, args: argparse.Namespace,
                              zone_path: Optional[Path], zone_source: str,
                              *, warn) -> tuple[str, str]:
    """The camera placement this run will score with, and where it came from.

    Returns:
        `(placement, source)`. `source` is "cli", "preset" or "default", and it
        reaches both the inputs block and the envelope. It exists because
        "this store scored quiet" and "this store inherited the wrong angle"
        are the same output otherwise.

    Precedence is flag, then preset, then `CAMERA_PLACEMENT_DEFAULT`. The flag
    wins outright: somebody who types an angle has looked at the picture more
    recently than whoever saved the file, and a preset silently overriding a
    typed flag is the one behaviour nobody would predict.

    Raises:
        ZoneSpecError: The preset records a placement that is not one of the
            five. Refused rather than ignored, for the reason `_placement`
            gives: an unknown placement reads as "outside" at the INBOUND kill
            switch, so falling back would score the clip from the wrong side of
            the door and produce a quiet run rather than a failed one.
    """
    from engine import zone_presets

    if args.camera_placement is not None:
        return args.camera_placement, "cli"

    saved: Optional[str] = None
    if zone_path is not None:
        try:
            saved = zone_presets.camera_placement_of(zone_path)
        except zone_presets.PresetError as e:
            raise ZoneSpecError(
                f"{Path(zone_path).name}: {e}",
                "pass --camera-placement to override it, or resave the zones "
                "in the demo.") from e

    if saved is not None:
        # --zones takes any path, including a set saved for a different clip,
        # and load_preset does not check the recorded basename the way
        # list_presets does. Borrowing another clip's polygons is a deliberate
        # thing someone might do -- two cameras on the same doorway -- but
        # inheriting its camera angle at the same time is not, and a wrong
        # angle is silent. Said out loud rather than refused, because refusing
        # would block the deliberate case for a field the flag can override.
        if zone_source == "preset":
            try:
                recorded = zone_presets.read_preset_info(zone_path).video
            except zone_presets.PresetError:
                recorded = ""              # already reported by the read above
            if recorded and recorded != video.name:
                warn(f"--zones was drawn on {recorded}, not {video.name}; "
                     f'taking its camera placement "{saved}" as well. Pass '
                     f"--camera-placement to set it yourself.")
        return saved, "preset"

    cfg = _config()
    return cfg.CAMERA_PLACEMENT_DEFAULT, "default"


def save_zones(video: Path, label: str, zones: list, frame_shape: tuple,
               camera_placement: Optional[str] = None,
               *, warn) -> Optional[Path]:
    """Write the assembled zones back to zone_presets/ for --auto-zones.

    Same folder and same schema as the demo's Save zone set button, so the two
    tools stay interchangeable rather than each having a private zone format.

    `camera_placement` is the one this run actually scored with, whatever chose
    it. That is what closes the loop: run once with --camera-placement and
    --save-zones, and the next --auto-zones on the same clip picks up the
    angle along with the polygons, so the long command only has to be typed
    once.

    Raises:
        ZoneSpecError: The label is not usable as a filename.
    """
    from engine import zone_presets

    if not zones:
        warn("--save-zones: nothing to save, no zones were loaded.")
        return None
    try:
        return Path(zone_presets.save_preset(str(video), label, zones,
                                             frame_shape=frame_shape,
                                             camera_placement=camera_placement))
    except zone_presets.PresetError as e:
        raise ZoneSpecError(f"--save-zones: {e}") from e


# ---------------------------------------------------------------------------
# The clip
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """The clip's digest. Recorded for the same reason the golden baseline
    records it: it is the only way to be sure two JSONs describe the same
    footage."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> tuple[dict, tuple]:
    """Open the clip, read frame 0, and return `(info, frame_shape)`.

    Runs BEFORE the model loads, so a typo'd path or an unreadable file fails
    in a second rather than after the weights are on the card.

    Raises:
        InputError: Missing, a directory, empty, or not decodable video.

    Uses `zone_editor.extract_first_frame`, which already tolerates a couple of
    undecodable leading frames -- the thing that makes a perfectly good clip
    look broken.
    """
    # The three filesystem checks come FIRST, above the imports, because they
    # are free and the imports are not: engine.zone_editor pulls the package,
    # and torch and ultralytics behind it, for ~2.7 s. A typo'd filename is the
    # most common mistake there is, and it has no business waiting for a model
    # runtime to load before being told it is a typo.
    if not path.exists():
        raise InputError(f"no such file: {path}")
    if path.is_dir():
        raise InputError(f"{path} is a directory, not a clip")
    try:
        size = path.stat().st_size
    except OSError as e:
        raise InputError(f"cannot stat {path}: {e.strerror or e}") from e
    if size == 0:
        raise InputError(f"{path} is empty (0 bytes)")

    import cv2
    from engine.zone_editor import extract_first_frame

    frame = extract_first_frame(str(path))
    if frame is None:
        raise InputError(
            f"{path.name} has no decodable frames",
            "is it really a video? OpenCV could not read a frame from it.")

    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    info = {
        "path": str(path),
        "name": path.name,
        "sha256": sha256_file(path),
        "size": size,
        "width": width or int(frame.shape[1]),
        "height": height or int(frame.shape[0]),
        "fps": fps,
        "total_frames": total,
    }
    return info, frame.shape


def provenance() -> dict:
    """Everything outside the clip that can change the numbers.

    The same four fields tests/make_golden_baseline.py records, assembled the
    same way -- `environment_fingerprint()` returns a "|"-joined string, not a
    dict, so it is parsed rather than used directly.
    """
    from engine.trajectory_cache import environment_fingerprint

    parts = {}
    for chunk in environment_fingerprint().split("|"):
        key, _, value = chunk.partition("=")
        parts[key] = value
    try:
        import ultralytics
        version = ultralytics.__version__
    except Exception:                     # pragma: no cover - reported, not fatal
        version = parts.get("ultralytics", "unknown")
    return {
        "ultralytics": version,
        "weights": parts.get("weights"),
        "tracker_cfg": parts.get("tracker_cfg"),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Console: phases driven by the engine's own progress events
# ---------------------------------------------------------------------------
#: The engine's five `desc=` strings, mapped to the headings printed for them.
#: "Building tracking JSON" and "Rendering panels" share one heading on purpose:
#: they are consecutive, both are post-loop bookkeeping, and folding them is
#: what keeps the phase count at 5 without a case report and 6 with one -- so
#: the denominator never promises a phase that will not run.
#:
#: These strings are engine internals. tests/test_cli_console.py scans
#: engine/tracker.py for all five, so a rename fails a test instead of silently
#: degrading this console into one undifferentiated phase.
PHASE_HEADINGS = {
    "Processing frames": "Detection and tracking",
    "Computing analytics": "Analytics and rules",
    "Encoding video": "Annotated video and heat map",
    "Building tracking JSON": "Tracking JSON and panels",
    "Rendering panels": "Tracking JSON and panels",
}

#: Phases this file prints itself, around the engine's.
PHASE_INPUTS = "Inputs"
PHASE_CASE_REPORT = "Case report"


def phase_total(case_report: bool, video: bool = True) -> int:
    """How many phases this run will have: Inputs, the engine's distinct
    headings, and optionally the case report.

    `video` False drops "Annotated video and heat map": with write_video off the
    engine never emits the "Encoding video" description, so counting it would
    leave the denominator promising a phase that cannot run, the exact thing
    PHASE_HEADINGS' own note says the fold of the last two headings exists to
    avoid.
    """
    headings = set(PHASE_HEADINGS.values())
    if not video:
        headings.discard(PHASE_HEADINGS["Encoding video"])
    return 1 + len(headings) + (1 if case_report else 0)


class PhaseReporter:
    """The `progress=` callable handed to `process_video()`.

    `process_video` reports as `progress(frac_or_(done, total), desc=...)`, and
    the five descriptions arrive in a fixed order. This opens a new console
    phase whenever the heading changes and closes the previous one with its
    MEASURED elapsed time -- so phase timings are timed, never estimated.

    An unrecognised description opens a phase using the raw string as its
    heading and widens the denominator. Information is never dropped because a
    heading was not in the map.

    A callable object rather than a closure so `index`, `total` and the open
    phase's start time are inspectable from a test.
    """

    def __init__(self, total: int, quiet: bool = False, index: int = 0):
        self.total = total
        self.index = index
        self.quiet = quiet
        self.heading: Optional[str] = None
        self.started = 0.0
        #: Seconds spent in each heading, in the order they opened.
        self.timings: list[tuple[str, float]] = []

    def open(self, heading: str) -> None:
        """Close whatever phase is running and start `heading`."""
        if heading == self.heading:
            return
        self.close()
        self.index += 1
        if self.index > self.total:
            # An unmapped description, or a case report nobody counted. Widen
            # rather than print [7/6].
            self.total = self.index
        self.heading = heading
        self.started = time.perf_counter()
        ui.phase(self.index, self.total, heading, rule=False)

    def close(self) -> None:
        """End the open phase, printing how long it took."""
        if self.heading is None:
            return
        elapsed = time.perf_counter() - self.started
        self.timings.append((self.heading, elapsed))
        ui.progress_end()
        ui.phase_end(elapsed)
        self.heading = None

    def __call__(self, value=None, desc=None, total=None, unit=None,
                 _tqdm=None, **kwargs) -> None:
        """Accepts gr.Progress's call signature; ignores what it does not use."""
        if desc:
            heading = PHASE_HEADINGS.get(desc, desc)
            self.open(heading)
        if self.quiet:
            return
        # A (done, total) tuple is the frame loop; a float is a phase marker
        # with nothing to count.
        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                done, count = int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return
            ui.progress_bar(done, count, "frames")


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------
#: Engine output slots carrying rendered HTML, mapped to their key in the
#: envelope's `html` block. Everything the run rendered -- a caller building a
#: report should not have to re-implement ui_builder. `json_output` is excluded
#: because it is the tracking document, which has its own block, and
#: `heatmap_image` because it is an ndarray whose readable form is the PNG.
HTML_SLOTS = {
    "video_info_html": "video_info",
    "detection_html": "detection",
    "config_html": "config",
    "legend_html": "legend",
    "pops_html": "pops",
    "events_html": "events",
    "case_report_html": "case_report",
    "analytics_summary_html": "analytics_summary",
    "spikes_html": "spikes",
    "dwell_html": "dwell",
    "journey_html": "journey",
    "alert_banner_html": "alert_banner",
    "ops_alerts_html": "ops_alerts",
    "run_summary_html": "run_summary",
    "tab_counts_html": "tab_counts",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")


def zones_block(zones: Iterable, notes: list[str], source: str,
                path: Optional[Path], units: str) -> dict:
    """`run.zones` -- the polygons themselves, not a record of what was asked for.

    A record of what was requested cannot explain a finding; a record of what
    ran can. `units` is here because --zone-units norm deliberately bypasses
    load_preset's frame-size refusal, and that trade has to be visible
    afterwards.
    """
    import numpy as np

    listed = list(zones)
    return {
        "source": source,
        "path": str(path) if path else None,
        "units": units,
        "count": len(listed),
        "notes": list(notes),
        "zones": [{
            "name": z.name,
            "kind": getattr(z, "kind", "analytics"),
            "applies_to": z.applies_to,
            "polygon": [[int(x), int(y)]
                        for x, y in np.asarray(z.polygon).tolist()],
        } for z in listed],
    }


def analytics_block(result) -> Optional[dict]:
    """`analytics` -- only the structured data that had NO JSON representation
    before `TrackingEngine._last_analytics` existed.

    The journey matrix, edges, labels, dwell rows, insight text and rule
    diagnostics. Nothing here is duplicated from `tracking`, which already
    carries rule_findings, rule_engine, operational_highlights, queue_spikes and
    dwell_summary.

    `heatmap_array` and `heatmap_composite` are excluded and that is not
    conditional on --heatmap-path: they are a (H, W) float array and a
    (H, W, 3) uint8 image, megabytes of numbers whose readable form is the PNG.
    A wall of raw densities is not a substitute for the picture, so there is no
    "put it in the JSON instead" fallback.

    Returns None when the run left no analytics behind, which is the honest
    answer for a run that failed before analytics ran.
    """
    if result is None:
        return None
    matrix = getattr(result, "journey_matrix", None)
    return {
        "journey_labels": list(getattr(result, "journey_labels", []) or []),
        "journey_matrix": matrix.tolist() if matrix is not None else None,
        "journey_edges": [dataclasses.asdict(e)
                          for e in getattr(result, "journey_edges", []) or []],
        "dwell_rows": [dataclasses.asdict(r)
                       for r in getattr(result, "dwell_rows", []) or []],
        "insight_text": getattr(result, "insight_text", "") or "",
        "rule_diagnostics": list(getattr(result, "rule_diagnostics", []) or []),
    }


def build_envelope(*, argv: list[str], started_at: str, finished_at: str,
                   wall_s: float, video_s: Optional[float],
                   case_report_s: Optional[float], exit_state: str,
                   run: dict, tracking: Optional[dict] = None,
                   analytics: Optional[dict] = None,
                   case_report: Optional[dict] = None,
                   artifacts: Optional[dict] = None,
                   html: Optional[dict] = None,
                   error: Optional[dict] = None) -> dict:
    """Assemble the document. Every block is optional except `cli` and `run`,
    so a failed run writes the same shape with the parts it got to."""
    envelope: dict[str, Any] = {
        "schema": SCHEMA,
        "cli": {
            "argv": list(argv),
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": round(wall_s, 3),
            "video_seconds": round(video_s, 3) if video_s is not None else None,
            "case_report_seconds": (round(case_report_s, 3)
                                    if case_report_s is not None else None),
            "exit": exit_state,
        },
        "run": run,
    }
    if error is not None:
        envelope["cli"]["error"] = error
    envelope["tracking"] = tracking
    envelope["analytics"] = analytics
    envelope["case_report"] = case_report
    envelope["artifacts"] = artifacts or {}
    if html is not None:
        envelope["html"] = html
    return envelope


def write_envelope(path: Path, envelope: dict, indent: int) -> int:
    """Write the envelope atomically. Returns the byte count.

    Write-then-replace, the same way zone_presets.save_preset does, and it is
    load-bearing rather than tidy: the overwrite carve-out in `is_tombstone`
    needs a failure envelope to PARSE as JSON. A Ctrl-C or a full disk part-way
    through a direct write would leave a truncated file that does not, which
    would then require --force -- exactly the sequence the carve-out exists to
    prevent.

    Raises:
        OutputPathError: The file could not be written.
    """
    text = json.dumps(envelope, indent=indent or None, default=str)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise OutputPathError(f"could not write {path}: {e.strerror or e}",
                              "check the path and the free space on it.") from e
    return len(text.encode("utf-8"))


def keep_artifact(produced: Optional[str], target: Optional[Path], name: str,
                  not_kept: list[str], *, warn) -> Optional[str]:
    """Copy one engine-produced file to where it was asked for.

    Three states, because "no file" has more than one meaning and a reader must
    be able to tell them apart:

      * a path        -- it was produced and kept
      * None, listed in `not_kept` -- produced, and no path flag asked for it
      * None, absent from `not_kept` -- never produced

    That last state has two causes and the envelope does not distinguish them,
    so do not read it as failure. For the heat map it still means what it always
    did: the engine tried and could not. For the annotated video it is now the
    NORMAL case -- since the encode is skipped entirely without --video-path
    (see wants_video), a default run produces no MP4 and therefore lists none
    and reports none. `run.enable_pose` is the field that distinguishes the two
    runs from outside: an encode that was never attempted also never ran pose.

    A copy, not a move. The engine's run directory prunes itself, and moving a
    file out from under a path the engine returned invites a surprise if
    anything downstream reads it. copy2 also handles the cross-drive case that
    is normal here (%TEMP% on C:, output on D:) and that os.replace would not.
    """
    if not produced or not os.path.exists(produced):
        return None
    if target is None:
        not_kept.append(name)
        return None
    try:
        shutil.copy2(produced, target)
    except OSError as e:
        # A finished run must not be lost over a copy. The engine's own file is
        # still where it was; say so and carry on.
        warn(f"could not keep {name}: {e.strerror or e} "
             f"(the engine's copy is still at {produced})")
        not_kept.append(name)
        return None
    return str(target)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Plan:
    """Everything validated before a single weight loads.

    Built by `preflight()` so a typo costs a second rather than four minutes,
    and so `--dry-run` can print exactly what a real run would do.
    """
    video: Path
    video_info: dict
    frame_shape: tuple
    zones: list
    zone_notes: list[str]
    zone_source: str
    zone_path: Optional[Path]
    #: The placement this run scores with, and which of "cli", "preset" or
    #: "default" produced it. Settled here rather than read off `args` at each
    #: of the three use sites, for the reason `resolve_pose` gives: one
    #: resolution point cannot disagree with itself.
    camera_placement: str
    camera_placement_source: str
    thresholds: dict
    outputs: dict                       # dest -> Path or None
    warnings: list[str]


def preflight(args: argparse.Namespace, origin: Path, note=None) -> Plan:
    """Validate every input and resolve every output path.

    Raises:
        InputError, ZoneSpecError, OutputPathError: see each.

    Order matters. The clip is opened first because it is the cheapest check and
    the most common mistake; zones need its frame size; the overwrite checks run
    last but still before the model loads, so a run cannot spend four minutes
    and then refuse to write.

    That ordering is why nothing is imported out of engine/ at the top of this
    function any more. `from engine.rules import resolve_thresholds` used to sit
    there, which meant the ~2.7 s engine load happened before the free checks
    rather than after them, and `gk_pops.py nosuchfile.mp4` took three and a
    half seconds to say "no such file". Each import now sits at its use site.

    Args:
        note: Called once with a short string just before the first thing that
            loads torch, so the console can explain the pause instead of going
            quiet under a freshly printed banner. Optional -- a caller that does
            not pass one simply gets no line.
    """
    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(message)

    video = _absolute(args.video, origin)
    # After _absolute and before probe_video's decode: probe_video runs its own
    # filesystem checks before it imports anything, so a missing clip still
    # fails without the wait this line is announcing.
    if note is not None and video.exists() and not video.is_dir():
        note("loading torch and the model runtime")
    video_info, frame_shape = probe_video(video)

    zones, notes, source, zone_path = resolve_zones(
        video, args, frame_shape, origin, warn=warn)

    # After the zones, because the preset they came from is where an unflagged
    # placement is read from.
    placement, placement_source = resolve_placement_for_run(
        video, args, zone_path, source, warn=warn)

    from engine.rules import resolve_thresholds
    thresholds = resolve_thresholds({
        "blocked_door_s": args.blocked_door_s,
        "static_cart_s": args.static_cart_s,
        "abandoned_cart_s": args.abandoned_cart_s,
    })

    # The threshold flags are inert without zones, and a run that exits 0 with
    # an empty findings list is how that goes unnoticed.
    if not zones and any(v is not None for v in (args.blocked_door_s,
                                                 args.static_cart_s,
                                                 args.abandoned_cart_s)):
        warn("threshold flags were passed but no zones were loaded; the rule "
             "engine reads door zones for the blocked-door family and "
             "aisle/analytics for static-cart, so it will report nothing. "
             "Pass --zones, --auto-zones or --zone.")

    values = {
        "json_path": args.json_path if args.json_path is not None else DERIVE,
        "video_path": args.video_path,
        "heatmap_path": args.heatmap_path,
        "engine_json_path": args.engine_json_path,
        "case_report_path": args.case_report_path,
    }

    outputs = {dest: resolve_output_path(value, dest, video, origin, warn=warn)
               for dest, value in values.items()}

    for dest, path in outputs.items():
        check_overwrite(path, dest.replace("_", "-"), args.force)

    return Plan(video=video, video_info=video_info, frame_shape=frame_shape,
                zones=zones, zone_notes=notes, zone_source=source,
                zone_path=zone_path, camera_placement=placement,
                camera_placement_source=placement_source,
                thresholds=thresholds, outputs=outputs, warnings=warnings)


def on_stderr():
    """Print a console_ui block to stderr instead of stdout.

    The phases and the outcome belong on the stream a caller redirects to a
    log. Notes, warnings and errors do not: they are what an operator has to
    NOTICE, and a run whose only complaint scrolled past in a redirected stdout
    has effectively not complained.

    Redirects console_ui rather than sys.stdout, because sys.stdout is the
    engine's sink for most of a run (see EngineLog) and a warning raised from
    inside that span would otherwise be swallowed by it. ui.stream() restores
    the previous console_ui stream, not the process's, so the nesting holds.
    """
    return ui.stream(sys.stderr)


# ---------------------------------------------------------------------------
# The engine's own output
# ---------------------------------------------------------------------------
#: How many captured lines are kept in memory for the failure dump. The full
#: stream still reaches --engine-log and -v; this bound only limits what a
#: crash prints back to the terminal, and what a 30-minute clip can cost in RAM.
ENGINE_LOG_LINES = 4000

#: How many of those are printed when a run fails. Enough to carry the last
#: model load, the last frame and whatever raised; not so many that the problem
#: block scrolls off the top of the window.
ENGINE_LOG_TAIL = 40

#: Longest run of engine output with no newline in it before it is banked as
#: a line anyway. The ring below bounds whole lines; this bounds the fragment
#: still being assembled, so a writer that never emits one cannot escape it.
_PARTIAL_MAX = 64 * 1024

#: The EngineLog of the run in progress, so the closing block can report how
#: much was suppressed. A module global rather than an argument because the
#: capture is opened in main(), around run(), and threading it through six
#: call sites to print one status row is not worth the churn.
_ENGINE_LOG = None


class EngineLog:
    """Stands in for sys.stdout while the engine runs.

    The engine narrates itself to stdout -- [VOTE] per cart, [POPS] per score,
    [PERF] breakdowns, [CACHE] keys, a 45-entry classification history. That is
    the right amount of detail for someone debugging a vote and the wrong
    amount for someone reading a result, and worse, it arrives DURING the frame
    loop, so it lands in the middle of the redrawn progress bar.

    Nothing in engine/ is changed to achieve this. Every print still happens,
    every frame is still processed and still logged; this object decides where
    the text goes. That keeps tests/test_cli_parity.py honest -- the engine
    behaves identically whether it is driven by this CLI or by the app.

    Where it goes:

      * -v / --verbose echoes it straight through, unchanged and interleaved,
        exactly as it printed before this class existed.
      * --engine-log PATH writes all of it to a file.
      * a failed run prints the last ENGINE_LOG_TAIL lines, always. Suppressing
        output by default is only safe if a crash still leaves its breadcrumbs.
      * otherwise it is counted and dropped, and the count is reported.

    Duck-types a text stream well enough for torch, ultralytics and OpenCV,
    which probe it. `fileno()` deliberately raises rather than returning the
    real descriptor: handing out fd 1 would let a C-level writer bypass this
    object entirely, which is the leak it exists to close.
    """

    encoding = "utf-8"
    errors = "replace"
    closed = False

    def __init__(self, echo=None, path: Optional[Path] = None,
                 limit: int = ENGINE_LOG_LINES):
        #: A stream to pass everything through to, for --verbose.
        self.echo = echo
        self.limit = limit
        self.lines: list[str] = []
        #: Lines dropped off the front of the ring, so the count stays honest
        #: on a long clip.
        self.dropped = 0
        self._partial = ""
        #: Where --engine-log is writing, for the closing block to name.
        self.path = path
        self._file = None
        if path is not None:
            self._file = open(path, "w", encoding="utf-8", errors="replace")

    # -- the stream protocol ------------------------------------------------
    def write(self, text: str) -> int:
        if self.echo is not None:
            self.echo.write(text)
        if self._file is not None:
            self._file.write(text)
        self._partial += text
        if "\n" in self._partial:
            *whole, self._partial = self._partial.split("\n")
            for line in whole:
                self._append(line)
        elif len(self._partial) > _PARTIAL_MAX:
            # A writer that emits only carriage returns -- a progress bar of
            # its own -- would otherwise grow this forever while self.lines
            # stays bounded, which defeats the ring. Bank it as a line.
            self._append(self._partial)
            self._partial = ""
        return len(text)

    def flush(self) -> None:
        for stream in (self.echo, self._file):
            if stream is not None:
                stream.flush()

    def isatty(self) -> bool:
        #: False, so a library that formats differently for a terminal picks
        #: the plain branch -- there is no terminal on this end of the pipe.
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        import io
        raise io.UnsupportedOperation("EngineLog has no file descriptor")

    # -- what we do with it -------------------------------------------------
    def _append(self, line: str) -> None:
        self.lines.append(line)
        if len(self.lines) > self.limit:
            del self.lines[:len(self.lines) - self.limit]
            self.dropped += 1

    @property
    def count(self) -> int:
        """Every line the engine printed, including any rotated out."""
        return len(self.lines) + self.dropped

    def close(self) -> None:
        if self._partial:
            self._append(self._partial)
            self._partial = ""
        if self._file is not None:
            self._file.close()
            self._file = None

    def tail(self, n: int = ENGINE_LOG_TAIL) -> list[str]:
        return [line for line in self.lines[-n:] if line.strip()]


def _clip(line: str) -> str:
    """One engine line, cut to the width of a detail() row.

    Cut rather than wrapped, and marked where it was cut. These lines are a
    45-entry classification history and a [VOTE] dict; wrapping one fills the
    screen with the least useful part of a failure report, and truncating it
    silently would let a reader mistake a fragment for the whole line. -v and
    --engine-log are where the untouched text lives.
    """
    room = ui.WIDTH - len(ui.PAD) - 1
    return line if len(line) <= room else line[:room - 1] + "\u2026"


@contextlib.contextmanager
def engine_output(args: argparse.Namespace, path: Optional[Path] = None):
    """Send the engine's prints to an EngineLog and keep ours on the terminal.

    Two swaps, and they are not the same swap. sys.stdout becomes the sink, so
    anything in engine/ that prints is captured. console_ui is PINNED to the
    real terminal, so the phase headings and the frame counter -- which are
    emitted by the progress callback, from inside the engine's own call stack --
    still reach the screen. Without the pin, console_ui would follow sys.stdout
    into the sink and the run would print nothing at all.

    On the way out the last lines are printed if the block raised: default
    suppression is only defensible while a failure still says what it saw.
    """
    global _ENGINE_LOG
    terminal = sys.stdout
    log = EngineLog(echo=terminal if args.verbose else None, path=path)
    ui.set_stream(terminal)
    sys.stdout = log
    _ENGINE_LOG = log
    for text in _PREAMBLE:
        if text:
            log.write(text)
    failed = False
    try:
        yield log
    except BaseException as e:
        # A usage error -- a path that already exists, a zone file that will
        # not load -- is raised by preflight, before the engine has done
        # anything. There are no breadcrumbs to show, and the message already
        # says precisely what is wrong; a tail underneath it is noise.
        failed = not (isinstance(e, CliError) and e.exit_code == EXIT_USAGE)
        raise
    finally:
        sys.stdout = terminal
        ui.set_stream(None)
        _ENGINE_LOG = None
        log.close()
        if failed and not args.verbose and log.tail():
            tail = log.tail()
            warn_row("engine output",
                     f"the last {len(tail)} of {log.count} captured lines")
            with on_stderr():
                for line in tail:
                    ui.detail(_clip(line))


#: Where a note() line starts, so a path put there can be measured against it.
_NOTE_INDENT = len(ui.PAD) + 1 + ui.BADGE_W + 2 + ui.LABEL_W

#: The folder printed once at the top of the run, that every path below it is
#: then printed relative to. A module global because it is one fact about the
#: whole run; threading it through every row would say nothing extra.
_ROOT = None


def _short(path, width: int = 0) -> str:
    """A path for a status row: relative to _ROOT, elided to fit the column."""
    return ui.shorten(path, _ROOT, width or VALUE_W)


def ok_path(label: str, path) -> None:
    """A row whose value is a path that must stay whole and absolute.

    Only the run's root uses this. Every other path is printed relative to it
    and may be elided; this one is what those are relative TO, so eliding it
    would make the rest unresolvable.
    """
    text = str(path)
    fits = len(text) <= VALUE_W
    ui.ok(label, text if fits else "")
    if not fits:
        ui.note(text)


#: Longest value a status row can hold before the line runs past WIDTH. The
#: label column is fixed, so this is simply what is left of it.
VALUE_W = ui.WIDTH - len(ui.PAD) - 1 - ui.BADGE_W - 2 - ui.LABEL_W


def warn_row(label: str, message: str) -> None:
    """A warning, on stderr, without saying it twice.

    Every one of these used to print `message[:48]` in the row and then the
    whole message again underneath it, so a long warning appeared as a
    truncated fragment followed by its own full text. A message that fits goes
    in the row; one that does not leaves the row to the label and goes
    underneath in full, where note() wraps it on word boundaries.
    """
    with on_stderr():
        fits = len(message) <= VALUE_W
        ui.warn(label, message if fits else "")
        if not fits:
            ui.note(message)


def zone_mix(zones: Iterable) -> str:
    """`3 loaded · 1 door · 2 analytics`, for the Inputs phase."""
    listed = list(zones)
    if not listed:
        return "none"
    counts: dict[str, int] = {}
    for z in listed:
        kind = getattr(z, "kind", "analytics")
        counts[kind] = counts.get(kind, 0) + 1
    parts = [f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f"{len(listed)} loaded  {ui.G['sep']}  " + f"  {ui.G['sep']}  ".join(parts)


def print_inputs(plan: Plan, args: argparse.Namespace, reporter: PhaseReporter,
                 api_key: str) -> None:
    """The [1/n] Inputs phase: what this run is about to do, before it does it.

    Under --quiet the status rows are dropped and the phase heading, the zone
    notes and the warnings are not. The rows are a description of a run that has
    not gone wrong yet; the notes and warnings are the two things whose absence
    would let a degraded run read as a clean one.
    """
    global _ROOT
    info = plan.video_info
    reporter.open(PHASE_INPUTS)
    # The project. zone_presets/, sample_videos/ and weights/ all hang off it,
    # so it is the prefix that actually repeats down the run. An output written
    # somewhere else entirely -- another drive, a scratch folder -- is left
    # absolute and elided in the middle instead.
    _ROOT = REPO_ROOT

    if not reporter.quiet:
        duration = info["total_frames"] / info["fps"] if info["fps"] else 0.0
        sep = f" {ui.G['sep']} "
        ok_path("project", _ROOT)
        ui.ok("clip", _short(plan.video))
        ui.ok("video", sep.join([
            f"{info['width']}x{info['height']}",
            f"{info['fps']:.1f} fps",
            f"{info['total_frames']} frames",
            f"{duration:.1f}s"]))
        # The source is on the line because an unrecognised placement is read
        # as "outside" downstream, so "which angle" and "who chose it" are the
        # two halves of the same question.
        ui.ok("camera placement",
              f"{plan.camera_placement}{sep}from the zone preset"
              if plan.camera_placement_source == "preset"
              else f"{plan.camera_placement}{sep}built-in default"
              if plan.camera_placement_source == "default"
              else plan.camera_placement)
        (ui.ok if plan.zones else ui.warn)("zones", zone_mix(plan.zones))
        if plan.zone_path:
            ui.note(_short(plan.zone_path, ui.WIDTH - _NOTE_INDENT))
        ui.ok("thresholds", sep.join([
            f"door {plan.thresholds['blocked_door_s']:g}s",
            f"static {plan.thresholds['static_cart_s']:g}s",
            f"abandoned {plan.thresholds['abandoned_cart_s']:g}s"]))
        # "auto" is the question, not the answer. The row is printed after the
        # engine has been built and resolved it -- except on a dry run, which
        # never builds one, where the request is the only device fact there is.
        if args.device != "auto" or args.dry_run:
            ui.ok("device",
                  args.device + ("" if args.pose else "  (pose off)"))
        if wants_case_report(args):
            ui.ok("case report", args.vlm_backend
                  + ("  (api key set)" if api_key else ""))
        # "writes " is not decoration: the row above is already labelled
        # video, and an unprefixed video_path row reads as a second statement
        # about the INPUT clip.
        for dest, path in plan.outputs.items():
            if path is not None:
                ui.ok("writes " + dest.replace("_path", "").replace("_", " "),
                      _short(path))

    # The record of what was NOT loaded -- dropped zones, unknown kinds, a
    # coerced applies_to. Also in the envelope under run.zones.notes; on stderr
    # here because it is the difference between "no findings" and "the geometry
    # you thought was loaded was not".
    for note in plan.zone_notes:
        warn_row("zone note", note)
    for message in plan.warnings:
        warn_row("check", message)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _run_block(args: argparse.Namespace, plan: Plan, device: str,
               argv: Sequence[str]) -> dict:
    """`run` -- what actually ran, not what was asked for.

    Thresholds AFTER resolve_thresholds(), the device the engine chose after
    "auto", and the zone polygons themselves.
    """
    return {
        "video": plan.video_info,
        "camera_placement": plan.camera_placement,
        # "cli", "preset" or "default". Recorded because "this store scored
        # quiet" and "this store inherited the wrong angle" are the same
        # output otherwise, and a placement can now come from a file nobody
        # looked at during this run.
        "camera_placement_source": plan.camera_placement_source,
        "enable_pose": bool(args.pose),
        "device": device,
        "rule_thresholds": dict(plan.thresholds),
        "zones": zones_block(plan.zones, plan.zone_notes, plan.zone_source,
                             plan.zone_path, args.zone_units),
        "vlm": {"case_report": wants_case_report(args),
                "backend": (args.vlm_backend if wants_case_report(args)
                            else None)},
        "provenance": provenance(),
    }


def run(args: argparse.Namespace, argv: list[str]) -> int:
    """Validate, run the pipeline, write the envelope. Returns the exit code.

    Raises:
        CliError: Any failure this tool reports rather than tracebacks. main()
            turns it into a message and a status.
    """
    origin = Path.cwd()
    started_at = _now()
    t_start = time.perf_counter()

    # The phase is opened HERE, not inside print_inputs, because preflight()
    # below is the first thing in the process to touch engine/ -- and torch,
    # torchvision and ultralytics behind it are ~2.7 s on a warm cache. The
    # banner is on screen by now (see _config() for why it is no longer three
    # seconds late), so without this the run shows a header and then stops dead
    # with nothing to say what it is doing. print_inputs() opens the same
    # heading and PhaseReporter.open() is a no-op when the heading has not
    # changed, so this adds a line, not a phase.
    reporter = PhaseReporter(phase_total(wants_case_report(args),
                                         wants_video(args)),
                             quiet=args.quiet)
    reporter.open(PHASE_INPUTS)
    plan = preflight(args, origin,
                     note=None if args.quiet else ui.detail)

    # AFTER preflight: every user-supplied path is absolute by now, and
    # engine/config.py's MODEL_PATH and TRACKER_CONFIG are repo-root-relative.
    os.chdir(REPO_ROOT)

    from engine import TrackingEngine
    from engine.cancellation import RunCancelled, RunSuperseded
    from engine.config import JSON_EVERY_N_FRAMES
    from engine.run_outputs import as_dict

    json_target = plan.outputs["json_path"]
    warnings = list(plan.warnings)

    def warn(message: str) -> None:
        """A degraded-but-continuing outcome. Never suppressed by --quiet."""
        warnings.append(message)
        warn_row("warning", message)

    def fail_envelope(state: str, error: Optional[dict], video_s, case_s,
                      device: str) -> None:
        """Write what we have. A batch or a retry loop needs the failure on disk
        next to the successes, not only in a terminal nobody is watching."""
        envelope = build_envelope(
            argv=argv, started_at=started_at, finished_at=_now(),
            wall_s=time.perf_counter() - t_start, video_s=video_s,
            case_report_s=case_s, exit_state=state,
            run=_run_block(args, plan, device, argv),
            error=error,
            artifacts={"json": str(json_target), "annotated_video": None,
                       "heatmap_png": None, "case_report_html": None,
                       "engine_json": None, "not_kept": []})
        write_envelope(json_target, envelope, args.indent)

    api_key = resolve_api_key(args.vlm_api_key)
    print_inputs(plan, args, reporter, api_key)

    if args.dry_run:
        reporter.close()
        ui.outcome(f"Dry run  {ui.G['sep']}  nothing was executed", "", [])
        ui.detail("Every input validated and every output path resolved. "
                  "Drop --dry-run to execute.")
        return EXIT_OK

    if args.save_zones:
        saved = save_zones(plan.video, args.save_zones, plan.zones,
                           plan.frame_shape, plan.camera_placement, warn=warn)
        if saved and not args.quiet:
            ui.ok("zones saved", _short(saved))

    # Inside its own guard: loading the weights is the first expensive thing
    # that happens, and the common way it fails is a git-lfs pointer file, whose
    # torch error names neither git nor lfs. An unguarded raise here would leave
    # main() with an exception it does not recognise and no envelope on disk.
    try:
        engine = TrackingEngine(device=args.device)
    except Exception as e:
        device = args.device
        fail_envelope("error", {"type": type(e).__name__, "message": str(e),
                                "traceback": traceback.format_exc()},
                      None, None, device)
        raise PipelineFailed(
            f"the engine could not be built: {type(e).__name__}: {e}",
            "the usual cause is model weights that are git-lfs pointer files "
            "rather than the models. Run `git lfs install && git lfs pull`. "
            f"A failure envelope was written to {json_target}.") from e
    device = getattr(engine, "device", args.device)
    if not args.quiet and args.device == "auto":
        ui.ok("device", str(device) + ("" if args.pose else "  (pose off)"))

    video_s: Optional[float] = None
    case_s: Optional[float] = None
    t_video = time.perf_counter()
    try:
        result = engine.process_video(
            str(plan.video),
            camera_placement=plan.camera_placement,
            vlm_backend=args.vlm_backend,
            vlm_api_key=api_key,
            zones=list(plan.zones),
            # Always deferred, whether or not a report is wanted. Off means the
            # payload is stashed and never consumed, so no VLM runs at all --
            # the path tests/test_golden_clip.py already exercises. On means the
            # report is a SEPARATE, announced step, so the wait after the video
            # is explainable rather than one opaque blocking call.
            defer_case_report=True,
            rule_thresholds={
                "blocked_door_s": args.blocked_door_s,
                "static_cart_s": args.static_cart_s,
                "abandoned_cart_s": args.abandoned_cart_s,
            },
            enable_pose=bool(args.pose),
            # The annotated MP4 is encoded only when --video-path asked for
            # one. It used to be produced unconditionally and then deleted by
            # keep_artifact below: an XVID frame per loop iteration plus a full
            # decode-and-re-encode pass, 10% of the run, for a file with no
            # reader. Nothing else changes: frames are still drawn, so the case
            # report's evidence images are unaffected.
            write_video=wants_video(args),
            progress=reporter,
        )
    except (RunCancelled, RunSuperseded) as e:
        # Caught by name. This CLI never supersedes itself, but process_video
        # raises these from its own checkpoints and a traceback would
        # misdescribe what happened.
        video_s = time.perf_counter() - t_video
        reporter.close()
        fail_envelope("cancelled", {"type": type(e).__name__, "message": str(e),
                                    "traceback": None}, video_s, None, device)
        raise RunInterrupted(f"the run was cancelled: {e}") from e
    except KeyboardInterrupt:
        video_s = time.perf_counter() - t_video
        reporter.close()
        fail_envelope("cancelled", {"type": "KeyboardInterrupt",
                                    "message": "interrupted at the keyboard",
                                    "traceback": None}, video_s, None, device)
        raise RunInterrupted("interrupted") from None
    except Exception as e:
        video_s = time.perf_counter() - t_video
        reporter.close()
        fail_envelope("error", {"type": type(e).__name__, "message": str(e),
                                "traceback": traceback.format_exc()},
                      video_s, None, device)
        raise PipelineFailed(
            f"the run failed: {type(e).__name__}: {e}",
            f"a failure envelope was written to {json_target}; re-running "
            f"overwrites it without --force.") from e

    video_s = time.perf_counter() - t_video
    out = as_dict(result)
    reporter.close()


    # --- The case report ---------------------------------------------------
    # With defer, the engine's case_report_html slot holds an animated
    # "Generating case report…" block. That is a live UI placeholder and a false
    # statement in a finished document, so it is ALWAYS replaced -- guarded on
    # the flag, never on the string's content.
    case_report: dict[str, Any] = {"generated": False, "backend": None,
                                   "seconds": None, "html_path": None}
    case_report_html: Optional[str] = None
    case_report_file: Optional[str] = None
    if wants_case_report(args):
        # These two lines survive --quiet deliberately. They are the answer to
        # "why is this still running": the engine emits no progress at all
        # during the VLM pass, so announcing the phase boundary is the only
        # honest way to report the wait. Decoration would be the frame counter;
        # this is not that.
        reporter.open(PHASE_CASE_REPORT)
        ui.detail("the frame work above is finished; this phase is the VLM "
                  "alone, and it emits no progress")
        ui.ok("model", args.vlm_backend)
        t_case = time.perf_counter()
        try:
            case_report_html, case_report_file = engine.finalize_case_report()
        except (RunCancelled, RunSuperseded) as e:
            case_report = {"generated": False, "backend": args.vlm_backend,
                           "seconds": round(time.perf_counter() - t_case, 3),
                           "html_path": None, "error": str(e)}
            warn(f"the case report was cancelled: {e}")
        else:
            case_s = time.perf_counter() - t_case
            # finalize_case_report catches a generation failure itself and
            # returns a red banner with a None file. Either way that is a report
            # which did not generate -- recorded, nothing written, and the run
            # still exits 0. A missing case report never costs a completed run
            # its POPS, rules and analytics.
            generated = bool(case_report_file)
            case_report = {"generated": generated,
                           "backend": args.vlm_backend,
                           "seconds": round(case_s, 3),
                           "html_path": None}
            if not generated:
                case_report["error"] = _text_of(case_report_html)
                warn("the case report did not generate: "
                     + (case_report["error"] or "no reason given"))
        reporter.close()
    else:
        case_report_html = None

    # --- Keep what was asked for ------------------------------------------
    not_kept: list[str] = []
    artifacts = {
        "json": str(json_target),
        "annotated_video": keep_artifact(out["video_output"],
                                         plan.outputs["video_path"],
                                         "video", not_kept, warn=warn),
        "heatmap_png": keep_artifact(out["heatmap_file"],
                                     plan.outputs["heatmap_path"],
                                     "heatmap", not_kept, warn=warn),
        "engine_json": keep_artifact(out["json_download"],
                                     plan.outputs["engine_json_path"],
                                     "engine-json", not_kept, warn=warn),
        "case_report_html": keep_artifact(case_report_file,
                                          plan.outputs["case_report_path"],
                                          "case-report", not_kept, warn=warn),
        "not_kept": not_kept,
    }
    case_report["html_path"] = artifacts["case_report_html"]

    if wants_video(args) and out["video_output"] is None:
        # engine/tracker.py swallows a failed encode and returns None on
        # purpose: POPS, the rules and the analytics are all complete without a
        # playable video. Not a failure, and not listed in not_kept either --
        # "you didn't ask for this" and "this run couldn't make one" are
        # different facts.
        #
        # Guarded on wants_video: None now has a third meaning, "no video was
        # asked for so none was encoded", and that is the DEFAULT. Unguarded,
        # every JSON-only run reported a failed encode it never attempted.
        warn("the annotated video could not be encoded; the rest of the run is "
             "unaffected (see the engine's [ERROR] lines above for ffmpeg's "
             "own reason).")

    # --- The document ------------------------------------------------------
    tracking = json.loads(out["json_output"])
    if args.frames == "none":
        n_frames = len(tracking.get("frames") or [])
        tracking["frames"] = {"_dropped": True, "count": n_frames,
                              "json_every_n": JSON_EVERY_N_FRAMES}

    html = None
    if not args.no_html:
        html = {key: out[slot] for slot, key in HTML_SLOTS.items()}
        # The deferred placeholder is never passed through: null when no report
        # was asked for, the real document when one was.
        html["case_report"] = case_report_html

    envelope = build_envelope(
        argv=argv, started_at=started_at, finished_at=_now(),
        wall_s=time.perf_counter() - t_start, video_s=video_s,
        case_report_s=case_s, exit_state="ok",
        run=_run_block(args, plan, device, argv),
        tracking=tracking,
        analytics=analytics_block(getattr(engine, "_last_analytics", None)),
        case_report=case_report, artifacts=artifacts, html=html)
    n_bytes = write_envelope(json_target, envelope, args.indent)

    _print_outcome(envelope, tracking, n_bytes, artifacts, warnings,
                   getattr(getattr(engine, "_last_analytics", None),
                           "rule_findings", None))
    return EXIT_OK


def _text_of(html: Optional[str]) -> Optional[str]:
    """The visible text of a one-line HTML banner, for an error field.

    The engine's failure paths return a `<p>…</p>` banner rather than raising,
    and a document that stores the markup makes a reader parse HTML to find out
    what went wrong.
    """
    if not html:
        return None
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split()) or None


#: Per-severity badge and colour. The mark is what carries the urgency at a
#: glance -- two marks for the one that stops the store, one for the one that
#: needs somebody, a dash for the one that is only worth knowing. The word
#: beside it is there for a reader who is not reading the marks.
#:
#: Keys are engine.highlights.OPS_SEVERITY_ORDER's, which is where the set of
#: severities is actually decided; an unknown one falls through to INFO rather
#: than vanishing.
FIND_MARKS = {
    "SAFETY": ("[!!]", "RED"),
    "ACTION": ("[!] ", "YELLOW"),
    "WATCH":  ("[-] ", "CYAN"),
    "INFO":   ("[·] ", "DIM"),
}

#: Column widths inside the findings frame. Severity is the widest word;
#: label is the widest RuleFinding.label once its qualifier is stripped;
#: duration is "999.9s". The cart-and-zone column takes whatever is left,
#: because it is the only part that can lose its tail and still identify the
#: finding.
FIND_SEV_W = 6
FIND_LABEL_W = 18
FIND_DUR_W = 6


def _finding_label(label: str) -> str:
    """`UNATTENDED CART (OPS)` -> `Unattended cart`.

    The (OPS) qualifier exists because "ABANDONED CART" is also a POPS event
    label and the two must not be confused in the JSON or the case report.
    Inside this frame every row IS an ops rule, so the qualifier distinguishes
    nothing and costs four columns that the zone name wants.
    """
    text = str(label or "")
    if "(" in text:
        text = text[:text.index("(")]
    return text.strip().capitalize()


def _finding_rows(findings) -> None:
    """The findings, framed, worst first.

    `findings` must be the RuleFinding DATACLASSES off
    `TrackingEngine._last_analytics`, not the dicts in the tracking JSON. The
    ordering comes from engine.highlights, which reads its fields with
    getattr() -- against a dict that returns the default for every one of
    them, so every finding would sort as INFO and the order would be
    arbitrary. The two also spell the fields differently (cart_display_id vs
    cart_id), and this renders the dataclass's names.
    """
    from engine.highlights import order_findings

    ordered = order_findings(findings or [])
    if not ordered:
        return
    # Two spaces of frame padding, the mark, and the space after it.
    room = ui.TABLE_W - 2 - 2 - 4 - 1
    where_w = room - FIND_SEV_W - 1 - FIND_LABEL_W - 1 - FIND_DUR_W - 1

    rows = []
    for f in ordered:
        severity = str(getattr(f, "severity", "INFO") or "INFO").upper()
        mark, tone = FIND_MARKS.get(severity, FIND_MARKS["INFO"])
        where = f"Cart {getattr(f, 'cart_display_id', '?')}"
        zone = getattr(f, "zone_name", None)
        if zone:
            where += f" / {zone}"
        secs = float(getattr(f, "duration_s", 0.0) or 0.0)
        rows.append((
            mark,
            getattr(ui, tone, ui.DIM),
            severity[:FIND_SEV_W].ljust(FIND_SEV_W)
            + " " + _finding_label(getattr(f, "label", ""))[:FIND_LABEL_W]
                    .ljust(FIND_LABEL_W)
            + " " + where[:where_w].ljust(where_w)
            + " " + f"{secs:.1f}s".rjust(FIND_DUR_W),
        ))
    ui.table("operational findings", rows, f"{len(rows)} total")


#: Column widths inside the POPS frame. The event is the widest label
#: classify_event() can return ("MEDIUM PRIORITY"); the cart column holds a
#: three-digit display id; score and time are numbers and are right-aligned so
#: they can be compared down the column. Contents takes what is left.
POPS_SCORE_W = 3
POPS_EVENT_W = 15
POPS_CART_W = 8
POPS_TIME_W = 6

#: Rows before the frame starts eliding. A busy clip can carry dozens of
#: carts, and forty rows of LOW PRIORITY buries the one that is not. The
#: remainder is reported in the header rather than dropped silently.
POPS_MAX_ROWS = 10


#: Readings that are not readings. The classifier spells the placeholder BOTH
#: ways -- engine/classifier.py emits "non-applicable" and "not_applicable"
#: from different paths -- so this compares on letters alone rather than
#: picking a winner between them. "unknown" joins them: all three mean the
#: question was not answered, and printing any of them spends a third of the
#: contents column saying nothing.
_NOT_A_READING = {"notapplicable", "nonapplicable", "unknown", "none", ""}


def _reading(value) -> str:
    """A classifier value if it says something, "" if it does not."""
    text = str(value or "").strip()
    if "".join(c for c in text.lower() if c.isalpha()) in _NOT_A_READING:
        return ""
    return text


def _pops_mark(score: int):
    """The badge for a POPS score, by the engine's own tier boundaries.

    Read from engine.scoring rather than retyped. Those numbers are described
    there as "the numbers that get retuned", and a copy here would keep
    rendering the old tiers after a retune with nothing to say it had.
    """
    from engine.scoring import HIGH_SCORE, MEDIUM_SCORE

    if score >= HIGH_SCORE:
        return "[!!]", ui.RED
    if score >= MEDIUM_SCORE:
        return "[!] ", ui.YELLOW
    if score > 0:
        return "[-] ", ui.CYAN
    return f"[{ui.G['sep']}] ", ui.DIM


def _pops_rows(pops_summary: dict) -> None:
    """One row per cart the run scored, worst first.

    Beside the ops findings rather than merged into them, because the two
    answer different questions and are computed by different halves of the
    engine: a rule finding is an operational fact about a zone over an
    interval, a POPS row is a theft-risk score for one cart. Merging them
    would need a severity scale that spans both, and there isn't one.
    """
    carts = [(str(key), value) for key, value in (pops_summary or {}).items()
             if isinstance(value, dict)]
    if not carts:
        return

    def _id(key: str) -> int:
        digits = "".join(c for c in key if c.isdigit())
        return int(digits) if digits else 0

    carts.sort(key=lambda kv: (-int(kv[1].get("max_score") or 0), _id(kv[0])))
    shown, extra = carts[:POPS_MAX_ROWS], max(0, len(carts) - POPS_MAX_ROWS)

    room = ui.TABLE_W - 2 - 2 - 4 - 1
    fill_w = (room - POPS_SCORE_W - 1 - POPS_EVENT_W - 1 - POPS_CART_W - 1
              - POPS_TIME_W - 1)

    rows = []
    for key, cart in shown:
        score = int(cart.get("max_score") or 0)
        mark, colour = _pops_mark(score)
        contents = [c for c in (_reading(cart.get("fill")),
                                _reading(cart.get("bag"))) if c]
        if not contents:
            # Nothing was read. WHY nothing was read is the useful thing to
            # say, and for an unclear cart the quality field is what says it;
            # a blank column would read as "empty and unbagged".
            quality = _reading(cart.get("quality"))
            if quality and quality != "valid_cart":
                contents = [quality.replace("_", " ")]
        seconds = float(cart.get("peak_timestamp") or 0.0)
        rows.append((
            mark, colour,
            str(score).rjust(POPS_SCORE_W)
            + " " + str(cart.get("peak_event") or "")[:POPS_EVENT_W]
                    .ljust(POPS_EVENT_W)
            + " " + f"Cart {_id(key)}"[:POPS_CART_W].ljust(POPS_CART_W)
            + " " + " / ".join(contents)[:fill_w].ljust(fill_w)
            + " " + f"{seconds:.1f}s".rjust(POPS_TIME_W),
        ))

    peak = int(shown[0][1].get("max_score") or 0)
    trailer = f"peak {peak}" + (f"  {ui.G['sep']}  +{extra} more" if extra else "")
    ui.table("POPS findings", rows, trailer)


#: How an unkept artifact is NAMED on screen. The keys are the strings
#: keep_artifact() records, and those are also what lands in the envelope's
#: `artifacts.not_kept` -- which is part of the gk-pops-cli/1 document and may
#: be being parsed. So the console spells them out and the document does not
#: move; an unmapped name falls through unchanged rather than disappearing.
NOT_KEPT_WORDS = {
    "video": "annotated video",
    "heatmap": "heat map",
    "engine-json": "engine JSON",
    "case-report": "case report",
}


def _counted(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _generated_at(iso: str) -> str:
    """`2026-09-11T15:28:18.402Z` -> `2026-09-11 20:58:18 +0530`.

    The envelope stores UTC, which is right for a document two machines might
    compare and wrong for a person reading their own console: nobody wants to
    do the offset in their head to find out whether this is the run they just
    started. Converted to the reader's own clock, with the offset kept so a
    screenshot mailed to another timezone is still unambiguous.

    Derived from the envelope rather than read off the clock again, so the row
    and `cli.finished_at` can never disagree by the milliseconds between them.
    Falls back to the raw string if it will not parse -- a stamp in a format
    this cannot read is still better than no stamp.
    """
    try:
        when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(iso or "")
    return when.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _print_outcome(envelope: dict, tracking: dict, n_bytes: int,
                   artifacts: dict, warnings: list[str],
                   findings_obj=None) -> None:
    """The closing block: the verdict, what was found, then every file kept.

    The per-section counts are here rather than under the section that
    produced them, and that is a constraint rather than a choice: sections 2
    to 4 are opened and closed by the engine's own progress callback, from
    inside process_video(), and none of these numbers exist until it returns.
    Printing them beside the section would mean buffering the whole run and
    showing nothing until it finished.
    """
    pops = tracking.get("pops_summary") or {}
    peak = max((int(v.get("max_score") or 0) for v in pops.values()
                if isinstance(v, dict)), default=0)
    peak_event = next((v.get("peak_event") for v in pops.values()
                       if isinstance(v, dict)
                       and int(v.get("max_score") or 0) == peak and peak), "")
    findings = len(tracking.get("rule_findings") or [])
    reason = (tracking.get("rule_engine") or {}).get("unavailable_reason")
    sep = f"  {ui.G['sep']}  "

    parts = [f"Complete in {envelope['cli']['wall_seconds']:.1f}s"]
    if pops:
        parts.append(f"peak POPS {peak}" + (f" {peak_event}" if peak_event else ""))
    if reason:
        parts.append("rules did not run")
    # Green only for a run that found nothing and had nothing to complain
    # about. A clean-looking verdict over three SAFETY findings is the one
    # thing a glance must never be able to get wrong.
    severities = {str(getattr(f, "severity", "")).upper()
                  for f in (findings_obj or [])}
    tone = ("fail" if reason or "SAFETY" in severities
            else "warn" if findings or warnings
            else "ok")
    # Labelled, because the extension does not say which file is which: the
    # run writes two .json documents and only one of them is this tool's. The
    # "generated at" is the same sentence for every artifact -- what it is,
    # that this run produced it, and where it put it -- so a reader who has
    # read one row can read the rest without re-parsing them.
    ui.outcome(sep.join(parts), "",
               [(f"{label} generated at", artifacts[key])
                for label, key in (
                   ("report", "json"),
                   ("video", "annotated_video"),
                   ("heat map", "heatmap_png"),
                   ("case report", "case_report_html"),
                   ("engine json", "engine_json")) if artifacts[key]],
               root=_ROOT, tone=tone)

    # The artifact rows above carry the clock alone. This is the row that
    # carries the DATE, so it has to stay -- without it a console pasted into
    # a ticket next week says 21:00 and nothing else.
    ui.ok("run completed", _generated_at((envelope.get("cli") or {})
                                         .get("finished_at")))

    # What the run actually found, which is what someone ran it for. A row
    # that reports only a duration reports nothing.
    summary = tracking.get("summary") or {}
    tracked = [_counted(int(summary.get("total_carts_seen") or 0), "cart"),
               _counted(int(summary.get("total_people_seen") or 0), "person",
                        "people"),
               _counted(int(summary.get("total_links_established") or 0),
                        "link")]
    ui.ok("tracked", sep.join(tracked))

    zones = len(tracking.get("dwell_summary") or [])
    spikes = len(tracking.get("queue_spikes") or [])
    events = len(tracking.get("events") or [])
    ui.ok("analytics", sep.join([_counted(zones, "dwell zone"),
                                 _counted(spikes, "queue spike"),
                                 _counted(events, "POPS event")]))

    if reason:
        # Part of the verdict, not a row: a run that scored nothing because the
        # rules never ran must never read as a quiet run.
        warn_row("rules unavailable", str(reason))
    elif not findings:
        ui.ok("findings", "none")
    if artifacts["not_kept"]:
        ui.info("not kept", ", ".join(NOT_KEPT_WORDS.get(n, n)
                                      for n in artifacts["not_kept"]))
        ui.note("produced by the run and left to the engine's own tidy-up; "
                "name a path to keep one.")
    if warnings:
        ui.warn("warnings", str(len(warnings)))

    # Say that output was suppressed. Silence about the suppression is what
    # turns a tidy console into a console that is hiding something.
    log = _ENGINE_LOG
    if log is not None and log.count and log.echo is None:
        ui.info("engine log", f"{log.count} lines captured")
        ui.note(str(log.path) if log.path else
                "the engine's own per-cart narration. Re-run with -v to watch "
                "it, or --engine-log PATH to keep it.")

    # Last, deliberately. Everything above is an account of the run; these are
    # what the run was for, and the last thing on screen is what a reader
    # takes away from it. A count is a promise to go and read the JSON.
    #
    # POPS first: it is the per-cart theft-risk scoring this tool is named
    # for. The ops findings are the zone rules, and they are second because a
    # blocked door is a store problem rather than the question that was asked.
    _pops_rows(pops)
    if findings and not reason:
        _finding_rows(findings_obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _configure_console(args: argparse.Namespace) -> None:
    """Colour and carriage returns, from --log or from whether stdout is a TTY."""
    plain = (args.log == "plain") if args.log else not sys.stdout.isatty()
    ui.set_plain(plain)


def _git_short() -> str:
    """The checkout's short SHA for the banner, or "" when this is not a
    checkout. Read from .git directly -- shelling out to git for one decoration
    is not worth a subprocess."""
    head = REPO_ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            ref = (REPO_ROOT / ".git" / ref[5:]).read_text(encoding="utf-8")
        return ref.strip()[:7]
    except OSError:
        return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, run, and map any failure onto an exit code.

    Args:
        argv: The full argv INCLUDING the program name, as `sys.argv`. None
            reads sys.argv.

    Returns:
        0, 2, 3 or 130. Never 1 -- that stays the signal for a process that died
        before this handler ran.
    """
    argv = list(sys.argv if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    # Immediately after parse_args and before anything reads it: --pose is
    # tri-state on the namespace (None = "nobody said"), and every later
    # reader -- print_inputs, _run_block's `enable_pose`, the engine call --
    # wants the resolved bool. Settled here so there is one resolution point
    # rather than three.
    args.pose = resolve_pose(args)
    _configure_console(args)

    if not args.quiet:
        ui.logo("headless run", os.path.basename(args.video), _git_short())
    if args.verbose:
        ui.detail("resolved arguments:")
        for key, value in sorted(vars(args).items()):
            if key == "vlm_api_key":
                value = "<set>" if resolve_api_key(value) else "<unset>"
            ui.detail(f"  {key} = {value!r}")

    # The whole run, not just process_video: the engine narrates from its
    # import (engine/tracker.py's ultralytics-compat line), from the
    # constructor (the three model loads), from the frame loop, and from the
    # artifact tidy-up. One span is the only shape that catches all four.
    try:
        engine_log = None
        if args.engine_log:
            engine_log = Path(args.engine_log).resolve()
            try:
                engine_log.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise CliError(f"--engine-log {engine_log} is not writable: {e}",
                               "name a path inside a folder this account can "
                               "write to.") from e
        with engine_output(args, engine_log):
            return run(args, argv)
    except CliError as e:
        ui.progress_end()
        body = [str(e)]
        if e.hint:
            body += ["", e.hint]
        if args.verbose and e.__cause__ is not None:
            body += [""] + traceback.format_exception(
                type(e.__cause__), e.__cause__,
                e.__cause__.__traceback__)[-6:]
        _problem_to_stderr(type(e).__name__, body)
        return e.exit_code
    except KeyboardInterrupt:
        ui.progress_end()
        _problem_to_stderr("Interrupted", ["Stopped at the keyboard."])
        return EXIT_INTERRUPTED
    except Exception:
        # Anything that is not a CliError is a bug in this file rather than a
        # mistake by the caller, so the traceback is printed unconditionally --
        # there is nothing actionable to say instead of it. Exit 3, the internal
        # failure code, never 1: 1 stays the signal that the process died before
        # reaching here at all.
        ui.progress_end()
        _problem_to_stderr(
            "Internal error",
            ["gk_pops.py itself failed. This is a bug, not a usage mistake.",
             ""] + traceback.format_exc().splitlines())
        return EXIT_FAILED


def _spaced(name: str) -> str:
    """`OutputPathError` -> `Output Path Error`, for a heading.

    The headings are the exception class names, which is the right source --
    there is no second list to drift. Run together and upper-cased they read
    as one long word, so the word boundaries are put back first.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _problem_to_stderr(heading: str, lines: list[str]) -> None:
    """console_ui prints to stdout; errors belong on stderr."""
    with on_stderr():
        ui.problem(_spaced(heading), lines)


if __name__ == "__main__":
    sys.exit(main())
