"""Regenerate a golden baseline under tests/fixtures/golden/.

    python tests/make_golden_baseline.py <clip.mp4> [--placement "Outside (facing entrance)"]
                                                    [--case <name>]

Writes `baseline.json` by default. `--case outside` writes
`baseline_outside.json` instead, which is how a second clip gets pinned —
tests/test_golden_clip.py discovers every `baseline*.json` and checks each
against its own clip.

Run this ONLY after a deliberate behaviour change, and read the resulting diff
before committing it — the baseline is the thing that would otherwise have told
you the change was unintended.

The clip itself is never copied into the repo (real store CCTV with
identifiable people). Only the per-frame counts, the scoring decisions, and the
clip's sha256 are recorded, so the test can confirm it is scoring the same
footage.
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Both so this works whether invoked as `python tests/make_golden_baseline.py`
# or from the repo root — tests/ is not a package.
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

GOLDEN_DIR = os.path.join(REPO, "tests", "fixtures", "golden")

from test_golden_clip import run_pipeline, sha256  # noqa: E402


def out_path(case):
    """`baseline.json` for the primary case, `baseline_<case>.json` otherwise."""
    name = "baseline.json" if not case else f"baseline_{case}.json"
    return os.path.join(GOLDEN_DIR, name)


def provenance():
    from engine.trajectory_cache import environment_fingerprint
    import ultralytics

    fingerprint = dict(
        part.split("=", 1) for part in environment_fingerprint(refresh=True).split("|")
    )
    return {
        "ultralytics": ultralytics.__version__,
        "tracker_config_sha": fingerprint.get("tracker_cfg"),
        "weights": fingerprint.get("weights"),
        "python": sys.version.split()[0],
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    clip = os.path.abspath(argv[1])
    if not os.path.exists(clip):
        print(f"no such clip: {clip}")
        return 2

    placement = "Outside (facing entrance)"
    if "--placement" in argv:
        placement = argv[argv.index("--placement") + 1]
    case = ""
    if "--case" in argv:
        case = argv[argv.index("--case") + 1]
    OUT = out_path(case)

    settings = {"camera_placement": placement, "zones": [], "enable_pose": True}

    print(f"scoring {clip} ...")
    frames, decisions = run_pipeline(clip, settings, capture_decisions=True)

    baseline = {
        "_comment": (
            "Per frame: counts [persons, carts, links] from the HUD, plus the "
            "display IDs actually drawn. IDs are recorded because counts alone "
            "cannot see one object being tracked under two IDs. `decisions` "
            "records the scoring layer — per-cart peak event, owner, "
            "abandonment, merch_removed, and the finalised event rows — because "
            "counts and IDs alone cannot see a cart linked to the wrong person "
            "and scored a tier too low. Regenerate with "
            "tests/make_golden_baseline.py; see tests/test_golden_clip.py."
        ),
        "clip": {
            "path": clip,
            "name": os.path.basename(clip),
            "sha256": sha256(clip),
            "size": os.path.getsize(clip),
        },
        "settings": settings,
        "provenance": provenance(),
        "frames": {str(k): v for k, v in sorted(frames.items())},
        "decisions": decisions,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=1, sort_keys=True)

    linked = sum(1 for v in frames.values() if v["counts"][2] > 0)
    people = sum(1 for v in frames.values() if v["counts"][0] > 0)
    carts = sorted({i for v in frames.values() for i in v["carts"]})
    persons = sorted({i for v in frames.values() for i in v["persons"]})
    print(f"wrote {OUT}")
    print(f"  {len(frames)} frames | person>0 on {people} | link>0 on {linked}")
    print(f"  distinct cart IDs: {carts}")
    print(f"  distinct person IDs: {persons}")
    print(f"  provenance: {baseline['provenance']}")
    print(f"  events: {len(decisions['events'])} | {decisions['summary']}")
    for cd, info in decisions["carts"].items():
        print(f"    {cd}: {info['max_score']:>3} {info['peak_event']:<14} "
              f"owner={info['owner']} abandoned={info['abandoned']} "
              f"merch_removed={info['merch_removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
