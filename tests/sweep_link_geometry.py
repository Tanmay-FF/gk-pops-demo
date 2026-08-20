# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Sweep the ground-plane link gate over every link a run actually made.

    python tests/sweep_link_geometry.py <tracking.json> [<tracking.json> ...]
                                        [--band 0.45]

This is the tool that tuned LINK_GROUND_BAND, and it is how a change to the gate
gets checked against real footage instead of against a handful of chosen frames.
It exists because the first design of that gate — a per-frame veto at 0.35 — was
plausible, passed every hand-picked case, and would have broken the primary golden
clip's own pushout link on 42% of its frames. The sweep is what said so.

`foot_ratio = (person_y2 - cart_y2) / cart_height`: how far the person's feet fall
in front of (positive) or behind (negative) the cart's base, scaled by the cart's
own apparent size so it means the same thing near and far from the camera.

Read the MEAN column. Single frames swing wide in both directions — a pusher is
legitimately nearer the camera than their cart whenever the cart is being pushed
away from it — so a per-frame reading says very little. On the two clips this was
built against, mean separated every legitimate link (worst +0.36) from every
mislink (best +0.64) with nothing in between.

Any pair listed as REJECT that you believe is a real owner is a regression the
gate is about to ship. Give it a verdict before changing the threshold.
"""
import sys
import os
import json
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def foot_ratios(doc):
    """{(person_key, cart_key): [foot_ratio per linked frame]} for one run."""
    out = defaultdict(list)
    for frame in (doc.get("frames") or {}).values():
        for link in (frame.get("links") or {}).values():
            pk = f"P{link['person_id']}"
            ck = f"C{link['cart_id']}"
            person = (frame.get("people") or {}).get(pk)
            cart = (frame.get("carts") or {}).get(ck)
            if not person or not cart:
                continue
            p, c = person["bbox"], cart["bbox"]
            height = c["y2"] - c["y1"]
            if height <= 0:
                continue
            out[(pk, ck)].append((p["y2"] - c["y2"]) / height)
    return out


def sweep(paths, band):
    print(f"ground band = {band:+.2f}  (reject when mean > band, or mean < -1.0)")
    print(f"{'run':22} {'pair':12} {'n':>5} {'mean':>7} {'median':>7} "
          f"{'min':>6} {'max':>6}  gate")
    flagged = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        run = os.path.basename(path).replace("_tracking.json", "")[:22]
        for (pk, ck), ratios in sorted(foot_ratios(doc).items()):
            mean = statistics.fmean(ratios)
            if mean > band:
                gate = "REJECT-foreground"
            elif mean < -1.0:
                gate = "REJECT-behind"
            else:
                gate = "keep"
            if gate != "keep":
                flagged.append((run, pk, ck, mean, gate))
            print(f"{run:22} {pk + '->' + ck:12} {len(ratios):>5} {mean:>+7.2f} "
                  f"{statistics.median(ratios):>+7.2f} {min(ratios):>+6.2f} "
                  f"{max(ratios):>+6.2f}  {gate}")
    print()
    if not flagged:
        print("no link would be rejected — the gate is a no-op on this footage, "
              "which means it is either safe or untested. Check that a KNOWN "
              "mislink is present before trusting that.")
    else:
        print(f"{len(flagged)} pair(s) would be rejected:")
        for run, pk, ck, mean, gate in flagged:
            print(f"  {run} {pk}->{ck} mean {mean:+.2f} {gate}")
        print("Each one is either a mislink you meant to kill or a regression.")
    return 0


def main(argv):
    paths = [a for a in argv[1:] if not a.startswith("--")
             and not (len(argv) > 1 and argv[argv.index(a) - 1] == "--band")]
    band = 0.45
    if "--band" in argv:
        band = float(argv[argv.index("--band") + 1])
    if not paths:
        print(__doc__)
        return 2
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("no such tracking JSON: " + ", ".join(missing))
        return 2
    return sweep(paths, band)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
