# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The rule badges have to survive the round trip onto the annotated video.

Run with:  python tests/test_rule_overlay_video.py

WHY THIS EXISTS
---------------
Operational rule findings are post-hoc: no frame in the tracking loop can know
that a cart is about to stand in the doorway long enough to matter, so the
badges cannot be drawn while the AVI is being written. They are painted on a
second pass, during the encode that was already happening
(`video_io.reencode_to_mp4(avi_path, frame_hook=...)`).

That pass is the piece most likely to fail quietly. `_process_video` swallows an
encode exception on purpose — a broken encoder must not throw away a good run —
so a mistake here does not raise anywhere the user can see it. It empties the
video panel instead. Hence a real ffmpeg round trip rather than a stub:

  * every frame of the AVI reaches the encoder (a dropped frame shortens the
    clip and desynchronises it from the event log's timestamps),
  * the hook is called with the SAME 1-based numbering the tracking loop used,
    which is what makes `rules.overlay_index()`'s frame keys mean anything,
  * a hook that raises fails loudly rather than producing a silently unbadged
    video.

Needs ffmpeg — imageio_ffmpeg ships it, and the app cannot start without it.
Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.video_io import reencode_to_mp4          # noqa: E402
from engine.renderer import draw_rule_badges         # noqa: E402

W, H, FPS, N = 320, 240, 20, 40

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def _write_avi(dir_path):
    """An AVI shaped like the one the tracking loop writes."""
    avi = os.path.join(dir_path, "pops_demo_raw.avi")
    wr = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"XVID"), FPS, (W, H))
    for _ in range(N):
        wr.write(np.full((H, W, 3), 40, np.uint8))
    wr.release()
    return avi


def test_every_frame_reaches_the_encoder():
    print("\n=== the overlay pass drops no frames and counts from 1 ===")
    d = tempfile.mkdtemp(prefix="pops_overlay_")
    try:
        avi = _write_avi(d)
        seen = []
        mp4 = reencode_to_mp4(avi, frame_hook=lambda f, i: seen.append(i))
        check("the hook saw every frame", len(seen) == N, f"{len(seen)} of {N}")
        check("numbering is 1-based, matching the tracking loop",
              seen[:1] == [1] and seen[-1:] == [N], f"{seen[:2]}..{seen[-2:]}")
        check("the MP4 was written", os.path.exists(mp4))
        cap = cv2.VideoCapture(mp4)
        got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        check("the MP4 is the same length as the AVI", got == N, f"{got} of {N}")
        check("the intermediate AVI is cleaned up", not os.path.exists(avi))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_badge_actually_lands_in_the_pixels():
    print("\n=== a badge drawn by the hook survives into the MP4 ===")
    d = tempfile.mkdtemp(prefix="pops_overlay_")
    try:
        avi = _write_avi(d)
        badge = [{"bbox": np.array([80.0, 120.0, 200.0, 200.0], np.float32),
                  "text": "BLOCKED DOOR", "zone_name": "Main Door",
                  "severity": "SAFETY", "degraded": False, "elapsed_s": 7.0}]
        # Only the back half is badged, so the front half is the control: any
        # difference between them can only have come from the hook.
        half = N // 2

        def hook(frame, i):
            if i > half:
                draw_rule_badges(frame, badge)

        mp4 = reencode_to_mp4(avi, frame_hook=hook)
        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        check("all frames decode back", len(frames) == N, f"{len(frames)}")
        if len(frames) == N:
            # Measured as "differs from the flat source", not as a colour: the
            # badge's styling is a design decision and may change, but a badge
            # that leaves the frame untouched is always a bug. Lossy encoding
            # moves every pixel a little, hence the margin.
            def marked(f):
                return int((np.abs(f.astype(int) - 40).max(axis=2) > 60).sum())
            plain = marked(frames[2])
            badged = marked(frames[-2])
            check("the unbadged half is untouched", plain < 50, str(plain))
            check("the badged half carries the badge", badged > 500,
                  f"{badged} px")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_badge_never_leaves_the_frame():
    print("\n=== a cart at the frame edge still gets a readable badge ===")
    # The cart worth badging is often the one half out of a doorway, at the
    # very edge of the frame. Anchoring the chip to bbox[0] alone clipped it to
    # "UNATTENDED CAR" against the right border on a real clip.
    text = {"text": "UNATTENDED CART (OPS)", "zone_name": None,
            "severity": "ACTION", "degraded": False, "elapsed_s": 12.0}
    corners = {
        "right edge":  (W - 40.0, 100.0, W - 5.0, 180.0),
        "left edge":   (-30.0, 100.0, 40.0, 180.0),
        "top edge":    (100.0, 0.0, 200.0, 60.0),
        "bottom edge": (100.0, H - 60.0, 200.0, H - 2.0),
        # The only box that exercises BOTH vertical fallbacks: no room above
        # it, and no room below it either, so the chip has to sit inside the
        # cart. Covering part of a cart beats not saying the door is blocked.
        "full height": (100.0, 0.0, 200.0, H - 1.0),
    }
    for name, box in corners.items():
        im = np.full((H, W, 3), 40, np.uint8)
        draw_rule_badges(im, [dict(text, bbox=np.array(box, np.float32))])
        painted = np.abs(im.astype(int) - 40).max(axis=2) > 60
        check(f"{name}: the badge is drawn", int(painted.sum()) > 300,
              str(int(painted.sum())))
        cols = np.flatnonzero(painted.any(axis=0))
        rows = np.flatnonzero(painted.any(axis=1))
        check(f"{name}: and lies inside the frame",
              cols.size > 0 and cols[0] >= 0 and cols[-1] <= W - 1
              and rows.size > 0 and rows[0] >= 0 and rows[-1] <= H - 1,
              f"x {cols[:1]}..{cols[-1:]} y {rows[:1]}..{rows[-1:]}")
        # Inside the frame is not enough on its own — a clipped chip is also
        # inside it. The badge has to be as WIDE as the same badge drawn with
        # room to spare, which is what says no text was cut off.
        mid = np.full((H, W, 3), 40, np.uint8)
        draw_rule_badges(mid, [dict(text, bbox=np.array(
            [40.0, 100.0, 140.0, 180.0], np.float32))])
        mid_cols = np.flatnonzero((np.abs(mid.astype(int) - 40).max(axis=2) > 60).any(axis=0))
        check(f"{name}: nothing was clipped off it",
              (cols[-1] - cols[0]) == (mid_cols[-1] - mid_cols[0]),
              f"{cols[-1] - cols[0]} vs {mid_cols[-1] - mid_cols[0]}")


def _painted(im):
    return int((np.abs(im.astype(int) - 40).max(axis=2) > 60).sum())


def test_side_by_side_carts_do_not_erase_each_other():
    print("\n=== two carts abreast: neither badge is drawn over the other ===")
    # Two carts parked side by side in one doorway is the NORMAL case for these
    # rules, and their badges are far wider than the carts. Anchoring each to
    # its own bbox.x put both in the same strip, so whichever drew second
    # erased the first and one of two flagged carts carried no label at all.
    left  = np.array([10.0, 120.0, 150.0, 220.0], np.float32)
    right = np.array([90.0, 120.0, 230.0, 220.0], np.float32)

    def badge(bbox, cid, sev, txt):
        return {"bbox": bbox, "cart_display_id": cid, "text": txt,
                "zone_name": "Aisle 1", "severity": sev, "degraded": False,
                "elapsed_s": 20.0}

    a = badge(left, 2, "WATCH", "STATIC CART")
    b = badge(right, 1, "WATCH", "STATIC CART")

    only_a = np.full((H, W, 3), 40, np.uint8)
    draw_rule_badges(only_a, [a])
    only_b = np.full((H, W, 3), 40, np.uint8)
    draw_rule_badges(only_b, [b])
    both = np.full((H, W, 3), 40, np.uint8)
    draw_rule_badges(both, [a, b])

    # Solid chips, so any overlap SUBTRACTS from the total painted area. Equal
    # areas is the same claim as "neither was drawn over", measured without
    # depending on where the placement chose to put them.
    check("each badge draws on its own",
          _painted(only_a) > 300 and _painted(only_b) > 300,
          f"{_painted(only_a)} / {_painted(only_b)}")
    check("both badges survive together",
          _painted(both) == _painted(only_a) + _painted(only_b),
          f"{_painted(both)} vs {_painted(only_a)} + {_painted(only_b)}")

    # Four badges, two rules on each of two carts — the screenshot case.
    quad = [badge(right, 1, "ACTION", "UNATTENDED CART (OPS)"),
            badge(left, 2, "ACTION", "UNATTENDED CART (OPS)"),
            badge(right, 1, "WATCH", "STATIC CART"),
            badge(left, 2, "WATCH", "STATIC CART")]
    im4 = np.full((H, W, 3), 40, np.uint8)
    draw_rule_badges(im4, quad)
    singles = 0
    for one in quad:
        tmp = np.full((H, W, 3), 40, np.uint8)
        draw_rule_badges(tmp, [one])
        singles += _painted(tmp)
    check("all four stay clear of one another",
          _painted(im4) == singles, f"{_painted(im4)} vs {singles}")


def test_a_badge_says_which_cart_it_is_about():
    print("\n=== a displaced badge still names its cart ===")
    # Once a badge has been nudged off its own cart to find clear air, position
    # no longer identifies it. The cart id has to be in the chip.
    bbox = np.array([40.0, 120.0, 140.0, 200.0], np.float32)
    base = {"bbox": bbox, "text": "STATIC CART", "zone_name": None,
            "severity": "WATCH", "degraded": False, "elapsed_s": 20.0}
    without = np.full((H, W, 3), 40, np.uint8)
    draw_rule_badges(without, [dict(base)])
    with_id = np.full((H, W, 3), 40, np.uint8)
    draw_rule_badges(with_id, [dict(base, cart_display_id=7)])

    def width(im):
        cols = np.flatnonzero((np.abs(im.astype(int) - 40).max(axis=2) > 60).any(axis=0))
        return int(cols[-1] - cols[0]) if cols.size else 0

    check("the id widens the chip, so it is really drawn",
          width(with_id) > width(without),
          f"{width(with_id)} vs {width(without)}")
    check("a badge with no cart id still renders",
          _painted(without) > 300, str(_painted(without)))


def test_a_raising_hook_is_not_swallowed_here():
    print("\n=== a broken hook fails loudly, it does not ship a bare video ===")
    # _process_video's try/except around the encode turns any failure into "no
    # video, run intact". That is the right call there and the wrong one here:
    # if this function ate the exception itself, a badge bug would ship a
    # correct-looking video with no badges on it and nothing in the log.
    d = tempfile.mkdtemp(prefix="pops_overlay_")
    try:
        avi = _write_avi(d)

        def boom(frame, i):
            raise ValueError("badge index is wrong")

        raised = None
        try:
            reencode_to_mp4(avi, frame_hook=boom)
        except Exception as e:                                   # noqa: BLE001
            raised = e
        check("the hook's own exception propagates",
              isinstance(raised, ValueError), repr(raised))
        check("and the AVI is kept, so the run still has its video",
              os.path.exists(avi))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    test_every_frame_reaches_the_encoder()
    test_the_badge_actually_lands_in_the_pixels()
    test_a_badge_never_leaves_the_frame()
    test_side_by_side_carts_do_not_erase_each_other()
    test_a_badge_says_which_cart_it_is_about()
    test_a_raising_hook_is_not_swallowed_here()
    print("\n" + "=" * 62)
    print(f"PASSED {len(_PASS)} / {len(_PASS) + len(_FAIL)}")
    if _FAIL:
        print("FAILED:")
        for name in _FAIL:
            print("  -", name)
        return 1
    print("All green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
