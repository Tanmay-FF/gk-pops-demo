# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Static-latch tests — pure logic, no GPU and no video decode.

Run with:  python tests/test_static_latch.py

The fixture is real. It is the per-frame window displacement of Cart 21 and
Person 17 on 1764099569430_B8A44F3CB0B9-medium-OUTSIDE.mp4, frames 158-175,
read out of both runs of that clip: one under ultralytics 8.4.19 / conda, one
under 8.4.40 / venv, on the same machine against the same footage.

The two runs produced the same tracks and the same detections to within 0.005
px. They disagreed about one number: on frame 162 the cart's six-frame
displacement measured 5.001559 px in one and 4.999244 px in the other, against
COMOVEMENT_STATIC_PX = 5. Above the bar the cart reads "moving" beside a static
person, which is the bystander branch of are_co_moving(), so that frame
contributed no evidence to the link. The candidate reached 19 frames against
the contested bar of 20, then hit a barren gap and lost all 19 to
LINK_CANDIDATE_PATIENCE. The link was established at frame 181 instead of 167,
and the two builds disagreed on 14 frames of output.

Nothing about that was version-specific — any run-to-run float wobble can
straddle a bar a track happens to be sitting on. StaticLatch gives the call
hysteresis so a track hovering at the bar keeps its state.

Deliberately stdlib only, no pytest — matches the rest of the repo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import (  # noqa: E402
    COMOVEMENT_STATIC_PX, COMOVEMENT_STATIC_EXIT_PX, COMOVEMENT_WINDOW,
)
from engine.motion import StaticLatch, are_co_moving  # noqa: E402

_passed = _failed = 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f" — {extra}" if extra else ""))


def section(title):
    print(f"\n{title}")


#: Cart 21's six-frame displacement, frames 158-175, as measured under
#: ultralytics 8.4.19 (conda) and 8.4.40 (venv). Only frame 162 straddles the
#: bar; every other frame agrees to within 0.02 px.
CART_CONDA = [3.4123, 3.2207, 2.2929, 3.6254, 5.0016, 2.2561, 1.0501, 1.9031,
              3.7528, 4.4052, 5.4755, 6.9051, 7.9719, 12.6765, 11.1314, 2.5278,
              7.8356, 10.5295]
CART_VENV = [3.4113, 3.2186, 2.2921, 3.6246, 4.9992, 2.2498, 1.0533, 1.9045,
             3.7544, 4.4069, 5.4756, 6.9133, 7.9865, 12.6852, 11.1377, 2.5168,
             7.8268, 10.5173]
FIRST_FRAME = 158


def run(mags):
    """The static/moving call per frame, as the latch resolves it."""
    latch = StaticLatch()
    return [latch.resolve("cart21", m) for m in mags]


# ---------------------------------------------------------------------------
section("the divergence that cost the clip 14 frames of agreement")
# ---------------------------------------------------------------------------
i162 = 162 - FIRST_FRAME
check("the fixture straddles the bar exactly as the two builds did",
      CART_CONDA[i162] > COMOVEMENT_STATIC_PX > CART_VENV[i162],
      f"conda={CART_CONDA[i162]} venv={CART_VENV[i162]} bar={COMOVEMENT_STATIC_PX}")
check("the bare threshold disagrees on that frame",
      (CART_CONDA[i162] < COMOVEMENT_STATIC_PX)
      != (CART_VENV[i162] < COMOVEMENT_STATIC_PX),
      "this is the whole bug: 0.0023 px decided a link")

latched_conda, latched_venv = run(CART_CONDA), run(CART_VENV)
check("the latch agrees on every frame of the window",
      latched_conda == latched_venv,
      f"conda={latched_conda}\n      venv={latched_venv}")
check("...including frame 162, which both now read as static",
      latched_conda[i162] and latched_venv[i162],
      "the cart was at 2-4 px on the frames either side of it")

# ---------------------------------------------------------------------------
section("the band is a band, not a higher bar")
# ---------------------------------------------------------------------------
latch = StaticLatch()
check("a track first seen inside the band is judged by the lower bar",
      not latch.resolve("fresh", (COMOVEMENT_STATIC_PX + COMOVEMENT_STATIC_EXIT_PX) / 2),
      "no prior state means no hysteresis to apply — same answer as before")

latch = StaticLatch()
latch.resolve("cart", 1.0)                       # established static
check("a static track survives a wobble into the band",
      latch.resolve("cart", COMOVEMENT_STATIC_PX + 0.01))
check("...and keeps surviving it",
      latch.resolve("cart", COMOVEMENT_STATIC_EXIT_PX - 0.01))
check("but a track that breaks out of the band is moving",
      not latch.resolve("cart", COMOVEMENT_STATIC_EXIT_PX + 0.01),
      "a cart genuinely rolling away must not be held static")
check("and a moving track needs the LOWER bar to be called static again",
      latch.resolve("cart", COMOVEMENT_STATIC_PX - 0.01) is True)

latch = StaticLatch()
latch.resolve("cart", 50.0)                      # established moving
check("a moving track inside the band stays moving",
      not latch.resolve("cart", COMOVEMENT_STATIC_PX + 0.01),
      "hysteresis has to work in both directions or it is just a raised bar")

# ---------------------------------------------------------------------------
section("cart 21's real trace, frame by frame")
# ---------------------------------------------------------------------------
# 168-171 is the barren gap the accumulated evidence died in. The cart is
# genuinely leaving there — 5.5, 6.9, 8.0, 12.7 px and climbing — so the latch
# must let go of it, just not on frame 162.
moving_frames = [FIRST_FRAME + i for i, st in enumerate(latched_conda) if not st]
check("the latch releases the cart as it actually rolls away",
      all(f >= 170 for f in moving_frames),
      f"first frame called moving: {min(moving_frames) if moving_frames else None}")
check("and it does call the cart moving eventually",
      moving_frames, "a latch that never lets go would link a passer-by to it")

# ---------------------------------------------------------------------------
section("are_co_moving is unchanged for callers that pass no latch")
# ---------------------------------------------------------------------------
_still = [(100.0, 100.0)] * COMOVEMENT_WINDOW
_rolling = [(100.0 + 30 * i, 100.0) for i in range(COMOVEMENT_WINDOW)]
check("static cart, moving person, static_a_ok — still allowed",
      are_co_moving(_still, _rolling, static_a_ok=True))
check("static person, moving cart — still the rejected bystander",
      not are_co_moving(_rolling, _still, static_a_ok=True))
check("two tracks moving together — still co-moving",
      are_co_moving(_rolling, _rolling))
# A displacement inside the band, against a static partner: the one shape where
# engaging the latch and not engaging it give opposite answers.
_in_band = [(100.0 + (COMOVEMENT_STATIC_PX + 1.0) * i / (COMOVEMENT_WINDOW - 1), 100.0)
            for i in range(COMOVEMENT_WINDOW)]
_half_wired = StaticLatch()
_half_wired.resolve("a", 1.0)          # "a" is established static
check("the bare test rejects a partner that has drifted into the band",
      not are_co_moving(_in_band, _still, static_a_ok=True),
      "moving A beside static B is the bystander branch")
check("passing only one key is not enough to engage the latch",
      not are_co_moving(_in_band, _still, static_a_ok=True,
                        latch=_half_wired, key_a="a"),
      "a half-wired caller must fall back to the pure test, not to a wrong one")
check("...while a fully wired caller does engage it",
      are_co_moving(_in_band, _still, static_a_ok=True,
                    latch=_half_wired, key_a="a", key_b="b"),
      "A was static a frame ago and has only drifted into the band since")

# ---------------------------------------------------------------------------
section("state is per track and per run")
# ---------------------------------------------------------------------------
latch = StaticLatch()
latch.resolve("cart", 1.0)
check("one track's state does not answer for another",
      not latch.resolve("other", COMOVEMENT_STATIC_PX + 0.01))

latch.reset()
check("reset drops every opinion", not latch.resolve("cart", COMOVEMENT_STATIC_PX + 0.01),
      "a fresh run must not inherit the last one's tracks")

latch = StaticLatch()
latch.resolve("old_raw", 1.0)
latch.rename("old_raw", "new_raw")
check("a re-identified cart carries its state to the new raw id",
      latch.resolve("new_raw", COMOVEMENT_STATIC_PX + 0.01),
      "re-ID resets the position history; losing the latch too would call the "
      "same physical cart moving on the frame it was re-found")
latch.forget("new_raw")
check("forget drops it", not latch.resolve("new_raw", COMOVEMENT_STATIC_PX + 0.01))


print(f"\n{_passed} passed, {_failed} failed")
if _failed:
    raise SystemExit(1)
print("All green.")
