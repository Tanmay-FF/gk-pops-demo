# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Motion analysis — speed, direction labels, co-movement detection.

All functions operate on raw position/timestamp history dicts to avoid
coupling to any particular tracker class.
"""
import bisect
import math
from .config import (
    SPEED_STATIC, SPEED_SLOW, SPEED_MEDIUM,
    COMOVEMENT_MIN_POSITIONS, COMOVEMENT_WINDOW,
    COMOVEMENT_STATIC_PX, COMOVEMENT_STATIC_EXIT_PX, COMOVEMENT_COS_THRESH,
    DIRECTION_MIN_POSITIONS, DIRECTION_MIN_DY,
)


def compute_motion(positions: list, timestamps: list, speeds: list, fps: float):
    """Compute speed, direction angle, speed status, acceleration.

    Returns (speed, direction_deg, status_str, acceleration).
    """
    n_pos = len(positions)
    if n_pos < 2:
        return 0.0, 0.0, "STATIC", 0.0

    n = min(5, n_pos)
    recent = positions[-n:]
    ts     = timestamps[-n:]
    dt = ts[-1] - ts[0]
    if dt < 0.01:
        return 0.0, 0.0, "STATIC", 0.0

    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    dist = math.sqrt(dx * dx + dy * dy)
    speed = dist / dt
    direction = math.degrees(math.atan2(dy, dx)) % 360

    accel = 0.0
    if len(speeds) >= 2 and dt > 0:
        accel = (speeds[-1] - speeds[-2]) * fps

    if speed < SPEED_STATIC:
        status = "STATIC"
    elif speed < SPEED_SLOW:
        status = "SLOW"
    elif speed < SPEED_MEDIUM:
        status = "MEDIUM"
    else:
        status = "FAST"

    return speed, direction, status, accel


def _window_start(positions: list, timestamps: list | None,
                  window_s: float | None) -> int:
    """Index to measure the direction delta FROM.

    0 (whole history) unless a time window is requested, in which case it is
    the first sample within `window_s` of the newest one — always leaving at
    least two samples so the delta is never degenerate.

    Callers that already hand in a pre-sliced window (rules._incoming_empty_rule
    slices to RULE_ENTRY_WINDOW_S itself, and says so) simply omit both
    arguments and keep the whole-list behaviour.
    """
    n = len(positions)
    if not window_s or not timestamps or n < 2:
        return 0
    n = min(n, len(timestamps))
    if n < 2:
        return 0
    cutoff = timestamps[n - 1] - window_s
    # timestamps are appended in frame order, so bisect beats a linear scan on
    # the long histories this exists to protect against.
    lo = bisect.bisect_left(timestamps, cutoff, 0, n)
    return min(lo, n - 2)


def compute_direction_label(positions: list, camera_placement: str,
                            timestamps: list | None = None,
                            window_s: float | None = None) -> str:
    """Determine INBOUND / OUTBOUND / UNKNOWN from position delta.

    With `timestamps` + `window_s`, the delta is measured over the last
    `window_s` seconds instead of the whole track. See DIRECTION_WINDOW_S in
    config for why that matters: `_obj_positions` is never trimmed, so a
    whole-track delta cancels out for anyone who enters and leaves through the
    same door — the single most important case this label feeds.

    DIRECTION_MIN_POSITIONS still gates on the FULL history: it exists to
    reject a track too new to have a heading at all, which is a different
    question from how far back to measure.
    """
    if len(positions) < DIRECTION_MIN_POSITIONS:
        return "UNKNOWN"
    i0 = _window_start(positions, timestamps, window_s)
    dx = positions[-1][0] - positions[i0][0]
    dy = positions[-1][1] - positions[i0][1]

    if camera_placement == "Inside (exit on right)":
        if abs(dx) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        return "OUTBOUND" if dx > 0 else "INBOUND"
    elif camera_placement == "Inside (exit on left)":
        if abs(dx) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        return "OUTBOUND" if dx < 0 else "INBOUND"
    elif camera_placement == "Inside (exit on both sides)":
        if abs(dx) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        return "OUTBOUND"  # moving left or right toward either exit
    else:
        # Vertical axis: "Outside (facing entrance)" or "Inside (facing exit)"
        if abs(dy) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        if camera_placement == "Outside (facing entrance)":
            return "OUTBOUND" if dy > 0 else "INBOUND"
        return "OUTBOUND" if dy < 0 else "INBOUND"


def outbound_axis_value(cx: float, cy: float, camera_placement: str):
    """Signed scalar that GROWS as an object moves toward the exit.

    Mirrors compute_direction_label()'s placement branches, so the two cannot
    disagree about which way "out" is. Returns None when the placement has no
    single outbound axis - "exit on both sides" is genuinely two axes, and a
    cart there can retreat from one exit while advancing on the other.

    Read by DirectionLatch, which needs a POSITION and not a heading: a heading
    says which way the cart is pointing this instant, and the question the latch
    asks on a resolved INBOUND frame is whether the cart has actually travelled
    back from where it got to.
    """
    if camera_placement == "Inside (exit on right)":
        return cx
    if camera_placement == "Inside (exit on left)":
        return -cx
    if camera_placement == "Inside (exit on both sides)":
        return None
    if camera_placement == "Outside (facing entrance)":
        return cy
    return -cy


class DirectionLatch:
    """Holds a cart's last SUSTAINED outbound heading through UNKNOWN frames.

    compute_direction_label() answers "which way has this object moved over the
    last DIRECTION_WINDOW_S seconds", and for a cart that reaches the door and
    stops, the honest answer becomes UNKNOWN the moment that window empties of
    the approach. On the 1763942209000 OUTSIDE clip it happened at frame 169:
    the cart parked at y~290 from frame 105, the trailing 4s delta fell under
    DIRECTION_MIN_DY (17px), and the label read UNKNOWN for the remaining 13s.

    That is not a cosmetic label change. The UNKNOWN branch of compute_pops()
    caps an abandoned loaded cart at 65, so a cart standing outside the door
    full of unbagged merchandise scored ABANDONED CART on every one of those
    frames, while the end-of-run finaliser — which reads direction off the peak
    EVENT, logged at frame 104 while the cart was still moving — reported
    PUSHOUT ALERT. One cart, two directions, and the annotated video and the
    POPS table disagreed for 13 of the clip's 22 seconds. The alert appeared
    for 15 frames: 153-167, the only frames where `abandoned` and OUTBOUND
    overlapped.

    A cart that has stopped has not changed its mind about where it was going,
    so the last resolved heading is held while the label reads UNKNOWN. Three
    rules keep that from becoming a licence to over-score:

      * OUTBOUND only. INBOUND is never latched. It is a kill switch worth
        INBOUND_SCORE (5) before contents, speed or abandonment are looked at,
        so holding it would silence the UNKNOWN abandonment tier for any cart
        that ever rolled inbound — including a cart wheeled in, loaded, and
        left.
      * A RESOLVED label always beats the latch, and a resolved INBOUND clears
        it. That is what keeps a staff member wheeling the cart back inside
        from being scored: their push resolves INBOUND and the score drops to
        the kill switch on the same frame it would have without this class.
      * Latching needs `min_frames` CONSECUTIVE resolved OUTBOUND frames
        (DIRECTION_LATCH_FRAMES). DIRECTION_MIN_DY is 20px, so a single twitch
        of the trailing window can resolve OUTBOUND on a cart parked inside the
        store, and holding that forever would move a shopper who parks at a
        shelf and steps away from 65 (ABANDONED CART) to 75 (PUSHOUT ALERT) —
        precisely the false positive that cap exists to prevent.

    Keyed by whatever the caller passes. tracker.py passes the cart DISPLAY id,
    matching _walkaway_frames and _cart_cls_history, so the latch survives a
    re-ID that hands the same cart a new raw tracker id — the case where the
    position history resets and the heading would be lost anyway.

    Stateful, so it lives here as a class rather than as another argument to
    compute_direction_label(); the label function stays pure.
    """

    def __init__(self, min_frames: int, reversal_px: float = None):
        """reversal_px: how far back along the outbound axis a cart must have
        travelled before a resolved INBOUND frame is allowed to clear its latch.
        None (the default) keeps the original rule: the first resolved INBOUND
        clears it. config.LATCH_REVERSAL_PX ships as 40.

        The distinction matters for a cart that reaches a doorway and settles.
        DIRECTION_MIN_DY is 20 px, so a 25 px drift back inside resolves INBOUND
        without the cart having gone anywhere, and clearing the latch on that
        costs the cart the OUTBOUND branch of compute_pops() - the +15 base, the
        loose-merchandise term and the 75 abandonment floor - for the rest of
        the run.
        """
        self._min_frames = min_frames
        self._reversal_px = reversal_px
        #: key -> consecutive resolved-OUTBOUND frames seen so far
        self._run: dict = {}
        #: keys whose run reached _min_frames
        self._latched: set = set()
        #: key -> furthest-out axis value the key reached while OUTBOUND,
        #: on the axis outbound_axis_value() defines. Only consulted when
        #: _reversal_px is set.
        self._peak_pos: dict = {}

    def resolve(self, key, label: str, pos: float = None) -> str:
        """Return the heading to score `key` with this frame.

        Pass the label AFTER any other adjustment the caller makes (tracker.py
        stamps a linked person's heading onto their cart first), so the latch
        records the heading actually used.
        """
        if label == "OUTBOUND":
            run = self._run.get(key, 0) + 1
            self._run[key] = run
            if run >= self._min_frames:
                self._latched.add(key)
            if pos is not None and pos > self._peak_pos.get(key, float("-inf")):
                self._peak_pos[key] = pos
            return "OUTBOUND"

        if label == "INBOUND":
            # Resolved reversal - forget everything about the outbound leg,
            # but, when a reversal distance is configured, only once the cart
            # has actually TRAVELLED back that far. A cart with no recorded peak
            # never latched in the first place, so there is nothing to protect.
            retreated = True
            if (self._reversal_px is not None and pos is not None
                    and key in self._peak_pos):
                retreated = (self._peak_pos[key] - pos) >= self._reversal_px
            self._run.pop(key, None)
            if retreated:
                self._latched.discard(key)
                self._peak_pos.pop(key, None)
            return "INBOUND"

        # UNKNOWN. The run must be CONSECUTIVE, so an outbound/unknown flicker
        # never accumulates its way to a latch; an undetected frame calls
        # nothing and leaves the run intact, which is the intent.
        self._run.pop(key, None)
        return "OUTBOUND" if key in self._latched else label

    def is_latched(self, key) -> bool:
        """Whether `key`'s heading is currently being held. For diagnostics."""
        return key in self._latched


class StaticLatch:
    """Hysteresis over "is this track standing still", keyed by track id.

    A bare `displacement < COMOVEMENT_STATIC_PX` is a knife edge, and a track
    parked with its wheels being nudged sits right on it. On
    1764099569430_B8A44F3CB0B9-medium-OUTSIDE.mp4 Cart 21 hovered at 2-4 px of
    window displacement, touched 5.0016 px on frame 162 under one torch/CUDA
    build and 4.9992 px under another, and that one frame moved the cart's link
    by 14 frames — see COMOVEMENT_STATIC_EXIT_PX for the full chain. The band
    here is what stops a wobble at the bar from being a decision.

    Static below COMOVEMENT_STATIC_PX, moving above COMOVEMENT_STATIC_EXIT_PX,
    and inside the band the track keeps the state it already had. A track seen
    for the first time inside the band is judged by the lower bar, which is the
    same answer the un-latched test gave.

    Stateful, so it lives here as a class rather than as another argument to
    are_co_moving(); the function stays pure when no latch is passed.
    """

    __slots__ = ("_static",)

    def __init__(self):
        #: key -> whether that track was last judged static
        self._static: dict = {}

    def reset(self):
        self._static.clear()

    def forget(self, key):
        """Drop a track that is gone, so ids are not held forever."""
        self._static.pop(key, None)

    def rename(self, old_key, new_key):
        """Carry a track's state to the id it was re-identified under."""
        if old_key in self._static:
            self._static[new_key] = self._static.pop(old_key)

    def resolve(self, key, magnitude: float) -> bool:
        """Whether `key` counts as static this frame, given its displacement."""
        was_static = self._static.get(key)
        if was_static is None:
            is_static = magnitude < COMOVEMENT_STATIC_PX
        elif was_static:
            is_static = magnitude <= COMOVEMENT_STATIC_EXIT_PX
        else:
            is_static = magnitude < COMOVEMENT_STATIC_PX
        self._static[key] = is_static
        return is_static


def are_co_moving(pos_a: list, pos_b: list, static_a_ok: bool = False,
                  latch: "StaticLatch | None" = None,
                  key_a=None, key_b=None) -> bool:
    """Check if two tracked objects share similar velocity direction.

    A bystander standing still while a cart rolls past will return False.

    `static_a_ok` relaxes exactly one case: A is static while B moves. The
    linker passes it with the CART as A, because "parked cart, person moving
    around it" is what loading or unloading looks like and it is not evidence
    against ownership — while "static person, cart rolling past" is the bystander
    the asymmetric rule exists to reject, and that case is B static, so it still
    returns False.

    Without this the linker could not see the 1764099569430 clip's real handler
    at all: over the 49 frames he spent at that cart, the plain rule rejected 19
    of them, including the unbroken run 48-59 where his overlap with the cart was
    at its strongest (IoU 0.24-0.42) because the cart had come to rest and he was
    working at it. The cart was left with no owner, and an ownerless cart cannot
    be scored as abandoned.
    """
    min_pos = COMOVEMENT_MIN_POSITIONS
    if not pos_a or not pos_b or len(pos_a) < min_pos or len(pos_b) < min_pos:
        return True  # not enough history — allow overlap-only linking

    n = min(COMOVEMENT_WINDOW, len(pos_a), len(pos_b))
    vax = pos_a[-1][0] - pos_a[-n][0]
    vay = pos_a[-1][1] - pos_a[-n][1]
    vbx = pos_b[-1][0] - pos_b[-n][0]
    vby = pos_b[-1][1] - pos_b[-n][1]

    mag_a = math.sqrt(vax * vax + vay * vay)
    mag_b = math.sqrt(vbx * vbx + vby * vby)

    # With a latch, "static" carries hysteresis, so a track sitting on the bar
    # cannot flip the answer frame to frame — see StaticLatch. Without one the
    # test is the bare threshold it always was, which keeps this function pure
    # for every caller that does not track identities.
    if latch is not None and key_a is not None and key_b is not None:
        a_static = latch.resolve(key_a, mag_a)
        b_static = latch.resolve(key_b, mag_b)
    else:
        a_static = mag_a < COMOVEMENT_STATIC_PX
        b_static = mag_b < COMOVEMENT_STATIC_PX

    if a_static and b_static:
        return True
    if a_static != b_static:
        return bool(static_a_ok and a_static)

    dot = vax * vbx + vay * vby
    cos_sim = dot / (mag_a * mag_b + 1e-9)
    return cos_sim > COMOVEMENT_COS_THRESH
