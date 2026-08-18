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
    COMOVEMENT_STATIC_PX, COMOVEMENT_COS_THRESH,
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


def are_co_moving(pos_a: list, pos_b: list) -> bool:
    """Check if two tracked objects share similar velocity direction.

    A bystander standing still while a cart rolls past will return False.
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

    a_static = mag_a < COMOVEMENT_STATIC_PX
    b_static = mag_b < COMOVEMENT_STATIC_PX

    if a_static and b_static:
        return True
    if a_static != b_static:
        return False

    dot = vax * vbx + vay * vby
    cos_sim = dot / (mag_a * mag_b + 1e-9)
    return cos_sim > COMOVEMENT_COS_THRESH
