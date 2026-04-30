# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Per-object orientation provider for the 2D BEV.

v1 uses *movement heading* as a proxy for body-facing yaw. A stationary
object holds its last non-stationary heading so the rendered box doesn't
flicker back to 0 deg whenever the shopper pauses.

To swap in a real pose-derived body yaw later, replace
`attach_orientations` with a version that reads from a pose channel.
The renderer only consumes the `o` field on each slim-frame object;
no other code changes are required.
"""
from __future__ import annotations


_DEFAULT_STATIC = "STATIC"


def attach_orientations(slim_frames: list[dict],
                        static_status: str = _DEFAULT_STATIC) -> None:
    """Mutate slim_frames in place: set `o` (degrees) on every person and
    cart entry, holding the last non-stationary heading when an object's
    `s` (speed_status) is `static_status`."""
    last_p: dict[str, float] = {}
    last_c: dict[str, float] = {}

    for fr in slim_frames:
        for k, obj in fr.get("p", {}).items():
            last_p[k] = _next_orientation(obj, last_p.get(k), static_status)
            obj["o"] = last_p[k]
        for k, obj in fr.get("c", {}).items():
            last_c[k] = _next_orientation(obj, last_c.get(k), static_status)
            obj["o"] = last_c[k]


def _next_orientation(obj: dict, prev: float | None,
                      static_status: str) -> float:
    d = obj.get("d")
    s = obj.get("s")
    if d is None:
        return prev if prev is not None else 0.0
    if s == static_status and prev is not None:
        return prev
    return float(d)
