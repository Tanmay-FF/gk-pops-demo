# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The one name-to-position map for `TrackingEngine.process_video()`'s output.

`process_video()` returns a positional tuple. Until this module existed the only
map from a name to a slot in it lived in `app_poc_v2.py`, behind a module-scope
`gr.Blocks(...)`, so importing it built the whole Gradio app, and any caller
that was not the demo had to count the slots by hand.

`api/main.py:207` is what counting by hand costs. It positionally unpacks 19
names from a tuple the engine grew to 21, so `/api/run` raises `ValueError`
before returning anything. The comment above that line says so. `as_dict()`
below is the guard it lacks: a length mismatch raises here, naming both counts,
instead of silently mapping values onto the wrong names.

The names are the UI's, not the engine's local variable names (`out_path`,
`json_path`, `json_str`, `case_report_file`). That is deliberate: the map has
to match the consumer that already has call sites, not the producer.
`app_poc_v2` reads these names in ~15 places, derives `FLUSH_OUTPUT_NAMES` from
them, and asserts key-set equality against both `_blank_panels()` and
`RUN_COMPONENTS`, so renaming one would be a `KeyError` plus two failed
import-time asserts.

Deliberately import-free. Everything else under `engine/` pulls in torch,
torchvision or ultralytics by way of `engine/__init__.py`; this module holds a
list of strings, and a test that only wants to check the tuple's length should
not pay six seconds for it.

Adding an output slot to `_process_video`'s return means adding its name here,
in the same position. Both callers then either keep working or fail loudly.
"""
from __future__ import annotations

from typing import Any, Sequence

#: Every slot `_process_video` returns, in order. `app_poc_v2.RUN_OUTPUT_NAMES`
#: is this list plus the UI-only `result_tabs` at the end.
ENGINE_OUTPUT_NAMES: tuple[str, ...] = (
    "video_output", "json_download", "json_output",
    "video_info_html", "detection_html", "config_html", "legend_html",
    "pops_html", "events_html",
    "case_report_html", "case_report_download",
    "analytics_summary_html", "spikes_html", "dwell_html", "journey_html",
    "heatmap_image", "heatmap_file",
    "alert_banner_html", "ops_alerts_html",
    "run_summary_html", "tab_counts_html",
)

#: name -> index in the returned tuple.
ENGINE_IDX: dict[str, int] = {name: i for i, name in enumerate(ENGINE_OUTPUT_NAMES)}

N_ENGINE_OUTPUTS = len(ENGINE_OUTPUT_NAMES)


def as_dict(result: Sequence[Any]) -> dict[str, Any]:
    """Map one `process_video()` return value onto `ENGINE_OUTPUT_NAMES`.

    Args:
        result: The tuple (or any sequence) `process_video()` returned.

    Returns:
        `{name: value}` over every slot, in `ENGINE_OUTPUT_NAMES` order.

    Raises:
        RuntimeError: The sequence is not `N_ENGINE_OUTPUTS` long. Both counts
            are named, because the useful question when this fires is which
            side moved.
    """
    values = list(result)
    if len(values) != N_ENGINE_OUTPUTS:
        raise RuntimeError(
            f"process_video returned {len(values)} values; "
            f"engine.run_outputs.ENGINE_OUTPUT_NAMES has {N_ENGINE_OUTPUTS}. "
            f"The engine's return tuple and this map have drifted apart. Add "
            f"the new slot's name to ENGINE_OUTPUT_NAMES in the same position.")
    return dict(zip(ENGINE_OUTPUT_NAMES, values))
