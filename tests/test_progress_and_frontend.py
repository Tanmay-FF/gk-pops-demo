# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Regressions for the two defects that made a finished run look like a hung
page. Both are invisible in code review and produce no error anywhere.

Run with:  python tests/test_progress_and_frontend.py

  1. PROGRESS LEAK. `gradio.helpers.Progress` keeps `self.iterables` on the
     instance and pops an entry only when the tracked iterator raises
     StopIteration. The frame loop breaks out early (CAP_PROP_FRAME_COUNT
     over-reports), so every run left one entry behind — and with
     `progress=gr.Progress()` as a DEFAULT ARGUMENT there was one instance for
     the whole process. Progress reports the entire list on every step, so run
     N drew N progress bars, N-1 of them frozen mid-run forever.

  2. OBSERVER AMPLIFICATION. static/app.js binds behaviour from a
     MutationObserver over the whole body subtree, and its callback walks the
     whole document. Uncoalesced, rendering a result set multiplies thousands
     of Svelte mutations by a full-document pass each, on the main thread.

Deliberately stdlib + gradio only, no pytest — matches the other suites.
"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name, cond, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title):
    print(f"\n=== {title} ===")


_APP_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "static", "app.js")


def test_no_shared_progress_default():
    section("The frame loop's progress tracker cannot outlive its run")
    from engine.tracker import TrackingEngine

    sig = inspect.signature(TrackingEngine.process_video)
    default = sig.parameters["progress"].default
    check("process_video has no gr.Progress() default argument",
          default is None,
          f"default={type(default).__name__}")

    # Comments explain why tqdm is avoided, so match against code lines only.
    src = inspect.getsource(TrackingEngine.process_video)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("#"))
    check("the frame loop does not iterate progress.tqdm(...)",
          "progress.tqdm(" not in code)
    check("progress is reported by call, which never appends to iterables",
          "progress((" in code)


def test_progress_call_never_accumulates():
    section("Breaking out of the frame loop leaves no stale progress bars")
    import gradio as gr
    from gradio.helpers import Progress

    sent = []
    # Stand in for blocks._queue.set_progress(event_id, ...) — outside an event
    # context the real callback is None and Progress becomes a no-op, which
    # would make this test pass for the wrong reason.
    Progress._progress_callback = staticmethod(
        lambda: (lambda iterables: sent.append(len(iterables))))

    # Our pattern: three runs, every one breaking early.
    ours = gr.Progress()
    for _ in range(3):
        for i in range(417):
            if i > 399:
                break                      # cap.read() returned ok=False
            ours((i + 1, 417), desc="Processing frames")
    check("no iterables retained after three early-breaking runs",
          len(ours.iterables) == 0, f"{len(ours.iterables)} retained")
    check("every run reports exactly one progress bar",
          sent and max(sent) == 1, f"max bars = {max(sent) if sent else 0}")

    # The pattern we moved away from, for contrast. Informational: if a future
    # gradio pops on close as well, this stops leaking and the note is stale.
    legacy = gr.Progress()
    for _ in range(3):
        for n, _v in enumerate(legacy.tqdm(range(417), desc="Processing frames")):
            if n > 399:
                break
    print(f"      (for contrast: progress.tqdm + break retained "
          f"{len(legacy.iterables)} across 3 runs)")


def test_observer_is_coalesced():
    section("The DOM binder cannot run once per mutation")
    js = open(_APP_JS, encoding="utf-8").read()

    check("observer callback is the scheduler, not the binder directly",
          "new MutationObserver(_schedule)" in js)
    check("passes are coalesced onto an animation frame",
          "requestAnimationFrame(_runBind)" in js)
    check("only one pass can be queued at a time",
          "if (_queued" in js and "_queued = true;" in js)
    check("the observer detaches while the binder writes to the DOM",
          "obs.disconnect()" in js and "obs.takeRecords()" in js)
    check("re-observes after the pass, or binding dies after the first run",
          js.count("obs.observe(document.body") >= 2)

    # The expensive selector must shrink as the page settles. Three of its
    # matchers are case-insensitive substring attribute tests, which have no
    # fast path — every element in the document is tested on every pass.
    sel_start = js.find("var defensive = document.querySelectorAll(")
    sel = js[sel_start:js.find(");", sel_start)]
    n_matchers = sel.count(":not([data-gk-oc])")
    n_total = sel.count(",") + 1          # comma-separated selector list
    check("every defensive matcher excludes already-stamped buttons",
          n_matchers == n_total, f"{n_matchers} of {n_total} guarded")
    check("buttons are stamped so they drop out of the match set",
          "setAttribute('data-gk-oc'" in js)
    check("binding cost is measurable from the console",
          "__gkBindStats" in js)


def test_json_preview_is_small():
    section("The JSON viewer gets a glance, not a document")
    import app_poc_v2 as app

    check("preview cap is a few hundred lines, not a few thousand",
          app.JSON_PREVIEW_BYTES <= 50_000, f"{app.JSON_PREVIEW_BYTES} bytes")
    big = "\n".join('  {"frame": %d, "x": 1234},' % i for i in range(100_000))
    prev = app._json_preview(big)
    check("a huge document is actually truncated",
          len(prev) < app.JSON_PREVIEW_BYTES + 1000,
          f"{len(prev)} bytes out")
    check("the truncation is disclosed, not silent",
          "Preview" in prev and "truncated" in prev)


def test_badge_guard_survives_rerender():
    section("Tab badges cannot re-apply themselves in a loop")
    js = open(_APP_JS, encoding="utf-8").read()

    check("the applied-payload guard is window-scoped, not on a Gradio node",
          "raw === window.__gkBadgesApplied" in js)
    check("the guard is set BEFORE the DOM writes, so re-entry cannot loop",
          js.index("window.__gkBadgesApplied = raw")
          < js.index("_allTabButtons().forEach"))
    check("no guard is read off host.dataset before mutating",
          "raw === host.dataset.gkApplied" not in js)

    # The stronger invariant, and the one that actually mattered. Appending a
    # <span> into a Svelte-rendered tab button made Svelte's reconciliation
    # effect re-run, which re-triggered the write, until Svelte 5 threw
    # `effect_update_depth_exceeded` from inside flush() — hundreds of times.
    # An exception mid-flush aborts the update, leaving pending overlays that
    # never clear and toasts that will not dismiss: the page looked frozen at
    # 100% with a ~40 KB payload and a server that had already finished.
    #
    # The guard above only stopped OUR re-entry. It could not stop Svelte's,
    # because the loop ran inside Svelte. Not writing children is what stops it.
    # CODE only — the comments in that function necessarily mention appendChild
    # to explain why it must not be used, and a substring test cannot tell prose
    # from an actual call.
    _start = js.index("function _syncTabBadges")
    _end = js.index("\n    function ", _start + 1)
    badge_fn = "\n".join(
        ln for ln in js[_start:_end].splitlines()
        if not ln.strip().startswith("//")
    )
    check("badges never append a child into a Svelte-owned tab button",
          "appendChild" not in badge_fn)
    check("badges never create an element at all",
          "createElement" not in badge_fn)
    check("badges are written as an attribute instead",
          "setAttribute('data-gk-badge'" in badge_fn)
    check("the attribute write is skipped when unchanged",
          "getAttribute('data-gk-badge') !== text" in badge_fn)
    check("host.dataset is not written either",
          "host.dataset.gkApplied = raw" not in js)

    css = open(os.path.join(os.path.dirname(_APP_JS), "app.css"),
               encoding="utf-8").read()
    # ::before, not ::after — the badge sits ahead of the label in flex order
    # (see app.css). The pseudo-element it uses is incidental; that it is a
    # pseudo-element fed by the attribute, and not a child node, is the point.
    check("the badge is rendered by CSS from the attribute",
          "[data-gk-badge]::before" in css and "content: attr(data-gk-badge)" in css)

    section("A runaway binder disables itself instead of locking the tab")
    check("there is a time budget", "BIND_BUDGET_MS" in js)
    check("there is a call-count budget", "BIND_BUDGET_CALLS" in js)
    check("over budget the observer is not re-attached",
          "__gkBindGaveUp = true" in js
          and "// deliberately not re-observed" in js)
    check("the scheduler also stops once it has given up",
          "if (_queued || window.__gkBindGaveUp) return;" in js)
    check("giving up is reported, not silent",
          "re-binding disabled after" in js)


def test_toggles_fire_once_per_click():
    section("A toggle bound twice is a toggle that does nothing")
    js = open(_APP_JS, encoding="utf-8").read()

    # The Zone Editor's Fullscreen button did nothing for exactly this reason.
    # _bindAll() did BOTH of these to the same element:
    #     el.onclick = fn;  el.addEventListener('click', fn);
    # The onclick IDL attribute registers an internal wrapper, not `fn`, so
    # addEventListener's same-function dedup does not apply — two entries in
    # the listener list, two runs per click. classList.toggle twice is a no-op,
    # it happens synchronously so there is no flicker, and nothing errors.
    # The only safe rule is: the sole thing this file ever assigns to .onclick
    # is the no-op that keeps Gradio's router from throwing.
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("//"))
    handlers = re.findall(r"\.onclick\s*=\s*([A-Za-z_$][\w$.\[\]']*)", code)
    check("nothing but NOOP is ever assigned to .onclick",
          handlers and all(h == "NOOP" for h in handlers),
          f"assigned: {sorted(set(handlers))}")
    check("no per-element click listener is registered alongside it",
          "el.addEventListener('click'" not in code)

    # Real work is delegated off `document`, which is registered once and
    # survives both Gradio re-rendering the gr.HTML and the circuit breaker
    # switching re-binding off. Binding in _bindAll alone would not.
    check("there is exactly one delegated document click listener",
          code.count("document.addEventListener('click'") == 1)
    for sel, what in (("#gk-zone-fs-btn, .gk-fs-btn", "zone fullscreen"),
                      ("#gk-theme-toggle, .gk-theme-toggle", "theme"),
                      ("#json-fullscreen-close", "JSON overlay close")):
        check(f"{what} is reached by delegation", f"closest('{sel}')" in code)

    # Buttons the substring matchers cannot see must be listed by hand, or
    # they keep onclick === null and Gradio's router throws on them.
    sel_start = code.find("var defensive = document.querySelectorAll(")
    sel = code[sel_start:code.find(");", sel_start)]
    for name in (".gk-fs-btn", ".gk-theme-toggle", "#json-fullscreen-close"):
        check(f"{name} still gets a callable onclick", name in sel)
    check("no button is skipped out of the defensive pass",
          "continue;" not in sel and "b.id === 'gk-zone-fs-btn'" not in code)

    # Page-level overlay, not the browser's own fullscreen — Escape is the
    # only exit an operator will reach for by reflex.
    check("Escape leaves zone fullscreen",
          "ev.key === 'Escape'" in code and "_toggleZoneFs()" in code)


def test_fullscreen_actually_enlarges_the_frame():
    section("Zone fullscreen must grow the frame, not just centre it")
    raw = open(os.path.join(os.path.dirname(_APP_JS), "app.css"),
               encoding="utf-8").read()
    # CODE only. The comments below necessarily name `.gradio-image` and
    # `calc(100vh - ...)` to explain why neither is used, and a substring test
    # cannot tell an explanation from a rule.
    css = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)

    # Measured in Edge against real Gradio 6.12: before this, fullscreen left
    # the frame at 807x454 painted — the exact size it is in the normal view,
    # just centred on black. The rules meant to enlarge it were written against
    # `.gradio-image`, a class Gradio 6 does not emit, so they matched nothing.
    # Gradio 6.12 renders gr.Image as
    #     div#zone-canvas-img.block > div.image-container > button >
    #         div.image-frame > img
    check("no rule targets .gradio-image, which Gradio 6 never emits",
          ".gradio-image" not in css)

    fs = css[css.find("#zone-editor-fs-wrap.gk-fs-active"):
             css.find("body.gk-zone-fs")]
    for cls in (".image-container", ".image-frame"):
        check(f"fullscreen sizes the real Gradio 6 node {cls}", cls in fs)

    # gr.Image(height=460) lands as an INLINE style on the block, so only an
    # !important rule outranks it. Without this the frame cannot exceed 460px
    # however much room the overlay has.
    block = fs[fs.find("#zone-editor-fs-wrap.gk-fs-active #zone-canvas-img {"):]
    block = block[:block.find("}")]
    check("the inline height=460 on the image block is overridden",
          "height: auto !important" in block)

    # scale-down refuses to enlarge past natural size — the one thing
    # fullscreen exists to do. contain enlarges and still letterboxes, which
    # is what Gradio's click->pixel maths assumes; fill/cover would put every
    # polygon vertex in the wrong place.
    check("the frame is fitted with contain, not scale-down",
          "object-fit: contain !important" in fs)
    check("nothing re-pins the frame to a constant viewport offset",
          "calc(100vh" not in fs)

    # The <img> box must shrink-wrap the frame, or the crosshair (and the
    # click target) extend over the letterbox, inviting vertices that Gradio
    # silently drops.
    check("the frame is not stretched past its aspect ratio",
          "width: auto !important" in fs and "max-width: 100% !important" in fs)
    # Gradio's floating Download/Share sit on the drawing surface. The wrapper
    # is .icon-button-wrapper; the older rule guessed .icon-buttons and missed.
    check("Gradio's floating icons are off the drawing surface",
          ".icon-button-wrapper" in fs)


def test_bisection_switches():
    section("A locked tab can be bisected from outside the browser")
    import app_poc_v2 as app

    for name in ("GK_SAFE_MODE", "GK_NO_APP_JS", "GK_NO_JSON_VIEWER",
                 "GK_NO_VIDEO", "GK_NO_HEATMAP", "GK_NO_PANELS"):
        src = inspect.getsource(app)
        check(f"{name} is documented in the module", name in src)

    # Switches must only blank the OUTGOING values, never the pipeline.
    apply_src = inspect.getsource(app._apply_switches)
    check("switches act on the output list only",
          "out[_IDX[" in apply_src and "process_video" not in apply_src)
    check("every blanked panel slot is a real output name",
          all(n in app._IDX for n in app._PANEL_SLOTS),
          str([n for n in app._PANEL_SLOTS if n not in app._IDX]))

    # Blanking must not change the arity — a safe-mode run has to fill every
    # output or Gradio maps values onto the wrong components.
    stub = list(range(app.N_RUN_OUTPUTS - 1))
    app._apply_switches(stub)
    check("blanking preserves the output count",
          len(stub) == app.N_RUN_OUTPUTS - 1, f"{len(stub)} slots")


def test_new_run_flushes_the_previous_one():
    section("A new run cannot render on top of the last run's findings")
    import gradio as gr
    import app_poc_v2 as app

    # ── shape ────────────────────────────────────────────────────────────
    check("_blank_panels covers exactly the run's output slots",
          set(app._blank_panels()) == set(app.RUN_OUTPUT_NAMES),
          str(set(app._blank_panels()) ^ set(app.RUN_OUTPUT_NAMES)))
    flushed = app.flush_results("clip.mp4")
    check("flush_results returns one value per flushed component",
          len(flushed) == len(app.FLUSH_OUTPUT_NAMES),
          f"{len(flushed)} values, {len(app.FLUSH_OUTPUT_NAMES)} components")
    check("the early-exit stub still fills every slot",
          len(app._empty_run_outputs("x", "y")) == app.N_RUN_OUTPUTS)

    by_name = dict(zip(app.FLUSH_OUTPUT_NAMES, flushed))

    # ── what must actually be gone ───────────────────────────────────────
    # The panels that carry the findings. A stale POPS table or event log read
    # as the CURRENT run's — that is the whole defect.
    for name in ("pops_html", "events_html", "ops_alerts_html",
                 "analytics_summary_html"):
        check(f"{name} is replaced, not left untouched",
              isinstance(by_name[name], str) and by_name[name].strip() != ""
              and "gk-empty" in by_name[name])
    for name in ("video_output", "json_download", "case_report_download",
                 "heatmap_image", "heatmap_file"):
        check(f"{name} is dropped", by_name[name] is None)
    check("the sticky alert banner is emptied AND hidden",
          by_name["alert_banner_html"] == gr.update(value="", visible=False))
    check("tab badges go back to no counts",
          "data-counts='{}'" in by_name["tab_counts_html"])
    check("the run summary strip is cleared", by_name["run_summary_html"] == "")

    # ── what must survive ────────────────────────────────────────────────
    # A gr.Tabs config update is the one thing that wedges Gradio's client
    # here, and the legend is static reference content, not a finding.
    check("the flush never touches the Tabs component",
          "result_tabs" not in app.FLUSH_OUTPUT_NAMES)
    check("the flush never blanks the legend",
          "legend_html" not in app.FLUSH_OUTPUT_NAMES)

    # ── a cleared page and a fresh page are the same page ────────────────
    fresh = app._blank_panels(has_video=False)
    idle = app.flush_results(None)
    check("flushing with no video reproduces the initial values exactly",
          all(fresh[n] == v for n, v in zip(app.FLUSH_OUTPUT_NAMES, idle)))

    # ── wiring: the flush must be SEQUENCED before the run, not parallel ──
    # Two handlers on one trigger are dispatched together, so a second
    # run_btn.click() could land after the results and wipe them. The order
    # only holds if the run is a .then() of the flush.
    deps = app.demo.get_config_file()["dependencies"]
    by_id = {d["id"]: d for d in deps}
    flush_deps = [d for d in deps
                  if len(d["outputs"]) == len(app.FLUSH_OUTPUT_NAMES)
                  and len(d["inputs"]) == 1]
    check("a flush event exists", len(flush_deps) >= 1, f"{len(flush_deps)} found")
    run_deps = [d for d in deps if len(d["outputs"]) == app.N_RUN_OUTPUTS]
    check("exactly one event writes every run output", len(run_deps) == 1)
    run_dep = run_deps[0]
    parent = by_id.get(run_dep.get("trigger_after"))
    check("the run is chained AFTER a flush, not fired alongside it",
          parent is not None
          and len(parent["outputs"]) == len(app.FLUSH_OUTPUT_NAMES))
    check("the run button itself triggers the flush, not the run",
          parent is not None and parent["targets"]
          and parent["targets"][0][1] == "click")
    # The two follow-ups (js tab switch, deferred case report) must still hang
    # off the RUN, or the case report never runs at all.
    followups = [d for d in deps if d.get("trigger_after") == run_dep["id"]]
    check("both run follow-ups still chain off the run event",
          len(followups) == 2, f"{len(followups)} found")
    check("the flush paints no progress trackers",
          parent is not None and parent.get("show_progress") == "hidden",
          str(parent.get("show_progress")) if parent else "no parent")


def main():
    print("=" * 66)
    print("PROGRESS / FRONT-END REGRESSION TESTS")
    print("=" * 66)
    test_observer_is_coalesced()
    test_badge_guard_survives_rerender()
    test_toggles_fire_once_per_click()
    test_fullscreen_actually_enlarges_the_frame()
    test_bisection_switches()
    test_new_run_flushes_the_previous_one()
    test_json_preview_is_small()
    test_no_shared_progress_default()
    test_progress_call_never_accumulates()

    print("\n" + "=" * 66)
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
