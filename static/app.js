(function () {
    if (window.__gkAppJs) return;
    window.__gkAppJs = true;

    // ─────────────────────────────────────────────────────────────────
    // GK POPS front-end behaviour.
    //
    // SELF-INVOKING, and it has to be. Gradio 6's launch(js=...) injects
    // this file VERBATIM into a <script> tag — it does not wrap it and it
    // does not call it. This file used to open with a bare `() => {`, which
    // as a script body is just an expression that evaluates to a function
    // and discards it: no error, no console warning, and every behaviour
    // below silently absent in the browser. The jsdom tests kept passing
    // because they invoke the function themselves.
    //
    // Everything here is delegated off `document`, because Gradio
    // re-renders panels on every run and per-element listeners would die
    // with them.
    //
    // Two Gradio-6 quirks shape this file:
    //   1. Inline onclick="..." attributes are stripped by the HTML
    //      sanitizer, so buttons rendered from Python HTML get their
    //      behaviour here, keyed by class or id.
    //   2. Gradio's Svelte router calls `element.onclick()` directly on a
    //      clicked button. If the property is null it throws
    //      "TypeError: i.onclick is not a function" and kills the click
    //      pipeline for the entire page. Every button we render therefore
    //      gets a real (no-op) .onclick, with the real work done in the
    //      delegated listeners below.
    // ─────────────────────────────────────────────────────────────────

    // MUST NOT return false. This is assigned to `.onclick` on buttons that
    // may be GRADIO'S OWN — the defensive pass below matches any button whose
    // aria-label/title contains "ull", which is how Gradio labels the
    // Fullscreen icon on video and image components, document-wide.
    //
    // Returning false from an onclick handler calls preventDefault(), so this
    // was cancelling the click on every one of those buttons: the whole point
    // of stamping them is to give Gradio's router something callable instead
    // of null, NOT to suppress what they do. Measured on a live run —
    // `dispatchEvent` came back `defaultPrevented: true` on Gradio's own
    // Fullscreen control. Returning undefined keeps it callable and inert.
    const NOOP = function () { /* deliberately empty — see above */ };

    // ── Zone editor fullscreen ────────────────────────────────────────
    // Bound by DELEGATION only (see the click listener below), never by
    // `el.onclick = _toggleZoneFs` + `addEventListener('click', _toggleZoneFs)`.
    // Those are two entries in the same listener list — the onclick IDL
    // attribute registers an internal wrapper, so addEventListener's
    // same-function dedup does not apply — and both fire on one click. Two
    // runs of a toggle is zero runs of a toggle, synchronously, with no
    // flicker to see: the Fullscreen button looked completely dead.
    // Temporarily replace the header-bar hint, then put it back. The zone
    // editor has no toast channel of its own, and gr.Info() is Python-side.
    var _hintTimer = null;
    function _flashZoneHint(msg) {
        var bar = document.querySelector('.zone-hdr-bar span');
        if (!bar) return;
        if (_hintTimer) { clearTimeout(_hintTimer); }
        else { bar.dataset.gkOrig = bar.textContent; }
        bar.textContent = msg;
        bar.classList.add('gk-hint-flash');
        _hintTimer = setTimeout(function () {
            bar.textContent = bar.dataset.gkOrig || '';
            bar.classList.remove('gk-hint-flash');
            _hintTimer = null;
        }, 2600);
    }

    // The panel to blow up, found WITHOUT relying on the elem_id.
    //
    // `getElementById('zone-editor-fs-wrap')` was the only lookup, and when it
    // came back null the handler logged and gave up — a dead button with an
    // explanation only visible in the console. The id is set on a gr.Group in
    // Python, so anything that re-renders or restructures that Group (and this
    // tab is re-rendered constantly: the zone pool alone drives 54 outputs
    // through it) can take the id with it.
    //
    // The button is INSIDE the panel by construction, so climbing from the
    // button cannot go out of date. The id stays as the first choice because
    // it is the most precise; the rest are fallbacks in decreasing specificity,
    // and the canvas is what fullscreen is actually for.
    function _zoneFsWrap(fromBtn) {
        var byId = document.getElementById('zone-editor-fs-wrap');
        if (byId) return byId;
        var btn = fromBtn || document.getElementById('gk-zone-fs-btn')
                  || document.querySelector('.gk-fs-btn');
        var canvas = document.getElementById('zone-canvas-img');
        if (!btn) return canvas ? canvas.parentElement : null;
        // Climb to the OUTERMOST group that still holds the canvas, not the
        // first one. Gradio stamps elem_id on both the outer and the inner
        // group div of a gr.Group, and getElementById returns the outer — so
        // the overlay CSS, which is written in terms of `> *`, is calibrated
        // against the outer one. Returning the inner div also goes fullscreen
        // but sizes the frame differently (measured: 519px against 717px).
        var node = btn.parentElement, outerGroup = null, firstHolder = null;
        while (node && node !== document.body) {
            if (!canvas || node.contains(canvas)) {
                if (!firstHolder) firstHolder = node;
                if (/gr-group|gradio-group/.test((node.className || '').toString())) {
                    outerGroup = node;          // keep going: want the last one
                }
            }
            node = node.parentElement;
        }
        return outerGroup || firstHolder
               || (btn.closest ? btn.closest('.gr-group, .block') : null);
    }

    function _toggleZoneFs(ev) {
        if (ev && ev.preventDefault) ev.preventDefault();
        if (ev && ev.stopPropagation) ev.stopPropagation();
        var btn  = (ev && ev.target && ev.target.closest)
                   ? ev.target.closest('#gk-zone-fs-btn, .gk-fs-btn') : null;
        btn = btn || document.getElementById('gk-zone-fs-btn');
        var wrap = _zoneFsWrap(btn);
        if (!wrap) {
            console.warn('[gk] zone editor panel not found - cannot go fullscreen');
            return false;
        }
        // The overlay styling keys off the id, so if we got here by climbing
        // rather than by lookup, put the id back. Next click is a plain
        // getElementById hit and the CSS matches as written.
        if (!wrap.id) { wrap.id = 'zone-editor-fs-wrap'; }
        // ALWAYS TOGGLE. An earlier version returned early here when no frame
        // was loaded, on the reasoning that a black overlay around Gradio's
        // empty-image placeholder is indistinguishable from a dead button.
        // That reasoning was right about the symptom and wrong about the cure:
        // a control that declines to act is ALSO indistinguishable from a dead
        // button, and it fails that way even when the user did nothing wrong.
        // The overlay opens either way; if there is nothing in it, it says so.
        var active = wrap.classList.toggle('gk-fs-active');
        if (active && !document.querySelector('#zone-canvas-img img')) {
            _flashZoneHint('No frame yet — upload a video to draw zones on.');
        }
        document.body.classList.toggle('gk-zone-fs', active);
        if (btn) {
            btn.innerHTML = active
                ? '☒︎&nbsp; Exit fullscreen'
                : '⛶︎&nbsp; Fullscreen';
            btn.classList.toggle('exiting', active);
        }
        window.dispatchEvent(new Event('resize'));
        return false;
    }

    // ── Theme ─────────────────────────────────────────────────────────
    function _prefersDark() {
        // Not every embedding browser exposes matchMedia (some kiosk shells
        // and webviews don't). An exception here used to abort the whole
        // IIFE, taking the table/seek/tab handlers down with it.
        try {
            return !!(window.matchMedia &&
                      window.matchMedia('(prefers-color-scheme: dark)').matches);
        } catch (e) { return false; }
    }
    function _effectiveTheme() {
        var root = document.documentElement;
        var body = document.body;
        if (root.classList.contains('gk-dark') || root.classList.contains('dark')
            || (body && body.classList.contains('dark'))) return 'dark';
        if (root.classList.contains('gk-light')) return 'light';
        return _prefersDark() ? 'dark' : 'light';
    }
    function _applyTheme(t) {
        var root = document.documentElement;
        var body = document.body;
        root.classList.remove('gk-dark', 'gk-light', 'dark');
        if (body) body.classList.remove('dark');
        if (t === 'dark') {
            root.classList.add('gk-dark', 'dark');
            if (body) body.classList.add('dark');
        } else if (t === 'light') {
            root.classList.add('gk-light');
        }
    }
    function _syncThemeBtn() {
        try {
            var btn = document.getElementById('gk-theme-toggle');
            if (!btn) return;
            var cur = _effectiveTheme();
            btn.textContent = cur === 'dark' ? '☀' : '\u{1F319}';
            btn.title = cur === 'dark' ? 'Switch to light mode'
                                       : 'Switch to dark mode';
        } catch (e) { /* cosmetic only — never break binding over it */ }
    }
    function _toggleTheme() {
        var next = _effectiveTheme() === 'dark' ? 'light' : 'dark';
        _applyTheme(next);
        try { localStorage.setItem('gk-theme', next); } catch (e) {}
        _syncThemeBtn();
        return false;
    }

    function _closeJsonOverlay() {
        var ov = document.getElementById('json-fullscreen-overlay');
        if (ov) ov.classList.remove('active');
        return false;
    }

    // ─────────────────────────────────────────────────────────────────
    // Tables — sort, filter, CSV export, show-all
    // ─────────────────────────────────────────────────────────────────
    function _bodyRows(table) {
        var tb = table.tBodies[0];
        return tb ? Array.prototype.slice.call(tb.rows) : [];
    }

    function _sortValue(td) {
        if (!td) return '';
        var v = td.getAttribute('data-v');
        return v !== null ? v : (td.textContent || '').trim();
    }

    function _sortTable(table, colIdx, kind, dir) {
        var rows = _bodyRows(table);
        var mult = dir === 'desc' ? -1 : 1;
        rows.sort(function (a, b) {
            var av = _sortValue(a.cells[colIdx]);
            var bv = _sortValue(b.cells[colIdx]);
            if (kind === 'num') {
                var an = parseFloat(String(av).replace(/[^0-9.eE+-]/g, ''));
                var bn = parseFloat(String(bv).replace(/[^0-9.eE+-]/g, ''));
                if (isNaN(an) && isNaN(bn)) return 0;
                if (isNaN(an)) return 1;      // blanks always sink
                if (isNaN(bn)) return -1;
                return (an - bn) * mult;
            }
            return String(av).localeCompare(String(bv), undefined,
                                            { numeric: true, sensitivity: 'base' }) * mult;
        });
        var tb = table.tBodies[0];
        rows.forEach(function (r) { tb.appendChild(r); });
        // Sorting means the visible window is no longer "the first N rows",
        // so reveal everything rather than showing an arbitrary slice.
        _showAll(table);
        _restripe(table);
    }

    // Zebra striping starts as plain CSS nth-child so tables look right even
    // if this file never runs. That breaks once rows are reordered or
    // filtered, so the first restripe stamps data-gk-striped on the table,
    // which switches the stylesheet over to the .gk-stripe class we manage
    // against the *visible* sequence.
    function _restripe(table) {
        var i = 0;
        _bodyRows(table).forEach(function (r) {
            if (r.classList.contains('gk-row-hidden') ||
                r.classList.contains('gk-filtered-out')) return;
            r.classList.toggle('gk-stripe', i % 2 === 1);
            i++;
        });
        table.setAttribute('data-gk-striped', '1');
    }

    function _showAll(table) {
        _bodyRows(table).forEach(function (r) { r.classList.remove('gk-row-hidden'); });
        var wrap = table.closest('.gk-table-wrap');
        if (!wrap) return;
        var btn = wrap.querySelector('.gk-table-showall');
        if (btn) btn.remove();
        var count = wrap.querySelector('.gk-table-count');
        if (count && count.dataset.total) {
            count.textContent = Number(count.dataset.total).toLocaleString() + ' rows';
        }
    }

    function _filterTable(table, query) {
        var q = (query || '').trim().toLowerCase();
        var shown = 0;
        _bodyRows(table).forEach(function (r) {
            var hit = !q || (r.textContent || '').toLowerCase().indexOf(q) !== -1;
            r.classList.toggle('gk-filtered-out', !hit);
            r.style.display = hit ? '' : 'none';
            if (hit) shown++;
        });
        if (q) _showAll(table);
        _restripe(table);
        var wrap = table.closest('.gk-table-wrap');
        var count = wrap && wrap.querySelector('.gk-table-count');
        if (count) {
            count.textContent = q
                ? shown.toLocaleString() + ' matching'
                : (Number(count.dataset.total || shown)).toLocaleString() + ' rows';
        }
    }

    function _exportCsv(table, filename) {
        var lines = [];
        var heads = Array.prototype.slice.call(table.tHead ? table.tHead.rows[0].cells : []);
        lines.push(heads.map(function (th) {
            var clone = th.cloneNode(true);
            var sub = clone.querySelector('.gk-th-sub');
            if (sub) sub.remove();
            var ind = clone.querySelector('.gk-sort-ind');
            if (ind) ind.remove();
            return _csvCell((clone.textContent || '').trim());
        }).join(','));
        _bodyRows(table).forEach(function (r) {
            if (r.classList.contains('gk-filtered-out')) return;   // respect the filter
            var cells = Array.prototype.slice.call(r.cells).map(function (td) {
                return _csvCell((td.textContent || '').trim().replace(/\s+/g, ' '));
            });
            lines.push(cells.join(','));
        });
        var blob = new Blob(['﻿' + lines.join('\r\n')],
                            { type: 'text/csv;charset=utf-8;' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename || 'export.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    }

    function _csvCell(s) {
        if (/[",\r\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
        return s;
    }

    // ─────────────────────────────────────────────────────────────────
    // Video seek — an events row knows its timestamp, so clicking it should
    // land the operator on the frame instead of making them scrub.
    // ─────────────────────────────────────────────────────────────────
    function _seekVideo(seconds) {
        var vid = document.querySelector('#gk-main video')
               || document.querySelector('.gradio-container video');
        if (!vid) return false;
        try {
            vid.currentTime = Math.max(0, seconds);
            var p = vid.play();
            if (p && p.catch) p.catch(function () {});
        } catch (e) { return false; }
        vid.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return true;
    }

    // ─────────────────────────────────────────────────────────────────
    // Tabs — programmatic switching + count badges
    // ─────────────────────────────────────────────────────────────────
    // Gradio 6.12 renders tabs as .tab-wrapper > .tab-container >
    // button[role=tab], with a duplicate .tab-container.visually-hidden it
    // measures widths against. There is no `.tab-nav` — that was Gradio 4/5.
    // Clicking must target the real (role=tab) buttons; badges go on BOTH
    // copies so Gradio's overflow-menu width maths stays honest.
    function _tabButtons() {
        var real = document.querySelectorAll('#gk-tabs button[role="tab"]');
        if (real.length) return Array.prototype.slice.call(real);
        return Array.prototype.slice.call(
            document.querySelectorAll('#gk-tabs .tab-nav button'));   // legacy
    }
    function _allTabButtons() {
        var all = document.querySelectorAll(
            '#gk-tabs .tab-container button, #gk-tabs .tab-nav button');
        return Array.prototype.slice.call(all);
    }
    function _labelOf(btn) {
        // Badges are ::after pseudo-elements now, and pseudo-element content is
        // not part of textContent — so the label needs no cleaning in the normal
        // case. Skipping the clone matters: this runs per tab button on every
        // bind pass, and cloneNode(true) was deep-copying the whole button each
        // time, which is main-thread cost the circuit breaker then had to fight.
        var b = btn.querySelector('.gk-tab-badge');
        if (!b) return (btn.textContent || '').trim();
        var clone = btn.cloneNode(true);          // legacy <span>, if one lingers
        var cb = clone.querySelector('.gk-tab-badge');
        if (cb) cb.remove();
        return (clone.textContent || '').trim();
    }
    function _clickTab(label) {
        if (!label) return false;
        var target = _tabButtons().filter(function (b) {
            return _labelOf(b).toLowerCase() === String(label).toLowerCase();
        })[0];
        if (!target) return false;
        target.click();
        return true;
    }
    // Tabs are a single flat row, so this is just "click that tab".
    function _gotoTab(label) {
        return _clickTab(label);
    }

    function _syncTabBadges() {
        var host = document.getElementById('gk-tab-counts');
        if (!host) return;
        var raw = host.getAttribute('data-counts');
        // Guard on WINDOW, not on the host element.
        //
        // This used to be `host.dataset.gkApplied`, which put the "already
        // done" flag on a node Gradio owns and replaces wholesale whenever it
        // re-renders that gr.HTML. Every replacement produced a fresh host with
        // no flag, so the badges went back on — and badges change tab-button
        // widths, which makes Gradio re-measure and re-render the tab bar,
        // which is another mutation, which brought us back here with the flag
        // gone again. A closed loop, running full-document passes, with nothing
        // in it that ever terminates.
        //
        // Window scope survives every re-render, so a given payload is applied
        // once per session no matter how often Gradio rebuilds the DOM.
        if (!raw || raw === window.__gkBadgesApplied) return;
        var counts;
        try { counts = JSON.parse(raw); } catch (e) { return; }
        window.__gkBadgesApplied = raw;          // set BEFORE mutating, so a
                                                 // re-entrant call cannot loop
        // ATTRIBUTES ONLY — never append a child here.
        //
        // This used to create a <span class="gk-tab-badge"> and appendChild it
        // into the tab button. Those buttons are rendered by Svelte, so a
        // foreign child made Svelte's reconciliation effect for the tab bar
        // re-run, which re-triggered this write, until Svelte 5's loop guard
        // threw `effect_update_depth_exceeded` from inside its own flush(). The
        // throw aborts the flush mid-update, so pending overlays never cleared
        // and toasts stopped dismissing: the page looked frozen at 100% while
        // the server had finished and sent only ~40 KB.
        //
        // The badge is now a ::before pseudo-element fed by data-gk-badge (see
        // app.css). Attribute writes create no node for Svelte to reconcile,
        // and they do not fire the childList observer below either, so this
        // cannot feed a loop from either direction.
        _allTabButtons().forEach(function (btn) {
            var label = _labelOf(btn);
            var info = counts[label];
            if (!info || !info.n) {
                if (btn.hasAttribute('data-gk-badge')) {
                    btn.removeAttribute('data-gk-badge');
                    btn.removeAttribute('data-gk-tone');
                }
                return;
            }
            var text = info.n > 99 ? '99+' : String(info.n);
            var tone = info.tone || 'NEUTRAL';
            // Write only on change: a no-op setAttribute still notifies
            // observers and dirties Svelte's tracking for no reason.
            if (btn.getAttribute('data-gk-badge') !== text) {
                btn.setAttribute('data-gk-badge', text);
            }
            if (btn.getAttribute('data-gk-tone') !== tone) {
                btn.setAttribute('data-gk-tone', tone);
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────
    // One delegated click listener for everything rendered from Python
    // ─────────────────────────────────────────────────────────────────
    if (!window.__gkDelegated) {
        document.addEventListener('click', function (ev) {
            var t = ev.target;
            if (!t || !t.closest) return;

            var el;

            // -- zone editor: fullscreen toggle --
            // Matched by CLASS as well as id: the button lives in a gr.HTML
            // that Gradio re-renders, and delegation off `document` is the
            // only binding that survives both that and the circuit breaker
            // below turning re-binding off.
            el = t.closest('#gk-zone-fs-btn, .gk-fs-btn');
            if (el) { _toggleZoneFs(ev); return; }

            // -- theme toggle --
            el = t.closest('#gk-theme-toggle, .gk-theme-toggle');
            if (el) { ev.preventDefault(); _toggleTheme(); return; }

            // -- JSON overlay: close --
            el = t.closest('#json-fullscreen-close');
            if (el) { ev.preventDefault(); _closeJsonOverlay(); return; }

            // -- table: sort --
            el = t.closest('.gk-table th[data-sort]');
            if (el) {
                var table = el.closest('table');
                var idx = parseInt(el.getAttribute('data-col'), 10);
                var kind = el.getAttribute('data-sort');
                var dir = el.getAttribute('data-dir') === 'asc' ? 'desc' : 'asc';
                Array.prototype.slice.call(el.parentNode.cells).forEach(function (th) {
                    if (th !== el) th.removeAttribute('data-dir');
                });
                el.setAttribute('data-dir', dir);
                _sortTable(table, idx, kind, dir);
                return;
            }

            // -- table: export --
            el = t.closest('.gk-table-export');
            if (el) {
                ev.preventDefault();
                var tex = document.getElementById(el.getAttribute('data-target'));
                if (tex) _exportCsv(tex, el.getAttribute('data-name'));
                return;
            }

            // -- table: show all --
            el = t.closest('.gk-table-showall');
            if (el) {
                ev.preventDefault();
                var tsa = document.getElementById(el.getAttribute('data-target'));
                if (tsa) { _showAll(tsa); _restripe(tsa); }
                return;
            }

            // -- alert banner: dismiss --
            el = t.closest('.gk-alert-dismiss');
            if (el) {
                ev.preventDefault();
                var banner = el.closest('.gk-alert-banner');
                if (banner) banner.style.display = 'none';
                return;
            }

            // -- alert banner chip → jump to the tab that explains it --
            el = t.closest('.gk-alert-chip[data-gk-tab]');
            if (el) {
                ev.preventDefault();
                _gotoTab(el.getAttribute('data-gk-tab'));
                return;
            }

            // -- generic "go to tab" link --
            el = t.closest('[data-gk-goto]');
            if (el) {
                ev.preventDefault();
                _gotoTab(el.getAttribute('data-gk-goto'));
                return;
            }

            // -- seekable row → move the tracked video --
            el = t.closest('[data-gk-seek]');
            if (el) {
                var secs = parseFloat(el.getAttribute('data-gk-seek'));
                if (!isNaN(secs) && _seekVideo(secs)) {
                    el.classList.remove('gk-seek-flash');
                    void el.offsetWidth;                 // restart the animation
                    el.classList.add('gk-seek-flash');
                }
                return;
            }
        }, false);

        // Keyboard parity for the two custom-role controls.
        document.addEventListener('keydown', function (ev) {
            // Escape leaves zone fullscreen. It is a page-level overlay, not
            // the browser's own fullscreen, so nothing else gives the operator
            // a way out except finding the button again.
            if (ev.key === 'Escape' || ev.key === 'Esc') {
                var fsWrap = document.getElementById('zone-editor-fs-wrap');
                if (fsWrap && fsWrap.classList.contains('gk-fs-active')) {
                    ev.preventDefault();
                    _toggleZoneFs();
                }
                return;
            }
            if (ev.key !== 'Enter' && ev.key !== ' ') return;
            var t = ev.target;
            if (!t || !t.closest) return;
            if (t.closest('[data-gk-seek]') || t.closest('.gk-table th[data-sort]')) {
                ev.preventDefault();
                t.click();
            }
        }, false);

        document.addEventListener('input', function (ev) {
            var el = ev.target && ev.target.closest && ev.target.closest('.gk-table-filter');
            if (!el) return;
            var table = document.getElementById(el.getAttribute('data-target'));
            if (table) _filterTable(table, el.value);
        }, false);

        // ── Stamp on pointerdown, not only in the sweep ──────────────
        // Gradio's router calls `element.onclick()` directly on the element
        // that was clicked; a null there throws "i.onclick is not a function"
        // and kills the click. The sweep in _bindAll() is supposed to prevent
        // that, but it misses two whole categories:
        //
        //   * elements that are not <button>. The video player's fullscreen
        //     control is `<div role="button" aria-label="full-screen">`, and
        //     every selector in that sweep starts with `button`.
        //   * controls that mount late. That same control bar is built when
        //     the pointer enters the player and torn down when it leaves, so
        //     a document-wide pass can plausibly never see it at all.
        //
        // Both were true of the player's Fullscreen button, which is why it
        // did nothing. pointerdown fires on the exact element about to be
        // clicked, in the capture phase, before any click handling — so this
        // is the only stamping guaranteed to be in time, and it costs one
        // closest() per interaction instead of a document walk per mutation.
        document.addEventListener('pointerdown', function (ev) {
            var t = ev.target;
            if (!t || !t.closest) return;
            var el = t.closest('button, [role="button"]');
            if (el && typeof el.onclick !== 'function') {
                el.onclick = NOOP;
                el.setAttribute('data-gk-oc', '1');
            }
        }, true);

        window.__gkDelegated = true;
    }

    // ─────────────────────────────────────────────────────────────────
    // Defensive .onclick assignment
    // ─────────────────────────────────────────────────────────────────
    // There is deliberately NO id -> handler map here any more. It used to
    // do both of these to the same element:
    //
    //     el.onclick = fn;
    //     el.addEventListener('click', fn);
    //
    // which registers the handler twice and runs it twice per click. The
    // JSON overlay's close is idempotent so it survived that, but the two
    // real toggles — zone fullscreen and dark mode — flipped and unflipped
    // in the same tick and looked broken. Real work now happens in the
    // delegated `document` listener above; the only thing assigned to
    // .onclick anywhere in this file is NOOP, and NOOP cancels nothing.

    // Instrumentation, kept in: this function is called from a MutationObserver
    // on the whole document, and when it got expensive the symptom was a page
    // that stopped repainting with no error anywhere. `window.__gkBindStats`
    // makes that measurable from the console instead of inferable.
    var _bindStats = { calls: 0, ms: 0, warned: false };
    window.__gkBindStats = _bindStats;

    function _bindAll() {
        var _t0 = (window.performance && performance.now) ? performance.now() : 0;
        // Give Gradio's router something callable on every button we render
        // from Python, plus its own fullscreen icons (see header note).
        //
        // #gk-zone-fs-btn and #gk-theme-toggle are listed explicitly. Neither
        // is caught by the substring matchers — `gk-fs-btn` does not contain
        // "fullscreen", and the zone button carries no title/aria-label — so
        // leaving them out would leave onclick === null and hand Gradio's
        // router the `i.onclick is not a function` throw this pass exists to
        // prevent.
        //
        // `:not([data-gk-oc])` is load-bearing, not tidiness. This selector
        // carries three case-insensitive substring attribute matchers, which
        // have no fast path — the engine tests every element in the document.
        // Stamping each button as it is handled means the match set shrinks to
        // nothing once the page settles, instead of re-walking every button on
        // every mutation for the life of the session.
        var defensive = document.querySelectorAll(
            'button[aria-label*="ull" i]:not([data-gk-oc]), ' +
            'button[title*="ull" i]:not([data-gk-oc]), ' +
            'button[class*="fullscreen" i]:not([data-gk-oc]), ' +
            '[role="button"][aria-label*="ull" i]:not([data-gk-oc]), ' +
            '.gk-fs-btn:not([data-gk-oc]), ' +
            '.gk-theme-toggle:not([data-gk-oc]), ' +
            '#json-fullscreen-close:not([data-gk-oc]), ' +
            '.gk-btn-mini:not([data-gk-oc]), ' +
            '.gk-alert-dismiss:not([data-gk-oc]), ' +
            'button.gk-alert-chip:not([data-gk-oc])'
        );
        for (var i = 0; i < defensive.length; i++) {
            var b = defensive[i];
            if (typeof b.onclick !== 'function') b.onclick = NOOP;
            b.setAttribute('data-gk-oc', '1');
        }
        _syncTabBadges();
        // Freshly rendered tables get JS striping once; _restripe stamps the
        // attribute that hands striping over from the CSS fallback.
        var tables = document.querySelectorAll('.gk-table:not([data-gk-striped])');
        for (var j = 0; j < tables.length; j++) {
            _restripe(tables[j]);
        }

        if (_t0) {
            _bindStats.calls++;
            _bindStats.ms += performance.now() - _t0;
            // One warning per session, when the cost stops being incidental.
            if (!_bindStats.warned && _bindStats.ms > 2000) {
                _bindStats.warned = true;
                console.warn('[gk] DOM binding has cost ' +
                    Math.round(_bindStats.ms) + 'ms over ' + _bindStats.calls +
                    ' passes — this is main-thread time and will stall the page.');
            }
        }
    }

    // ── Circuit breaker ───────────────────────────────────────────────
    // Convenience bindings are NOT worth a dead tab. Every loop in here is a
    // loop between our code and Gradio's re-rendering, so a fix that assumes I
    // found all of them is a fix that can still lock the page hard enough that
    // DevTools will not open — at which point there is no way to even see what
    // went wrong.
    //
    // So this is a hard stop, not a warning: past the budget the observer is
    // disconnected for good. The page loses badges and re-binding on
    // newly-rendered content; it keeps the delegated click/keydown/input
    // handlers, which are attached to `document` once and never re-run. That
    // degradation is invisible to most of the UI, and it is always the right
    // trade against a browser that has to be force-closed.
    var BIND_BUDGET_MS = 4000;
    var BIND_BUDGET_CALLS = 600;

    function _overBudget() {
        return _bindStats.ms > BIND_BUDGET_MS ||
               _bindStats.calls > BIND_BUDGET_CALLS;
    }

    // Exports go up FIRST. They used to sit at the very bottom, so anything
    // that threw above them (see _prefersDark) silently left the page with no
    // seek/tab helpers at all.
    window.gkToggleZoneFullscreen = _toggleZoneFs;
    window.gkToggleTheme          = _toggleTheme;
    window.gkSyncThemeBtn         = _syncThemeBtn;
    window.gkEffectiveTheme       = _effectiveTheme;
    window.gkGotoTab              = _gotoTab;
    window.gkSeekVideo            = _seekVideo;
    window.gkRestripe             = _restripe;

    try {
        _bindAll();
    } catch (e) { console.warn('[gk] initial bind failed', e); }

    if (!window.__gkBindingObserver) {
        // COALESCED, and it has to be. This observer watches childList over the
        // entire body subtree, and _bindAll() walks the whole document. Run
        // one-to-one, the two multiply: rendering a result set is thousands of
        // Svelte mutations, the JSON viewer's editor churns the DOM while it
        // lays out and highlights, and every progress tick rewrites the status
        // bars. Each of those used to buy another full-document pass, on the
        // main thread, while the page was trying to paint the results the user
        // was waiting for — which looks exactly like a hung tab, with no error
        // in the console and the server long since finished.
        //
        // requestAnimationFrame collapses a storm into at most one pass per
        // frame, and the pass is idempotent, so coalescing loses nothing.
        var obs;
        var _queued = false;

        function _runBind() {
            _queued = false;
            // Detach for the duration: _syncTabBadges() appends and removes
            // badge nodes, which are themselves childList mutations inside the
            // observed subtree. takeRecords() drops what our own writes queued
            // so reconnecting cannot immediately re-fire on them.
            if (obs) obs.disconnect();
            try {
                _bindAll();
            } catch (e) {
                /* logged once at install */
            } finally {
                if (obs) {
                    obs.takeRecords();
                    if (_overBudget()) {
                        window.__gkBindGaveUp = true;
                        console.warn('[gk] re-binding disabled after ' +
                            Math.round(_bindStats.ms) + 'ms over ' +
                            _bindStats.calls + ' passes. The page stays usable ' +
                            '(clicks/sort/seek are delegated on document); tab ' +
                            'badges and newly-rendered buttons stop updating. ' +
                            'This is the guard against a locked tab — please ' +
                            'report it.');
                        return;                  // deliberately not re-observed
                    }
                    obs.observe(document.body, { childList: true, subtree: true });
                }
            }
        }

        function _schedule() {
            if (_queued || window.__gkBindGaveUp) return;
            _queued = true;
            if (window.requestAnimationFrame) requestAnimationFrame(_runBind);
            else setTimeout(_runBind, 16);
        }

        obs = new MutationObserver(_schedule);
        var start = function () {
            obs.observe(document.body, { childList: true, subtree: true });
            window.__gkBindingObserver = obs;
            try { _bindAll(); } catch (e) { console.warn('[gk] bind failed', e); }
            _syncThemeBtn();
            console.log('[gk] observer installed (rAF-coalesced)');
        };
        if (document.body) start();
        else document.addEventListener('DOMContentLoaded', start);
    }

    // Restore saved theme on first load.
    try {
        var saved = localStorage.getItem('gk-theme');
        if (saved === 'dark' || saved === 'light') {
            if (document.body) {
                _applyTheme(saved);
            } else {
                document.documentElement.classList.add('gk-' + saved);
                document.addEventListener('DOMContentLoaded', function () {
                    _applyTheme(saved);
                });
            }
        }
    } catch (e) {}
})();
