# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Canvas-based 2D Bird's Eye View builder.

Generates a self-contained HTML string with an embedded Canvas2D player
that renders rotated shopper boxes colored by velocity, fixture rectangles,
recent trail polylines, and a click-to-inspect impression-history panel.

Mirrors the public surface of bev3d_builder.build_3d_bev_html: returns a
single HTML document string injected into the Gradio iframe via
contentDocument.write() (see app_poc_v2.py for the wiring).
"""
from __future__ import annotations

import json

from .bev3d_builder import slim_frame_for_3d


def slim_frame_for_2d(fj: dict) -> dict:
    """Like slim_frame_for_3d but also carries continuous velocity `v`
    (px/s) so the renderer can map it to a colormap. Orientation `o` is
    attached later by `bev2d_orientation.attach_orientations`."""
    base = slim_frame_for_3d(fj)
    for k, p in fj.get("people", {}).items():
        if k in base["p"]:
            base["p"][k]["v"] = p["motion"]["speed"]
    for k, c in fj.get("carts", {}).items():
        if k in base["c"]:
            base["c"][k]["v"] = c["motion"]["speed"]
    return base


def _percentile_speed(slim_frames: list[dict], q: float = 0.95) -> float:
    speeds: list[float] = []
    for fr in slim_frames:
        for obj in fr.get("p", {}).values():
            v = obj.get("v")
            if v:
                speeds.append(v)
        for obj in fr.get("c", {}).values():
            v = obj.get("v")
            if v:
                speeds.append(v)
    if not speeds:
        return 1.0
    speeds.sort()
    idx = max(0, min(len(speeds) - 1, int(len(speeds) * q)))
    return max(1.0, speeds[idx])


def _build_canvas_document(frames_json: str, fixtures_json: str,
                           impressions_json: str,
                           video_width: int, video_height: int,
                           fps: float, vmax: float) -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; overflow:hidden; font-family:'Segoe UI',Tahoma,sans-serif; color:#e2e8f0; }}
  #scene {{ position:relative; width:100vw; height:100vh; }}
  #stage {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; }}
  canvas#bev {{ background:#0f172a; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,0.4); display:block; }}
  #controls {{
    position:absolute; bottom:0; left:0; right:0;
    background:rgba(15,15,30,0.92); padding:8px 16px;
    display:flex; align-items:center; gap:12px; z-index:10;
    border-top:1px solid rgba(255,255,255,0.08);
  }}
  #controls button {{
    background:#3b82f6; color:#fff; border:none; border-radius:6px;
    padding:6px 14px; font-weight:700; cursor:pointer; font-size:13px;
  }}
  #controls button:hover {{ background:#2563eb; }}
  #frame-slider {{ flex:1; accent-color:#3b82f6; cursor:pointer; }}
  #frame-info {{
    color:#94a3b8; font-size:12px; min-width:130px; text-align:right;
    font-variant-numeric:tabular-nums;
  }}
  #speed-select {{
    background:#1e293b; color:#e2e8f0; border:1px solid #334155;
    border-radius:4px; padding:4px 6px; font-size:12px;
  }}
  #legend {{
    position:absolute; top:12px; left:12px; background:rgba(15,15,30,0.88);
    border:1px solid rgba(255,255,255,0.1); border-radius:8px;
    padding:8px 12px; font-size:12px; color:#cbd5e1; line-height:1.6;
    pointer-events:none;
  }}
  #legend .swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px;
    vertical-align:middle; margin-right:6px; }}
  #colorbar {{
    position:absolute; top:12px; right:12px; background:rgba(15,15,30,0.88);
    border:1px solid rgba(255,255,255,0.1); border-radius:8px;
    padding:10px 10px 10px 14px; font-size:11px; color:#cbd5e1;
    display:flex; gap:8px; align-items:stretch;
  }}
  #colorbar canvas {{ width:14px; height:120px; border-radius:3px; }}
  #colorbar .ticks {{ display:flex; flex-direction:column; justify-content:space-between; height:120px; }}
  #inspector {{
    position:absolute; right:12px; bottom:60px; max-width:300px;
    background:rgba(15,15,30,0.92); border:1px solid rgba(255,255,255,0.1);
    border-radius:8px; padding:10px 12px; font-size:12px; color:#cbd5e1;
    display:none; line-height:1.5;
  }}
  #inspector h4 {{ margin-bottom:6px; color:#93c5fd; font-size:12px; letter-spacing:0.04em; }}
  #inspector ul {{ list-style:none; padding-left:0; }}
  #inspector li {{ padding:2px 0; border-bottom:1px solid rgba(255,255,255,0.05); }}
  #inspector .empty {{ color:#64748b; font-style:italic; }}
  #inspector .close {{ float:right; cursor:pointer; color:#64748b; }}
  #inspector .close:hover {{ color:#f87171; }}
</style>
</head>
<body>
<div id="scene">
  <div id="stage"><canvas id="bev"></canvas></div>
  <div id="legend">
    <div><span class="swatch" style="background:#22c55e;"></span>Person</div>
    <div><span class="swatch" style="background:#f59e0b;"></span>Cart</div>
    <div><span class="swatch" style="background:#7dd3fc;"></span>Fixture</div>
    <div style="margin-top:4px;color:#64748b;font-size:11px;">Click a person for impressions</div>
  </div>
  <div id="colorbar">
    <canvas id="cbar" width="14" height="120"></canvas>
    <div class="ticks"><div>Fast</div><div>Slow</div></div>
  </div>
  <div id="inspector">
    <span class="close" id="inspector-close">x</span>
    <h4 id="inspector-title">Person</h4>
    <div id="inspector-body"></div>
  </div>
  <div id="controls">
    <button id="play">Pause</button>
    <input id="frame-slider" type="range" min="0" max="0" value="0">
    <select id="speed-select">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <span id="frame-info">0 / 0</span>
  </div>
</div>
<script id="bev2d-frames" type="application/json">{frames_json}</script>
<script id="bev2d-fixtures" type="application/json">{fixtures_json}</script>
<script id="bev2d-impressions" type="application/json">{impressions_json}</script>
<script>
(function() {{
  const FRAMES = JSON.parse(document.getElementById('bev2d-frames').textContent || '[]');
  const FIXTURES = JSON.parse(document.getElementById('bev2d-fixtures').textContent || '[]');
  const IMPRESSIONS = JSON.parse(document.getElementById('bev2d-impressions').textContent || '{{}}');
  const VW = {video_width};
  const VH = {video_height};
  const FPS = {fps};
  const VMAX = {vmax};

  /* viridis colormap: 0 (dark blue) -> 1 (yellow) */
  const VIRIDIS = [
    [68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],
    [39,173,129],[92,200,99],[170,220,50],[253,231,37]
  ];
  function colormap(t) {{
    if (!isFinite(t)) t = 0;
    t = Math.max(0, Math.min(1, t));
    const f = t * (VIRIDIS.length - 1);
    const i = Math.floor(f);
    const a = VIRIDIS[i];
    const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
    const u = f - i;
    const r = Math.round(a[0] + (b[0] - a[0]) * u);
    const g = Math.round(a[1] + (b[1] - a[1]) * u);
    const bl = Math.round(a[2] + (b[2] - a[2]) * u);
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
  }}

  /* layout: fit canvas to video aspect inside the stage */
  const canvas = document.getElementById('bev');
  const ctx = canvas.getContext('2d');
  let scale = 1;
  function resize() {{
    const stage = document.getElementById('stage');
    const sw = stage.clientWidth - 24;
    const sh = stage.clientHeight - 24;
    const vAspect = VW / VH;
    let cw = sw, ch = sw / vAspect;
    if (ch > sh) {{ ch = sh; cw = sh * vAspect; }}
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    canvas.width = VW;
    canvas.height = VH;
    scale = cw / VW;
    draw(currentIdx);
  }}

  /* paint colorbar once */
  (function() {{
    const cb = document.getElementById('cbar');
    const cctx = cb.getContext('2d');
    for (let y = 0; y < cb.height; y++) {{
      const t = 1 - y / (cb.height - 1);
      cctx.fillStyle = colormap(t);
      cctx.fillRect(0, y, cb.width, 1);
    }}
  }})();

  /* fixtures drawn once per frame as backdrop */
  function drawFixtures() {{
    ctx.save();
    ctx.lineWidth = 2;
    for (const fx of FIXTURES) {{
      ctx.fillStyle = 'rgba(125,211,252,0.10)';
      ctx.strokeStyle = 'rgba(125,211,252,0.55)';
      const w = fx.x2 - fx.x1, h = fx.y2 - fx.y1;
      ctx.fillRect(fx.x1, fx.y1, w, h);
      ctx.strokeRect(fx.x1, fx.y1, w, h);
      ctx.fillStyle = 'rgba(186,230,253,0.85)';
      ctx.font = '14px Segoe UI, sans-serif';
      ctx.fillText(fx.label, fx.x1 + 6, fx.y1 + 18);
    }}
    ctx.restore();
  }}

  /* trails: last TRAIL_LEN frames per object, dotted with viridis-by-step */
  const TRAIL_LEN = 30;
  function buildTrails(idx) {{
    const start = Math.max(0, idx - TRAIL_LEN + 1);
    const trailsP = {{}};
    const trailsC = {{}};
    for (let i = start; i <= idx; i++) {{
      const fr = FRAMES[i];
      if (!fr) continue;
      for (const k in fr.p) {{
        if (!trailsP[k]) trailsP[k] = [];
        trailsP[k].push(fr.p[k]);
      }}
      for (const k in fr.c) {{
        if (!trailsC[k]) trailsC[k] = [];
        trailsC[k].push(fr.c[k]);
      }}
    }}
    return {{ p: trailsP, c: trailsC }};
  }}

  function drawTrails(trails) {{
    ctx.save();
    for (const set of [trails.p, trails.c]) {{
      for (const k in set) {{
        const pts = set[k];
        for (let i = 1; i < pts.length; i++) {{
          const a = pts[i - 1], b = pts[i];
          const v = b.v || 0;
          const t = Math.min(1, v / VMAX);
          ctx.strokeStyle = colormap(t);
          ctx.globalAlpha = 0.25 + 0.75 * (i / pts.length);
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }}
      }}
    }}
    ctx.globalAlpha = 1;
    ctx.restore();
  }}

  /* rotated boxes for people/carts; person clickable for impressions */
  let personHitboxes = [];   /* [{{key, displayId, x, y, w, h, theta}}] */
  function drawObjects(fr) {{
    personHitboxes = [];
    if (!fr) return;
    const PW = 34, PH = 20;     /* box size in source-pixel space */
    /* people */
    for (const k in fr.p) {{
      const o = fr.p[k];
      const v = o.v || 0;
      const t = Math.min(1, v / VMAX);
      const fill = colormap(t);
      const theta = (o.o || 0) * Math.PI / 180;
      drawRotatedBox(o.x, o.y, PW, PH, theta, fill, '#0f172a', k);
      personHitboxes.push({{
        key: k,
        displayId: parseInt(k.replace(/[^0-9]/g, ''), 10),
        x: o.x, y: o.y, w: PW, h: PH, theta: theta,
      }});
    }}
    /* carts */
    for (const k in fr.c) {{
      const o = fr.c[k];
      const theta = (o.o || 0) * Math.PI / 180;
      drawRotatedBox(o.x, o.y, PW + 6, PH + 4, theta, '#f59e0b', '#7c2d12', k);
    }}
    /* link lines */
    for (const k in fr.p) {{
      const p = fr.p[k];
      const lc = p.lc;
      if (lc == null) continue;
      const cKey = 'C' + lc;
      const cart = fr.c[cKey];
      if (!cart) continue;
      ctx.save();
      ctx.strokeStyle = 'rgba(236,72,153,0.85)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(cart.x, cart.y);
      ctx.stroke();
      ctx.restore();
    }}
  }}

  function drawRotatedBox(cx, cy, w, h, theta, fill, stroke, label) {{
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(theta);
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.fillRect(-w / 2, -h / 2, w, h);
    ctx.strokeRect(-w / 2, -h / 2, w, h);
    /* short heading tick */
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(w / 2 + 8, 0);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
    /* label outside the rotation so it stays upright */
    ctx.save();
    ctx.font = 'bold 11px Segoe UI, sans-serif';
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = 3;
    ctx.strokeText(label, cx + 8, cy - 8);
    ctx.fillText(label, cx + 8, cy - 8);
    ctx.restore();
  }}

  /* clicks: locate nearest person hit and toggle inspector */
  canvas.addEventListener('click', function(ev) {{
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / scale;
    const y = (ev.clientY - rect.top) / scale;
    let best = null, bestD = Infinity;
    for (const hb of personHitboxes) {{
      const d = (hb.x - x) ** 2 + (hb.y - y) ** 2;
      if (d < bestD && d < (Math.max(hb.w, hb.h) ** 2)) {{
        bestD = d; best = hb;
      }}
    }}
    if (best) showInspector(best);
    else hideInspector();
  }});

  function showInspector(hb) {{
    const ins = document.getElementById('inspector');
    document.getElementById('inspector-title').textContent = 'Person ' + hb.key;
    const body = document.getElementById('inspector-body');
    const list = IMPRESSIONS[hb.displayId] || [];
    if (!list.length) {{
      body.innerHTML = '<div class="empty">No fixture engagements detected.</div>';
    }} else {{
      const items = list.map(function(im) {{
        return '<li><b>' + im.fixture_label + '</b> '
             + '<span style="color:#64748b;">@ ' + im.enter_t.toFixed(1) + 's</span> '
             + '<span style="color:#94a3b8;">' + im.dwell_s.toFixed(1) + 's</span></li>';
      }}).join('');
      body.innerHTML = '<div style="color:#64748b;margin-bottom:4px;">Impression History</div><ul>' + items + '</ul>';
    }}
    ins.style.display = 'block';
  }}
  function hideInspector() {{
    document.getElementById('inspector').style.display = 'none';
  }}
  document.getElementById('inspector-close').addEventListener('click', hideInspector);

  /* render */
  let currentIdx = 0;
  function draw(idx) {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFixtures();
    if (FRAMES.length === 0) return;
    const trails = buildTrails(idx);
    drawTrails(trails);
    drawObjects(FRAMES[idx]);
  }}

  /* controls */
  const slider = document.getElementById('frame-slider');
  const info = document.getElementById('frame-info');
  const playBtn = document.getElementById('play');
  const speedSel = document.getElementById('speed-select');
  slider.max = Math.max(0, FRAMES.length - 1);
  function updateInfo() {{
    const fr = FRAMES[currentIdx];
    const t = fr ? fr.t : 0;
    info.textContent = 'Frame ' + (fr ? fr.f : 0) + ' (t=' + t.toFixed(1) + 's)';
  }}
  slider.addEventListener('input', function() {{
    currentIdx = parseInt(slider.value, 10) || 0;
    draw(currentIdx); updateInfo();
  }});
  let playing = true;
  let speed = 1;
  playBtn.addEventListener('click', function() {{
    playing = !playing;
    playBtn.textContent = playing ? 'Pause' : 'Play';
  }});
  speedSel.addEventListener('change', function() {{
    speed = parseFloat(speedSel.value) || 1;
  }});

  let lastTime = 0;
  function tick(ts) {{
    requestAnimationFrame(tick);
    if (playing && FRAMES.length > 1) {{
      const interval = 1000 / (FPS * speed);
      if (ts - lastTime >= interval) {{
        currentIdx = (currentIdx + 1) % FRAMES.length;
        slider.value = currentIdx;
        draw(currentIdx);
        updateInfo();
        lastTime = ts;
      }}
    }}
  }}

  window.addEventListener('resize', resize);
  resize();
  updateInfo();
  requestAnimationFrame(tick);
}})();
</script>
</body>
</html>'''


def build_2d_bev_html(slim_frames: list, fixtures: list[dict],
                      impressions: dict[int, list[dict]],
                      video_width: int, video_height: int,
                      fps: float, total_frames: int) -> str:
    """Public entry point. Returns the inner HTML document string for the
    Gradio iframe wiring (mirrors `build_3d_bev_html`)."""
    if not slim_frames:
        return ""

    vmax = _percentile_speed(slim_frames, q=0.95)
    frames_json = json.dumps(slim_frames, separators=(',', ':'))
    fixtures_json = json.dumps(fixtures or [], separators=(',', ':'))
    impressions_json = json.dumps(
        {str(k): v for k, v in (impressions or {}).items()},
        separators=(',', ':'),
    )
    return _build_canvas_document(
        frames_json, fixtures_json, impressions_json,
        video_width, video_height, fps, vmax,
    )
