# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Three.js 3D Bird's Eye View builder.

Generates a self-contained HTML string with an embedded Three.js scene
for interactive playback of tracked person / cart data.

Uses Three.js r128 with classic <script> tags (no ES modules) for
maximum compatibility across file://, iframe, and Gradio contexts.
"""
import json
from pathlib import Path

# Inline procedural model JS so the iframe is self-contained.
# Loaded lazily at first use; falls back to an empty string if missing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INLINE_JS_CACHE: dict[str, str] = {}


def _load_inline_js(filename: str) -> str:
    if filename not in _INLINE_JS_CACHE:
        path = _PROJECT_ROOT / filename
        try:
            _INLINE_JS_CACHE[filename] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"[bev3d] {filename} not found at {path} - feature disabled.")
            _INLINE_JS_CACHE[filename] = ""
    return _INLINE_JS_CACHE[filename]


def _load_cart_model_js() -> str:
    return _load_inline_js("cart-model.js")


def _load_store_fixtures_js() -> str:
    return _load_inline_js("store-fixtures.js")


def _load_person_model_js() -> str:
    return _load_inline_js("person-model.js")


def slim_frame_for_3d(fj: dict, pose_kps: dict | None = None,
                      person_bboxes: dict | None = None) -> dict:
    """Extract minimal per-frame data for the 3D viewer.

    pose_kps and person_bboxes are keyed by person display id (e.g. "P1")
    and only included when pose estimation is enabled. kp is a list of 17
    [x, y] pairs (or null for low-confidence keypoints), bb is [x1,y1,x2,y2].
    """
    people = {}
    for k, p in fj.get("people", {}).items():
        entry = {
            "x": p["centroid"]["x"], "y": p["centroid"]["y"],
            "d": p["motion"]["direction"],
            "s": p["motion"]["speed_status"],
            "lc": p["linking"].get("linked_cart_id"),
        }
        if pose_kps and k in pose_kps:
            entry["kp"] = pose_kps[k]
            if person_bboxes and k in person_bboxes:
                entry["bb"] = person_bboxes[k]
        people[k] = entry
    carts = {}
    for k, c in fj.get("carts", {}).items():
        carts[k] = {
            "x": c["centroid"]["x"], "y": c["centroid"]["y"],
            "d": c["motion"]["direction"],
            "s": c["motion"]["speed_status"],
            "lp": c["linking"].get("linked_person_id"),
            "ps": c.get("pops", {}).get("score", 0),
            "pe": c.get("pops", {}).get("event", ""),
        }
    return {"f": fj["frame_number"], "t": fj["timestamp"],
            "p": people, "c": carts}


def _zones_to_payload(zones, video_width, video_height):
    """Convert Zone dataclasses to a plain JSON-serialisable list for the
    Three.js scene. Polygons stay in pixel coords (the JS side maps them
    through toWorld using VW/VH)."""
    if not zones:
        return []
    out = []
    for z in zones:
        poly = getattr(z, "polygon", None)
        if poly is None or len(poly) < 3:
            continue
        # Zone.color is BGR; convert to "#rrggbb" for CSS/Three.js.
        b, g, r = z.color if hasattr(z, "color") else (0, 200, 255)
        out.append({
            "name": z.name,
            "applies": getattr(z, "applies_to", "person"),
            "kind": getattr(z, "kind", "analytics"),
            "color": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
            "poly": [[int(x), int(y)] for x, y in poly.tolist()],
        })
    return out


def _build_threejs_document(frames_json: str,
                            video_width: int, video_height: int,
                            fps: float,
                            zones_json: str = "[]",
                            enable_pose: bool = False) -> str:
    """Build a complete HTML document string containing the Three.js scene."""

    aspect = video_height / video_width if video_width else 1.0
    scene_w = 20
    scene_d = round(20 * aspect, 2)
    # `enable_pose` is accepted for backward compat but no longer used:
    # the procedural shopper from person-model.js now stands in for the
    # COCO-17 skeleton in this view.

    # Using classic <script> tags with Three.js r128 (global builds)
    # so it works from file://, iframes, and Gradio contexts.
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; overflow:hidden; font-family:'Segoe UI',Tahoma,sans-serif; }}
  #scene-container {{ width:100vw; height:100vh; position:relative; }}
  canvas {{ display:block; }}
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
    padding:10px 14px; color:#e2e8f0; font-size:12px; z-index:10;
    line-height:1.7;
  }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
          vertical-align:middle; margin-right:6px; }}
  .sline {{ display:inline-block; width:16px; height:3px;
           vertical-align:middle; margin-right:6px; border-radius:2px; }}
  #title {{
    position:absolute; top:12px; right:16px; color:#e2e8f0;
    font-size:15px; font-weight:800; z-index:10;
    text-shadow:0 1px 4px rgba(0,0,0,0.5);
  }}
</style>
<!-- Three.js r128: classic global builds (no ES modules needed) -->
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<!-- Procedural shopping-cart model — inlined from cart-model.js -->
<script>/*__CART_MODEL_JS__*/</script>
<!-- Procedural store fixtures (wall/door/aisle) — inlined from store-fixtures.js -->
<script>/*__STORE_FIXTURES_JS__*/</script>
<!-- Procedural shopper model — inlined from person-model.js -->
<script>/*__PERSON_MODEL_JS__*/</script>
</head>
<body>
<div id="scene-container">
  <div id="legend">
    <div><span class="dot" style="background:#00e676"></span> Person</div>
    <div><span class="dot" style="background:#87ceeb"></span> Cart</div>
    <div><span class="sline" style="background:#ff32ff"></span> Linked</div>
    <div><span class="sline" style="background:#22d3ee"></span> Zone outline</div>
    <div><span class="sline" style="background:#eae6df"></span> Wall &#183; Door &#183; Aisle</div>
    <div style="font-size:11px;color:#94a3b8;margin-top:4px">Drag to rotate &#183; Scroll to zoom</div>
  </div>
  <div id="title">3D Bird&#8217;s Eye View</div>
  <div id="controls">
    <button id="play-btn">&#9654; Play</button>
    <input id="frame-slider" type="range" min="0" max="0" value="0">
    <span id="frame-info">Frame 0 / 0</span>
    <select id="speed-select">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
  </div>
</div>

<script>
(function() {{
  /* DATA */
  var FRAMES = {frames_json};
  var ZONES = {zones_json};
  var VW = {video_width}, VH = {video_height};
  var FPS = {fps}, SW = {scene_w}, SD = {scene_d};
  var POOL = 20, TRAIL_LEN = 30;

  function toWorld(vx, vy) {{
    return new THREE.Vector3(
      (vx / VW) * SW - SW / 2,
      0,
      (vy / VH) * SD - SD / 2
    );
  }}

  /* SCENE */
  var container = document.getElementById('scene-container');
  var renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x1a1a2e);
  container.appendChild(renderer.domElement);

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(50,
    container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(0, 18, 12);
  camera.lookAt(0, 0, 0);

  var controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(5, 15, 5);
  scene.add(dirLight);

  /* Ground */
  var groundGeo = new THREE.PlaneGeometry(SW, SD);
  var groundMat = new THREE.MeshStandardMaterial({{ color: 0x1e1e2e, roughness: 0.9 }});
  var ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  scene.add(new THREE.GridHelper(Math.max(SW, SD), 20, 0x333355, 0x222244));

  /* Zones — drawn as semi-transparent floor polygons + outline + label */
  function addZones() {{
    if (!ZONES || !ZONES.length) return;
    for (var zi = 0; zi < ZONES.length; zi++) {{
      var z = ZONES[zi];

      /* Project polygon into world coords once, compute centroid + edges */
      var wpts = [];
      for (var pi = 0; pi < z.poly.length; pi++) {{
        wpts.push(toWorld(z.poly[pi][0], z.poly[pi][1]));
      }}
      var cx = 0, cz = 0;
      for (var ci = 0; ci < wpts.length; ci++) {{ cx += wpts[ci].x; cz += wpts[ci].z; }}
      cx /= wpts.length; cz /= wpts.length;

      var col = new THREE.Color(z.color);

      /* Find longest edge — gives orientation + length for line-like fixtures */
      var bestI = 0, bestLen = 0;
      var edgeLens = [];
      for (var ei = 0; ei < wpts.length; ei++) {{
        var ej = (ei + 1) % wpts.length;
        var dx = wpts[ej].x - wpts[ei].x;
        var dz = wpts[ej].z - wpts[ei].z;
        var L = Math.sqrt(dx*dx + dz*dz);
        edgeLens.push(L);
        if (L > bestLen) {{ bestLen = L; bestI = ei; }}
      }}
      var pa = wpts[bestI], pb = wpts[(bestI + 1) % wpts.length];
      var theta = Math.atan2(pb.z - pa.z, pb.x - pa.x);
      var midX = (pa.x + pb.x) * 0.5, midZ = (pa.z + pb.z) * 0.5;

      var kind = z.kind || 'analytics';

      if (kind === 'wall' && typeof window.createStoreWall === 'function') {{
        var wall = window.createStoreWall({{
          length: Math.max(0.5, bestLen),
          height: 3.0,
          color: col.getHex(),
        }});
        wall.position.set(midX, 0, midZ);
        wall.rotation.y = -theta;
        scene.add(wall);
        var wlabel = makeLabel(z.name);
        wlabel.position.set(midX, 3.3, midZ);
        wlabel.scale.set(1.2, 0.55, 1);
        scene.add(wlabel);

      }} else if (kind === 'door' && typeof window.createStoreDoor === 'function') {{
        var door = window.createStoreDoor({{
          width: Math.max(0.8, bestLen),
          height: 2.4,
          open: 0.0,
        }});
        door.position.set(midX, 0, midZ);
        door.rotation.y = -theta;
        scene.add(door);
        var dlabel = makeLabel(z.name);
        dlabel.position.set(midX, 2.7, midZ);
        dlabel.scale.set(1.2, 0.55, 1);
        scene.add(dlabel);

      }} else if (kind === 'aisle' && typeof window.createStoreAisle === 'function') {{
        /* For ~rectangular zones, longest edge = aisle length, shortest = lane width */
        var sortedLens = edgeLens.slice().sort(function(a,b){{ return a - b; }});
        var laneW = Math.max(0.6, sortedLens[0]);
        var aisleL = Math.max(1.0, bestLen);
        /* Lane is the open walkway between gondolas; cap so gondolas don't
           occupy the entire zone width. */
        var aisle = window.createStoreAisle({{
          length: aisleL,
          laneWidth: Math.max(0.5, laneW * 0.6),
          shelfDepth: Math.max(0.25, laneW * 0.18),
          shelfHeight: 1.95,
          levels: 4,
          productDensity: 0.85,
        }});
        aisle.position.set(cx, 0, cz);
        aisle.rotation.y = -theta;
        scene.add(aisle);
        var alabel = makeLabel(z.name);
        alabel.position.set(cx, 2.3, cz);
        alabel.scale.set(1.2, 0.55, 1);
        scene.add(alabel);

      }} else {{
        /* Default: flat ground polygon (analytics zones, fixtures, etc.) */
        var shape = new THREE.Shape();
        for (var pi2 = 0; pi2 < wpts.length; pi2++) {{
          if (pi2 === 0) shape.moveTo(wpts[pi2].x, wpts[pi2].z);
          else shape.lineTo(wpts[pi2].x, wpts[pi2].z);
        }}
        shape.closePath();
        var fillMat = new THREE.MeshBasicMaterial({{
          color: col, transparent: true, opacity: 0.18,
          side: THREE.DoubleSide, depthWrite: false,
        }});
        var fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), fillMat);
        fill.rotation.x = -Math.PI / 2;
        fill.position.y = 0.005;
        scene.add(fill);

        /* Outline */
        var olpts = [];
        for (var pi3 = 0; pi3 < wpts.length; pi3++) {{
          olpts.push(new THREE.Vector3(wpts[pi3].x, 0.01, wpts[pi3].z));
        }}
        olpts.push(olpts[0].clone());
        scene.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(olpts),
          new THREE.LineBasicMaterial({{ color: col }})
        ));

        var zlabel = makeLabel(z.name);
        zlabel.position.set(cx, 0.4, cz);
        zlabel.scale.set(1.1, 0.55, 1);
        scene.add(zlabel);
      }}
    }}
  }}

  /* Materials */
  var personMat = new THREE.MeshStandardMaterial({{ color: 0x00e676 }});
  var cartMat = new THREE.MeshStandardMaterial({{ color: 0xffffff }});
  var coneMat = new THREE.MeshStandardMaterial({{ color: 0xcccccc }});
  var linkMat = new THREE.LineBasicMaterial({{ color: 0xff32ff }});
  var pTrailMat = new THREE.LineBasicMaterial({{ color: 0x00e676, transparent: true, opacity: 0.5 }});
  var cTrailMat = new THREE.LineBasicMaterial({{ color: 0xffa500, transparent: true, opacity: 0.5 }});

  /* Sprite label helper */
  function makeLabel(text) {{
    var canvas = document.createElement('canvas');
    canvas.width = 128; canvas.height = 64;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.roundRect(0, 0, 128, 64, 8);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 28px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);
    var tex = new THREE.CanvasTexture(canvas);
    var mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
    var sprite = new THREE.Sprite(mat);
    sprite.scale.set(0.8, 0.4, 1);
    return sprite;
  }}

  function updateLabel(sprite, text, bgColor) {{
    var canvas = sprite.material.map.image;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 128, 64);
    ctx.fillStyle = bgColor || 'rgba(0,0,0,0.6)';
    if (ctx.roundRect) {{ ctx.roundRect(0, 0, 128, 64, 8); ctx.fill(); }}
    else {{ ctx.fillRect(0, 0, 128, 64); }}
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);
    sprite.material.map.needsUpdate = true;
  }}

  /* Object pools */
  function makePersonGroup() {{
    var g = new THREE.Group();
    /* Procedural shopper from person-model.js, with a cylinder+sphere
       fallback if the JS wasn't inlined. */
    var bodyVisual;
    if (typeof window.createPerson === 'function') {{
      bodyVisual = window.createPerson({{
        scale: 0.5,            // ~0.86 world units tall, matches cart scale 0.7
        shirt: 0x00e676,       // POPS person-green
        castShadow: false, receiveShadow: false,
      }});
    }} else {{
      bodyVisual = new THREE.Group();
      var _b = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.6, 8), personMat);
      _b.position.y = 0.3; bodyVisual.add(_b);
      var _h = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 6), personMat);
      _h.position.y = 0.72; bodyVisual.add(_h);
    }}
    g.add(bodyVisual);
    var cone = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.2, 6), coneMat);
    cone.position.y = 0.05; cone.rotation.x = Math.PI / 2; cone.visible = false;
    g.add(cone);
    var label = makeLabel('P?');
    label.position.set(0, 1.1, 0);
    g.add(label);
    g.visible = false;

    g.userData = {{ id: null, cone: cone, label: label, bodyVisual: bodyVisual }};
    scene.add(g);
    return g;
  }}

  /* Build a single template shopping cart (procedural mesh from
     cart-model.js) and clone it per pool slot — geometries are shared
     across clones, materials are shared, only the Mesh wrappers are new. */
  var _cartTemplate = null;
  if (typeof window.createShoppingCart === 'function') {{
    _cartTemplate = window.createShoppingCart({{
      scale: 0.7,
      // Lower wire density vs defaults — visually nearly identical, ~half the meshes
      wireBarsLong: 12, wireBarsShort: 8,
      floorBarsLong: 8, floorBarsCross: 12,
      wireRingsHoriz: 4,
      // Light-blue cart livery: dark rails, sky-blue basket, blue wheels
      frameColor:  0x14161a,   // darker rails (down from default 0x2a2c30)
      basketColor: 0x87ceeb,   // light/sky-blue wire basket
      handleColor: 0x4a90c2,   // medium-blue plastic grip
      seatColor:   0x141518,   // black child seat for contrast
      wheelColor:  0x2563eb,   // blue wheels
      castShadow: false, receiveShadow: false,
    }});
  }}
  var CART_TOP_Y = 0.74;   // approx cart top in scene units (1.02m * 0.7 + a bit)

  function makeCartGroup() {{
    var g = new THREE.Group();
    var cartVisual;
    if (_cartTemplate) {{
      cartVisual = _cartTemplate.clone(true);
    }} else {{
      // Fallback if cart-model.js wasn't inlined — keep the simple box.
      cartVisual = new THREE.Group();
      var body = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.3, 0.35), cartMat);
      body.position.y = 0.15; cartVisual.add(body);
      var handle = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.25, 0.35), cartMat);
      handle.position.set(0.26, 0.27, 0); cartVisual.add(handle);
    }}
    g.add(cartVisual);

    var cone = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.2, 6), coneMat);
    cone.position.y = 0.05; cone.rotation.x = Math.PI / 2; cone.visible = false;
    g.add(cone);
    var label = makeLabel('C?');
    label.position.set(0, CART_TOP_Y + 0.32, 0);
    g.add(label);
    var badge = makeLabel('');
    badge.position.set(0, CART_TOP_Y + 0.05, 0);
    badge.scale.set(0.7, 0.35, 1);
    badge.visible = false;
    g.add(badge);
    g.visible = false;
    g.userData = {{ id: null, cone: cone, label: label, badge: badge,
                   cartVisual: cartVisual }};
    scene.add(g);
    return g;
  }}

  var personPool = []; for (var i = 0; i < POOL; i++) personPool.push(makePersonGroup());
  var cartPool = [];   for (var i = 0; i < POOL; i++) cartPool.push(makeCartGroup());

  /* Link lines */
  var linkLines = [];
  for (var i = 0; i < POOL; i++) {{
    var geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3()
    ]);
    var line = new THREE.Line(geo, linkMat);
    line.visible = false;
    scene.add(line);
    linkLines.push(line);
  }}

  /* Trail buffers */
  var trailData = {{}};
  function getTrail(id, isPerson) {{
    if (trailData[id]) return trailData[id];
    var positions = new Float32Array(TRAIL_LEN * 3);
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setDrawRange(0, 0);
    var line = new THREE.Line(geo, isPerson ? pTrailMat : cTrailMat);
    scene.add(line);
    trailData[id] = {{ pts: [], line: line, geo: geo }};
    return trailData[id];
  }}

  /* Render zones now that toWorld + makeLabel are defined */
  addZones();

  /* PLAYBACK */
  var currentIdx = 0, playing = false, speed = 1.0, lastTime = 0;
  var slider = document.getElementById('frame-slider');
  var frameInfo = document.getElementById('frame-info');
  var playBtn = document.getElementById('play-btn');
  var speedSel = document.getElementById('speed-select');

  slider.max = FRAMES.length - 1;
  frameInfo.textContent = 'Frame 0 / ' + FRAMES.length;

  playBtn.onclick = function() {{
    playing = !playing;
    playBtn.innerHTML = playing ? '&#9646;&#9646; Pause' : '&#9654; Play';
  }};
  slider.oninput = function() {{
    currentIdx = parseInt(slider.value);
    updateScene(currentIdx);
  }};
  speedSel.onchange = function() {{ speed = parseFloat(speedSel.value); }};

  /* UPDATE SCENE */
  var activeTrails = {{}};

  function updateScene(idx) {{
    if (idx < 0 || idx >= FRAMES.length) return;
    var frame = FRAMES[idx];
    slider.value = idx;
    frameInfo.textContent = 'Frame ' + frame.f + ' / ' + FRAMES[FRAMES.length - 1].f;
    activeTrails = {{}};

    /* People */
    var pKeys = Object.keys(frame.p);
    for (var i = 0; i < POOL; i++) {{
      var g = personPool[i];
      if (i < pKeys.length) {{
        var k = pKeys[i];
        var d = frame.p[k];
        var pos = toWorld(d.x, d.y);
        g.position.set(pos.x, 0, pos.z);
        g.visible = true;
        g.userData.id = k;
        updateLabel(g.userData.label, k, 'rgba(0,0,0,0.6)');
        if (d.s !== 'STATIC') {{
          g.userData.cone.visible = true;
          var rad = d.d * Math.PI / 180;
          g.userData.cone.position.set(Math.cos(rad) * 0.4, 0.05, Math.sin(rad) * 0.4);
          g.userData.cone.rotation.z = -rad + Math.PI / 2;
        }} else {{ g.userData.cone.visible = false; }}

        /* Procedural shopper: walk cycle + face direction of motion. */
        var bv = g.userData.bodyVisual;
        if (bv.userData && bv.userData.updateWalk) {{
          var walkAmt = (d.s !== 'STATIC') ? 1.0 : 0.0;
          var rad2 = (d.d || 0) * Math.PI / 180;
          // Person rests facing +X; align with the BEV motion vector
          // (cos(rad), 0, sin(rad)) by rotating around +Y.
          if (d.s !== 'STATIC') bv.rotation.y = -rad2;
          bv.userData.updateWalk(idx / FPS, walkAmt);
        }}

        var trail = getTrail(k, true);
        trail.pts.push(pos.x, 0.02, pos.z);
        if (trail.pts.length > TRAIL_LEN * 3) trail.pts.splice(0, 3);
        var pa = trail.geo.getAttribute('position');
        for (var j = 0; j < trail.pts.length; j++) pa.array[j] = trail.pts[j];
        pa.needsUpdate = true;
        trail.geo.setDrawRange(0, trail.pts.length / 3);
        trail.line.visible = true;
        activeTrails[k] = true;
      }} else {{
        g.visible = false; g.userData.id = null;
      }}
    }}

    /* Carts */
    var cKeys = Object.keys(frame.c);
    for (var i = 0; i < POOL; i++) {{
      var g = cartPool[i];
      if (i < cKeys.length) {{
        var k = cKeys[i];
        var d = frame.c[k];
        var pos = toWorld(d.x, d.y);
        g.position.set(pos.x, 0, pos.z);
        g.visible = true;
        g.userData.id = k;
        updateLabel(g.userData.label, k, 'rgba(0,0,0,0.6)');
        if (d.ps > 0) {{
          var bg = d.ps >= 71 ? '#ff1744' : d.ps >= 31 ? '#ff9100' : '#00c853';
          updateLabel(g.userData.badge, 'POPS:' + d.ps, bg);
          g.userData.badge.visible = true;
        }} else {{ g.userData.badge.visible = false; }}
        if (d.s !== 'STATIC') {{
          g.userData.cone.visible = true;
          var rad = d.d * Math.PI / 180;
          g.userData.cone.position.set(Math.cos(rad) * 0.4, 0.05, Math.sin(rad) * 0.4);
          g.userData.cone.rotation.z = -rad + Math.PI / 2;
          /* Cart faces +X by default; rotate around +Y so its forward
             aligns with the BEV motion vector (cos(rad), 0, sin(rad)). */
          if (g.userData.cartVisual) g.userData.cartVisual.rotation.y = -rad;
        }} else {{ g.userData.cone.visible = false; }}
        var trail = getTrail(k, false);
        trail.pts.push(pos.x, 0.02, pos.z);
        if (trail.pts.length > TRAIL_LEN * 3) trail.pts.splice(0, 3);
        var pa = trail.geo.getAttribute('position');
        for (var j = 0; j < trail.pts.length; j++) pa.array[j] = trail.pts[j];
        pa.needsUpdate = true;
        trail.geo.setDrawRange(0, trail.pts.length / 3);
        trail.line.visible = true;
        activeTrails[k] = true;
      }} else {{ g.visible = false; g.userData.id = null; }}
    }}

    /* Hide unused trails */
    for (var tid in trailData) {{
      if (!activeTrails[tid]) trailData[tid].line.visible = false;
    }}

    /* Link lines */
    var li = 0;
    for (var ci = 0; ci < cKeys.length; ci++) {{
      var cd = frame.c[cKeys[ci]];
      if (cd.lp != null && li < linkLines.length) {{
        var pKey = 'P' + cd.lp;
        if (frame.p[pKey]) {{
          var cp = toWorld(cd.x, cd.y);
          var pp = toWorld(frame.p[pKey].x, frame.p[pKey].y);
          var lpos = linkLines[li].geometry.getAttribute('position');
          lpos.array[0] = cp.x; lpos.array[1] = 0.1; lpos.array[2] = cp.z;
          lpos.array[3] = pp.x; lpos.array[4] = 0.1; lpos.array[5] = pp.z;
          lpos.needsUpdate = true;
          linkLines[li].visible = true;
          li++;
        }}
      }}
    }}
    for (; li < linkLines.length; li++) linkLines[li].visible = false;
  }}

  /* ANIMATE */
  if (FRAMES.length > 0) updateScene(0);

  function animate(ts) {{
    requestAnimationFrame(animate);
    if (playing && FRAMES.length > 1) {{
      var elapsed = ts - lastTime;
      var interval = 1000 / (FPS * speed);
      if (elapsed >= interval) {{
        currentIdx = (currentIdx + 1) % FRAMES.length;
        lastTime = ts;
        updateScene(currentIdx);
      }}
    }}
    controls.update();
    renderer.render(scene, camera);
  }}
  requestAnimationFrame(animate);

  /* Resize */
  window.addEventListener('resize', function() {{
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }});
}})();
</script>
</body>
</html>'''
    # Inject the inline JS content via non-brace markers so we don't have
    # to escape every `{`/`}` inside the f-string.
    html = html.replace("/*__CART_MODEL_JS__*/", _load_cart_model_js())
    html = html.replace("/*__STORE_FIXTURES_JS__*/", _load_store_fixtures_js())
    html = html.replace("/*__PERSON_MODEL_JS__*/", _load_person_model_js())
    return html


def build_3d_bev_html(slim_frames: list, video_width: int,
                      video_height: int, fps: float,
                      total_frames: int,
                      zones: list | None = None,
                      enable_pose: bool = False) -> str:
    """Public entry point.  Returns the raw inner HTML document string.

    In Gradio 6+, gr.HTML() uses js_on_load to inject this as an
    iframe via contentDocument.write() — see app_poc_v2.py for the wiring.
    """
    if not slim_frames:
        return ""

    frames_json = json.dumps(slim_frames, separators=(',', ':'))
    zones_json  = json.dumps(_zones_to_payload(zones, video_width, video_height),
                             separators=(',', ':'))
    return _build_threejs_document(frames_json, video_width,
                                   video_height, fps,
                                   zones_json=zones_json,
                                   enable_pose=enable_pose)
