import React, { useRef, useState, useEffect, useLayoutEffect } from 'react';
import { Icons } from '../icons';
import { useAppStore } from '../store';
import { makeZone } from '../api';
import { bgrToCss } from '../types';
import s from './ZoneEditor.module.css';

type Tool = 'cursor' | 'poly' | 'rect' | 'pan';

export const ZoneEditor: React.FC = () => {
  const { frameUrl, frameSize, zones, currentPoly, addZone, setCurrentPoly, removeZone } = useAppStore();
  const [tool, setTool]         = useState<Tool>('poly');
  const [zoneName, setZoneName] = useState('Aisle 1');
  const [appliesTo, setAppliesTo] = useState('person');
  const [fullscreen, setFullscreen] = useState(false);

  // Rect tool — tracks start + live preview end while dragging
  const [rectStart,   setRectStart]   = useState<[number,number] | null>(null);
  const [rectPreview, setRectPreview] = useState<[number,number] | null>(null);

  // Vertex drag
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const didDragRef = useRef(false);   // suppress click after pointer drag

  const imgRef  = useRef<HTMLImageElement>(null);
  const [boxSize, setBoxSize] = useState<{ w: number; h: number } | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  // Compute the contained-fit dimensions of the frame inside the active wrapper.
  // The wrapper is whichever element holds the frameContent (canvasWrap in normal
  // mode, fsFrameBox in fullscreen). Using a callback ref + ResizeObserver makes
  // this deterministic regardless of CSS aspect-ratio quirks.
  const setWrapNode = React.useCallback((node: HTMLDivElement | null) => {
    wrapRef.current = node;
  }, []);

  useLayoutEffect(() => {
    const node = wrapRef.current;
    if (!node || !frameUrl) { setBoxSize(null); return; }
    const measure = () => {
      const cs = window.getComputedStyle(node);
      const padX = parseFloat(cs.paddingLeft || '0') + parseFloat(cs.paddingRight  || '0');
      const padY = parseFloat(cs.paddingTop  || '0') + parseFloat(cs.paddingBottom || '0');
      const cw = node.clientWidth  - padX;
      const ch = node.clientHeight - padY;
      if (cw <= 0 || ch <= 0) return;
      const fa = frameSize.w / frameSize.h;
      const ca = cw / ch;
      const w = fa > ca ? cw : ch * fa;
      const h = fa > ca ? cw / fa : ch;
      setBoxSize({ w, h });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(node);
    return () => ro.disconnect();
  }, [frameUrl, frameSize.w, frameSize.h, fullscreen]);

  // Escape exits fullscreen
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFullscreen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [fullscreen]);

  // ── coordinate helpers ──────────────────────────────────────────────
  function getSourceCoords(e: React.PointerEvent | React.MouseEvent): [number,number] | null {
    const img = imgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    const scaleX = frameSize.w / rect.width;
    const scaleY = frameSize.h / rect.height;
    return [
      Math.max(0, Math.min(frameSize.w, Math.round((e.clientX - rect.left) * scaleX))),
      Math.max(0, Math.min(frameSize.h, Math.round((e.clientY - rect.top)  * scaleY))),
    ];
  }

  function toSvgPct(pts: [number,number][]): string {
    return pts.map(([x, y]) => `${(x / frameSize.w) * 100},${(y / frameSize.h) * 100}`).join(' ');
  }

  // ── pointer events on canvas ────────────────────────────────────────
  function onCanvasPointerDown(e: React.PointerEvent) {
    if (!frameUrl || e.button !== 0) return;
    didDragRef.current = false;

    if (tool === 'rect') {
      e.preventDefault();
      const coords = getSourceCoords(e);
      if (!coords) return;
      setRectStart(coords);
      setRectPreview(coords);
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    }
  }

  function onCanvasPointerMove(e: React.PointerEvent) {
    if (!frameUrl) return;
    const coords = getSourceCoords(e);
    if (!coords) return;

    if (tool === 'rect' && rectStart) {
      setRectPreview(coords);
      return;
    }

    if (draggingIdx !== null) {
      didDragRef.current = true;
      const next = [...currentPoly] as [number,number][];
      next[draggingIdx] = coords;
      setCurrentPoly(next);
    }
  }

  function onCanvasPointerUp(e: React.PointerEvent) {
    if (tool === 'rect' && rectStart) {
      const end = getSourceCoords(e) ?? rectStart;
      const [x1, y1] = rectStart;
      const [x2, y2] = end;
      setCurrentPoly([
        [Math.min(x1, x2), Math.min(y1, y2)],
        [Math.max(x1, x2), Math.min(y1, y2)],
        [Math.max(x1, x2), Math.max(y1, y2)],
        [Math.min(x1, x2), Math.max(y1, y2)],
      ]);
      setRectStart(null);
      setRectPreview(null);
      didDragRef.current = true; // prevent click from also firing
      return;
    }
    setDraggingIdx(null);
  }

  function onCanvasClick(e: React.MouseEvent) {
    if (didDragRef.current) { didDragRef.current = false; return; }
    if (tool !== 'poly' || !frameUrl) return;
    const coords = getSourceCoords(e);
    if (!coords) return;
    setCurrentPoly([...currentPoly, coords]);
  }

  // ── vertex drag handles ─────────────────────────────────────────────
  function startVertexDrag(e: React.PointerEvent, idx: number) {
    e.stopPropagation();
    e.preventDefault();
    didDragRef.current = false;
    setDraggingIdx(idx);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }

  // ── close polygon → call API ────────────────────────────────────────
  async function handleClosePolygon() {
    if (currentPoly.length < 3) {
      alert('Need at least 3 vertices to close a polygon.');
      return;
    }
    try {
      const zone = await makeZone({
        name: zoneName,
        points: currentPoly,
        applies_to: appliesTo,
        zone_index: zones.length,
      });
      addZone(zone);
      setCurrentPoly([]);
      setZoneName(`Aisle ${zones.length + 2}`);
    } catch (err) {
      alert(`Zone error: ${err}`);
    }
  }

  // ── derived display values ──────────────────────────────────────────
  const inProgressPct = toSvgPct(currentPoly);

  // Rect preview in SVG % space
  const rectSvg = rectStart && rectPreview ? (() => {
    const rx = (Math.min(rectStart[0], rectPreview[0]) / frameSize.w) * 100;
    const ry = (Math.min(rectStart[1], rectPreview[1]) / frameSize.h) * 100;
    const rw = (Math.abs(rectPreview[0] - rectStart[0]) / frameSize.w) * 100;
    const rh = (Math.abs(rectPreview[1] - rectStart[1]) / frameSize.h) * 100;
    return { rx, ry, rw, rh };
  })() : null;

  // ── shared canvas content ───────────────────────────────────────────
  const canvasCursor = draggingIdx !== null ? 'grabbing'
    : tool === 'rect' && rectStart ? 'crosshair'
    : tool === 'poly' || tool === 'rect' ? 'crosshair'
    : 'default';

  const frameContent = frameUrl ? (
    <div className={s.frameBox}
      style={boxSize ? { width: boxSize.w, height: boxSize.h } : { visibility: 'hidden' }}>
      <img ref={imgRef} src={frameUrl} alt="First frame" className={s.frame} draggable={false} />

      {/* SVG polygon overlays */}
      <svg className={s.overlay} viewBox="0 0 100 100" preserveAspectRatio="none">
        {/* Committed zones */}
        {zones.map((z) => {
          const pct = toSvgPct(z.polygon);
          const css = bgrToCss(z.color);
          return (
            <polygon key={z.zone_id} points={pct}
              fill={`${css}28`} stroke={css} strokeWidth="0.4"
              vectorEffect="non-scaling-stroke" />
          );
        })}
        {/* In-progress polygon */}
        {currentPoly.length > 1 && (
          <polygon points={inProgressPct}
            fill="color-mix(in oklch, var(--accent) 18%, transparent)"
            stroke="var(--accent-hi)" strokeWidth="0.5"
            strokeDasharray="2 1" vectorEffect="non-scaling-stroke" />
        )}
        {/* Rect preview */}
        {rectSvg && (
          <rect x={rectSvg.rx} y={rectSvg.ry} width={rectSvg.rw} height={rectSvg.rh}
            fill="color-mix(in oklch, var(--accent) 12%, transparent)"
            stroke="var(--accent-hi)" strokeWidth="0.5"
            strokeDasharray="2 1" vectorEffect="non-scaling-stroke" />
        )}
      </svg>

      {/* Zone labels */}
      {zones.map((z) => {
        const cx = z.polygon.reduce((sum, p) => sum + p[0], 0) / z.polygon.length;
        const cy = z.polygon.reduce((sum, p) => sum + p[1], 0) / z.polygon.length;
        return (
          <div key={z.zone_id} className={s.zoneLabel}
            style={{ left: `${(cx / frameSize.w) * 100}%`, top: `${(cy / frameSize.h) * 100}%`, background: bgrToCss(z.color) }}>
            {z.name} [{z.applies_to}]
            <button className={s.labelX} onClick={(e) => { e.stopPropagation(); removeZone(z.zone_id); }}>×</button>
          </div>
        );
      })}

      {/* Draggable vertex handles for in-progress polygon */}
      {currentPoly.map(([x, y], i) => (
        <div key={i} className={s.vertex}
          style={{
            left: `${(x / frameSize.w) * 100}%`,
            top:  `${(y / frameSize.h) * 100}%`,
            pointerEvents: 'auto',
            cursor: draggingIdx === i ? 'grabbing' : 'grab',
          }}
          onPointerDown={(e) => startVertexDrag(e, i)}
        />
      ))}
    </div>
  ) : (
    <div className={s.noFrame}>
      <Icons.cam size={28} style={{ color: 'var(--text-3)', marginBottom: 10 }} />
      <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>No video loaded</div>
      <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 4 }}>Upload a video to start drawing zones</div>
    </div>
  );

  // ── tool pill bar (shared between normal + fullscreen) ──────────────
  const toolBar = (
    <div className={s.tools}>
      {(['cursor','poly','rect','pan'] as Tool[]).map((t) => {
        const I = t === 'cursor' ? Icons.cursor : t === 'poly' ? Icons.poly : t === 'rect' ? Icons.rect : Icons.hand;
        return (
          <button key={t} className={`${s.tool} ${tool === t ? s.active : ''}`}
            onClick={() => setTool(t)} title={t}>
            <I size={13} />
          </button>
        );
      })}
    </div>
  );

  // ── form row (shared) ───────────────────────────────────────────────
  const formRow = (
    <>
      <div className={s.formField}>
        <label className={s.formLabel}>Zone name</label>
        <input className={s.input} value={zoneName} onChange={(e) => setZoneName(e.target.value)} placeholder="e.g. Checkout, Entrance…" />
      </div>
      <div className={s.formField}>
        <label className={s.formLabel}>Applies to</label>
        <select className={s.select} value={appliesTo} onChange={(e) => setAppliesTo(e.target.value)}>
          <option>person</option><option>cart</option><option>both</option>
        </select>
      </div>
      <button className={`${s.btn} ${s.primary}`} onClick={handleClosePolygon} disabled={currentPoly.length < 3}>
        <Icons.check size={12} /> Close polygon
      </button>
      <button className={s.btn} onClick={() => setCurrentPoly(currentPoly.slice(0, -1))}>
        <Icons.undo size={12} /> Undo
      </button>
      <button className={s.btn} onClick={() => setCurrentPoly([])}>
        <Icons.x size={12} /> Clear
      </button>
      <button className={`${s.btn} ${s.danger}`} onClick={() => { setCurrentPoly([]); useAppStore.getState().setZones([]); }}>
        <Icons.trash size={12} /> Clear all
      </button>
    </>
  );

  // ── fullscreen mode ─────────────────────────────────────────────────
  if (fullscreen) {
    return (
      <div className={s.fsOverlay}
        style={{ cursor: canvasCursor }}
        onClick={onCanvasClick}
        onPointerDown={onCanvasPointerDown}
        onPointerMove={onCanvasPointerMove}
        onPointerUp={onCanvasPointerUp}
      >
        {/* Floating top HUD */}
        <div className={s.fsHud}
          onClick={(e) => e.stopPropagation()}
          onPointerDown={(e) => e.stopPropagation()}
          onPointerMove={(e) => e.stopPropagation()}
          onPointerUp={(e) => e.stopPropagation()}
        >
          {toolBar}
          <div className={s.hint}>
            {tool === 'rect'
              ? 'Drag to draw rectangle · release to commit'
              : 'Click to place vertices · drag handles to adjust'}
          </div>
          <div style={{ flex: 1 }} />
          {currentPoly.length > 0 && (
            <span className={s.vertexChip}><span className={s.chipDot} />{currentPoly.length} pts</span>
          )}
          <button className={s.fsClose} onClick={() => setFullscreen(false)} title="Exit fullscreen (Esc)">
            <Icons.x size={14} />
          </button>
        </div>

        {/* Frame */}
        <div className={s.fsFrameBox} ref={setWrapNode}>
          {frameContent}
        </div>

        {/* Floating bottom form */}
        <div className={s.fsForm}
          onClick={(e) => e.stopPropagation()}
          onPointerDown={(e) => e.stopPropagation()}
          onPointerMove={(e) => e.stopPropagation()}
          onPointerUp={(e) => e.stopPropagation()}
        >
          {formRow}
        </div>
      </div>
    );
  }

  // ── normal (embedded) mode ──────────────────────────────────────────
  return (
    <div className={s.root}>
      <div className={s.toolbar}>
        {toolBar}
        <div className={s.hint}>
          {tool === 'rect'
            ? <>Drag to draw a rectangle · <kbd>Enter</kbd> close</>
            : <>Click to place vertices · <kbd>Enter</kbd> close · <kbd>⌫</kbd> undo</>
          }
        </div>
        <div style={{ flex: 1 }} />
        {currentPoly.length > 0 && (
          <span className={s.vertexChip}><span className={s.chipDot} />{currentPoly.length} vertices</span>
        )}
        <button className={s.expandBtn} onClick={() => setFullscreen(true)} title="Fullscreen">
          <Icons.expand size={13} />
        </button>
      </div>

      <div className={s.canvasWrap}
        ref={setWrapNode}
        style={{ cursor: canvasCursor }}
        onClick={onCanvasClick}
        onPointerDown={onCanvasPointerDown}
        onPointerMove={onCanvasPointerMove}
        onPointerUp={onCanvasPointerUp}
      >
        {frameContent}
      </div>

      <div className={s.form}>
        {formRow}
      </div>
    </div>
  );
};
