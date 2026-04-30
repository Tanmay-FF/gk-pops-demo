import React from 'react';
import { useAppStore } from '../store';
import { COLOR_PERSON, COLOR_CART, COLOR_LINK, COLOR_PUSHOUT, COLOR_SUSPICIOUS, COLOR_CLEAR, bgrToCss } from '../types';
import s from './LegendTab.module.css';

export const LegendTab: React.FC = () => {
  const { zones } = useAppStore();

  return (
    <div className={s.root}>
      <div className={s.content}>
        <div className={s.grid}>
          <div className={s.card}>
            <h4 className={s.h4}>Annotations</h4>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: COLOR_PERSON }} />
              <span className={s.rowLabel}>Green box</span>
              <span className={s.rowNote}>Person</span>
            </div>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: COLOR_CART }} />
              <span className={s.rowLabel}>Orange box</span>
              <span className={s.rowNote}>Cart</span>
            </div>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: COLOR_LINK }} />
              <span className={s.rowLabel}>Magenta line</span>
              <span className={s.rowNote}>Person-cart link</span>
            </div>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: '#34d399' }} />
              <span className={s.rowLabel}>Green label</span>
              <span className={s.rowNote}>Valid cart</span>
            </div>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: '#ef4444' }} />
              <span className={s.rowLabel}>Red label</span>
              <span className={s.rowNote}>Unclear cart</span>
            </div>
          </div>

          <div className={s.card}>
            <h4 className={s.h4}>POPS Score</h4>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: COLOR_PUSHOUT }} />
              <span className={s.rowLabel}>71-100 · High</span>
              <span className={s.rowNote}>PUSHOUT</span>
            </div>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: COLOR_SUSPICIOUS }} />
              <span className={s.rowLabel}>31-70 · Medium</span>
              <span className={s.rowNote}>SUSPICIOUS</span>
            </div>
            <div className={s.row}>
              <div className={s.swatch} style={{ background: COLOR_CLEAR }} />
              <span className={s.rowLabel}>0-30 · Low</span>
              <span className={s.rowNote}>MONITORING</span>
            </div>
          </div>

          <div className={s.card}>
            <h4 className={s.h4}>Zones</h4>
            {zones.length === 0 ? (
              <div className={s.emptyZones}>No zones drawn — use Zone Editor to add zones</div>
            ) : (
              zones.map(z => (
                <div key={z.zone_id} className={s.row}>
                  <div className={s.swatch} style={{ background: bgrToCss(z.color) }} />
                  <span className={s.rowLabel}>{z.name}</span>
                  <span className={s.rowNote}>[{z.applies_to}]</span>
                </div>
              ))
            )}
          </div>

          <div className={s.card}>
            <h4 className={s.h4}>Keyboard shortcuts</h4>
            <div className={s.row}><kbd className={s.kbd}>Space</kbd><span className={s.rowLabel}>Play / pause</span></div>
            <div className={s.row}><kbd className={s.kbd}>P</kbd><span className={s.rowLabel}>Polygon tool</span></div>
            <div className={s.row}><kbd className={s.kbd}>R</kbd><span className={s.rowLabel}>Run analysis</span></div>
            <div className={s.row}><kbd className={s.kbd}>F</kbd><span className={s.rowLabel}>Fullscreen viewer</span></div>
            <div className={s.row}><kbd className={s.kbd}>⌫</kbd><span className={s.rowLabel}>Undo last vertex</span></div>
            <div className={s.row}><kbd className={s.kbd}>Esc</kbd><span className={s.rowLabel}>Cancel polygon</span></div>
          </div>
        </div>
      </div>
    </div>
  );
};
