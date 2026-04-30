import React from 'react';
import { useAppStore } from '../store';
import { popsColor, COLOR_PUSHOUT, COLOR_SUSPICIOUS } from '../types';
import { Icons } from '../icons';
import s from './POPSTab.module.css';

export const POPSTab: React.FC = () => {
  const { runResult } = useAppStore();
  const carts = runResult?.pops_carts ?? [];

  const high = carts.filter(c => c.max_pops >= 71).length;
  const med  = carts.filter(c => c.max_pops >= 31 && c.max_pops < 71).length;

  const sorted = [...carts].sort((a, b) => b.max_pops - a.max_pops);

  return (
    <div className={s.root}>
      <div className={s.content}>
        <div className={s.kpiGrid}>
          <div className={s.kpi}>
            <span className={s.lbl}>Carts tracked</span>
            <span className={s.val}>{carts.length}</span>
            <span className={s.delta}>→ this run</span>
          </div>
          <div className={s.kpi}>
            <span className={s.lbl}>High priority (POPS 71+)</span>
            <span className={s.val} style={{ color: COLOR_PUSHOUT }}>{high}</span>
            <span className={s.deltaUp}>↑ this run</span>
          </div>
          <div className={s.kpi}>
            <span className={s.lbl}>Medium (POPS 31–70)</span>
            <span className={s.val} style={{ color: COLOR_SUSPICIOUS }}>{med}</span>
            <span className={s.delta}>→ stable</span>
          </div>
        </div>

        {carts.length === 0 ? (
          <div className={s.empty}>
            <Icons.cart size={28} style={{ color: 'var(--text-3)', marginBottom: 10 }} />
            <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>No carts tracked yet</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 4 }}>Run analysis to see POPS scores</div>
          </div>
        ) : (
          <div className={s.surface}>
            <table className={s.table}>
              <thead>
                <tr>
                  <th>Cart</th>
                  <th>Max POPS</th>
                  <th>Peak event</th>
                  <th>Quality</th>
                  <th>Fill</th>
                  <th>Bag</th>
                  <th>Direction</th>
                  <th>Score bar</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(c => {
                  const color = popsColor(c.max_pops);
                  return (
                    <tr key={c.id}>
                      <td className={s.mono} style={{ color: 'var(--text-0)' }}>{c.id}</td>
                      <td className={s.mono} style={{ color, fontWeight: 600, fontSize: 13 }}>{c.max_pops}</td>
                      <td>
                        <span className={s.chip} style={{ color, borderColor: `${color}55`, background: `${color}15` }}>
                          {c.peak_event}
                        </span>
                      </td>
                      <td className={s.upper11} style={{ color: c.quality === 'valid' ? 'var(--good)' : 'var(--bad)' }}>
                        {c.quality}
                      </td>
                      <td className={s.upper11}>{c.fill}</td>
                      <td className={s.upper11}>{c.bag}</td>
                      <td className={s.upper11}>{c.direction}</td>
                      <td style={{ width: 140 }}>
                        <div className={s.progress}>
                          <div style={{ width: `${c.max_pops}%`, background: color }} />
                        </div>
                      </td>
                      <td>
                        <button className={s.iconBtn}><Icons.more size={12} /></button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
