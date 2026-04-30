import React from 'react';
import { useAppStore } from '../store';
import { popsColor } from '../types';
import { Icons } from '../icons';
import s from './EventsTab.module.css';

export const EventsTab: React.FC = () => {
  const { runResult } = useAppStore();
  const events = runResult?.tracking_json?.events ?? [];

  const high = events.filter(e =>
    ['PUSHOUT ALERT', 'HIGH PRIORITY'].includes((e.event_type ?? e.evt ?? '').toUpperCase())
  ).length;

  return (
    <div className={s.root}>
      <div className={s.content}>
        <div className={high > 0 ? s.bannerAlert : s.bannerOk}>
          <Icons.alert size={14} />
          {high > 0
            ? `ALERT: ${high} HIGH-PRIORITY EVENT${high > 1 ? 'S' : ''} DETECTED`
            : 'NO HIGH-RISK EVENTS'}
        </div>

        {events.length === 0 ? (
          <div className={s.empty}>
            <Icons.pulse size={28} style={{ color: 'var(--text-3)', marginBottom: 10 }} />
            <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>No events logged</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 4 }}>Run analysis to populate the event timeline</div>
          </div>
        ) : (
          <div className={s.surface}>
            <table className={s.table}>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Cart</th>
                  <th>Event</th>
                  <th>POPS</th>
                  <th>Fill</th>
                  <th>Bag</th>
                  <th>Speed</th>
                  <th>Direction</th>
                  <th>Link</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e, i) => {
                  const evtType = ((e as any).event_type ?? e.event ?? (e as any).evt ?? '').toUpperCase();
                  const pops = e.pops_score ?? (e as any).pops ?? 0;
                  const color = popsColor(pops);
                  const linked = e.linked;
                  const fill = (e.fill ?? '').toUpperCase();
                  const bag  = (e.bag  ?? '').toUpperCase();
                  const speed = (e.speed_status ?? '').toUpperCase();
                  const dir   = (e.direction ?? '').toUpperCase();
                  const t     = e.timestamp ?? '';
                  const frame = e.frame ?? '';
                  const cart  = e.cart_id ?? '';

                  return (
                    <tr key={i}>
                      <td className={s.mono}>
                        {t} {frame ? <span className={s.dim}>(F{frame})</span> : null}
                      </td>
                      <td className={s.mono} style={{ color: 'var(--text-0)' }}>{cart}</td>
                      <td>
                        <span className={s.chip} style={{ color, borderColor: `${color}55`, background: `${color}15` }}>
                          {evtType}
                        </span>
                      </td>
                      <td className={s.mono} style={{ color, fontWeight: 600 }}>{pops}</td>
                      <td className={s.upper11}>{fill}</td>
                      <td className={s.upper11}>{bag}</td>
                      <td className={s.upper11} style={{ color: speed === 'FAST' ? 'var(--bad)' : 'var(--text-1)' }}>{speed}</td>
                      <td className={s.upper11}>{dir}</td>
                      <td>
                        {linked
                          ? <span className={`${s.chip} ${s.good}`}><Icons.link size={9} /> LINKED</span>
                          : <span className={`${s.chip} ${s.bad}`}>NO LINK</span>
                        }
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
