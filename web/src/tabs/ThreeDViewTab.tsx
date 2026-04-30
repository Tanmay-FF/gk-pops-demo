import React from 'react';
import { useAppStore } from '../store';
import { Icons } from '../icons';
import { COLOR_PERSON, COLOR_CART } from '../types';
import s from './ThreeDViewTab.module.css';

export const ThreeDViewTab: React.FC = () => {
  const { runResult } = useAppStore();
  const bev3dHtml = runResult?.bev3d_html;

  if (bev3dHtml) {
    return (
      <div className={s.root}>
        <div dangerouslySetInnerHTML={{ __html: bev3dHtml }} className={s.htmlWrap} />
      </div>
    );
  }

  return (
    <div className={s.root}>
      <div className={s.stage}>
        <div className={s.floor} />

        {/* Placeholder track pillars */}
        {[
          { id: 'P-001', color: COLOR_PERSON, left: '28%', top: '38%' },
          { id: 'C-007', color: COLOR_CART,   left: '42%', top: '52%' },
          { id: 'P-002', color: COLOR_PERSON, left: '56%', top: '42%' },
          { id: 'C-011', color: COLOR_CART,   left: '68%', top: '36%' },
          { id: 'P-003', color: COLOR_PERSON, left: '36%', top: '60%' },
        ].map(tr => (
          <div key={tr.id} className={s.pillar}
            style={{ left: tr.left, top: tr.top, background: tr.color, boxShadow: `0 24px 32px -12px ${tr.color}` }}>
            <div className={s.pillarLabel} style={{ borderColor: tr.color }}>{tr.id}</div>
          </div>
        ))}

        <div className={s.hudCards}>
          <div className={s.hudCard}>
            <span className={s.hudLbl}>View</span>
            <span className={s.hudVal}>BEV · Orbit 64°</span>
          </div>
          <div className={s.hudCard}>
            <span className={s.hudLbl}>Persons</span>
            <span className={s.hudVal} style={{ color: COLOR_PERSON }}>
              {runResult?.tracking_json?.summary?.unique_persons ?? '—'}
            </span>
          </div>
          <div className={s.hudCard}>
            <span className={s.hudLbl}>Carts</span>
            <span className={s.hudVal} style={{ color: COLOR_CART }}>
              {runResult?.tracking_json?.summary?.unique_carts ?? '—'}
            </span>
          </div>
        </div>

        <div className={s.viewButtons}>
          <button className={s.viewBtn}>Top</button>
          <button className={s.viewBtn}>Iso</button>
          <button className={s.viewBtn}>Front</button>
          <button className={s.viewBtn}><Icons.refresh size={11} /> Reset</button>
        </div>

        {!runResult && (
          <div className={s.overlay}>
            <Icons.cube size={28} style={{ color: 'var(--text-3)', marginBottom: 10 }} />
            <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>3D BEV renders after analysis</div>
          </div>
        )}
      </div>
    </div>
  );
};
