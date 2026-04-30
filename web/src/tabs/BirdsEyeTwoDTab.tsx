import React from 'react';
import { useAppStore } from '../store';
import { Icons } from '../icons';
import s from './BirdsEyeTwoDTab.module.css';

export const BirdsEyeTwoDTab: React.FC = () => {
  const { runResult } = useAppStore();
  const bev2dHtml = runResult?.bev2d_html;

  if (bev2dHtml) {
    return (
      <div className={s.root}>
        <iframe
          className={s.frame}
          title="Bird's-Eye 2D"
          srcDoc={bev2dHtml}
          sandbox="allow-scripts allow-same-origin"
        />
      </div>
    );
  }

  return (
    <div className={s.root}>
      <div className={s.placeholder}>
        <Icons.cube size={28} style={{ color: 'var(--text-3)', marginBottom: 10 }} />
        <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>
          2D bird's-eye view renders after analysis
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 6, maxWidth: 320, textAlign: 'center' }}>
          Velocity-coloured shopper paths, fixture proximity, and per-shopper impression history.
        </div>
      </div>
    </div>
  );
};
