import React, { useState } from 'react';
import s from './ConfigTab.module.css';

interface Cfg {
  qualityThreshold: string;
  linkFrames: string;
  linkGrace: string;
}

export const ConfigTab: React.FC = () => {
  const [cfg, setCfg] = useState<Cfg>({
    qualityThreshold: '0.55',
    linkFrames: '15',
    linkGrace: '30',
  });

  const set = <K extends keyof Cfg>(k: K) => (v: Cfg[K]) => setCfg(c => ({ ...c, [k]: v }));

  return (
    <div className={s.root}>
      <div className={s.content}>
        <div className={s.surface}>
          <h3 className={s.h3}>Detection & Linking</h3>
          <div className={s.configRow}>
            <div>
              <div className={s.label}>Quality threshold</div>
              <div className={s.desc}>Min cart-quality score · 0–1</div>
            </div>
            <input className={s.inputMono} value={cfg.qualityThreshold}
              onChange={e => set('qualityThreshold')(e.target.value)} />
          </div>
          <div className={s.configRow}>
            <div>
              <div className={s.label}>Link confirmation frames</div>
              <div className={s.desc}>Co-movement frames before a person-cart link is confirmed</div>
            </div>
            <input className={s.inputMono} value={cfg.linkFrames}
              onChange={e => set('linkFrames')(e.target.value)} />
          </div>
          <div className={s.configRow}>
            <div>
              <div className={s.label}>Link grace period</div>
              <div className={s.desc}>Frames before the linker tries to attach a new cart</div>
            </div>
            <input className={s.inputMono} value={cfg.linkGrace}
              onChange={e => set('linkGrace')(e.target.value)} />
          </div>
        </div>
      </div>
    </div>
  );
};
