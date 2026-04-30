import React, { useRef } from 'react';
import { useTweaksStore } from '../store';
import type { Tweaks } from '../types';
import s from './TweaksPanel.module.css';

// Only rendered in dev mode
export const TweaksPanel: React.FC = () => {
  if (import.meta.env.PROD) return null;
  return <TweaksPanelInner />;
};

const TweaksPanelInner: React.FC = () => {
  const { tweaks, setTweak } = useTweaksStore();
  const [open, setOpen] = React.useState(false);
  const dragRef = useRef<HTMLDivElement>(null);

  if (!open) {
    return (
      <button className={s.fab} onClick={() => setOpen(true)} title="Tweaks">
        ⚙
      </button>
    );
  }

  return (
    <div ref={dragRef} className={s.panel}>
      <div className={s.hd}>
        <b>Tweaks</b>
        <button className={s.close} onClick={() => setOpen(false)}>✕</button>
      </div>
      <div className={s.body}>
        <Section label="Theme">
          <SliderRow label="Accent hue" value={tweaks.accentHue} min={0} max={360}
            onChange={(v) => { setTweak('accentHue', v); document.documentElement.style.setProperty('--accent-h', String(v)); }} />
          <RadioRow label="Density" value={tweaks.density}
            opts={['compact','cozy','comfortable'] as const}
            onChange={(v) => { setTweak('density', v as Tweaks['density']); document.documentElement.dataset.density = v; }} />
        </Section>
        <Section label="Layout">
          <RadioRow label="Rail side" value={tweaks.railSide} opts={['left','right'] as const}
            onChange={(v) => setTweak('railSide', v as 'left'|'right')} />
          <ToggleRow label="JSON dock" value={tweaks.showDock} onChange={(v) => setTweak('showDock', v)} />
        </Section>
        <Section label="Tracked output">
          <ToggleRow label="HUD overlays" value={tweaks.showHud} onChange={(v) => setTweak('showHud', v)} />
          <ToggleRow label="Zone overlays" value={tweaks.showZones} onChange={(v) => setTweak('showZones', v)} />
          <ToggleRow label="Camera PiP" value={tweaks.showPip} onChange={(v) => setTweak('showPip', v)} />
          <ToggleRow label="Animated tracks" value={tweaks.animated} onChange={(v) => setTweak('animated', v)} />
        </Section>
      </div>
    </div>
  );
};

const Section: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className={s.section}><div className={s.sectionLabel}>{label}</div>{children}</div>
);

const SliderRow: React.FC<{ label: string; value: number; min: number; max: number; onChange: (v: number) => void }> = ({ label, value, min, max, onChange }) => (
  <div className={s.row}>
    <div className={s.rowLabel}><span>{label}</span><span className={s.muted}>{value}</span></div>
    <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} className={s.slider} />
  </div>
);

const ToggleRow: React.FC<{ label: string; value: boolean; onChange: (v: boolean) => void }> = ({ label, value, onChange }) => (
  <div className={`${s.row} ${s.rowH}`}>
    <div className={s.rowLabel}><span>{label}</span></div>
    <button className={`${s.toggle} ${value ? s.on : ''}`} onClick={() => onChange(!value)}><i /></button>
  </div>
);

const RadioRow: React.FC<{ label: string; value: string; opts: readonly string[]; onChange: (v: string) => void }> = ({ label, value, opts, onChange }) => (
  <div className={s.row}>
    <div className={s.rowLabel}><span>{label}</span></div>
    <div className={s.seg}>
      {opts.map((o) => (
        <button key={o} className={`${s.segBtn} ${o === value ? s.active : ''}`} onClick={() => onChange(o)}>{o}</button>
      ))}
    </div>
  </div>
);
