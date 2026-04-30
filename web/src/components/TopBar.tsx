import React from 'react';
import { Icons } from '../icons';
import { useAppStore } from '../store';
import { runAnalysis, invalidateCache, recomputeAnalytics } from '../api';
import s from './TopBar.module.css';

interface Props {
  dockOpen: boolean;
  onDockToggle: () => void;
}

export const TopBar: React.FC<Props> = ({ dockOpen, onDockToggle }) => {
  const { runResult, isRunning, videoId, zones, cameraPlacement, vlmBackend, vlmApiKey,
          setIsRunning, setRunResult, applyRecompute } = useAppStore();

  const tracking = runResult?.tracking_json;
  const summary  = tracking?.summary;
  const high     = summary?.high_priority ?? 0;

  async function handleRun() {
    if (!videoId || isRunning) return;
    setIsRunning(true);
    try {
      const result = await runAnalysis({ video_id: videoId, camera_placement: cameraPlacement, vlm_backend: vlmBackend, vlm_api_key: vlmApiKey, zones });
      setRunResult(result);
    } catch (e) {
      console.error('[POPS] run failed:', e);
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <header className={s.topbar}>
      <div className={s.brand}>
        <span className={s.brandMark} />
        <span className={s.brandName}>Gatekeeper</span>
        <span className={s.brandSep}>/</span>
        <span className={s.brandProduct}>POPS</span>
      </div>
      <span className={s.divider} />
      <div className={s.sessionPill}>
        <span className={s.dot} />
        <span>Loss Prevention Console</span>
        <code>gk-pops-enhanced</code>
      </div>
      <div className={s.spacer} />

      {summary && (
        <>
          <div className={s.stat}><span className={s.lbl}>Persons</span><span className={s.val}>{summary.total_people_seen}</span></div>
          <div className={s.stat}><span className={s.lbl}>Carts</span><span className={`${s.val} ${s.accent}`}>{summary.total_carts_seen}</span></div>
          <div className={s.stat}><span className={s.lbl}>Alerts</span><span className={s.val} style={{ color: high > 0 ? 'var(--bad)' : 'var(--good)' }}>{high > 0 ? `${high} HIGH` : 'CLEAR'}</span></div>
          <div className={s.stat}><span className={s.lbl}>Links</span><span className={s.val}>{summary.total_links_established}</span></div>
          <span className={s.divider} />
        </>
      )}

      <button className={s.iconBtn} title="Search"><Icons.search size={14} /></button>
      <button className={s.iconBtn} title="History"><Icons.history size={14} /></button>
      <button className={`${s.iconBtn} ${dockOpen ? s.active : ''}`} title="Toggle JSON dock" onClick={onDockToggle}><Icons.layers size={14} /></button>
      <button
        className={s.runBtn}
        onClick={handleRun}
        disabled={!videoId || isRunning}
      >
        {isRunning
          ? <><span className={s.spinner} /> Processing…</>
          : <><Icons.bolt size={12} /> Run Analysis</>
        }
      </button>
    </header>
  );
};
