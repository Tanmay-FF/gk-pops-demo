import React, { useState } from 'react';
import { Icons } from '../icons';
import { useAppStore } from '../store';
import s from './JsonDock.module.css';

export const JsonDock: React.FC = () => {
  const [tab, setTab] = useState<'json' | 'log'>('json');
  const { runResult } = useAppStore();
  const json = runResult?.tracking_json;

  function copyJson() {
    if (json) navigator.clipboard.writeText(JSON.stringify(json, null, 2)).catch(() => {});
  }

  return (
    <aside className={s.dock}>
      <div className={s.header}>
        <div className={s.dockTabs}>
          <button className={`${s.dockTab} ${tab === 'json' ? s.active : ''}`} onClick={() => setTab('json')}>Tracking JSON</button>
          <button className={`${s.dockTab} ${tab === 'log' ? s.active : ''}`} onClick={() => setTab('log')}>Log</button>
        </div>
        <div className={s.tools}>
          <button className={s.iconBtn} title="Copy JSON" onClick={copyJson}><Icons.copy size={12} /></button>
          <button className={s.iconBtn} title="Expand"><Icons.expand size={12} /></button>
        </div>
      </div>

      <div className={s.body}>
        {tab === 'json' && (
          json
            ? <JsonView data={json} />
            : <pre className={s.jsonView} style={{ color: 'var(--text-3)', padding: 12 }}>{"// Run analysis to see tracking JSON"}</pre>
        )}
        {tab === 'log' && (
          <pre className={s.jsonView} style={{ color: 'var(--text-2)', fontSize: 11 }}>
            {json
              ? `# POPS tracking log\n[INFO]  video: ${json.video_info?.video_name}\n[INFO]  resolution: ${json.video_info?.width}×${json.video_info?.height} @ ${json.video_info?.fps?.toFixed(2)}fps\n[INFO]  frames: ${json.video_info?.total_frames}\n[INFO]  model: ${json.processing_info?.model}\n[INFO]  tracker: ${json.processing_info?.tracker}\n[INFO]  device: ${json.processing_info?.device}\n[INFO]  persons: ${json.summary?.total_people_seen}\n[INFO]  carts: ${json.summary?.total_carts_seen}\n[INFO]  links: ${json.summary?.total_links_established}\n[WARN]  high-priority events: ${json.summary?.high_priority}\n[INFO]  medium-priority events: ${json.summary?.medium_priority}`
              : '# Waiting for analysis...'
            }
          </pre>
        )}
      </div>

      <div className={s.footer}>
        <button className={s.btn} onClick={copyJson} disabled={!json}><Icons.copy size={11} /> Copy JSON</button>
        {json && (
          <a
            className={`${s.btn} ${s.primary}`}
            href={`data:application/json,${encodeURIComponent(JSON.stringify(json, null, 2))}`}
            download="pops_tracking.json"
          >
            <Icons.download size={11} /> Download
          </a>
        )}
      </div>
    </aside>
  );
};

// Simple syntax-highlighted JSON viewer
const JsonView: React.FC<{ data: unknown; depth?: number }> = ({ data, depth = 0 }) => {
  const str = JSON.stringify(data, null, 2);
  // Colorize via regex on the raw string
  const highlighted = str
    .replace(/("(?:[^"\\]|\\.)*")\s*:/g, '<span class="jk">$1</span>:')
    .replace(/:\s*("(?:[^"\\]|\\.)*")/g, ': <span class="js">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="jn">$1</span>')
    .replace(/:\s*(true|false)/g, ': <span class="jb">$1</span>')
    .replace(/:\s*(null)/g, ': <span class="jnl">$1</span>');

  return (
    <pre
      className="json-view"
      style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.55, padding: '10px 14px', whiteSpace: 'pre', color: 'var(--text-1)', margin: 0 }}
      dangerouslySetInnerHTML={{ __html: highlighted }}
    />
  );
};
