import React from 'react';
import { useAppStore } from '../store';
import { Icons } from '../icons';
import { bgrToCss } from '../types';
import s from './AnalyticsTab.module.css';

const ZonesOverview: React.FC = () => {
  const { frameUrl, frameSize, zones } = useAppStore();
  if (!frameUrl || zones.length === 0) return null;

  const toPct = (pts: [number, number][]) =>
    pts.map(([x, y]) => `${(x / frameSize.w) * 100},${(y / frameSize.h) * 100}`).join(' ');

  return (
    <div className={s.surface} style={{ padding: 14 }}>
      <div className={s.sectionTitle}>
        <h3>Defined zones</h3>
        <span className={s.sub}>{zones.length} zone{zones.length === 1 ? '' : 's'} on first frame</span>
      </div>

      <div className={s.zonesPreviewWrap}>
        <div className={s.zonesPreview} style={{ aspectRatio: `${frameSize.w} / ${frameSize.h}` }}>
          <img src={frameUrl} alt="First frame" className={s.zonesPreviewImg} draggable={false} />
          <svg className={s.zonesPreviewOverlay} viewBox="0 0 100 100" preserveAspectRatio="none">
            {zones.map((z) => {
              const css = bgrToCss(z.color);
              return (
                <polygon key={z.zone_id} points={toPct(z.polygon)}
                  fill={`${css}33`} stroke={css} strokeWidth="0.5"
                  vectorEffect="non-scaling-stroke" />
              );
            })}
          </svg>
          {zones.map((z) => {
            const cx = z.polygon.reduce((sum, p) => sum + p[0], 0) / z.polygon.length;
            const cy = z.polygon.reduce((sum, p) => sum + p[1], 0) / z.polygon.length;
            return (
              <div key={z.zone_id} className={s.zonesPreviewLabel}
                style={{
                  left: `${(cx / frameSize.w) * 100}%`,
                  top: `${(cy / frameSize.h) * 100}%`,
                  background: bgrToCss(z.color),
                }}>
                {z.name}
              </div>
            );
          })}
        </div>

        <ul className={s.zonesLegend}>
          {zones.map((z) => (
            <li key={z.zone_id}>
              <span className={s.zonesLegendSwatch} style={{ background: bgrToCss(z.color) }} />
              <span className={s.zonesLegendName}>{z.name}</span>
              <span className={s.zonesLegendMeta}>{z.applies_to}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

const Sparkline: React.FC<{ data: number[]; color?: string; width?: number; height?: number }> = ({
  data, color = 'var(--accent)', width = 180, height = 26,
}) => {
  if (data.length < 2) return null;
  const max = Math.max(...data), min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) =>
    `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * (height - 4) - 2}`
  ).join(' ');
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polygon points={`0,${height} ${pts} ${width},${height}`} fill={color} opacity="0.18" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
};

export const AnalyticsTab: React.FC = () => {
  const { runResult } = useAppStore();
  const analytics = runResult?.analytics;
  const zones = useAppStore(s => s.zones);

  const heatmapUrl = runResult?.heatmap_url;

  return (
    <div className={s.root}>
      <div className={s.content}>
        {analytics ? (
          <>
            <ZonesOverview />

            {analytics.spikes_html && (
              <div dangerouslySetInnerHTML={{ __html: analytics.spikes_html }} className={s.htmlBlock} />
            )}

            <div className={s.cols12}>
              <div className={s.surface} style={{ padding: 14 }}>
                <div className={s.sectionTitle}>
                  <h3>Zone dwell breakdown</h3>
                  <span className={s.sub}>per-zone analytics</span>
                </div>
                {analytics.dwell_html
                  ? <div dangerouslySetInnerHTML={{ __html: analytics.dwell_html }} className={s.htmlBlock} />
                  : <div className={s.emptySection}>No zone data — draw zones first</div>
                }
              </div>

              <div className={s.surface} style={{ padding: 14 }}>
                <div className={s.sectionTitle}>
                  <h3>Floor heatmap</h3>
                  <span className={s.sub}>cart-dwell weighted</span>
                </div>
                {heatmapUrl ? (
                  <div className={s.heatmapWrap}>
                    <img src={heatmapUrl} alt="Heatmap" className={s.heatmapImg} />
                    <a href={heatmapUrl} download="heatmap.png" className={s.downloadBtn}>
                      <Icons.download size={11} /> Download PNG
                    </a>
                  </div>
                ) : (
                  <div className={s.emptySection}>Heatmap renders after analysis</div>
                )}
              </div>
            </div>

            {analytics.journey_html && (
              <div className={s.surface} style={{ padding: 14 }}>
                <div className={s.sectionTitle}>
                  <h3>Zone transition matrix</h3>
                  <span className={s.sub}>journey edges</span>
                </div>
                <div dangerouslySetInnerHTML={{ __html: analytics.journey_html }} className={s.htmlBlock} />
              </div>
            )}
          </>
        ) : (
          <div className={s.empty}>
            <Icons.chart size={28} style={{ color: 'var(--text-3)', marginBottom: 10 }} />
            <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>No analytics yet</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 4 }}>
              Draw zones in the Zone Editor, then run analysis
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
