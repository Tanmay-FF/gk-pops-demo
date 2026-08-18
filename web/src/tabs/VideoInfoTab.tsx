import React from 'react';
import { useAppStore } from '../store';
import { Icons } from '../icons';
import { COLOR_PERSON, COLOR_CART, COLOR_LINK } from '../types';
import s from './VideoInfoTab.module.css';

export const VideoInfoTab: React.FC = () => {
  const { runResult } = useAppStore();
  const info = runResult?.tracking_json?.video_info;
  const summary = runResult?.tracking_json?.summary;
  const procInfo = runResult?.tracking_json?.processing_info;

  return (
    <div className={s.root}>
      <div className={s.content}>
        <div className={s.cols2}>
          <div className={s.surface}>
            <h3 className={s.h3}>Video Information</h3>
            {info ? (
              <dl className={s.dl}>
                <dt>Video Name</dt><dd className={s.mono}>{info.video_name ?? 'n/a'}</dd>
                <dt>Resolution</dt><dd>{info.width ?? 'n/a'} × {info.height ?? 'n/a'}</dd>
                <dt>Total Frames</dt><dd>{(info.total_frames ?? 0).toLocaleString()}</dd>
                <dt>Frames Processed</dt><dd>{(info.total_frames ?? 0).toLocaleString()}</dd>
                <dt>FPS</dt><dd>{info.fps ?? 'n/a'}</dd>
                <dt>Duration</dt><dd>{(info as any).duration ?? 'n/a'}</dd>
              </dl>
            ) : (
              <div className={s.empty}><Icons.cam size={22} style={{ color: 'var(--text-3)' }} /><span>No video loaded</span></div>
            )}

            <h3 className={s.h3} style={{ marginTop: 20 }}>Detection Summary</h3>
            {summary ? (
              <dl className={s.dl}>
                <dt>Unique Persons</dt><dd style={{ fontWeight: 600, color: COLOR_PERSON }}>{summary.total_people_seen ?? 'n/a'}</dd>
                <dt>Unique Carts</dt><dd style={{ fontWeight: 600, color: COLOR_CART }}>{summary.total_carts_seen ?? 'n/a'}</dd>
                <dt>Person-Cart Links</dt><dd style={{ fontWeight: 600, color: COLOR_LINK }}>{summary.total_links_established ?? 'n/a'}</dd>
              </dl>
            ) : (
              <div className={s.empty}><span>Run analysis to see stats</span></div>
            )}
          </div>

          <div className={s.surface}>
            <h3 className={s.h3}>Model Configuration</h3>
            {procInfo ? (
              <dl className={s.dl}>
                <dt>Detection Model</dt><dd>{procInfo.model ?? 'YOLOv26m (custom trained)'}</dd>
                <dt>Tracker</dt><dd>{procInfo.tracker ?? 'BoTSORT (retail tuned)'}</dd>
                <dt>Frames Processed</dt><dd>{(procInfo.total_frames_processed ?? 0).toLocaleString()}</dd>
                <dt>JSON Sample Rate</dt><dd>every {procInfo.json_every_n ?? '?'} frames</dd>
                <dt>Quality Threshold</dt><dd>{procInfo.quality_threshold ?? '0.55'}</dd>
                <dt>Device</dt><dd>{procInfo.device ?? 'n/a'}</dd>
              </dl>
            ) : (
              <dl className={s.dl}>
                <dt>Detection Model</dt><dd>YOLOv26m (custom trained)</dd>
                <dt>Tracker</dt><dd>BoTSORT (retail tuned)</dd>
                <dt>Link Confirmation</dt><dd>15 frames co-movement + overlap</dd>
                <dt>Link Grace Period</dt><dd>30 frames before re-link</dd>
                <dt>Quality Threshold</dt><dd>0.55</dd>
              </dl>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
