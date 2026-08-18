import React, { useRef, useEffect, useState } from 'react';
import { Icons } from '../icons';
import { useAppStore } from '../store';
import { COLOR_PERSON, COLOR_CART } from '../types';
import type { Tweaks } from '../types';
import s from './TrackedOutput.module.css';

interface Props { tweaks: Tweaks; }

export const TrackedOutput: React.FC<Props> = ({ tweaks }) => {
  const { runResult } = useAppStore();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState('00:00');
  const [duration, setDuration] = useState('00:00');

  const videoUrl = runResult?.video_url;
  const tracking = runResult?.tracking_json;
  const summary  = tracking?.summary;
  const vi       = tracking?.video_info;

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => {
      setProgress(v.duration ? v.currentTime / v.duration : 0);
      setCurrentTime(fmt(v.currentTime));
    };
    const onMeta = () => setDuration(fmt(v.duration || 0));
    const onPlay  = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    v.addEventListener('timeupdate', onTime);
    v.addEventListener('loadedmetadata', onMeta);
    v.addEventListener('play', onPlay);
    v.addEventListener('pause', onPause);
    return () => { v.removeEventListener('timeupdate', onTime); v.removeEventListener('loadedmetadata', onMeta); v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause); };
  }, [videoUrl]);

  function fmt(sec: number) {
    const m = Math.floor(sec / 60), ss = Math.floor(sec % 60);
    return `${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  }

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    playing ? v.pause() : v.play();
  }

  function seek(e: React.MouseEvent<HTMLDivElement>) {
    const v = videoRef.current;
    if (!v) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    v.currentTime = ratio * v.duration;
  }

  const fps  = vi?.fps?.toFixed(1) ?? 'n/a';
  const res  = vi ? `${vi.width}×${vi.height}` : 'n/a';
  const totalTracks = (summary?.total_people_seen ?? 0) + (summary?.total_carts_seen ?? 0);

  return (
    <div className={s.tracked}>
      <div className={s.paneHeader}>
        <div className={s.paneTitle}>
          <Icons.cube size={13} />
          Tracked Output
          <span className={s.sub}>· Camera + Bird's-Eye View</span>
        </div>
        {vi && <span className={s.paneTag}>{res} · {fps}fps</span>}
        <div className={s.paneSpaccer} />
        {runResult && <span className={s.chipAccent}><span className={s.liveDot} />Live</span>}
        <button className={s.iconBtn} title="Fullscreen" onClick={() => videoRef.current?.requestFullscreen?.()}>
          <Icons.expand size={13} />
        </button>
      </div>

      <div className={s.trackedBody}>
        {videoUrl ? (
          // Real tracked video from engine
          <div className={s.videoWrap}>
            <video ref={videoRef} src={videoUrl} className={s.video} loop playsInline />

            {tweaks.showHud && (
              <>
                <div className={`${s.hud} ${s.hudTl}`}>
                  <div className={s.hudRow}>
                    <div className={s.hudCard}><span className={s.hudLbl}>FPS</span><span className={s.hudVal}>{fps}</span></div>
                    <div className={s.hudCard}><span className={s.hudLbl}>Tracks</span><span className={s.hudVal}>{totalTracks}</span></div>
                    <div className={s.hudCard}><span className={s.hudLbl}>Persons</span><span className={s.hudVal} style={{ color: COLOR_PERSON }}>{summary?.total_people_seen ?? 0}</span></div>
                    <div className={s.hudCard}><span className={s.hudLbl}>Carts</span><span className={s.hudVal} style={{ color: COLOR_CART }}>{summary?.total_carts_seen ?? 0}</span></div>
                  </div>
                </div>
                <div className={`${s.hud} ${s.hudTr}`}>
                  <div className={s.hudRec}>
                    <span className={s.recDot} />
                    <span className={s.recText}>REC · {currentTime}</span>
                  </div>
                </div>
              </>
            )}

            {/* Timeline scrubber */}
            {tweaks.showHud && (
              <div className={s.timeline}>
                <button className={s.play} onClick={togglePlay}>
                  {playing ? <Icons.pause size={9} /> : <Icons.play size={9} />}
                </button>
                <span className={s.time}>{currentTime}</span>
                <div className={s.bar} onClick={seek}>
                  <div className={s.barFill} style={{ width: `${progress * 100}%` }} />
                </div>
                <span className={s.time} style={{ color: 'var(--text-3)' }}>{duration}</span>
              </div>
            )}
          </div>
        ) : (
          // Empty state — BEV floor schematic placeholder
          <div className={s.emptyCanvas}>
            <div className={s.floorGrid} />
            <div className={s.emptyMsg}>
              <Icons.cube size={32} style={{ color: 'var(--text-3)', marginBottom: 12 }} />
              <div style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>No video loaded</div>
              <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 4 }}>Upload a video from the left rail and click Run Analysis</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
