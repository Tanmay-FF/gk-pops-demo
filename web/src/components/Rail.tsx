import React, { useRef } from 'react';
import { Icons } from '../icons';
import { useAppStore } from '../store';
import { uploadVideo, invalidateCache, recomputeAnalytics } from '../api';
import { CAMERA_OPTIONS, VLM_OPTIONS, bgrToCss } from '../types';
import s from './Rail.module.css';

export const Rail: React.FC = () => {
  const { videoId, videoFile, zones, cameraPlacement, vlmBackend, vlmApiKey,
          isRunning, runResult,
          setVideoId, setVideoFile, setFrameUrl, setZones,
          setCameraPlacement, setVlmBackend, setVlmApiKey,
          applyRecompute, setIsRunning,
          removeZone } = useAppStore();
  const fileInput = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setVideoFile(file);
    try {
      const res = await uploadVideo(file);
      setVideoId(res.video_id);
      setFrameUrl(res.frame_url, res.width, res.height);
      setZones([]); // reset zones on new video
    } catch (e) {
      console.error('[POPS] upload failed:', e);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  async function handleRecompute() {
    if (!videoId || isRunning) return;
    setIsRunning(true);
    try {
      const r = await recomputeAnalytics({ video_id: videoId, zones });
      const { applyRecompute: apply } = useAppStore.getState();
      apply(r);
    } catch (e) { console.error(e); }
    finally { setIsRunning(false); }
  }

  return (
    <aside className={s.rail}>
      {/* Step 1: Video */}
      <div className={s.section}>
        <div className={s.head}><span><span className={s.step}>1</span>Video Source</span><Icons.cam size={11} /></div>
        <div className={s.body}>
          <div
            className={s.dropzone}
            onClick={() => fileInput.current?.click()}
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            <Icons.upload size={22} />
            <div className={s.dropT}>{videoFile ? videoFile.name : 'Drop video here'}</div>
            <div className={s.dropS}>or <span className={s.accentLink}>click to upload</span> · MP4, MOV</div>
          </div>
          <input ref={fileInput} type="file" accept="video/*" style={{ display: 'none' }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
          {videoFile && (
            <div className={s.field} style={{ marginTop: 8 }}>
              <label className={s.fieldLabel}>Loaded</label>
              <input className={`${s.input} ${s.mono}`} value={videoFile.name} readOnly />
            </div>
          )}
        </div>
      </div>

      {/* Step 2: Camera */}
      <div className={s.section}>
        <div className={s.head}><span><span className={s.step}>2</span>Camera Placement</span><Icons.pin size={11} /></div>
        <div className={s.body}>
          <div className={s.field}>
            <label className={s.fieldLabel}>Camera angle / orientation</label>
            <select className={s.select} value={cameraPlacement} onChange={(e) => setCameraPlacement(e.target.value)}>
              {CAMERA_OPTIONS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* Step 3: VLM */}
      <div className={s.section}>
        <div className={s.head}><span><span className={s.step}>3</span>VLM Backend</span><Icons.sparkles size={11} /></div>
        <div className={s.body}>
          <div className={s.field}>
            <label className={s.fieldLabel}>Model</label>
            <select className={s.select} value={vlmBackend} onChange={(e) => setVlmBackend(e.target.value)}>
              {VLM_OPTIONS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>
          {vlmBackend !== 'None (skip)' && (
            <div className={s.field}>
              <label className={s.fieldLabel}>API key <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>optional</span></label>
              <input className={`${s.input} ${s.mono}`} type="password" placeholder="Leave blank for OAuth token"
                value={vlmApiKey} onChange={(e) => setVlmApiKey(e.target.value)} />
            </div>
          )}
        </div>
      </div>

      {/* Step 4: Zones */}
      <div className={s.section}>
        <div className={s.head}><span><span className={s.step}>4</span>Zones</span><span className={s.zoneCnt}>{zones.length}</span></div>
        <div className={s.body}>
          {zones.length === 0
            ? <div className={s.emptyZones}>No zones yet - draw in Zone Editor tab</div>
            : (
              <div className={s.zoneList}>
                {zones.map((z) => (
                  <div key={z.zone_id} className={s.zoneItem}>
                    <span className={s.swatch} style={{ background: bgrToCss(z.color) }} />
                    <span className={s.zoneName}>{z.name}</span>
                    <span className={s.zoneMeta}>{z.applies_to}</span>
                    <button className={s.zoneX} onClick={() => removeZone(z.zone_id)} title="Remove zone">
                      <Icons.x size={11} />
                    </button>
                  </div>
                ))}
              </div>
            )
          }
        </div>
      </div>

      <div className={s.divider} />

      {/* Actions */}
      <div className={s.body} style={{ paddingTop: 12 }}>
        <div className={s.btnRow} style={{ gap: 6, marginTop: 6 }}>
          <button className={`${s.btn} ${s.sm}`} onClick={() => videoId && invalidateCache(videoId)}>
            Re-run detection
          </button>
          <button className={`${s.btn} ${s.sm}`} onClick={handleRecompute} disabled={!videoId || isRunning}>
            Recompute analytics
          </button>
        </div>
      </div>

      <div className={s.footer}>
        <span>Built by <a href="#">Tanmay Thaker</a></span>
        <span>Gatekeeper Systems</span>
      </div>
    </aside>
  );
};
