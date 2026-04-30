import { create } from 'zustand';
import type { RunResult, RecomputeResult, ZoneOut, Tweaks } from './types';
import { TWEAK_DEFAULTS } from './types';

// ── App state ──────────────────────────────────────────────────────────────
interface AppState {
  // Video + upload
  videoId: string | null;
  videoFile: File | null;
  frameUrl: string | null;
  frameSize: { w: number; h: number };

  // Zone editor
  zones: ZoneOut[];
  currentPoly: [number, number][]; // source-pixel coords in progress

  // Camera + model config
  cameraPlacement: string;
  vlmBackend: string;
  vlmApiKey: string;

  // Analysis results
  isRunning: boolean;
  runResult: RunResult | null;

  // UI
  activeTab: string;
  dockOpen: boolean;
  workspaceMode: 'split' | 'focus';

  // Actions
  setVideoId: (id: string | null) => void;
  setVideoFile: (f: File | null) => void;
  setFrameUrl: (url: string | null, w: number, h: number) => void;
  setZones: (zones: ZoneOut[]) => void;
  addZone: (z: ZoneOut) => void;
  removeZone: (zone_id: string) => void;
  setCurrentPoly: (pts: [number, number][]) => void;
  setCameraPlacement: (v: string) => void;
  setVlmBackend: (v: string) => void;
  setVlmApiKey: (v: string) => void;
  setIsRunning: (v: boolean) => void;
  setRunResult: (r: RunResult | null) => void;
  applyRecompute: (r: RecomputeResult) => void;
  setActiveTab: (tab: string) => void;
  setDockOpen: (open: boolean) => void;
  setWorkspaceMode: (m: 'split' | 'focus') => void;
}

// Tabs that auto-enter focus mode (content-heavy)
const FOCUS_TABS = new Set(['anal', 'rep', '3d']);

export const useAppStore = create<AppState>((set) => ({
  videoId: null,
  videoFile: null,
  frameUrl: null,
  frameSize: { w: 1920, h: 1080 },
  zones: [],
  currentPoly: [],
  cameraPlacement: 'Outside (facing entrance)',
  vlmBackend: 'None (skip)',
  vlmApiKey: '',
  isRunning: false,
  runResult: null,
  activeTab: 'zone',
  dockOpen: true,
  workspaceMode: 'split',

  setVideoId: (id) => set({ videoId: id }),
  setVideoFile: (f) => set({ videoFile: f }),
  setFrameUrl: (url, w, h) => set({ frameUrl: url, frameSize: { w, h } }),
  setZones: (zones) => set({ zones }),
  addZone: (z) => set((s) => ({ zones: [...s.zones, z] })),
  removeZone: (zone_id) => set((s) => ({ zones: s.zones.filter((z) => z.zone_id !== zone_id) })),
  setCurrentPoly: (pts) => set({ currentPoly: pts }),
  setCameraPlacement: (v) => set({ cameraPlacement: v }),
  setVlmBackend: (v) => set({ vlmBackend: v }),
  setVlmApiKey: (v) => set({ vlmApiKey: v }),
  setIsRunning: (v) => set({ isRunning: v }),
  setRunResult: (r) => set({ runResult: r }),
  applyRecompute: (r) =>
    set((s) => ({
      runResult: s.runResult
        ? {
            ...s.runResult,
            analytics: r.analytics,
            heatmap_url: r.heatmap_url ?? s.runResult.heatmap_url,
          }
        : null,
    })),
  setActiveTab: (tab) => set((s) => ({
    activeTab: tab,
    workspaceMode: FOCUS_TABS.has(tab) ? 'focus' : (FOCUS_TABS.has(s.activeTab) ? 'split' : s.workspaceMode),
  })),
  setDockOpen: (open) => set({ dockOpen: open }),
  setWorkspaceMode: (m) => set({ workspaceMode: m }),
}));

// ── Tweaks state (persisted to localStorage) ───────────────────────────────
interface TweaksState {
  tweaks: Tweaks;
  setTweak: <K extends keyof Tweaks>(key: K, value: Tweaks[K]) => void;
}

function loadTweaks(): Tweaks {
  try {
    const raw = localStorage.getItem('pops-tweaks');
    if (raw) return { ...TWEAK_DEFAULTS, ...JSON.parse(raw) };
  } catch {}
  return { ...TWEAK_DEFAULTS };
}

export const useTweaksStore = create<TweaksState>((set) => ({
  tweaks: loadTweaks(),
  setTweak: (key, value) =>
    set((s) => {
      const next = { ...s.tweaks, [key]: value };
      try { localStorage.setItem('pops-tweaks', JSON.stringify(next)); } catch {}
      return { tweaks: next };
    }),
}));
