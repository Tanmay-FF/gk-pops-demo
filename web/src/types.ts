// TypeScript interfaces matching api/models.py + engine JSON schema

export interface ZoneOut {
  zone_id: string;
  name: string;
  polygon: [number, number][];
  applies_to: string;
  color: [number, number, number]; // BGR
}

export interface VideoInfo {
  video_name: string;
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  processing_timestamp: string;
}

export interface EventLog {
  frame: number;
  timestamp: number;
  cart_id: string;
  event: string;
  pops_score: number;
  fill: string;
  bag: string;
  direction: string;
  linked: boolean;
  speed_status: string;
  abandoned: boolean;
}

export interface TrackingJSON {
  video_info: VideoInfo;
  frames: unknown[];
  events: EventLog[];
  cart_classifications: Record<string, unknown>;
  pops_summary: Record<string, { max_score: number; peak_event: string }>;
  summary: {
    total_people_seen: number;
    total_carts_seen: number;
    total_links_established: number;
    total_events: number;
    high_priority: number;
    medium_priority: number;
  };
  processing_info: {
    total_frames_processed: number;
    json_sampled_frames: number;
    json_every_n: number;
    device: string;
    model: string;
    tracker: string;
    quality_model: string;
    fill_model: string;
    quality_threshold: number;
  };
  _html?: {
    video_info: string;
    detection: string;
    config: string;
    legend: string;
  };
}

export interface POPSCart {
  id: string;
  max_pops: number;
  peak_event: string;
  quality: string;
  fill: string;
  bag: string;
  direction: string;
  speed_status: string;
  linked: boolean;
}

export interface AnalyticsHTML {
  summary_html: string;
  spikes_html: string;
  dwell_html: string;
  journey_html: string;
}

export interface RunResult {
  session_id: string;
  video_url: string;
  heatmap_url: string | null;
  tracking_json: TrackingJSON;
  pops_carts: POPSCart[];
  case_report_html: string;
  bev3d_html: string;
  bev2d_html: string;
  analytics: AnalyticsHTML;
}

export interface RecomputeResult {
  analytics: AnalyticsHTML;
  heatmap_url: string | null;
}

export interface UploadResult {
  video_id: string;
  frame_url: string;
  width: number;
  height: number;
}

export interface Tweaks {
  accentHue: number;
  density: 'compact' | 'cozy' | 'comfortable';
  showHud: boolean;
  showZones: boolean;
  showPip: boolean;
  animated: boolean;
  peopleCount: number;
  railSide: 'left' | 'right';
  showDock: boolean;
}

export const TWEAK_DEFAULTS: Tweaks = {
  accentHue: 254,
  density: 'cozy',
  showHud: true,
  showZones: true,
  showPip: true,
  animated: true,
  peopleCount: 5,
  railSide: 'left',
  showDock: true,
};

// POPS color constants (from engine/scoring.py)
export const COLOR_PERSON     = '#00e676';
export const COLOR_CART       = '#ffa500';
export const COLOR_LINK       = '#ff32ff';
export const COLOR_PUSHOUT    = '#ff0000';
export const COLOR_SUSPICIOUS = '#ff8c00';
export const COLOR_CLEAR      = '#00c853';

export const ZONE_PALETTE = ['#0098ff', '#ffc800', '#9370db', '#50c850', '#dc3cc8'];

export const CAMERA_OPTIONS = [
  'Outside (facing entrance)',
  'Inside (facing exit)',
  'Inside (exit on right)',
  'Inside (exit on left)',
  'Inside (exit on both sides)',
];

export const VLM_OPTIONS = [
  'None (skip)',
  'Claude (API)',
  'Moondream2 (local)',
  'Qwen2-VL-2B (local)',
  'InternVL2-2B (local)',
];

export function popsColor(score: number): string {
  return score >= 71 ? COLOR_PUSHOUT : score >= 31 ? COLOR_SUSPICIOUS : COLOR_CLEAR;
}

export function popsLabel(score: number): string {
  return score >= 71 ? 'High' : score >= 31 ? 'Medium' : 'Low';
}

export function fmtTimestamp(ts: number): string {
  const m = Math.floor(ts / 60);
  const s = Math.floor(ts % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function bgrToCss(color: [number, number, number]): string {
  const [b, g, r] = color;
  return `rgb(${r},${g},${b})`;
}
