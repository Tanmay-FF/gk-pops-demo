import type { RunResult, RecomputeResult, UploadResult, ZoneOut } from './types';

const BASE = '/api';

export async function uploadVideo(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface RunParams {
  video_id: string;
  camera_placement: string;
  vlm_backend: string;
  vlm_api_key: string;
  zones: ZoneOut[];
}

export async function runAnalysis(params: RunParams): Promise<RunResult> {
  const res = await fetch(`${BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface RecomputeParams {
  video_id: string;
  zones: ZoneOut[];
}

export async function recomputeAnalytics(params: RecomputeParams): Promise<RecomputeResult> {
  const res = await fetch(`${BASE}/recompute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function invalidateCache(video_id?: string): Promise<void> {
  await fetch(`${BASE}/invalidate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: video_id ?? null }),
  });
}

export interface MakeZoneParams {
  name: string;
  points: [number, number][];
  applies_to: string;
  zone_index: number;
}

export async function makeZone(params: MakeZoneParams): Promise<ZoneOut> {
  const res = await fetch(`${BASE}/zones/make`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.zone;
}
