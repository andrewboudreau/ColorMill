/**
 * ColorMill v2 — deep links (pure TS, tested).
 *
 * A mill start is everything needed to restart a simulation from a URL:
 * the pigment chunks to set down (`?drops=`, src/drops.ts), the batch size,
 * the quality preset and any mill parameter that differs from its default.
 * The same query keys work on the simulator (`index.html?…`) and on the
 * colour mixer (`mixer.html?mix=…&…`, which forwards them to the mill), and
 * only values that differ from the defaults are written, so most links stay
 * short: `index.html?drops=naphtholRed@2.m&gap=0.06&preset=high`.
 */
import { formatDrops, parseDrops, type Drop } from '../drops';
import {
  BATCH_LIMITS, DEFAULT_PARAMS, PARAM_LIMITS, QUALITY_PRESETS, omegaToRpm, type MillParams, type QualityPreset
} from './mill';

/** Mill parameters that a link may carry, in the order they are written (and shown in the drawer). */
export const LINK_PARAM_KEYS: readonly (keyof MillParams)[] = [
  'omega', 'frictionRatio', 'gap', 'dispersion', 'gravity', 'backFriction', 'logFeed'
];

/** Label and readout for each linkable parameter (the drawer and the mixer's mill settings share these). */
export const PARAM_LABELS: Readonly<Record<keyof MillParams, { label: string; format: (v: number) => string }>> = {
  omega: { label: 'Roller speed', format: (v) => `${omegaToRpm(v).toFixed(1)} rpm` },
  frictionRatio: { label: 'Friction ratio', format: (v) => `${v.toFixed(2)}×` },
  gap: { label: 'Nip gap', format: (v) => v.toFixed(3) },
  dispersion: { label: 'Dispersion', format: (v) => v.toFixed(2) },
  gravity: { label: 'Gravity', format: (v) => v.toFixed(1) },
  backFriction: { label: 'Back-roll friction', format: (v) => v.toFixed(2) },
  logFeed: { label: 'Log feed', format: (v) => (v > 0 ? `lower in ${v.toFixed(2)}/s` : 'drop') },
  tackCells: { label: 'Tack band', format: (v) => (v > 0 ? `${v.toFixed(1)} cells` : 'auto') }
};

export interface MillStart {
  drops: Drop[];
  /** bank volume multiplier; 1 is the default */
  batch: number;
  /** only when pinned on purpose (auto-select otherwise) */
  preset?: QualityPreset;
  /** only the parameters that differ from DEFAULT_PARAMS */
  params: Partial<MillParams>;
}

export function isPreset(v: unknown): v is QualityPreset {
  return typeof v === 'string' && v in QUALITY_PRESETS;
}

/** Snap to the parameter's step and clamp to its range. */
export function clampParam(key: keyof MillParams, v: number): number {
  const l = PARAM_LIMITS[key];
  const snapped = Math.round(v / l.step) * l.step;
  return Math.min(l.max, Math.max(l.min, parseFloat(snapped.toFixed(6))));
}

/** True when the value is the default to within half a step. */
export function isDefaultParam(key: keyof MillParams, v: number): boolean {
  return Math.abs(v - DEFAULT_PARAMS[key]) < 0.5 * PARAM_LIMITS[key].step;
}

/** Clamp a batch multiplier to the accepted range; anything unusable is the default 1. */
export function clampBatch(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  return Math.min(Math.max(v, BATCH_LIMITS.min), BATCH_LIMITS.max);
}

/** The mill parameters a query carries that differ from the defaults (finite, snapped, clamped). */
export function parseMillParams(query: URLSearchParams): Partial<MillParams> {
  const out: Partial<MillParams> = {};
  for (const key of LINK_PARAM_KEYS) {
    const raw = query.get(key);
    if (raw === null) continue;
    const v = parseFloat(raw);
    if (!Number.isFinite(v)) continue;
    const c = clampParam(key, v);
    if (!isDefaultParam(key, c)) out[key] = c;
  }
  return out;
}

/** Everything a URL says about how to start the mill. */
export function parseMillStart(search: string | URLSearchParams): MillStart {
  const q = typeof search === 'string' ? new URLSearchParams(search) : search;
  const preset = q.get('preset');
  return {
    drops: parseDrops(q.get('drops')),
    batch: clampBatch(parseFloat(q.get('batch') ?? '')),
    preset: isPreset(preset) ? preset : undefined,
    params: parseMillParams(q)
  };
}

function fmtNum(v: number): string {
  return parseFloat(v.toFixed(4)).toString();
}

/**
 * The query string (no '?') for a start: drops, then batch, preset and the
 * non-default parameters, each only when it says something. Built by hand:
 * ',', '@', ':' and '.' are legal in a query and read better than escapes.
 */
export function formatMillQuery(start: Partial<MillStart>, extra: readonly (readonly [string, string])[] = []): string {
  const parts: string[] = [];
  for (const [k, v] of extra) if (v) parts.push(`${k}=${v}`);
  const drops = start.drops && start.drops.length ? formatDrops(start.drops) : '';
  if (drops) parts.push(`drops=${drops}`);
  if (start.batch !== undefined && Math.abs(start.batch - 1) > 1e-9) parts.push(`batch=${fmtNum(clampBatch(start.batch))}`);
  if (start.preset) parts.push(`preset=${start.preset}`);
  const p = start.params ?? {};
  for (const key of LINK_PARAM_KEYS) {
    const v = p[key];
    if (v === undefined || !Number.isFinite(v)) continue;
    const c = clampParam(key, v);
    if (!isDefaultParam(key, c)) parts.push(`${key}=${fmtNum(c)}`);
  }
  return parts.join('&');
}

/** `base?query`, or just `base` when the start is all defaults. */
export function millLink(start: Partial<MillStart>, base = 'index.html'): string {
  const q = formatMillQuery(start);
  return q ? `${base}?${q}` : base;
}
