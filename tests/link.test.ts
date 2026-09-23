import { describe, expect, it } from 'vitest';
import { DEFAULT_PARAMS, PARAM_LIMITS } from '../src/config/mill';
import {
  EXPERIMENTAL_SCALE, LINK_PARAM_KEYS, clampBatch, clampParam, formatMillQuery, isDefaultParam, millLink, paramMax,
  parseExperimental, parseMillParams, parseMillStart
} from '../src/config/link';

describe('parseMillParams', () => {
  it('reads only the linkable keys, snapped and clamped, and drops defaults and junk', () => {
    const q = new URLSearchParams('gap=0.06&omega=99&gravity=2.0&dispersion=abc&backFriction=-1&tackCells=3&logFeed=0.3');
    expect(parseMillParams(q)).toEqual({ gap: 0.06, omega: PARAM_LIMITS.omega.max, backFriction: 0, logFeed: 0.3 });
    expect(parseMillParams(new URLSearchParams(''))).toEqual({});
  });

  it('treats a value within half a step of the default as the default', () => {
    expect(isDefaultParam('gap', DEFAULT_PARAMS.gap + 0.0004)).toBe(true);
    expect(isDefaultParam('gap', DEFAULT_PARAMS.gap + 0.002)).toBe(false);
    expect(clampParam('gap', 0.0611)).toBeCloseTo(0.062, 9);
    expect(clampBatch(NaN)).toBe(1);
    expect(clampBatch(0.1)).toBe(0.25);
    expect(clampBatch(9)).toBe(4);
  });
});

describe('formatMillQuery / millLink', () => {
  it('writes nothing for an all-default start', () => {
    expect(formatMillQuery({})).toBe('');
    expect(formatMillQuery({ drops: [], batch: 1, params: { ...DEFAULT_PARAMS } })).toBe('');
    expect(millLink({})).toBe('index.html');
  });

  it('writes drops, batch, preset and the changed parameters in a fixed order', () => {
    const q = formatMillQuery({
      drops: [{ pigment: 'naphtholRed', slot: 2, size: 'm' }],
      batch: 1.5,
      preset: 'high',
      params: { ...DEFAULT_PARAMS, gap: 0.06, omega: 2, gravity: DEFAULT_PARAMS.gravity }
    });
    expect(q).toBe('drops=naphtholRed@2.m&batch=1.5&preset=high&omega=2&gap=0.06');
    expect(millLink({ params: { logFeed: 0.15 } }, 'index.html')).toBe('index.html?logFeed=0.15');
  });

  it('round-trips through parseMillStart', () => {
    const start = {
      experimental: false,
      drops: [{ pigment: 'hansaYellow', slot: 1, size: 'l' as const }, { pigment: '#ff8a00', slot: 4, size: 's' as const }],
      batch: 2,
      preset: 'low' as const,
      params: { frictionRatio: 1.4, backFriction: 0.7 }
    };
    const back = parseMillStart(`?${formatMillQuery(start)}`);
    expect(back).toEqual(start);
    expect(parseMillStart('')).toEqual({ experimental: false, drops: [], batch: 1, preset: undefined, params: {} });
    expect(parseMillStart('?preset=insane&batch=x').preset).toBeUndefined();
  });

  it('keeps the linkable keys within the parameter table', () => {
    for (const k of LINK_PARAM_KEYS) expect(PARAM_LIMITS[k]).toBeDefined();
    expect(LINK_PARAM_KEYS).not.toContain('tackCells');
  });
});

describe('experimental mode', () => {
  it('lifts the upper caps in the URL but keeps the floors and the step', () => {
    const q = new URLSearchParams('experimental=1&omega=50&gap=0.4&frictionRatio=0.2&dispersion=1.2345');
    expect(parseExperimental(q)).toBe(true);
    expect(parseMillParams(q)).toEqual({ omega: 50, gap: 0.4, frictionRatio: PARAM_LIMITS.frictionRatio.min, dispersion: 1.235 });
    expect(parseExperimental(new URLSearchParams('x=1'))).toBe(true);
    expect(parseExperimental(new URLSearchParams('experimental=0'))).toBe(false);
    // without the flag the same values are capped as before
    expect(parseMillParams(new URLSearchParams('omega=50&gap=0.4'))).toEqual({ omega: PARAM_LIMITS.omega.max, gap: PARAM_LIMITS.gap.max });
    expect(clampParam('omega', 50, true)).toBe(50);
    expect(clampParam('omega', -3, true)).toBe(0);
  });

  it('stretches the sliders and writes the flag first in a link', () => {
    expect(paramMax('omega')).toBe(PARAM_LIMITS.omega.max);
    expect(paramMax('omega', true)).toBeCloseTo(PARAM_LIMITS.omega.max * EXPERIMENTAL_SCALE, 9);
    const start = parseMillStart('?experimental=1&omega=50&preset=low');
    expect(start.experimental).toBe(true);
    expect(formatMillQuery(start)).toBe('experimental=1&preset=low&omega=50');
    expect(parseMillStart(`?${formatMillQuery(start)}`)).toEqual(start);
    // a non-experimental start caps what it writes, so the link never carries a value its page would refuse
    expect(formatMillQuery({ params: { omega: 50 } })).toBe(`omega=${PARAM_LIMITS.omega.max}`);
  });
});
