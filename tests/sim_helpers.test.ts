import { describe, expect, it } from 'vitest';
import { FOLD_DURATION, WHITE_LATENT, dispatchSize, foldProfile, halfToFloat } from '../src/sim/mpm';

describe('solver helpers', () => {
  it('splits large particle dispatches into 2D so no dimension exceeds the limit', () => {
    expect(dispatchSize(0, 65535)).toEqual([1, 1]);
    expect(dispatchSize(473, 65535)).toEqual([473, 1]);
    const [x, y] = dispatchSize(200000, 65535);
    expect(x).toBeLessThanOrEqual(65535);
    expect(x * y).toBeGreaterThanOrEqual(200000);
    expect(x * y - 200000).toBeLessThan(x);
  });

  it('fold profile is a smoothstep with zero end velocities', () => {
    expect(foldProfile(0, FOLD_DURATION)).toEqual({ s: 0, dsdt: 0 });
    expect(foldProfile(FOLD_DURATION, FOLD_DURATION).s).toBeCloseTo(1, 9);
    expect(foldProfile(2 * FOLD_DURATION, FOLD_DURATION)).toEqual({ s: 1, dsdt: 0 });
    const mid = foldProfile(FOLD_DURATION / 2, FOLD_DURATION);
    expect(mid.s).toBeCloseTo(0.5, 9);
    expect(mid.dsdt).toBeCloseTo(1.5 / FOLD_DURATION, 9);
    // derivative check by finite difference
    const t = 0.3;
    const eps = 1e-5;
    const fd = (foldProfile(t + eps, FOLD_DURATION).s - foldProfile(t - eps, FOLD_DURATION).s) / (2 * eps);
    expect(foldProfile(t, FOLD_DURATION).dsdt).toBeCloseTo(fd, 5);
  });

  it('decodes IEEE half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x0000)).toBe(0);
    expect(halfToFloat(0x3555)).toBeCloseTo(1 / 3, 3);
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
  });

  it('white base latent has unit pigment weight on the white channel', () => {
    expect(WHITE_LATENT[0] + WHITE_LATENT[1] + WHITE_LATENT[2] + WHITE_LATENT[3]).toBeCloseTo(1, 9);
  });
});
