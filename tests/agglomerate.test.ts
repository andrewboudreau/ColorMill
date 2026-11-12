import { describe, expect, it } from 'vitest';
import { agglomerateLossRate, agglomerateStep, dispersionStats, effectiveLoad, flowStress, shearRate } from '../src/sim/agglomerate';
import { DEFAULT_AGGLOMERATE, DEFAULT_MATERIAL, DEFAULT_PARAMS, QUALITY_PRESETS, agglomerateRateScale, type AgglomerateConstants } from '../src/config/mill';

const M = DEFAULT_MATERIAL;
const A: AgglomerateConstants = { cohesiveStress: 1.5, erosionRate: 0.4, ruptureStress: 0, ruptureRate: 0 };
/** the shear rate the flow curve reaches the stress `tau` at */
const rateAt = (tau: number): number => Math.pow((tau - M.yieldStress) / M.consistency, 1 / M.flowIndex);
/** simple shear v = (rate y, 0, 0), row-major */
const shear = (rate: number): number[] => [0, rate, 0, 0, 0, 0, 0, 0, 0];

describe('agglomerate kinetics', () => {
  it('the shear rate of simple shear is its rate, and volume change and rotation do not count', () => {
    expect(shearRate(shear(7))).toBeCloseTo(7, 12);
    // a rigid rotation and a uniform dilation on top change nothing
    const L = shear(7);
    L[1] += 3; L[3] -= 3;
    L[0] += 2; L[4] += 2; L[8] += 2;
    expect(shearRate(L)).toBeCloseTo(7, 12);
    // planar extension at rate e: sqrt(2 (e^2 + e^2)) = 2 e
    expect(shearRate([4, 0, 0, 0, -4, 0, 0, 0, 0])).toBeCloseTo(8, 12);
    // read at an offset, as from a snapshot
    expect(shearRate([...shear(1), ...shear(5)], 9)).toBeCloseTo(5, 12);
  });

  it('the stress is the flow curve', () => {
    expect(flowStress(0, M)).toBeCloseTo(M.yieldStress, 12);
    expect(flowStress(rateAt(1.5), M)).toBeCloseTo(1.5, 9);
  });

  it('nothing disperses below the cohesive stress, however long', () => {
    for (const tau of [0.6, 1.0, 1.4, 1.4999]) {
      const rate = rateAt(tau);
      expect(agglomerateLossRate(rate, M, A)).toBe(0);
      let u = 12;
      for (let i = 0; i < 1000; i++) u = agglomerateStep(u, rate, 0.0128, M, A);
      expect(u).toBe(12);
    }
    // at rest, and with the kinetics off
    expect(agglomerateStep(12, 0, 1, M, A)).toBe(12);
    expect(agglomerateStep(12, 100, 1, M, { ...A, cohesiveStress: 0 })).toBe(12);
    expect(agglomerateStep(12, 100, 1, M, { ...A, erosionRate: 0 })).toBe(12);
  });

  it('above it the undispersed load only ever decreases, and never below zero', () => {
    const rate = rateAt(1.8);
    let u = 12;
    for (let i = 0; i < 400; i++) {
      const next = agglomerateStep(u, rate, 0.0128, M, A);
      expect(next).toBeLessThan(u);
      expect(next).toBeGreaterThan(0);
      u = next;
    }
    // a huge step does not overshoot
    expect(agglomerateStep(12, 1e4, 10, M, A)).toBeGreaterThanOrEqual(0);
    expect(agglomerateStep(0, rate, 1, M, A)).toBe(0);
  });

  it('the rate is the shear rate times the erosion constant times the excess stress over the cohesive stress', () => {
    for (const tau of [1.6, 1.8, 2.1]) {
      const rate = rateAt(tau);
      expect(agglomerateLossRate(rate, M, A)).toBeCloseTo(A.erosionRate * rate * (tau / A.cohesiveStress - 1), 9);
    }
    // linear in the constant, and more stress disperses faster per unit of strain
    const r = rateAt(1.8);
    expect(agglomerateLossRate(r, M, { ...A, erosionRate: 0.8 })).toBeCloseTo(2 * agglomerateLossRate(r, M, A), 12);
    expect(agglomerateLossRate(rateAt(2.1), M, A) / rateAt(2.1)).toBeGreaterThan(agglomerateLossRate(r, M, A) / r);
  });

  it('first order: the decay over a time does not depend on how it is stepped', () => {
    const rate = rateAt(1.9);
    let fine = 12;
    for (let i = 0; i < 160; i++) fine = agglomerateStep(fine, rate, 0.0128 / 16, M, A);
    let coarse = 12;
    for (let i = 0; i < 10; i++) coarse = agglomerateStep(coarse, rate, 0.0128, M, A);
    expect(fine).toBeCloseTo(coarse, 9);
    expect(coarse).toBeCloseTo(12 * Math.exp(-agglomerateLossRate(rate, M, A) * 0.128), 9);
  });

  it('the shear rate is read through the scale of the preset: the square root of its gap over the reference gap', () => {
    expect(agglomerateRateScale(QUALITY_PRESETS.low)).toBeCloseTo(1, 12);
    expect(agglomerateRateScale(QUALITY_PRESETS.high)).toBeCloseTo(Math.SQRT1_2, 12);
    // both the stress and the strain are read at the scaled rate
    expect(agglomerateStep(12, 30, 0.1, M, A, 0.5)).toBeCloseTo(agglomerateStep(12, 15, 0.1, M, A), 9);
  });

  it('rupture adds a second loss over its own, higher stress only', () => {
    const R: AgglomerateConstants = { ...A, ruptureStress: 2.0, ruptureRate: 3 };
    const below = rateAt(1.9), above = rateAt(2.2);
    expect(agglomerateLossRate(below, M, R)).toBeCloseTo(agglomerateLossRate(below, M, A), 12);
    expect(agglomerateLossRate(above, M, R)).toBeCloseTo(agglomerateLossRate(above, M, A) + 3 * above * (2.2 / 2.0 - 1), 9);
  });

  it('the default threshold sits over what the bank reaches and under what the wedge over a fast nip does (design §5)', () => {
    // the bank's median shear rate is about 3 /s and its 99th percentile at equal roll speeds 16 /s (1.76); the
    // wedge over the nip at a friction ratio of 1.6 runs at 14 to 27 /s
    // at the default putty stiffness, which multiplies the flow curve (and the threshold was set with)
    const S = { ...M, yieldStress: M.yieldStress * DEFAULT_PARAMS.stiffness, consistency: M.consistency * DEFAULT_PARAMS.stiffness };
    expect(agglomerateLossRate(3.5, S, DEFAULT_AGGLOMERATE)).toBe(0);
    expect(agglomerateLossRate(12, S, DEFAULT_AGGLOMERATE)).toBe(0);
    expect(agglomerateLossRate(20, S, DEFAULT_AGGLOMERATE)).toBeGreaterThan(0);
    expect(agglomerateLossRate(27, S, DEFAULT_AGGLOMERATE)).toBeGreaterThan(agglomerateLossRate(20, S, DEFAULT_AGGLOMERATE));
  });
});

describe('agglomerate appearance and measures', () => {
  it('undispersed pigment tints at a fraction of its strength', () => {
    expect(effectiveLoad(12, 0, 0.1)).toBeCloseTo(12, 12);
    expect(effectiveLoad(12, 12, 0.1)).toBeCloseTo(1.2, 12);
    expect(effectiveLoad(4, 2, 0.1)).toBeCloseTo(2 + 0.2, 12);
    // more than the load is the load
    expect(effectiveLoad(4, 9, 0.1)).toBeCloseTo(0.4, 12);
  });

  it('the dispersed fraction and its distribution are load-weighted over the pigment', () => {
    const snap = {
      count: 5,
      loads: new Float32Array([12, 12, 4, 0, 4]),
      undispersed: new Float32Array([12, 0, 2, 0, 1])
    };
    const s = dispersionStats(snap, 4);
    expect(s.totalLoad).toBeCloseTo(32, 6);
    expect(s.undispersedLoad).toBeCloseTo(15, 6);
    expect(s.dispersed).toBeCloseTo(1 - 15 / 32, 6);
    // by dispersed fraction: 0 (12), 0.5 (4), 0.75 (4), 1 (12)
    expect(s.histogram.map((v) => +v.toFixed(4))).toEqual([0.375, 0, 0.125, 0.5]);
    expect(s.raw).toBeCloseTo(12 / 32, 6);
    expect(s.developed).toBeCloseTo(12 / 32, 6);
    expect(s.p10).toBe(0);
    expect(s.median).toBeCloseTo(0.5, 6);
    expect(s.p90).toBe(1);
    // no pigment: nothing left to disperse; no state: all of it dispersed
    expect(dispersionStats({ count: 2, loads: new Float32Array(2), undispersed: new Float32Array(2) }).dispersed).toBe(1);
    expect(dispersionStats({ count: 1, loads: new Float32Array([3]) }).dispersed).toBe(1);
  });
});
