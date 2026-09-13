import { describe, expect, it } from 'vitest';
import {
  DEFAULT_PARAMS, GEOMETRY, QUALITY_PRESETS, gridDims, lameParameters, millTopSurfaceY,
  rollerAxisDistance, rollerPoses, rollerSurfaceVelocity, seedBankPositions
} from '../src/config/mill';

describe('mill geometry', () => {
  it('places the rollers symmetrically around the nip with the gap between surfaces', () => {
    const { back, front } = rollerPoses(DEFAULT_PARAMS);
    expect(front.axisZ - back.axisZ).toBeCloseTo(2 * GEOMETRY.radius + DEFAULT_PARAMS.gap, 6);
    expect((front.axisZ + back.axisZ) / 2).toBeCloseTo(GEOMETRY.nipZ, 6);
  });

  it('moves both roller surfaces downward through the nip', () => {
    const { back, front } = rollerPoses(DEFAULT_PARAMS);
    // point on the back roller surface facing the nip (toward +z)
    const vb = rollerSurfaceVelocity(back, back.axisY, back.axisZ + back.radius);
    const vf = rollerSurfaceVelocity(front, front.axisY, front.axisZ - front.radius);
    expect(vb[1]).toBeLessThan(0);
    expect(vf[1]).toBeLessThan(0);
    // tops move toward the nip
    expect(rollerSurfaceVelocity(back, back.axisY + back.radius, back.axisZ)[2]).toBeGreaterThan(0);
    expect(rollerSurfaceVelocity(front, front.axisY + front.radius, front.axisZ)[2]).toBeLessThan(0);
    // friction ratio: back is faster
    expect(Math.abs(back.omegaX)).toBeCloseTo(Math.abs(front.omegaX) * DEFAULT_PARAMS.frictionRatio, 6);
  });

  it('computes the grid dims of the presets', () => {
    const d = gridDims(QUALITY_PRESETS.high);
    expect([d.nx, d.ny, d.nz]).toEqual([97, 81, 97]);
    expect(d.h).toBeCloseTo(1 / 64, 9);
    const l = gridDims(QUALITY_PRESETS.low);
    expect([l.nx, l.ny, l.nz]).toEqual([49, 41, 49]);
  });

  it('seeds the bank above the rollers, outside them, at roughly 8 particles per cell', () => {
    const q = QUALITY_PRESETS.low;
    const pos = seedBankPositions(q, DEFAULT_PARAMS);
    const n = pos.length / 3;
    expect(n).toBeGreaterThan(20000);
    expect(n).toBeLessThan(70000);
    const { back, front } = rollerPoses(DEFAULT_PARAMS);
    for (let i = 0; i < n; i += 97) {
      const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
      expect(x).toBeGreaterThan(0);
      expect(x).toBeLessThan(GEOMETRY.length);
      expect(rollerAxisDistance(back, y, z)).toBeGreaterThanOrEqual(GEOMETRY.radius);
      expect(rollerAxisDistance(front, y, z)).toBeGreaterThanOrEqual(GEOMETRY.radius);
      expect(y).toBeGreaterThanOrEqual(millTopSurfaceY(z, DEFAULT_PARAMS));
    }
  });

  it('is deterministic for a given seed', () => {
    const a = seedBankPositions(QUALITY_PRESETS.low, DEFAULT_PARAMS, 7);
    const b = seedBankPositions(QUALITY_PRESETS.low, DEFAULT_PARAMS, 7);
    expect(a).toEqual(b);
  });

  it('derives Lame parameters', () => {
    const { mu, lambda } = lameParameters({ E: 60, nu: 0.35, thetaC: 0, thetaS: 0 });
    expect(mu).toBeCloseTo(22.22, 1);
    expect(lambda).toBeCloseTo(51.85, 1);
  });
});
