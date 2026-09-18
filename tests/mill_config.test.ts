import { describe, expect, it } from 'vitest';
import {
  DEFAULT_CHUNK_SIZE, DEFAULT_PARAMS, DROP_SLOTS, GEOMETRY, PIGMENT_CHUNK_RADIUS, PIGMENT_CHUNK_SIZES, QUALITY_PRESETS, bankTopY, bankVolumeBelow, dropSlotX, gridDims,
  lameParameters, millTopSurfaceY, rollerAxisDistance, rollerPoses, rollerSurfaceVelocity, seedBankPositions
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
    expect([d.nx, d.ny, d.nz]).toEqual([97, 145, 97]);
    expect(d.h).toBeCloseTo(1 / 64, 9);
    const l = gridDims(QUALITY_PRESETS.low);
    expect([l.nx, l.ny, l.nz]).toEqual([49, 73, 49]);
  });

  it('seeds the bank above the rollers, outside them, at roughly 8 particles per cell', () => {
    const q = QUALITY_PRESETS.low;
    const pos = seedBankPositions(q, DEFAULT_PARAMS);
    const n = pos.length / 3;
    expect(n).toBeGreaterThan(25000);
    expect(n).toBeLessThan(60000);
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

  it('scales the seeded bank volume with the batch size', () => {
    const one = seedBankPositions(QUALITY_PRESETS.low, DEFAULT_PARAMS, 1, 1).length / 3;
    const half = seedBankPositions(QUALITY_PRESETS.low, DEFAULT_PARAMS, 1, 0.5).length / 3;
    const twice = seedBankPositions(QUALITY_PRESETS.low, DEFAULT_PARAMS, 1, 2).length / 3;
    // batch is a volume multiplier (within the seeding skin around the rolls)
    expect(half / one).toBeGreaterThan(0.4);
    expect(half / one).toBeLessThan(0.6);
    expect(twice / one).toBeGreaterThan(1.8);
    expect(twice / one).toBeLessThan(2.2);
  });

  it('seeds about the nominal batch volume', () => {
    const { h } = gridDims(QUALITY_PRESETS.high);
    const n = seedBankPositions(QUALITY_PRESETS.high, DEFAULT_PARAMS, 1, 1).length / 3;
    const vol = (n * h * h * h) / 8;
    expect(vol).toBeGreaterThan(GEOMETRY.bankVolume * 0.85);
    expect(vol).toBeLessThan(GEOMETRY.bankVolume * 1.05);
    expect(bankVolumeBelow(bankTopY(1), DEFAULT_PARAMS)).toBeCloseTo(GEOMETRY.bankVolume, 5);
    expect(bankVolumeBelow(bankTopY(2), DEFAULT_PARAMS)).toBeCloseTo(2 * GEOMETRY.bankVolume, 5);
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

describe('pigment drop slots', () => {
  it('spaces the slots evenly along the roll, clear of the end guides', () => {
    const xs = Array.from({ length: DROP_SLOTS }, (_, i) => dropSlotX(i));
    for (const x of xs) {
      expect(x - PIGMENT_CHUNK_RADIUS).toBeGreaterThan(0);
      expect(x + PIGMENT_CHUNK_RADIUS).toBeLessThan(GEOMETRY.length);
    }
    for (let i = 1; i < xs.length; i++) expect(xs[i] - xs[i - 1]).toBeCloseTo(xs[1] - xs[0], 9);
    // mirror-symmetric about the middle of the roll
    for (let i = 0; i < xs.length; i++) expect(xs[i] + xs[xs.length - 1 - i]).toBeCloseTo(GEOMETRY.length, 9);
  });

  it('clamps out-of-range slots to the ends', () => {
    expect(dropSlotX(-3)).toBe(dropSlotX(0));
    expect(dropSlotX(99)).toBe(dropSlotX(DROP_SLOTS - 1));
  });
});

describe('pigment chunk sizes', () => {
  it('offers ascending sizes with the medium one as the default radius, all clear of the end guides', () => {
    const r = PIGMENT_CHUNK_SIZES.map((s) => s.radius);
    for (let i = 1; i < r.length; i++) expect(r[i]).toBeGreaterThan(r[i - 1]);
    expect(PIGMENT_CHUNK_SIZES[DEFAULT_CHUNK_SIZE].radius).toBe(PIGMENT_CHUNK_RADIUS);
    const rMax = r[r.length - 1];
    expect(dropSlotX(0) - rMax).toBeGreaterThan(0);
    expect(dropSlotX(DROP_SLOTS - 1) + rMax).toBeLessThan(GEOMETRY.length);
  });
});
