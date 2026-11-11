import { describe, expect, it } from 'vitest';
import { mixingMetrics, pigmentInventory, type PigmentSnapshot } from '../src/sim/metrics';

const H = 0.1;
const SPACING = H / 2;

/** A block of particles on the seeding lattice (two per cell per axis), `load(x, y, z, index)` each. */
function block(nx: number, ny: number, nz: number, load: (x: number, y: number, z: number, i: number) => number,
  latent: (i: number) => readonly number[] = () => [1, 0, 0, 0, 0, 0, 0]): PigmentSnapshot {
  const n = nx * ny * nz;
  const positions = new Float32Array(3 * n);
  const loads = new Float32Array(n);
  const latents = new Float32Array(7 * n);
  let p = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = (i + 0.5) * SPACING, y = (j + 0.5) * SPACING, z = (k + 0.5) * SPACING;
    positions.set([x, y, z], 3 * p);
    loads[p] = load(x, y, z, p);
    latents.set(latent(p), 7 * p);
    p++;
  }
  return { count: n, positions, loads, latents };
}

describe('pigment inventory', () => {
  it('sums load and load-weighted latent, ignoring the latent of clear particles', () => {
    const snap = block(4, 2, 2, (_x, _y, _z, i) => (i < 4 ? 12 : i < 8 ? 3 : 0),
      (i) => (i < 4 ? [1, 0, 0, 0, 0.1, 0, 0] : i < 8 ? [0, 1, 0, 0, 0, 0.2, 0] : [0, 0, 0, 1, 9, 9, 9]));
    const inv = pigmentInventory(snap);
    expect(inv.particles).toBe(16);
    expect(inv.pigmented).toBe(8);
    expect(inv.totalLoad).toBeCloseTo(60, 9);
    expect(inv.channels[0]).toBeCloseTo(48, 6);
    expect(inv.channels[1]).toBeCloseTo(12, 6);
    expect(inv.channels[3]).toBe(0);
    expect(inv.channels[4]).toBeCloseTo(4.8, 5);
    expect(inv.channels[5]).toBeCloseTo(2.4, 5);
    expect(inv.meanLatent[0]).toBeCloseTo(0.8, 6);
    expect(inv.meanLatent[1]).toBeCloseTo(0.2, 6);
    expect(inv.minLoad).toBe(0);
    expect(inv.maxLoad).toBe(12);
  });

  it('scales with the particle mass and is zero for a clear batch', () => {
    const snap = block(2, 2, 2, () => 6);
    expect(pigmentInventory(snap, 0.5).totalLoad).toBeCloseTo(24, 9);
    const clear = pigmentInventory(block(2, 2, 2, () => 0));
    expect(clear.totalLoad).toBe(0);
    expect(clear.meanLatent).toEqual([0, 0, 0, 0, 0, 0, 0]);
  });

  it('is invariant under a conservative exchange between two particles', () => {
    const snap = block(2, 1, 1, (_x, _y, _z, i) => (i === 0 ? 12 : 0), (i) => (i === 0 ? [0, 1, 0, 0, 0, 0, 0] : [0, 0, 0, 1, 0, 0, 0]));
    const before = pigmentInventory(snap);
    // particle 0 hands 5 units of its pigment to particle 1, which takes its colour
    snap.loads[0] = 7; snap.loads[1] = 5;
    snap.latents.set([0, 1, 0, 0, 0, 0, 0], 7);
    const after = pigmentInventory(snap);
    expect(after.totalLoad).toBeCloseTo(before.totalLoad, 9);
    for (let c = 0; c < 7; c++) expect(after.channels[c]).toBeCloseTo(before.channels[c], 9);
  });
});

describe('mixing metrics', () => {
  it('reads 1 at every scale for a fully segregated batch whose halves line up with the boxes', () => {
    // 16 x 8 x 8 particles = 8 x 4 x 4 cells; the left half (4 cells) neat masterbatch, the right clear
    const snap = block(16, 8, 8, (x) => (x < 4 * H ? 12 : 0));
    const m = mixingMetrics(snap, H);
    expect(m.mean).toBeCloseTo(0.5, 9);
    expect(m.sigma0Sq).toBeCloseTo(0.25, 9);
    expect(m.scales.map((s) => s.box)).toEqual([H, 2 * H, 4 * H]);
    for (const s of m.scales) expect(s.intensity).toBeCloseTo(1, 9);
    expect(m.scales[0].bins).toBe(8 * 4 * 4);
    expect(m.scales[2].bins).toBe(2);
    expect(m.axial).toBeCloseTo(1, 9);
    expect(m.axialSlabs).toBe(8);
  });

  it('reads 0 for a uniform batch, at any load', () => {
    const m = mixingMetrics(block(8, 8, 8, () => 3), H);
    expect(m.mean).toBeCloseTo(0.25, 9);
    for (const s of m.scales) expect(s.intensity).toBeCloseTo(0, 12);
    expect(m.axial).toBeCloseTo(0, 12);
    // and for a clear one (no pigment: nothing to segregate)
    const clear = mixingMetrics(block(4, 4, 4, () => 0), H);
    expect(clear.mean).toBe(0);
    expect(clear.scales[0].intensity).toBe(0);
    expect(clear.axial).toBe(0);
  });

  it('falls with the box size for a fine checkerboard, and has no axial part when every slab holds the same', () => {
    // alternate particles in y: every cell holds half masterbatch, half clear
    const snap = block(8, 8, 8, (_x, y) => (Math.floor(y / SPACING) % 2 === 0 ? 12 : 0));
    const m = mixingMetrics(snap, H, { boxes: [SPACING, H, 2 * H] });
    expect(m.scales[0].intensity).toBeCloseTo(1, 9);   // a box per particle sees the two levels
    expect(m.scales[1].intensity).toBeCloseTo(0, 12);  // a cell already averages them
    expect(m.scales[2].intensity).toBeCloseTo(0, 12);
    expect(m.axial).toBeCloseTo(0, 12);
  });

  it('separates axial segregation from segregation across the sheet', () => {
    // layered through y in whole cells: segregated at h, but every x slab has the same mean
    const layered = mixingMetrics(block(8, 8, 8, (_x, y) => (y < 2 * H ? 12 : 0)), H);
    expect(layered.scales[0].intensity).toBeCloseTo(1, 9);
    expect(layered.axial).toBeCloseTo(0, 12);
    // banded along x: the same intensity, all of it axial
    const banded = mixingMetrics(block(8, 8, 8, (x) => (x < 2 * H ? 12 : 0)), H);
    expect(banded.scales[0].intensity).toBeCloseTo(1, 9);
    expect(banded.axial).toBeCloseTo(1, 9);
  });

  it('reports the random-mixture floor: a random two-level mix reads about 1 / particles per box', () => {
    let seed = 12345;
    const rnd = (): number => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    const snap = block(32, 16, 16, () => (rnd() < 0.3 ? 12 : 0));
    const m = mixingMetrics(snap, H);
    expect(m.scales[0].floor).toBeCloseTo(1 / 8, 9);
    expect(m.scales[1].floor).toBeCloseTo(1 / 64, 9);
    expect(m.scales[0].intensity).toBeGreaterThan(0.09);
    expect(m.scales[0].intensity).toBeLessThan(0.16);
    expect(m.scales[1].intensity).toBeLessThan(m.scales[0].intensity);
    expect(m.scales[2].intensity).toBeLessThan(m.scales[1].intensity);
  });

  it('clamps the concentration to 0..1 and takes the full load as an option', () => {
    const m = mixingMetrics(block(4, 4, 4, (x) => (x < H ? 30 : 0)), H, { fullLoad: 6 });
    expect(m.mean).toBeCloseTo(0.5, 9);
    expect(m.scales[0].intensity).toBeCloseTo(1, 9);
  });
});
