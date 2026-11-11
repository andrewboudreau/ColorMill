import { describe, expect, it } from 'vitest';
import { DIAG_ARC, DIAG_X_BAND, nipDiagnostics, rollGhostShare, splineCdf, type DiagSnapshot } from '../src/sim/diagnostics';
import { GEOMETRY, QUALITY_PRESETS, gridDims, rollerPoses } from '../src/config/mill';

describe('splineCdf', () => {
  it('is the integral of the quadratic B-spline: 0, a half, 1, monotone', () => {
    expect(splineCdf(-2)).toBe(0);
    expect(splineCdf(0)).toBeCloseTo(0.5, 12);
    expect(splineCdf(2)).toBe(1);
    expect(splineCdf(-0.5)).toBeCloseTo(1 / 6, 12);
    expect(splineCdf(0.5)).toBeCloseTo(5 / 6, 12);
    let last = 0;
    for (let t = -1.6; t <= 1.6; t += 0.05) { const v = splineCdf(t); expect(v).toBeGreaterThanOrEqual(last); last = v; }
  });
  it('a node on a roll surface is half ghost, a node deep inside all ghost', () => {
    expect(rollGhostShare(0, 10)).toBeCloseTo(0.5, 12);
    expect(rollGhostShare(-3, 10)).toBe(1);
    expect(rollGhostShare(3, 10)).toBe(0);
  });
});

describe('the particle-level ghost share of g2p.wgsl', () => {
  it('a smoothstep of half-width 1.45 cells matches the gather of the ghost shares of the nodes', () => {
    const smooth = (s: number): number => { const t = Math.min(Math.max((s + 1.45) / 2.9, 0), 1); return 1 - t * t * (3 - 2 * t); };
    let worst = 0;
    // a flat wall at 0 along one axis, the particle at s cells outside it, the grid at every offset
    for (let s = -3; s <= 3; s += 0.05) {
      for (let shift = 0; shift < 1; shift += 0.1) {
        const g = s + shift + 10;          // particle's grid coordinate; the wall is at shift + 10
        const b0 = Math.floor(g - 0.5), f = g - b0;
        const w = [0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2];
        let exact = 0;
        for (let i = 0; i < 3; i++) exact += w[i] * splineCdf(-(b0 + i - shift - 10));
        worst = Math.max(worst, Math.abs(exact - smooth(s)));
      }
    }
    expect(worst).toBeLessThan(0.04);
  });
});

describe('nipDiagnostics', () => {
  const q = QUALITY_PRESETS.low;
  const dims = gridDims(q);
  const params = { gap: q.gap, omega: 3, frictionRatio: 1.25 };
  const { front } = rollerPoses(params);
  const R = GEOMETRY.radius;

  /** A sheet `thick` deep at `per`^3 particles a cell on the lower half of the front roll, F = J^(1/3) I. */
  const sheet = (thick: number, per: number, J: number): DiagSnapshot => {
    const h = dims.h, s = h / per;
    const pts: number[] = [];
    for (let x = 0.1 + s / 2; x < 1.4; x += s) {
      for (let y = 0.1 + s / 2; y < front.axisY; y += s) {
        for (let z = front.axisZ - R - 0.2 + s / 2; z < front.axisZ + R + 0.2; z += s) {
          const d = Math.hypot(y - front.axisY, z - front.axisZ);
          if (d > R && d < R + thick) pts.push(x, y, z);
        }
      }
    }
    const n = pts.length / 3;
    const F = new Float32Array(9 * n);
    const c = Math.cbrt(J);
    for (let i = 0; i < n; i++) { F[9 * i] = c; F[9 * i + 4] = c; F[9 * i + 8] = c; }
    return { count: n, positions: Float32Array.from(pts), velocities: new Float32Array(3 * n), deformation: F, flags: new Uint32Array(n) };
  };
  const opts = { dims, params, dt: q.dt, seconds: 1, soundSpeed: 11.5 };

  it('reads a gap-thick sheet at rest density as count 1, and J as stored', () => {
    const s = sheet(q.gap, 2, 0.9);
    const d = nipDiagnostics(s, s, opts);
    expect(d.sheetCount).toBeGreaterThan(0.93);
    expect(d.sheetCount).toBeLessThan(1.07);
    expect(d.sheet.J.mean).toBeCloseTo(0.9, 3);
    expect(d.thicknessGaps).toBeGreaterThan(0.8);
    expect(d.thicknessGaps).toBeLessThan(1);
    expect(d.flux).toBe(0);
    expect(d.tray).toBe(0);
    // the gather over a sheet 1.3 cells thick spills past its free surface: it under-reads, the ghost mass less so
    expect(d.sheet.rhoGhost.mean).toBeGreaterThan(d.sheet.rho.mean);
    expect(d.sheet.rhoGhost.mean).toBeLessThan(1.05);
  });

  it('a thick layer on the roll gathers rest density next to the roll once the ghost mass is in', () => {
    const s = sheet(6 * dims.h, 2, 1);
    const d = nipDiagnostics(s, s, { ...opts });
    // most of a six-cell layer is interior: the 95th percentile is the packed reading
    expect(d.sheet.rhoGhost.p95).toBeGreaterThan(0.97);
    expect(d.sheet.rhoGhost.p95).toBeLessThan(1.06);
    expect(d.sheetCount).toBeGreaterThan(1);
  });

  it('twice the particles is twice the count', () => {
    const a = nipDiagnostics(sheet(q.gap, 2, 1), sheet(q.gap, 2, 1), opts).sheetCount;
    const s = sheet(q.gap, 2 * Math.cbrt(2), 1);
    expect(nipDiagnostics(s, s, opts).sheetCount / a).toBeGreaterThan(1.8);
    expect(DIAG_ARC[1] - DIAG_ARC[0]).toBeGreaterThan(2);
    expect(DIAG_X_BAND[1] - DIAG_X_BAND[0]).toBeCloseTo(0.9, 12);
  });
});
