import { describe, expect, it } from 'vitest';
import { exchangeFraction, laminaStep, mixedAfter, striationStats, type LaminaState, type Vec3 } from '../src/sim/lamina';

/** Row-major velocity gradient L[r][c] = dv_r / dx_c. */
const mat = (rows: number[][]): number[] => rows.flat();

function run(state: LaminaState, L: number[], seconds: number, dt: number): LaminaState {
  let s = state;
  const steps = Math.round(seconds / dt);
  for (let i = 0; i < steps; i++) s = laminaStep(s, L, dt);
  return s;
}

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

describe('lamina kinematics', () => {
  // simple shear v = (rate * y, 0, 0): L[0][1] = rate
  const shear = (rate: number): number[] => mat([[0, rate, 0], [0, 0, 0], [0, 0, 0]]);

  it('simple shear thins laminae that face the flow as 1 / sqrt(1 + gamma^2), so as 1 / gamma at large strain', () => {
    // normal along the flow: stripes across it, which the shear tilts over and thins
    for (const gamma of [0.5, 3, 20, 200]) {
      const out = run({ n: [1, 0, 0], q: 0 }, shear(10), gamma / 10, 1e-3);
      const s = Math.exp(out.q);
      expect(s).toBeCloseTo(1 / Math.sqrt(1 + gamma * gamma), 6);
      // the normal turns toward the velocity-gradient direction: n is parallel to F^-T n0 = (1, -gamma, 0)
      expect(Math.abs(dot(out.n, [1 / Math.hypot(1, gamma), -gamma / Math.hypot(1, gamma), 0]))).toBeCloseTo(1, 6);
    }
    const s200 = Math.exp(run({ n: [1, 0, 0], q: 0 }, shear(10), 20, 1e-3).q);
    expect(s200 * 200).toBeCloseTo(1, 4);   // s -> s0 / gamma
  });

  it('simple shear does nothing to laminae that lie along the shear planes, or that are stacked along the vorticity', () => {
    for (const n of [[0, 1, 0], [0, 0, 1]] as Vec3[]) {
      const out = run({ n, q: 0 }, shear(10), 5, 1e-3);
      expect(out.q).toBeCloseTo(0, 12);
      expect(dot(out.n, n)).toBeCloseTo(1, 12);
    }
  });

  it('simple shear first thickens a lamina tilted against it, then thins it (the strain is reversible)', () => {
    // F^-T n0 for n0 = (1, 1, 0)/sqrt2 is (1, 1 - gamma, 0)/sqrt2: thickest at gamma = 1, back to s0 at gamma = 2
    const n0: Vec3 = [Math.SQRT1_2, Math.SQRT1_2, 0];
    expect(Math.exp(run({ n: n0, q: 0 }, shear(1), 1, 1e-3).q)).toBeCloseTo(Math.SQRT2, 6);
    expect(Math.exp(run({ n: n0, q: 0 }, shear(1), 2, 1e-3).q)).toBeCloseTo(1, 6);
    // and shearing back undoes it
    const there = run({ n: [1, 0, 0], q: 0 }, shear(4), 1, 1e-3);
    const back = run(there, shear(-4), 1, 1e-3);
    expect(back.q).toBeCloseTo(0, 9);
    expect(dot(back.n, [1, 0, 0])).toBeCloseTo(1, 9);
  });

  it('pure extension thins exponentially along the compressed axis and thickens along the stretched one', () => {
    const rate = 2;
    const L = mat([[rate, 0, 0], [0, -rate, 0], [0, 0, 0]]);   // planar extension, stretched along x
    for (const t of [0.25, 1, 2]) {
      const squeezed = run({ n: [0, 1, 0], q: 0 }, L, t, 1e-3);
      expect(squeezed.q).toBeCloseTo(-rate * t, 5);
      expect(dot(squeezed.n, [0, 1, 0])).toBeCloseTo(1, 9);
      const stretched = run({ n: [1, 0, 0], q: 0 }, L, t, 1e-3);
      expect(stretched.q).toBeCloseTo(Math.min(rate * t, 2), 5);   // q is capped at LAMINA_Q_MAX = 2
    }
    // an oblique lamina turns to face the compression and then thins at the same rate
    const a = run({ n: [Math.SQRT1_2, Math.SQRT1_2, 0], q: 0 }, L, 2, 1e-3);
    const b = run(a, L, 0.5, 1e-3);
    expect(Math.abs(a.n[1])).toBeGreaterThan(0.999);
    expect(b.q - a.q).toBeCloseTo(-rate * 0.5, 3);
  });

  it('uniaxial extension is exponential too, and the step length does not matter much', () => {
    const rate = 1.5;
    const L = mat([[rate, 0, 0], [0, -rate / 2, 0], [0, 0, -rate / 2]]);
    expect(run({ n: [0, 0, 1], q: 0 }, L, 2, 1e-3).q).toBeCloseTo(-rate, 5);
    // a frame-sized step at a nip-sized rate: 12.8 ms at 25 /s
    const coarse = run({ n: [0, 1, 0], q: 0 }, mat([[25, 0, 0], [0, -25, 0], [0, 0, 0]]), 0.128, 0.0128);
    expect(coarse.q).toBeCloseTo(-3.2, 1);
  });

  it('a rigid rotation turns the normal and leaves the thickness alone', () => {
    const w = 3;   // rad/s about x: v = w x r, L = [[0,0,0],[0,0,-w],[0,w,0]]
    const L = mat([[0, 0, 0], [0, 0, -w], [0, w, 0]]);
    const out = run({ n: [0, 1, 0], q: -0.7 }, L, 0.5, 1e-3);
    expect(out.q).toBeCloseTo(-0.7, 7);   // theta^4 / 8 a step
    expect(out.n[0]).toBeCloseTo(0, 9);
    expect(out.n[1]).toBeCloseTo(Math.cos(w * 0.5), 5);
    expect(out.n[2]).toBeCloseTo(Math.sin(w * 0.5), 5);
    // the axis itself does not move
    const axis = run({ n: [1, 0, 0], q: 0 }, L, 0.5, 1e-3);
    expect(dot(axis.n, [1, 0, 0])).toBeCloseTo(1, 12);
    // a whole roll revolution at the solver's substep: the drift in q is far below anything that matters
    const rev = run({ n: [0, 1, 0], q: 0 }, L, 2 * Math.PI / w, 1.6e-3);
    expect(Math.abs(rev.q)).toBeLessThan(1e-7);
  });

  it('volume change is not mixing: an isotropic squeeze changes nothing', () => {
    const L = mat([[-4, 0, 0], [0, -4, 0], [0, 0, -4]]);
    const out = run({ n: [0.6, 0.8, 0], q: 0 }, L, 0.3, 1e-3);
    expect(out.q).toBeCloseTo(0, 12);
    expect(dot(out.n, [0.6, 0.8, 0])).toBeCloseTo(1, 12);
  });

  it('matches s / s0 = 1 / |F^-T n0| for a mixed, volume-preserving gradient', () => {
    // shear plus a little planar extension; F integrated alongside with the same L
    const L = mat([[0.8, 6, 0], [0, -0.8, 0], [0, 0, 0]]);
    const dt = 2e-4;
    const n0: Vec3 = [0.36, 0.48, 0.8];
    let st: LaminaState = { n: n0, q: 0 };
    let m: [number, number, number] = [...n0];   // dm/dt = -L^T m, by RK4
    const f = (v: number[]): number[] => [-(L[0] * v[0] + L[3] * v[1] + L[6] * v[2]), -(L[1] * v[0] + L[4] * v[1] + L[7] * v[2]), -(L[2] * v[0] + L[5] * v[1] + L[8] * v[2])];
    for (let i = 0; i < 5000; i++) {
      st = laminaStep(st, L, dt);
      const k1 = f(m), k2 = f(m.map((x, j) => x + 0.5 * dt * k1[j])), k3 = f(m.map((x, j) => x + 0.5 * dt * k2[j])), k4 = f(m.map((x, j) => x + dt * k3[j]));
      m = m.map((x, j) => x + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])) as [number, number, number];
    }
    const len = Math.hypot(...m);
    expect(st.q).toBeCloseTo(-Math.log(len), 6);
    expect(Math.abs(dot(st.n, [m[0] / len, m[1] / len, m[2] / len]))).toBeCloseTo(1, 8);
  });
});

describe('exchange closure', () => {
  const lnDelta = Math.log(1 / 64);

  it('trades nothing while the laminae are thicker than the particle spacing', () => {
    expect(exchangeFraction(Math.log(0.1), Math.log(0.14), lnDelta)).toBe(0);
    expect(exchangeFraction(lnDelta + 1e-6, Math.log(0.14), lnDelta)).toBe(0);
  });

  it('trades 1 - s / delta on the step that crosses the spacing, and nothing on the way back up', () => {
    expect(exchangeFraction(lnDelta + Math.log(0.8), Math.log(0.14), lnDelta)).toBeCloseTo(0.2, 12);
    const mixed = mixedAfter(lnDelta + Math.log(0.8), Math.log(0.14));
    // thicker again, then thinner but not below the old minimum: nothing new
    expect(exchangeFraction(lnDelta + Math.log(0.9), mixed, lnDelta)).toBe(0);
    expect(exchangeFraction(lnDelta + Math.log(0.8), mixed, lnDelta)).toBe(0);
    expect(exchangeFraction(lnDelta + Math.log(0.4), mixed, lnDelta)).toBeCloseTo(0.5, 12);
  });

  it('leaves (sMin / delta)^k of the deviation however the thinning is split into steps, chatter included', () => {
    for (const k of [1, 0.5, 2]) {
      for (const steps of [1, 7, 300]) {
        let mixed = Math.log(0.2);
        let keep = 1;
        const target = lnDelta + Math.log(0.05);
        for (let i = 1; i <= steps; i++) {
          // a wobble on top of the trend, so many steps go the wrong way
          const lnS = Math.log(0.2) + (target - Math.log(0.2)) * (i / steps) + (i < steps ? 0.3 * Math.sin(1.7 * i) : 0);
          keep *= 1 - exchangeFraction(lnS, mixed, lnDelta, k);
          mixed = mixedAfter(lnS, mixed);
        }
        // the wobble can only have taken the laminae thinner than the target on the way
        expect(keep).toBeLessThanOrEqual(Math.pow(0.05, k) * (1 + 1e-9));
        expect(keep).toBeCloseTo(Math.exp(k * (Math.min(mixed, lnDelta) - lnDelta)), 12);
      }
    }
  });
});

describe('striation statistics', () => {
  it('weights by load and reports thickness in world units', () => {
    const spacing = 1 / 64;
    const lnS0 = Math.log(0.2);
    const snap = {
      count: 4,
      loads: new Float32Array([12, 12, 0, 1]),
      // q = ln(s / s0): 0.2, 0.02, (clear), 0.001
      laminae: new Float32Array([1, 0, 0, 0, 1, 0, 0, Math.log(0.1), 0, 1, 0, -3, 0, 0, 1, Math.log(0.005)]),
      mixState: new Float32Array([lnS0, lnS0, lnS0, lnS0, lnS0, lnS0, lnS0, lnS0])
    };
    const st = striationStats(snap, spacing);
    expect(st.pigmented).toBe(3);
    expect(st.median).toBeCloseTo(0.02, 6);
    expect(st.p90).toBeCloseTo(0.2, 6);
    expect(st.belowSpacing).toBeCloseTo(1 / 25, 9);
    expect(st.blended).toBeCloseTo(1 / 25, 9);
  });
});
