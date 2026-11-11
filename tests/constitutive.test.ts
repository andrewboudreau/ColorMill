import { describe, expect, it } from 'vitest';
import { DEFAULT_MATERIAL, QUALITY_PRESETS, bulkModulus, courantNumber, lameParameters, soundSpeed, type MaterialConstants } from '../src/config/mill';
import { flowCurve, hbShear, plasticReturn, principalStress, shearStress, updatePutty, type Mat3, type Vec3 } from '../src/sim/constitutive';

const m = DEFAULT_MATERIAL;
const { mu } = lameParameters(m);
const dt = QUALITY_PRESETS.high.dt;

/** The root of the scalar return by bisection. */
function bisect(trial: number, tauY: number, k: number, n: number, muDt: number): number {
  if (trial <= tauY) return trial;
  let lo = tauY, hi = trial;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (mid + muDt * Math.pow((mid - tauY) / k, 1 / n) > trial) hi = mid; else lo = mid;
  }
  return 0.5 * (lo + hi);
}

describe('hbShear', () => {
  it('leaves a stress inside the yield surface alone', () => {
    expect(hbShear(0.3, 0.58, 0.45, 0.35, mu * dt)).toBe(0.3);
    expect(hbShear(0.58, 0.58, 0.45, 0.35, mu * dt)).toBe(0.58);
  });

  it('is the root of the backward-Euler equation over the whole range of trial stresses, steps and indices', () => {
    let worst = 0;
    for (const n of [0.2, 0.3, 0.35, 0.5, 0.75, 1]) {
      for (const k of [0.05, 0.45, 2]) {
        for (const step of [7.1e-4, 1.6e-3, 5e-3]) {
          for (const over of [1e-5, 1e-3, 0.02, 0.3, 2, 20]) {
            const trial = 0.58 + over;
            const got = hbShear(trial, 0.58, k, n, mu * step);
            const want = bisect(trial, 0.58, k, n, mu * step);
            expect(got).toBeGreaterThanOrEqual(0.58);
            expect(got).toBeLessThanOrEqual(trial);
            worst = Math.max(worst, Math.abs(got - want) / (want - 0.58 + 1e-9));
          }
        }
      }
    }
    // four Newton steps from the bracketing start: within a percent of the overstress everywhere
    expect(worst).toBeLessThan(0.01);
  });

  it('is monotone in the trial stress', () => {
    let last = 0;
    for (let t = 0; t < 6; t += 0.01) {
      const v = hbShear(t, 0.58, 0.45, 0.35, mu * dt);
      expect(v).toBeGreaterThanOrEqual(last);
      last = v;
    }
  });
});

describe('plasticReturn', () => {
  const eps: Vec3 = [0.31, -0.08, -0.29];

  it('is radial in the deviatoric plane and leaves the volume alone', () => {
    const r = plasticReturn(eps, m, dt);
    const tr = (v: Vec3) => v[0] + v[1] + v[2];
    expect(tr(r)).toBeCloseTo(tr(eps), 12);
    const e0 = eps.map((v) => v - tr(eps) / 3), e1 = r.map((v) => v - tr(r) / 3);
    const s = e1[0] / e0[0];
    expect(s).toBeGreaterThan(0);
    expect(s).toBeLessThan(1);
    expect(e1[1] / e0[1]).toBeCloseTo(s, 10);
    expect(e1[2] / e0[2]).toBeCloseTo(s, 10);
    expect(shearStress(r, m)).toBeGreaterThan(m.yieldStress);
  });

  it('leaves a state inside the yield surface alone (elastic), and a pure volume change at any size', () => {
    const small: Vec3 = [0.01, -0.004, -0.02];
    expect(shearStress(small, m)).toBeLessThan(m.yieldStress);
    const kept = plasticReturn(small, m, dt);
    for (let i = 0; i < 3; i++) expect(kept[i]).toBeCloseTo(small[i], 14);
    const squeezed = plasticReturn([-0.1, -0.1, -0.1], m, dt);
    for (let i = 0; i < 3; i++) expect(squeezed[i]).toBeCloseTo(-0.1, 14);
  });

  it('the volume saturates at its bounds, the shape untouched', () => {
    const r = plasticReturn([-0.39, -0.4, -0.41], m, dt);
    expect(r[0] + r[1] + r[2]).toBeCloseTo(Math.log(m.volumeMin), 12);
    expect(r[0] - r[1]).toBeCloseTo(0.01, 12);
    const s = plasticReturn([0.3, 0.3, 0.3], m, dt);
    expect(s[0] + s[1] + s[2]).toBeCloseTo(Math.log(m.volumeMax), 12);
  });

  it('the density tie lowers J to what the measured density allows and never raises it', () => {
    const tied = { ...m, densityTolerance: 0.1 };
    const at = (lnJ: number, rho: number): number => { const r = plasticReturn([lnJ / 3, lnJ / 3, lnJ / 3], tied, dt, rho); return Math.exp(r[0] + r[1] + r[2]); };
    expect(at(Math.log(0.95), 1.375)).toBeCloseTo(0.8, 12);      // packed to 1.375: J = 1.1 / 1.375
    expect(at(Math.log(0.95), 1.0)).toBeCloseTo(0.95, 12);       // within the tolerance: untouched
    expect(at(Math.log(0.75), 1.2)).toBeCloseTo(0.75, 12);       // already below: untouched
    expect(at(Math.log(1.2), 0.6)).toBeCloseTo(1.2, 12);         // a free surface under-reads: never raised
    expect(at(Math.log(0.95), 3)).toBeCloseTo(tied.volumeMin, 12);
    expect(plasticReturn([0, 0, 0], m, dt, 2)).toEqual([0, 0, 0]);   // tolerance 0: off
  });

  it('is idempotent in the rate-independent limit: a returned state is on the yield surface and stays', () => {
    const inviscid: MaterialConstants = { ...m, consistency: 0 };
    const once = plasticReturn(eps, inviscid, dt);
    expect(shearStress(once, inviscid)).toBeCloseTo(m.yieldStress, 4);
    const twice = plasticReturn(once, inviscid, dt);
    for (let i = 0; i < 3; i++) expect(twice[i]).toBeCloseTo(once[i], 5);
  });

  it('with the overstress, repeated returns relax monotonically onto the yield surface and no further', () => {
    let s = eps, last = shearStress(eps, m);
    for (let i = 0; i < 20000; i++) {
      s = plasticReturn(s, m, dt);
      const t = shearStress(s, m);
      expect(t).toBeLessThanOrEqual(last);
      expect(t).toBeGreaterThanOrEqual(m.yieldStress);
      last = t;
    }
    expect(last).toBeLessThan(m.yieldStress * 1.05);
  });

  it('the pressure is K ln J, with no ceiling in compression', () => {
    const K = bulkModulus(m);
    expect(K).toBeCloseTo(122.2, 1);
    for (const J of [0.9, 0.5, 0.2]) {
      const l = Math.log(J) / 3;
      const t = principalStress([l, l, l], m);
      expect(t[0]).toBeCloseTo(K * Math.log(J), 9);
      // Cauchy pressure -tau / J grows without bound as J -> 0 (the old lambda (J - 1) J peaked at J = 1/2)
      expect(-t[0] / J).toBeGreaterThan(K * (1 - J));
    }
  });
});

describe('updatePutty', () => {
  /** Steady simple shear v_x = g y: the shear stress once the transient has gone. */
  function steadyShear(g: number, mat: MaterialConstants, step: number): { tau: number; inv: number; J: number } {
    let F: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const L: Mat3 = [0, g, 0, 0, 0, 0, 0, 0, 0];
    let tau = 0, inv = 0;
    const steps = Math.ceil(Math.max(3 / g, 0.5) / step);
    for (let i = 0; i < steps; i++) {
      const r = updatePutty(F, L, step, mat);
      F = r.F;
      tau = r.tau[1];
      inv = shearStress(r.eps, mat);
    }
    const J = F[0] * (F[4] * F[8] - F[5] * F[7]) - F[1] * (F[3] * F[8] - F[5] * F[6]) + F[2] * (F[3] * F[7] - F[4] * F[6]);
    return { tau, inv, J };
  }

  it('steady simple shear reproduces the Herschel-Bulkley flow curve', () => {
    for (const g of [0.3, 1, 3, 10, 30]) {
      const { tau, inv, J } = steadyShear(g, m, dt);
      const want = flowCurve(g, m);
      // the von Mises shear stress sits on the curve; the gap is second order in the elastic shear tau / mu
      // (0.12 - 0.28 here: the elastic stretch is not coaxial with the rate of deformation)
      expect(inv).toBeLessThanOrEqual(want);
      expect(inv).toBeGreaterThan(0.985 * want);
      // the xy component is lower again by the first normal stress difference
      expect(tau).toBeLessThanOrEqual(inv);
      expect(tau).toBeGreaterThan(0.94 * want);
      expect(J).toBeCloseTo(1, 6);
    }
  });

  it('and at every preset step, and for other indices', () => {
    for (const q of Object.values(QUALITY_PRESETS)) {
      for (const [n, g] of [[0.3, 5], [0.4, 5], [1, 0.5]]) {
        const mat = { ...m, flowIndex: n };
        const { inv } = steadyShear(g, mat, q.dt);
        expect(Math.abs(inv - flowCurve(g, mat)) / flowCurve(g, mat)).toBeLessThan(0.01);
      }
    }
  });

  it('below yield a sheared block is elastic: tau = mu gamma', () => {
    const g = 0.5, steps = 20;
    let F: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    let tau = 0;
    for (let i = 0; i < steps; i++) { const r = updatePutty(F, [0, g, 0, 0, 0, 0, 0, 0, 0], dt, m); F = r.F; tau = r.tau[1]; }
    expect(tau).toBeCloseTo(mu * g * steps * dt, 4);
    expect(tau).toBeLessThan(m.yieldStress);
  });
});

describe('stability numbers', () => {
  it('the sound speed and the Courant number of every preset', () => {
    expect(soundSpeed(m)).toBeCloseTo(11.5, 1);
    for (const q of Object.values(QUALITY_PRESETS)) {
      const c = courantNumber(q, m);
      expect(c).toBeGreaterThan(0.55);
      expect(c).toBeLessThan(0.62);
    }
    expect(courantNumber(QUALITY_PRESETS.low, { E: 22, nu: 0.49 })).toBeGreaterThan(0.95);
  });
});
