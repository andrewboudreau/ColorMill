/**
 * CPU mirror of the putty's constitutive update (design §4; g2p.wgsl and `hbShear` / `henckyStress`
 * in common.wgsl): Hencky elasticity with a von Mises radial return whose overstress follows a
 * Herschel-Bulkley law. Pure TypeScript in f64, for unit tests and for reading the flow curve off;
 * the solver itself runs the WGSL.
 *
 *   trial   F <- (I + dt C) F,  [U, S, V] = svd(F),  eps = log S
 *   split   lnJ = tr eps,  e = eps - lnJ / 3,  tau_tr = sqrt(2) mu |e|     (the shear stress of the state)
 *   return  tau_tr > tau_y:  solve  tau + mu dt ((tau - tau_y) / k)^(1/n) = tau_tr,  e <- e tau / tau_tr
 *   volume  lnJ <- clamp(min(lnJ, ln((1 + tol) / density)), ln volumeMin, ln volumeMax)
 *   F <- U exp(e + lnJ / 3) V^T,   Kirchhoff stress  U diag(2 mu e + K lnJ) U^T
 *
 * In steady simple shear at rate g the plastic rate equals g, so the shear stress is the
 * Herschel-Bulkley curve tau_y + k g^n (to first order in the elastic strain tau / 2 mu).
 */
import { bulkModulus, lameParameters, type MaterialConstants } from '../config/mill';

export type Vec3 = [number, number, number];
/** row-major 3x3 */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

/** Newton steps of the scalar return (mirrors HB_NEWTON in common.wgsl). */
export const HB_NEWTON = 4;

/**
 * The shear stress after one backward-Euler step of the Herschel-Bulkley overstress from a trial shear
 * stress `trial`: the root of  tau + muDt ((tau - tauY) / k)^(1/n) = trial  in [tauY, trial].
 * The left side is increasing and convex in tau (n <= 1), so Newton from above cannot overshoot; it
 * starts from the smaller of the trial value and the root of the power term alone, within a factor of
 * two of the root.
 */
export function hbShear(trial: number, tauY: number, k: number, n: number, muDt: number): number {
  if (!(trial > tauY)) return trial;
  const xt = trial - tauY;
  const kk = Math.max(k, 1e-6);
  const p = 1 / n;
  let x = Math.min(xt, kk * Math.pow(xt / muDt, n));
  for (let i = 0; i < HB_NEWTON; i++) {
    const w = muDt * Math.pow(x / kk, p);
    x -= (x - xt + w) / (1 + (p * w) / Math.max(x, 1e-20));
    x = Math.max(x, 0);
  }
  return tauY + Math.min(x, xt);
}

/** The deviatoric part of a principal Hencky strain, and its trace. */
function split(eps: Vec3): { e: Vec3; lnJ: number } {
  const lnJ = eps[0] + eps[1] + eps[2];
  return { e: [eps[0] - lnJ / 3, eps[1] - lnJ / 3, eps[2] - lnJ / 3], lnJ };
}

/** Shear stress of a principal Hencky strain: sqrt(2) mu |dev eps| (tau_xy of the equivalent simple shear). */
export function shearStress(eps: Vec3, m: MaterialConstants): number {
  const { e } = split(eps);
  return Math.SQRT2 * lameParameters(m).mu * Math.hypot(e[0], e[1], e[2]);
}

/**
 * The plastic return on principal Hencky strains: radial in the deviatoric plane; the volume is untouched
 * inside [volumeMin, volumeMax] and saturates at those bounds. `density` (the particle density the grid
 * measures, rest = 1) applies the density tie as well: J may not exceed (1 + densityTolerance) / density.
 */
export function plasticReturn(eps: Vec3, m: MaterialConstants, dt: number, density?: number): Vec3 {
  const { mu } = lameParameters(m);
  const { e, lnJ } = split(eps);
  let v = lnJ;
  if (density !== undefined && m.densityTolerance > 0) v = Math.min(v, Math.log((1 + m.densityTolerance) / Math.max(density, 1e-3)));
  v = Math.min(Math.max(v, Math.log(m.volumeMin)), Math.log(m.volumeMax)) / 3;
  const trial = Math.SQRT2 * mu * Math.hypot(e[0], e[1], e[2]);
  const s = trial > m.yieldStress ? hbShear(trial, m.yieldStress, m.consistency, m.flowIndex, mu * dt) / trial : 1;
  return [e[0] * s + v, e[1] * s + v, e[2] * s + v];
}

/** Principal Kirchhoff stresses of a principal Hencky strain: 2 mu dev eps + K ln J. */
export function principalStress(eps: Vec3, m: MaterialConstants): Vec3 {
  const { mu } = lameParameters(m);
  const { e, lnJ } = split(eps);
  const pv = bulkModulus(m) * lnJ;
  return [2 * mu * e[0] + pv, 2 * mu * e[1] + pv, 2 * mu * e[2] + pv];
}

/** The Herschel-Bulkley flow curve the return is built on. */
export function flowCurve(rate: number, m: MaterialConstants): number {
  return m.yieldStress + m.consistency * Math.pow(rate, m.flowIndex);
}

// ---- 3x3 helpers for the full update (tests drive one particle through it) ----

export function mul(a: Mat3, b: Mat3): Mat3 {
  const o = new Array<number>(9) as Mat3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  return o;
}

export function transpose(a: Mat3): Mat3 {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}

/** Eigen-decomposition of a symmetric 3x3 by cyclic Jacobi: A = Q diag(w) Q^T, the columns of Q the vectors. */
export function eigSym(A: Mat3): { w: Vec3; Q: Mat3 } {
  const a = A.slice() as Mat3;
  let Q: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 30; sweep++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[3 * p + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[3 * q + q] - a[3 * p + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      const J: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
      J[3 * p + p] = c; J[3 * q + q] = c; J[3 * p + q] = s; J[3 * q + p] = -s;
      const r = mul(mul(transpose(J), a), J);
      for (let i = 0; i < 9; i++) a[i] = r[i];
      Q = mul(Q, J);
    }
  }
  return { w: [a[0], a[4], a[8]], Q };
}

export interface PuttyState {
  /** elastic deformation gradient after the return */
  readonly F: Mat3;
  /** Kirchhoff stress */
  readonly tau: Mat3;
  /** principal Hencky strains after the return */
  readonly eps: Vec3;
}

/**
 * One substep of the constitutive update for a velocity gradient L (= C): the trial F, the return, the
 * stress. Works on the left stretch, b = F F^T = U S^2 U^T; the rotation is carried along as R = V_l^-1 F.
 */
export function updatePutty(F: Mat3, L: Mat3, dt: number, m: MaterialConstants): PuttyState {
  const I: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const Ft = mul(I.map((v, i) => v + dt * L[i]) as Mat3, F);
  const { w, Q } = eigSym(mul(Ft, transpose(Ft)));
  const S: Vec3 = [Math.sqrt(Math.max(w[0], 1e-12)), Math.sqrt(Math.max(w[1], 1e-12)), Math.sqrt(Math.max(w[2], 1e-12))];
  const eps0: Vec3 = [Math.log(S[0]), Math.log(S[1]), Math.log(S[2])];
  const eps = plasticReturn(eps0, m, dt);
  const diag = (d: Vec3): Mat3 => [d[0], 0, 0, 0, d[1], 0, 0, 0, d[2]];
  const Qt = transpose(Q);
  // F_new = U diag(S_new / S) U^T F_trial: the left stretch is rescaled, the rotation kept
  const scale = mul(mul(Q, diag([Math.exp(eps[0]) / S[0], Math.exp(eps[1]) / S[1], Math.exp(eps[2]) / S[2]])), Qt);
  const tau = mul(mul(Q, diag(principalStress(eps, m))), Qt);
  return { F: mul(scale, Ft), tau, eps };
}
