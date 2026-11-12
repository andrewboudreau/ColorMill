/**
 * Dispersive mixing: pigment agglomerates and the stress that breaks them (design §5, Agglomerates).
 * The CPU mirror of the kinetics in shaders/lamina.wgsl, and the measures of how far a batch is dispersed.
 *
 * Fresh masterbatch is pigment in agglomerates. A particle carries its UNDISPERSED load u, the part of
 * its pigment load w still in agglomerates (0 <= u <= w; the undispersed fraction is a = u / w). Only
 * dispersed pigment develops its full colour: the load the renderer tints with is w - (1 - weak) u.
 *
 * Kinetics, per particle, from the velocity gradient L it carries (the APIC matrix C):
 *
 *     rate = sqrt(2 D':D'),  D' the symmetric, traceless part of L       the shear rate
 *     tau  = tau_y + k rate^n                                             the flow curve (design §4)
 *     du/dt = -u rate [ k_e max(tau / tau_e - 1, 0) + k_r max(tau / tau_c - 1, 0) ]
 *
 * (`rate` is the particle's times `rateScale`, config agglomerateRateScale: 1 at the `low` preset.)
 * Nothing below the cohesive stress tau_e; over it the loss per unit of shear strain grows with the
 * excess stress (erosion), and over the higher tau_c a second term stands for rupture (tau_c = 0: none).
 * First order in u, so the step is the exact decay exp(-loss dt) and cannot overshoot.
 *
 * What is from the literature and what is not: that agglomerates break by erosion and by rupture, that
 * erosion is gradual, sets in at the lower stress and follows first-order kinetics, and that the applied
 * shear stress scaled by the cohesive strength is what governs it, is Rwei, Manas-Zloczower & Feke
 * (Polym. Eng. Sci. 1990, 1991; read at the level of their abstracts). The rate expression above (linear
 * in the strain rate and in the excess stress) and every constant are this project's own choice.
 */
import type { AgglomerateConstants, MaterialConstants } from '../config/mill';
import type { ParticleSnapshot } from './types';

export type FlowCurve = Pick<MaterialConstants, 'yieldStress' | 'consistency' | 'flowIndex'>;

/** The share of its tinting strength pigment still in agglomerates has (mirrors AGGLO_WEAK in raymarch.wgsl). */
export const AGGLOMERATE_TINT = 0.12;

/** The shear rate sqrt(2 D':D') of a row-major 3x3 velocity gradient (read at `offset` of `L`). */
export function shearRate(L: ArrayLike<number>, offset = 0): number {
  const l = (r: number, c: number): number => L[offset + 3 * r + c];
  const tr = (l(0, 0) + l(1, 1) + l(2, 2)) / 3;
  const dxx = l(0, 0) - tr, dyy = l(1, 1) - tr, dzz = l(2, 2) - tr;
  const dxy = 0.5 * (l(0, 1) + l(1, 0)), dxz = 0.5 * (l(0, 2) + l(2, 0)), dyz = 0.5 * (l(1, 2) + l(2, 1));
  return Math.sqrt(2 * (dxx * dxx + dyy * dyy + dzz * dzz + 2 * (dxy * dxy + dxz * dxz + dyz * dyz)));
}

/** The shear stress of the putty flowing at shear rate `rate`: the Herschel-Bulkley flow curve. */
export function flowStress(rate: number, m: FlowCurve): number {
  return m.yieldStress + m.consistency * Math.pow(Math.max(rate, 0), Math.min(Math.max(m.flowIndex, 0.05), 1));
}

/** The rate (1/s) the undispersed load decays at under the shear rate `rawRate` of the particle's velocity
 *  gradient, read at the scale `rateScale` (agglomerateRateScale); 0 below the cohesive stress. */
export function agglomerateLossRate(rawRate: number, m: FlowCurve, a: AgglomerateConstants, rateScale = 1): number {
  const rate = rawRate * rateScale;
  if (!(a.cohesiveStress > 0) || !(rate > 0) || !Number.isFinite(rate)) return 0;
  const tau = flowStress(rate, m);
  let loss = a.erosionRate * Math.max(tau / a.cohesiveStress - 1, 0);
  if (a.ruptureStress > 0) loss += a.ruptureRate * Math.max(tau / a.ruptureStress - 1, 0);
  loss *= rate;
  return Number.isFinite(loss) ? Math.max(loss, 0) : 0;
}

/** The undispersed load after `dt` at the shear rate `rate`. */
export function agglomerateStep(undispersed: number, rate: number, dt: number, m: FlowCurve, a: AgglomerateConstants, rateScale = 1): number {
  if (!(undispersed > 0)) return undispersed;
  const loss = agglomerateLossRate(rate, m, a, rateScale);
  return loss > 0 ? undispersed * Math.exp(-loss * dt) : undispersed;
}

/** The load that tints: dispersed pigment in full, agglomerates at the share `weak` of their strength. */
export function effectiveLoad(load: number, undispersed: number, weak: number = AGGLOMERATE_TINT): number {
  const u = Math.min(Math.max(undispersed, 0), Math.max(load, 0));
  return Math.max(load, 0) - (1 - weak) * u;
}

export interface DispersionStats {
  /** total pigment load and the part of it still in agglomerates (sums over particles, x particle mass 1) */
  readonly totalLoad: number;
  readonly undispersedLoad: number;
  /** the batch's dispersed fraction, 1 - undispersedLoad / totalLoad (1 when there is no pigment) */
  readonly dispersed: number;
  /** the dispersed fraction 1 - u / w over the pigment, load-weighted: 10th, 50th and 90th percentile */
  readonly p10: number;
  readonly median: number;
  readonly p90: number;
  /** load-weighted shares of the pigment by dispersed fraction, in `bins` equal bins from 0 to 1 */
  readonly histogram: readonly number[];
  /** the share of the pigment that is still raw (dispersed fraction under 0.1) and that is developed (over 0.9) */
  readonly raw: number;
  readonly developed: number;
}

/** How far the batch's pigment is dispersed, from a particle snapshot. */
export function dispersionStats(snap: Pick<ParticleSnapshot, 'count' | 'loads'> & { readonly undispersed?: Float32Array }, bins = 10): DispersionStats {
  const items: [number, number][] = [];
  const histogram = new Array<number>(bins).fill(0);
  let total = 0, und = 0, raw = 0, developed = 0;
  for (let p = 0; p < snap.count; p++) {
    const w = snap.loads[p];
    if (!(w > 0) || !Number.isFinite(w)) continue;
    const uRaw = snap.undispersed ? snap.undispersed[p] : 0;
    const u = Number.isFinite(uRaw) ? Math.min(Math.max(uRaw, 0), w) : 0;
    const d = 1 - u / w;
    total += w;
    und += u;
    items.push([d, w]);
    histogram[Math.min(Math.floor(d * bins), bins - 1)] += w;
    if (d < 0.1) raw += w;
    if (d > 0.9) developed += w;
  }
  items.sort((a, b) => a[0] - b[0]);
  const quant = (f: number): number => {
    let acc = 0;
    for (const [d, w] of items) { acc += w; if (acc >= f * total) return d; }
    return 1;
  };
  const share = (v: number): number => (total > 0 ? v / total : 0);
  return {
    totalLoad: total,
    undispersedLoad: und,
    dispersed: total > 0 ? 1 - und / total : 1,
    p10: quant(0.1), median: quant(0.5), p90: quant(0.9),
    histogram: histogram.map(share),
    raw: share(raw),
    developed: share(developed)
  };
}
