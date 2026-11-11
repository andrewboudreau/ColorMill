/**
 * Lamellar (striation) mixing model: the CPU mirror of src/sim/shaders/lamina.wgsl and of the exchange
 * closure in disperse.wgsl. Tests and diagnostics; nothing in the frame loop calls these.
 *
 * Pigment in a particle is a stack of laminae with unit normal n and thickness s. Both are carried
 * by the flow. With L the velocity gradient (L[r][c] = dv_r / dx_c; the solver's APIC matrix C):
 *
 *   A material surface element with area vector a obeys Nanson's relation a = J F^-T a0, so its
 *   direction m = F^-T n0 obeys dm/dt = -L^T m, and for the unit normal n = m / |m|:
 *       dn/dt = -(I - n n^T) L^T n
 *   Two neighbouring material surfaces a material vector dx apart are s = n . dx apart along the
 *   normal; d(dx)/dt = L dx, so
 *       d(ln s)/dt = n^T L n = n^T D n,   D = sym(L)
 *   Equivalently s / s0 = 1 / |F^-T n0| for an incompressible motion: d(ln |m|)/dt = -n^T L n.
 *   Volume change is not mixing, so L is replaced by its deviatoric part L' = L - tr(L)/3 I
 *   (the same n, and s without the isotropic factor J^(1/3)).
 *
 * One step integrates m, not n and s separately, so the pair stays consistent:
 *       m = (I - dt L'^T + dt^2/2 L'^T L'^T) n      (exp(-dt L'^T) to second order)
 *       q <- q - ln |m|,   n <- m / |m|             (q = ln(s / s0))
 * Exact for simple shear (L'^T is nilpotent); a rotation by theta a step changes q by theta^4 / 8.
 */

export type Vec3 = readonly [number, number, number];
/** Row-major 3x3: M[3 r + c]. */
export type Mat3 = ArrayLike<number>;

export interface LaminaState {
  /** unit normal of the laminae */
  readonly n: Vec3;
  /** q = ln(s / s0): the log of the striation thickness over the thickness the pigment arrived with */
  readonly q: number;
}

/** q is kept in this range (lamina.wgsl): beyond 1e-5 of the blob's size nothing reads it. */
export const LAMINA_Q_MIN = -12;
export const LAMINA_Q_MAX = 2;

/** One step of the lamina kinematics over dt under the velocity gradient L (row-major, the solver's C). */
export function laminaStep(state: LaminaState, L: Mat3, dt: number): LaminaState {
  const tr = (L[0] + L[4] + L[8]) / 3;
  // A = L'^T (row-major): A[r][c] = L'[c][r]
  const A = [L[0] - tr, L[3], L[6], L[1], L[4] - tr, L[7], L[2], L[5], L[8] - tr];
  const mul = (v: Vec3): Vec3 => [
    A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
    A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
  const n = state.n;
  const An = mul(n);
  const AAn = mul(An);
  const m: Vec3 = [
    n[0] - dt * An[0] + 0.5 * dt * dt * AAn[0],
    n[1] - dt * An[1] + 0.5 * dt * dt * AAn[1],
    n[2] - dt * An[2] + 0.5 * dt * dt * AAn[2]];
  const len = Math.hypot(m[0], m[1], m[2]);
  if (!(len > 1e-12) || !Number.isFinite(len)) return state;
  const q = Math.min(Math.max(state.q - Math.log(len), LAMINA_Q_MIN), LAMINA_Q_MAX);
  return { n: [m[0] / len, m[1] / len, m[2] / len], q };
}

/**
 * The exchange closure (disperse.wgsl): the share of its pigment a particle trades with its
 * neighbours this frame.
 *
 * A particle is a parcel one particle spacing (delta) across. While its laminae are thicker than
 * that, advection carries them and nothing is exchanged. Once s < delta the parcel is striped, and
 * a box delta wide that samples stripes s thick differs from the local mean in proportion to
 * s / delta. So the particle's deviation from its neighbourhood should be (sMin / delta)^k of what
 * it was when its laminae crossed delta, with sMin the thinnest they have been: blending is not
 * undone when the flow thickens a lamina again, and thinning it a second time to the same
 * thickness mixes nothing new. With `mixed` the log thickness already accounted for (the smaller
 * of ln delta and every earlier ln s), a step to a new minimum trades
 *       beta = 1 - (s / exp(mixed))^k
 * and the product of (1 - beta) over the steps is (sMin / delta)^k whatever the step length.
 * k (MillParams.dispersion) is 1 for the geometric argument above; it is the model's only number.
 *
 * @param lnS      ln of the striation thickness now (world units)
 * @param mixed    ln of the thickness the particle is already blended down to
 * @param lnDelta  ln of the particle spacing
 */
export function exchangeFraction(lnS: number, mixed: number, lnDelta: number, k = 1): number {
  const ref = Math.min(mixed, lnDelta);
  if (!(lnS < ref) || !(k > 0)) return 0;
  return Math.min(Math.max(1 - Math.exp(k * (lnS - ref)), 0), 1);
}

/** The reference the next step measures from: the smaller of the old one and the thickness now. */
export function mixedAfter(lnS: number, mixed: number): number {
  return Math.min(mixed, lnS);
}

export interface StriationStats {
  /** particles that carry pigment (load above minLoad) */
  readonly pigmented: number;
  /** load-weighted quantiles of the striation thickness s, world units */
  readonly p10: number;
  readonly median: number;
  readonly p90: number;
  /** load-weighted mean of ln(s0 / s): the log of how far the laminae have been thinned */
  readonly meanLogThinning: number;
  /** share of the pigment whose laminae are thinner than the particle spacing (it has begun to blend) */
  readonly belowSpacing: number;
  /** share of the pigment whose laminae are thinner than `visible` (default 1/10 of the spacing): blended to the eye */
  readonly blended: number;
}

/** The part of a snapshot the striation statistics read. */
export interface LaminaSnapshot {
  readonly count: number;
  readonly loads: Float32Array;
  /** 4 floats per particle: n xyz, q = ln(s / s0) */
  readonly laminae: Float32Array;
  /** 2 floats per particle: ln s0, ln of the thickness already blended down to */
  readonly mixState: Float32Array;
}

/** Load-weighted statistics of the striation thickness over the pigment-carrying particles. */
export function striationStats(snap: LaminaSnapshot, spacing: number, options: { minLoad?: number; visible?: number } = {}): StriationStats {
  const minLoad = options.minLoad ?? 0;
  const visible = options.visible ?? 0.1 * spacing;
  const items: [number, number][] = [];
  let total = 0, thin = 0, below = 0, blended = 0;
  for (let p = 0; p < snap.count; p++) {
    const w = snap.loads[p];
    if (!(w > minLoad)) continue;
    const q = snap.laminae[4 * p + 3];
    const lnS = q + snap.mixState[2 * p];
    if (!Number.isFinite(lnS)) continue;
    const s = Math.exp(lnS);
    items.push([s, w]);
    total += w;
    thin += -q * w;
    if (s < spacing) below += w;
    if (s < visible) blended += w;
  }
  if (!items.length || !(total > 0)) return { pigmented: 0, p10: NaN, median: NaN, p90: NaN, meanLogThinning: 0, belowSpacing: 0, blended: 0 };
  items.sort((a, b) => a[0] - b[0]);
  const quantile = (f: number): number => {
    let acc = 0;
    for (const [s, w] of items) { acc += w; if (acc >= f * total) return s; }
    return items[items.length - 1][0];
  };
  return {
    pigmented: items.length, p10: quantile(0.1), median: quantile(0.5), p90: quantile(0.9),
    meanLogThinning: thin / total, belowSpacing: below / total, blended: blended / total
  };
}
