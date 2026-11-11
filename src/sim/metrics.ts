/**
 * Pigment inventory and mixing metrics, computed on the CPU from a particle snapshot
 * (GpuMpmSim.readParticles). Tests and diagnostics; nothing in the frame loop calls these.
 *
 * Every particle has the same mass, so "mass-weighted" below is a plain count; the inventory
 * multiplies by the particle mass so its totals are in load x mass.
 */
import type { ParticleSnapshot } from './types';

/** The part of a snapshot these measures read. */
export type PigmentSnapshot = Pick<ParticleSnapshot, 'count' | 'positions' | 'loads' | 'latents'>;

export interface PigmentInventory {
  /** particles in the snapshot */
  readonly particles: number;
  /** particles with a pigment load above zero */
  readonly pigmented: number;
  /** sum over particles of load x mass: the conserved amount of pigment */
  readonly totalLoad: number;
  /** sum over particles of load x mass x latent, per latent channel (7): the conserved colour */
  readonly channels: readonly number[];
  /** channels / totalLoad: the batch's load-weighted mean latent (zeros when there is no pigment) */
  readonly meanLatent: readonly number[];
  /** smallest and largest particle load */
  readonly minLoad: number;
  readonly maxLoad: number;
}

/** Total pigment and its colour, summed in double precision. */
export function pigmentInventory(snap: PigmentSnapshot, particleMass = 1): PigmentInventory {
  const n = snap.count;
  const channels = [0, 0, 0, 0, 0, 0, 0];
  let total = 0;
  let pigmented = 0;
  let lo = n ? Infinity : 0;
  let hi = n ? -Infinity : 0;
  for (let p = 0; p < n; p++) {
    const w = snap.loads[p];
    if (w < lo) lo = w;
    if (w > hi) hi = w;
    if (!(w !== 0)) continue;   // a clear particle's latent has no weight (and NaN never passes for pigment)
    if (w > 0) pigmented++;
    total += w;
    for (let c = 0; c < 7; c++) channels[c] += w * snap.latents[7 * p + c];
  }
  total *= particleMass;
  for (let c = 0; c < 7; c++) channels[c] *= particleMass;
  const meanLatent = channels.map((v) => (total > 0 ? v / total : 0));
  return { particles: n, pigmented, totalLoad: total, channels, meanLatent, minLoad: lo, maxLoad: hi };
}

export interface SegregationAtScale {
  /** box edge in world units */
  readonly box: number;
  /** occupied boxes */
  readonly bins: number;
  /** Danckwerts intensity of segregation: variance of the box means over mean (1 - mean); 1 = segregated, 0 = uniform */
  readonly intensity: number;
  /** what the same index reads for a random mixture at this box size (Lacey's floor): the mean of 1 / particles per box */
  readonly floor: number;
}

export interface MixingMetrics {
  /** mean concentration: load / fullLoad averaged over the particles (0..1) */
  readonly mean: number;
  /** mean (1 - mean): the variance of a completely segregated two-level field of this mean */
  readonly sigma0Sq: number;
  /** intensity of segregation at each box size asked for, in that order */
  readonly scales: readonly SegregationAtScale[];
  /** axial segregation: variance over slabs along x of the slab's mean concentration, over sigma0Sq.
   *  The rolls alone barely change it; only the operator's moves mix along the roll. */
  readonly axial: number;
  /** slabs along x that hold material */
  readonly axialSlabs: number;
}

export interface MixingOptions {
  /** box edges in world units (default: h, 2h, 4h) */
  readonly boxes?: readonly number[];
  /** the load that counts as concentration 1 (PIGMENT_LOAD: neat masterbatch) */
  readonly fullLoad?: number;
  /** thickness of the axial slabs along x (default: h) */
  readonly slab?: number;
}

/** Mass-weighted variance of bin means about the global mean, given per-bin sums and counts. */
function binVariance(sums: Iterable<readonly [number, number]>, mean: number): { variance: number; bins: number; floor: number } {
  let acc = 0;
  let n = 0;
  let bins = 0;
  let inv = 0;
  for (const [sum, count] of sums) {
    const d = sum / count - mean;
    acc += count * d * d;
    n += count;
    bins++;
    inv += 1;   // count * (1 / count)
  }
  return { variance: n > 0 ? acc / n : 0, bins, floor: n > 0 ? inv / n : 0 };
}

/**
 * Danckwerts' intensity of segregation of the pigment load at several box sizes, and the axial
 * segregation number. The concentration c of a particle is load / fullLoad clamped to 0..1; a box's
 * concentration is the mean over its particles; averages are over particles (equal mass), so empty
 * boxes do not count. sigma0^2 = mean (1 - mean) is the variance before any mixing, when every
 * particle is either neat masterbatch or clear.
 *
 * `h` is the solver's cell (two particle spacings). The index depends on the box it is observed at,
 * hence the three sizes; a random mixture reads `floor`, not 0.
 */
export function mixingMetrics(snap: PigmentSnapshot, h: number, options: MixingOptions = {}): MixingMetrics {
  const boxes = options.boxes ?? [h, 2 * h, 4 * h];
  const full = options.fullLoad ?? 12;
  const slab = options.slab ?? h;
  const n = snap.count;
  const conc = new Float64Array(n);
  let total = 0;
  for (let p = 0; p < n; p++) {
    const c = Math.min(Math.max(snap.loads[p] / full, 0), 1);
    conc[p] = Number.isFinite(c) ? c : 0;
    total += conc[p];
  }
  const mean = n > 0 ? total / n : 0;
  const sigma0Sq = mean * (1 - mean);
  const index = (variance: number): number => (sigma0Sq > 1e-12 ? variance / sigma0Sq : 0);

  const scales: SegregationAtScale[] = [];
  for (const box of boxes) {
    const bins = new Map<string, [number, number]>();
    const inv = 1 / box;
    for (let p = 0; p < n; p++) {
      const key = `${Math.floor(snap.positions[3 * p] * inv)},${Math.floor(snap.positions[3 * p + 1] * inv)},${Math.floor(snap.positions[3 * p + 2] * inv)}`;
      const b = bins.get(key);
      if (b) { b[0] += conc[p]; b[1]++; } else bins.set(key, [conc[p], 1]);
    }
    const r = binVariance(bins.values(), mean);
    scales.push({ box, bins: r.bins, intensity: index(r.variance), floor: r.floor });
  }

  const slabs = new Map<number, [number, number]>();
  const invSlab = 1 / slab;
  for (let p = 0; p < n; p++) {
    const key = Math.floor(snap.positions[3 * p] * invSlab);
    const b = slabs.get(key);
    if (b) { b[0] += conc[p]; b[1]++; } else slabs.set(key, [conc[p], 1]);
  }
  const ax = binVariance(slabs.values(), mean);
  return { mean, sigma0Sq, scales, axial: index(ax.variance), axialSlabs: ax.bins };
}
