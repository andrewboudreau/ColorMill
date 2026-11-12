/**
 * ColorMill v2 — mill configuration and geometry (pure TypeScript, no GPU).
 *
 * This module is the single source of truth for the mill's geometry, the
 * material constants, and the quality presets. See docs/design-v2.md §1, §4, §7.
 * Coordinates: +y up, +z toward the viewer, roller axes parallel to x.
 */

export type QualityPreset = 'low' | 'medium' | 'high' | 'ultra';
/** How cut & roll puts the wound roll back (design §6): 'double' folds it in half first (the default);
 *  'long' stands the full-length roll up as one and feeds it in, which may reach the ceiling. */
export type RollStyle = 'double' | 'long';

export interface QualitySettings {
  readonly preset: QualityPreset;
  /** grid cells per sim unit (h = 1 / cellsPerUnit) */
  readonly cellsPerUnit: number;
  /** fixed substep length in sim seconds */
  readonly dt: number;
  /** substeps per rendered frame (fixed, deterministic) */
  readonly substepsPerFrame: number;
  /** cells per sim unit of the colour raster the renderer reads pigment from, at its finest: twice
   *  the solver's, the particle spacing (finer would leave holes between particles). The sheet is
   *  one or two solver cells thick, and a streak inside it is lost at the solver's resolution; the
   *  mechanics stay on the solver grid. The raster covers a box around the material, not the
   *  domain (design §3.6), and its cell grows when the material outgrows the box colourNodes buys. */
  readonly colourCellsPerUnit: number;
  /** node budget of the colour raster's box: what its accumulators and textures are sized for
   *  (64 bytes a node; the 7-channel accumulator must fit the default 128 MiB binding limit) */
  readonly colourNodes: number;
  /** the nip gap this preset starts with: about 1.3 cells, the thinnest sheet the grid carries (the low
   *  preset's 0.04 is that already). A real mill's nip is far narrower for its rolls than 0.04 is for these;
   *  a finer grid can afford to be nearer to it, and the narrower the nip, the thinner the sheet and the wider
   *  a chunk's pigment is spread (design §7). */
  readonly gap: number;
  /** the batch this preset starts with. The nip passes a sheet about 1.3 gaps thick at rest density, so a narrower
   *  gap carries less of the batch round the roll and leaves more in the bank; past roughly 0.45 of the batch the
   *  bank's back half stalls and chunks are no longer drawn in (tools/perf/bank.mjs). These keep the bank near 0.3. */
  readonly batch: number;
}

export const QUALITY_PRESETS: Readonly<Record<QualityPreset, QualitySettings>> = {
  low: { preset: 'low', cellsPerUnit: 32, dt: 1.6e-3, substepsPerFrame: 8, colourCellsPerUnit: 64, colourNodes: 1.82e6, gap: 0.04, batch: 1 },
  medium: { preset: 'medium', cellsPerUnit: 48, dt: 1.1e-3, substepsPerFrame: 12, colourCellsPerUnit: 96, colourNodes: 3.53e6, gap: 0.028, batch: 0.65 },
  high: { preset: 'high', cellsPerUnit: 64, dt: 8e-4, substepsPerFrame: 16, colourCellsPerUnit: 128, colourNodes: 4.69e6, gap: 0.02, batch: 0.5 },
  ultra: { preset: 'ultra', cellsPerUnit: 72, dt: 7.1e-4, substepsPerFrame: 18, colourCellsPerUnit: 144, colourNodes: 4.69e6, gap: 0.02, batch: 0.5 }
};

export interface MaterialConstants {
  /** Young's modulus */
  readonly E: number;
  /** Poisson ratio */
  readonly nu: number;
  /** yield stress in shear, tau_y: below it the putty is an elastic solid (the bank holds its shape) */
  readonly yieldStress: number;
  /** Herschel-Bulkley consistency k: flowing at shear rate g the shear stress is tau_y + k g^n */
  readonly consistency: number;
  /** Herschel-Bulkley index n (0 < n <= 1; below 1 the putty shear-thins) */
  readonly flowIndex: number;
  /** density tie: det F may not exceed (1 + this) / (density gathered from the grid, rest = 1); 0 turns it off */
  readonly densityTolerance: number;
  /** the elastic volume ratio J = det F saturates at these bounds (the pressure K ln J with it) */
  readonly volumeMin: number;
  readonly volumeMax: number;
}

/*
 * Uncured silicone millbase is a soft yield-stress paste. E = 22 (unit
 * density) lets the bank slump into the nip's V as a rolling bank and keeps
 * the nip fed so the sheet on the front roll is continuous, while a pulled
 * sheet (cut & fold) holds together as a slab with the pigment chunks still
 * lumps; at 15 it sagged as a film and the chunks smeared, at E = 60 the
 * bank is a rigid slab that starves the nip and the sheet comes out lacy
 * (validated on the GPU at the low preset, gravity 2, see docs/design-v2.md §4).
 */
export const DEFAULT_MATERIAL: MaterialConstants = {
  E: 22,
  /* nearly incompressible. With E = 22, nu 0.35 / 0.45 / 0.47 / 0.49 is a bulk modulus K = E / 3(1 - 2 nu) of
     24 / 73 / 122 / 367 and a sound speed sqrt(K + 4 mu / 3) of 5.9 / 9.1 / 11.5 / 19.4, which is a Courant
     number c dt / h of 0.30 / 0.47 / 0.59 / 0.99 on every preset (soundSpeed and courantNumber below; the
     measured limit is in design §4, Stability). */
  nu: 0.47,
  /* the bank holds its shape under gravity 2 and a sheet pulled by its edge folds as a slab */
  yieldStress: 0.58,
  /* consistency and index: design §4 has the measurements that chose them */
  consistency: 0.45,
  flowIndex: 0.35,
  /* the density tie (g2p.wgsl; design §4, Volume): 0.05 / 0.1 / 0.2 leave a sheet count of 1.06 / 1.08 / 1.11
     at `low` and 1.35 / 1.34 / 1.43 at `high` (calendering theory: 1.2 to 1.4; 1.09 and 1.83 with it off) */
  densityTolerance: 0.1,
  /* with J free the cut & fold blows up 10% past the preset's time step, with [0.5, 2] the fed log of a cut &
     roll does at `high`; [0.7, 1.4] is clean at 1.2x the step on every preset (design §4, Stability). Steady
     milling does not reach the bounds: the sheet's J is 0.76 to 1.06 (5th to 95th percentile, `high`). */
  volumeMin: 0.7,
  volumeMax: 1.4
};

/**
 * Dispersive mixing (design §5, Agglomerates): fresh masterbatch is pigment in agglomerates, which only
 * develop colour once shear stress over their cohesive strength has broken them down. Stresses are in the
 * material's units (the flow curve's: yieldStress + consistency * rate^flowIndex).
 */
export interface AgglomerateConstants {
  /** cohesive stress tau_e: below it agglomerates do not erode at all (0 turns the kinetics off) */
  readonly cohesiveStress: number;
  /** erosion: the undispersed fraction lost per unit of shear strain, per unit of excess stress tau / tau_e - 1 */
  readonly erosionRate: number;
  /** rupture stress tau_c (> tau_e): over it a second, faster loss sets in; 0 = no rupture */
  readonly ruptureStress: number;
  /** rupture: loss per unit of shear strain, per unit of tau / tau_c - 1 */
  readonly ruptureRate: number;
}

/*
 * The cohesive stress is set from the stress the simulation's particles flow at (tools/perf/stress.mjs; design
 * §5 has the table). At `low`, by region, 10th / 50th / 90th / 99th percentile of the flow-curve stress:
 *   bank, equal roll speeds     1.02 / 1.27 / 1.64 / 1.76      (friction ratio 1.6: 1.06 / 1.26 / 1.66 / 1.95)
 *   wedge just over the nip     1.15 / 1.48 / 1.71 / 1.76      (friction ratio 1.6: 1.65 / 1.71 / 1.92 / 2.01)
 *   nip                         0.96 / 1.12 / 1.28 / 1.40      (friction ratio 1.6: 1.68 / 1.70 / 1.71 / 1.74)
 * 1.75 is the bank's 99th percentile at equal speeds: under it the bank and an even-speed nip do nothing, and
 * what erodes is the layer dragged along the rolls into the wedge, which the friction ratio loads. The erosion
 * rate is set so a batch at the default ratio is mostly developed within a minute of milling (0.84 at `low`).
 * Both constants are this project's choice, not measured material data.
 * Those stresses are at putty stiffness 1 (MillParams.stiffness, which multiplies the whole flow curve). The default
 * stiffness is 1.25, so the cohesive stress is 1.25 * 1.75: the same place in the bank's distribution, and the
 * same dispersion at the defaults as the table was measured with. The stiffness slider then moves the putty's
 * stress against it: softer putty disperses less, stiffer more.
 */
export const DEFAULT_AGGLOMERATE: AgglomerateConstants = {
  cohesiveStress: 2.19,
  erosionRate: 1,
  ruptureStress: 0,
  ruptureRate: 0
};

/** The gap the agglomerate kinetics' shear rates are referred to: the `low` preset's. */
export const AGGLOMERATE_REFERENCE_GAP = 0.04;

/**
 * The factor on a particle's shear rate before the agglomerate kinetics see it: the square root of the preset's
 * nominal gap over the reference gap. The presets narrow the gap with the cell (1.28 cells at each), so the
 * nip's strain rate doubles from `low` to `high` while the bank's hardly changes; that is numerics, and the same
 * recipe should disperse alike at every preset. Dividing the gap out in full made `high` disperse at a third of
 * `low`'s rate, not at all three times faster: the square root is the measured middle (design §5), not a
 * derivation. The gap the user sets (MillParams.gap) is not divided out: a tighter nip does disperse more.
 */
export function agglomerateRateScale(quality: Pick<QualitySettings, 'gap'>): number {
  return Math.sqrt(quality.gap / AGGLOMERATE_REFERENCE_GAP);
}

/** Lame parameters derived from E and nu. */
export function lameParameters(m: Pick<MaterialConstants, 'E' | 'nu'>): { mu: number; lambda: number } {
  const mu = m.E / (2 * (1 + m.nu));
  const lambda = (m.E * m.nu) / ((1 + m.nu) * (1 - 2 * m.nu));
  return { mu, lambda };
}

/** Bulk modulus K = E / 3(1 - 2 nu) = lambda + 2 mu / 3: the Kirchhoff pressure is K ln J (design §4). */
export function bulkModulus(m: Pick<MaterialConstants, 'E' | 'nu'>): number {
  return m.E / (3 * (1 - 2 * m.nu));
}

/**
 * Elastic (P-wave) speed at unit density, sqrt(K + 4 mu / 3) = sqrt(lambda + 2 mu). The Kirchhoff stress
 * is linear in Hencky strain, so the stiffness the grid sees per unit of particle mass, and with it this
 * speed, does not change with compression (design §4, Stability).
 */
export function soundSpeed(m: Pick<MaterialConstants, 'E' | 'nu'>): number {
  const { mu, lambda } = lameParameters(m);
  return Math.sqrt(lambda + 2 * mu);
}

/** (c + |v|) dt / h for a preset: the Courant number of the explicit step. */
export function courantNumber(q: { readonly dt: number; readonly cellsPerUnit: number }, m: Pick<MaterialConstants, 'E' | 'nu'>, speed = 0): number {
  return (soundSpeed(m) + speed) * q.dt * q.cellsPerUnit;
}

/** Live, user-adjustable mill parameters (all in sim units). */
export interface MillParams {
  /** front roller angular speed, rad/s */
  omega: number;
  /** back roller speed / front roller speed (1 = pure counter-rotation) */
  frictionRatio: number;
  /** nip opening between roller surfaces */
  gap: number;
  /** gravity, sim units / s^2 */
  gravity: number;
  /** how stiff the putty is: multiplies its strength in flow (the yield stress and the Herschel-Bulkley consistency,
   *  DEFAULT_MATERIAL's), not its elastic moduli, so the sound speed and with it the time step's margin are the
   *  same at every setting. The agglomerates' cohesive stress is the pigment's and does not move with it: a
   *  stiffer putty loads them harder and disperses them sooner. */
  stiffness: number;
  /** lamellar mixing: the exponent k of the exchange closure (design §5, src/sim/lamina.ts exchangeFraction). A
   *  particle whose pigment laminae are thinned to s below the particle spacing keeps (s / spacing)^k of its
   *  difference from its neighbours. 1 is the geometric value; 0 switches the exchange off. */
  dispersion: number;
  /** back-roller Coulomb friction coefficient */
  backFriction: number;
  /** front-roll adhesion layer thickness in cells (0 = auto: gap/h + 1) */
  tackCells: number;
  /** cut & roll: how the log goes back on the mill. 0 = let go of the whole log at once
   *  (dropped onto the nip); > 0 = lowered in end-first at this speed (sim units / s along
   *  the log axis), released only where it touches the pile the nip is eating. */
  logFeed: number;
  /** cut & fold: how many passes one press makes (each a cut at the crown, the edge lifted, pulled back over
   *  the mill and dropped), 1 to 3 */
  foldPasses: number;
  /** colour threads: the longest line a pigment particle's laminae are drawn as, in particle spacings; 0 draws particles as points */
  streak: number;
  /** pigment chunks: how long a medium chunk takes to feed into the bank, in seconds (it is held and sinks in,
   *  let go where it touches material; bigger chunks take longer in proportion); 0 lets a chunk go whole */
  chunkFeed: number;
  /** side guides: the open width between the two plates as a fraction of the roll length (1 = the plates stand
   *  at the ends of the rolls). The plates are symmetric about the middle of the rolls and slide at GUIDE_SPEED
   *  toward where this puts them (guideSpan, grid.wgsl). */
  guideWidth: number;
}

export const DEFAULT_PARAMS: Readonly<MillParams> = {
  omega: 3.0,
  frictionRatio: 1.25,
  gap: 0.04,
  gravity: 2.0,
  /* a quarter stiffer than the flow curve design §4 chose: the owner's choice. The bank and the sheet measure the same there
     (tools/perf/bank.mjs: `low` slows its bank from 1.5 up, `high` is unmoved to 2) */
  stiffness: 1.25,
  /* not tuned: 1 is what the geometry gives (a parcel one spacing wide that samples stripes s thick differs
     from the local mean in proportion to s / spacing). It is not a rate, so it does not depend on the nip's
     strain rates or on the preset */
  dispersion: 1,
  backFriction: 0.4,
  tackCells: 0,
  /* drop: lowering the log in released it a thin slice at a time, and the nip spread and
     tore each slice before the rest arrived; the whole log set down at once holds together */
  logFeed: 0,
  foldPasses: 2,
  streak: 12,
  /* off: a chunk is let go whole when it is set down. Fed over about 6 s (three revolutions of the front roll
     at the default speed) the same pigment is laid round the roll as a thin layer; let go whole it goes round
     as one patch a third to a half of the circumference long and stays one for many revolutions */
  chunkFeed: 0,
  /* the whole roll: the plates are the end guides */
  guideWidth: 1
};

export const PARAM_LIMITS: Readonly<Record<keyof MillParams, { min: number; max: number; step: number }>> = {
  omega: { min: 0, max: 6, step: 0.05 },
  frictionRatio: { min: 1, max: 1.6, step: 0.01 },
  gap: { min: 0.02, max: 0.1, step: 0.002 },
  gravity: { min: 0, max: 6, step: 0.1 },
  stiffness: { min: 0.5, max: 2, step: 0.05 },
  dispersion: { min: 0, max: 2, step: 0.05 },
  backFriction: { min: 0, max: 1, step: 0.02 },
  tackCells: { min: 0, max: 6, step: 0.1 },
  logFeed: { min: 0, max: 0.6, step: 0.05 },
  foldPasses: { min: 1, max: 3, step: 1 },
  streak: { min: 0, max: 12, step: 1 },
  chunkFeed: { min: 0, max: 15, step: 0.5 },
  /* 0.4 leaves a strip two medium chunks wide between the drop margins */
  guideWidth: { min: 0.4, max: 1, step: 0.02 }
};

/** Fixed geometry (sim units). */
export const GEOMETRY = {
  /** domain size (x, y, z); origin at (0,0,0). y leaves headroom above the rolls
      for the operator's rolled-up sheet standing over the nip while it is fed in,
      and for pigment chunks to drop in; z leaves room in front of the front roll
      for the operator's hands (the cut & fold flap swings out there). */
  domain: [1.5, 2.25, 2.0] as readonly [number, number, number],
  /** roller length along x == domain x */
  length: 1.5,
  /** roller radius */
  radius: 0.32,
  /** height of both roller axes */
  axisY: 0.55,
  /** z of the nip mid-plane */
  nipZ: 0.75,
  /** initial bank: half-width in z around nipZ */
  bankHalfDepth: 0.25,
  /** volume (sim units³) of the default batch (batch = 1): the material seeded
      in the pocket between the rolls and the bank above it. 0.15 units³ is
      about 1.2 L at UNIT_METRES; a gap-thick sheet around the front roll
      (2πR·gap per unit length) takes ~0.11 units³, leaving a modest bank. */
  bankVolume: 0.15,
  /** x margin the bank keeps from the side guides at seeding */
  bankEndMargin: 0.05,
  /** particles per axis per cell at seeding (8 per cell) */
  seedPerAxis: 2
} as const;

/** How fast a side guide slides along the rolls when the operator moves it (sim units / s per plate): a sixth of
    the roll's surface speed at the default setting, so the plate ploughs the bank in front of it and the putty
    has time to flow (the full travel from the roll's end to the narrowest setting takes three seconds). */
export const GUIDE_SPEED = 0.15;

/** The open width between the side guides as a fraction of the roll length, from a setting: within the slider's
    range whatever was asked for (there is no roll past its ends). */
export function guideOpen(width: number | undefined): number {
  const l = PARAM_LIMITS.guideWidth;
  return width !== undefined && Number.isFinite(width) ? Math.min(Math.max(width, l.min), l.max) : 1;
}

/** x of the two side guides' inner faces for an open width (fraction of the roll length), symmetric about the
    middle of the rolls; [0, length] at 1. */
export function guideSpan(width?: number): [number, number] {
  const half = 0.5 * guideOpen(width) * GEOMETRY.length;
  return [0.5 * GEOMETRY.length - half, 0.5 * GEOMETRY.length + half];
}

/** Top of the side guides (y): a little above the settled bank of the default batch when they stand at the
    ends, and higher the further in they are, since a batch pushed into a narrower strip stands higher (the top
    of the default batch seeded 1 / width as deep, which is more than the bank holds once the sheet has taken its
    share). Above this the plates do not hold the putty. */
export function guideTopY(width?: number): number {
  // by the hundredth of the width: the solver asks every substep while a plate slides
  const key = Math.round(guideOpen(width) * 100);
  let top = guideTopCache.get(key);
  if (top === undefined) {
    top = solveBankTop(100 / key, DEFAULT_PARAMS) + 0.15;
    guideTopCache.set(key, top);
  }
  return top;
}
const guideTopCache = new Map<number, number>();

export interface RollerPose {
  /** axis point (any point on the axis); axis direction is +x */
  readonly axisY: number;
  readonly axisZ: number;
  readonly radius: number;
  /** angular velocity about +x (rad/s); sign per design §1 */
  readonly omegaX: number;
}

/** Roller axes for a given gap and speed setting. */
export function rollerPoses(params: Pick<MillParams, 'gap' | 'omega' | 'frictionRatio'>): {
  back: RollerPose;
  front: RollerPose;
} {
  const R = GEOMETRY.radius;
  const half = R + params.gap / 2;
  return {
    back: { axisY: GEOMETRY.axisY, axisZ: GEOMETRY.nipZ - half, radius: R, omegaX: +params.omega * params.frictionRatio },
    front: { axisY: GEOMETRY.axisY, axisZ: GEOMETRY.nipZ + half, radius: R, omegaX: -params.omega }
  };
}

/** Surface velocity of a roller at point p (sim units / s): v = omega x (p - axis). */
export function rollerSurfaceVelocity(r: RollerPose, py: number, pz: number): [number, number, number] {
  // omega = (wx, 0, 0), rel = (0, dy, dz) -> omega x rel = (0, -wx*dz, wx*dy)
  const dy = py - r.axisY;
  const dz = pz - r.axisZ;
  return [0, -r.omegaX * dz, r.omegaX * dy];
}

/** Distance from a point (y,z) to a roller axis. */
export function rollerAxisDistance(r: RollerPose, py: number, pz: number): number {
  const dy = py - r.axisY;
  const dz = pz - r.axisZ;
  return Math.hypot(dy, dz);
}

export interface GridDims {
  /** cell size */
  readonly h: number;
  /** node counts per axis (cells + 1) */
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly nodeCount: number;
}

/** Node grid for a quality preset: N = round(domain / h) + 1 per axis. */
export function gridDims(q: QualitySettings): GridDims {
  return gridDimsFor(q.cellsPerUnit);
}

/**
 * Node counts of the colour raster's box (design §3.5 / §3.6) and its finest cell: the whole
 * roll length along x, and as much of the domain's height and depth as colourNodes buys (both
 * scaled alike), all of it when the budget allows. The box follows the material; when the
 * material needs more than these nodes at the finest cell, the cell grows (box.wgsl).
 */
export function colourGridDims(q: QualitySettings): GridDims {
  const full = gridDimsFor(q.colourCellsPerUnit);
  if (full.nodeCount <= q.colourNodes) return full;
  const s = Math.sqrt(q.colourNodes / full.nodeCount);
  const ny = Math.max(8, Math.floor(full.ny * s));
  const nz = Math.max(8, Math.floor(full.nz * s));
  return { h: full.h, nx: full.nx, ny, nz, nodeCount: full.nx * ny * nz };
}

function gridDimsFor(cellsPerUnit: number): GridDims {
  const h = 1 / cellsPerUnit;
  const nx = Math.round(GEOMETRY.domain[0] / h) + 1;
  const ny = Math.round(GEOMETRY.domain[1] / h) + 1;
  const nz = Math.round(GEOMETRY.domain[2] / h) + 1;
  return { h, nx, ny, nz, nodeCount: nx * ny * nz };
}

/** Linear node index for node (i, j, k). */
export function nodeIndex(d: GridDims, i: number, j: number, k: number): number {
  return (k * d.ny + j) * d.nx + i;
}

/**
 * Height of the top surface of the mill at depth z: the roller surface under
 * z, or the nip floor (axisY) inside the gap. Used for seeding the bank.
 */
export function millTopSurfaceY(z: number, params: Pick<MillParams, 'gap' | 'omega' | 'frictionRatio'>): number {
  const { back, front } = rollerPoses(params);
  const R = GEOMETRY.radius;
  let y: number = GEOMETRY.axisY;
  for (const r of [back, front]) {
    const dz = z - r.axisZ;
    if (Math.abs(dz) < R) y = Math.max(y, r.axisY + Math.sqrt(R * R - dz * dz));
  }
  return y;
}

type BankParams = Pick<MillParams, 'gap' | 'omega' | 'frictionRatio'> & Partial<Pick<MillParams, 'guideWidth'>>;

/**
 * Volume (sim units³) of the bank slab between the mill top surface and y = top,
 * over the bank's x/z footprint (a 1-D quadrature over z of the free height).
 */
export function bankVolumeBelow(top: number, params: BankParams): number {
  const z0 = GEOMETRY.nipZ - GEOMETRY.bankHalfDepth;
  const z1 = GEOMETRY.nipZ + GEOMETRY.bankHalfDepth;
  const n = 400;
  const dz = (z1 - z0) / n;
  let area = 0;
  for (let i = 0; i < n; i++) {
    const z = z0 + (i + 0.5) * dz;
    area += Math.max(0, top - millTopSurfaceY(z, params)) * dz;
  }
  return area * (GEOMETRY.length - 2 * GEOMETRY.bankEndMargin);
}

// Geometry depends only on batch volume and gap. Keep a small bounded cache so
// substep uniforms and the renderer do not repeat the numerical integration.
const bankTopCache = new Map<string, number>();

/**
 * Top of the seeded bank for a batch size: the height at which the bank slab
 * holds `batch × GEOMETRY.bankVolume` of material. Batch is a volume (mass)
 * multiplier, not a height multiplier, so 2× really is twice the putty.
 */
export function bankTopY(batch: number, params: BankParams = DEFAULT_PARAMS): number {
  const key = `${batch}:${params.gap}`;
  const cached = bankTopCache.get(key);
  if (cached !== undefined) return cached;
  const top = solveBankTop(batch, params);
  if (bankTopCache.size >= 32) bankTopCache.delete(bankTopCache.keys().next().value!);
  bankTopCache.set(key, top);
  return top;
}

function solveBankTop(batch: number, params: BankParams): number {
  const target = GEOMETRY.bankVolume * Math.max(batch, 0);
  let lo: number = GEOMETRY.axisY;
  let hi: number = GEOMETRY.domain[1];
  for (let i = 0; i < 48; i++) {
    const mid = 0.5 * (lo + hi);
    if (bankVolumeBelow(mid, params) < target) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/** Litres of putty in a batch (HUD/panel readout). */
export function batchLitres(batch: number): number {
  return GEOMETRY.bankVolume * batch * UNIT_METRES ** 3 * 1000;
}

/** Axis-aligned bounds of the initial bank (x, y, z ranges): between the side guides along the roll. The batch
    is a charge for the whole roll length (bankTopY), so a narrower strip is seeded to the same height with
    proportionally less putty: the bank and the sheet are the same in section at any guide width. */
export function bankBounds(params: BankParams, batch = 1): {
  x: [number, number];
  y: [number, number];
  z: [number, number];
} {
  const top = bankTopY(batch, params);
  const [x0, x1] = guideSpan(params.guideWidth);
  return {
    x: [x0 + GEOMETRY.bankEndMargin, x1 - GEOMETRY.bankEndMargin],
    y: [GEOMETRY.axisY, top],
    z: [GEOMETRY.nipZ - GEOMETRY.bankHalfDepth, GEOMETRY.nipZ + GEOMETRY.bankHalfDepth]
  };
}

/**
 * Deterministic seeding of the initial bank: jittered lattice, seedPerAxis
 * per axis per cell, skipping points inside either roller or below the mill
 * top surface. Returns a flat xyz array. Pure and testable (used by the GPU
 * sim to fill its position buffer and by tests to check counts).
 */
export function seedBankPositions(q: QualitySettings, params: MillParams, seed = 1234, batch = 1): Float32Array {
  const { h } = gridDims(q);
  const per = GEOMETRY.seedPerAxis;
  const b = bankBounds(params, batch);
  const { back, front } = rollerPoses(params);
  const R = GEOMETRY.radius;
  const out: number[] = [];
  let s = seed >>> 0;
  const rnd = (): number => {
    // xorshift32
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return (s >>> 0) / 4294967296;
  };
  const cx0 = Math.floor(b.x[0] / h), cx1 = Math.ceil(b.x[1] / h);
  const cy0 = Math.floor(b.y[0] / h), cy1 = Math.ceil(b.y[1] / h);
  const cz0 = Math.floor(b.z[0] / h), cz1 = Math.ceil(b.z[1] / h);
  const jitter = 0.35;
  for (let cz = cz0; cz < cz1; cz++) {
    for (let cy = cy0; cy < cy1; cy++) {
      for (let cx = cx0; cx < cx1; cx++) {
        for (let sz = 0; sz < per; sz++) {
          for (let sy = 0; sy < per; sy++) {
            for (let sx = 0; sx < per; sx++) {
              const x = (cx + (sx + 0.5 + (rnd() - 0.5) * jitter) / per) * h;
              const y = (cy + (sy + 0.5 + (rnd() - 0.5) * jitter) / per) * h;
              const z = (cz + (sz + 0.5 + (rnd() - 0.5) * jitter) / per) * h;
              if (x < b.x[0] || x > b.x[1] || z < b.z[0] || z > b.z[1] || y > b.y[1]) continue;
              if (y < millTopSurfaceY(z, params) + 0.25 * h) continue;
              // not inside the nip channel itself: material seeded in the throat
              // over-fills the first wrap (1.5x rest) and the wrap then runs short
              // and leaves a bare stretch of roll for the first few seconds
              if (y < GEOMETRY.axisY + 0.06 && Math.abs(z - GEOMETRY.nipZ) < params.gap) continue;
              if (rollerAxisDistance(back, y, z) < R + 0.25 * h) continue;
              if (rollerAxisDistance(front, y, z) < R + 0.25 * h) continue;
              out.push(x, y, z);
            }
          }
        }
      }
    }
  }
  return Float32Array.from(out);
}

/** Extra particle capacity, as a fraction of the seeded bank, reserved for pigment chunks. */
export const PIGMENT_POOL_FRACTION = 0.5;

/** Radius of one pigment chunk (a dollop of coloured putty dropped on the bank): the medium size. */
export const PIGMENT_CHUNK_RADIUS = 0.1;

/** Chunk sizes the operator can pick (radius in sim units; the medium one is PIGMENT_CHUNK_RADIUS).
    A large chunk holds about 2.7x the particles of a medium one, a small one about a fifth. */
export const PIGMENT_CHUNK_SIZES = [
  { key: 's', label: 'Small', radius: 0.06 },
  { key: 'm', label: 'Medium', radius: PIGMENT_CHUNK_RADIUS },
  { key: 'l', label: 'Large', radius: 0.14 }
] as const;
export const DEFAULT_CHUNK_SIZE = 1;

/** Number of fixed pigment drop positions along the roll. A tap sets its chunk
    down over the nip at the chosen slot (left to right as the viewer sees the
    mill) rather than at a random x, so an operator can lay stripes on purpose
    and stack chunks on top of each other. */
export const DROP_SLOTS = 6;

/** x of drop slot `slot` (0-based, clamped): the slots are evenly spaced
    between margins that keep a whole chunk clear of the side guides, which
    stand `guideWidth` of the roll length apart (the ends of the rolls at 1). */
export function dropSlotX(slot: number, slots = DROP_SLOTS, guideWidth = 1): number {
  const margin = PIGMENT_CHUNK_RADIUS + 0.05;
  const i = Math.min(slots - 1, Math.max(0, Math.round(slot)));
  const [x0, x1] = guideSpan(guideWidth);
  return x0 + margin + ((i + 0.5) * (x1 - x0 - 2 * margin)) / slots;
}

/** Estimated particle count for a preset (batch volume / (h^3/8)), for UI/preset selection. */
export function estimateParticleCount(q: QualitySettings, batch = q.batch): number {
  const { h } = gridDims(q);
  return Math.round((GEOMETRY.bankVolume * batch) / (h * h * h / 8));
}

/** Convert roller angular speed to the rpm shown in the HUD. */
export function omegaToRpm(omega: number): number {
  return (omega * 60) / (2 * Math.PI);
}

export interface MillConfig {
  readonly quality: QualitySettings;
  readonly material: MaterialConstants;
  /** pigment agglomerates (DEFAULT_AGGLOMERATE when absent) */
  readonly agglomerate?: AgglomerateConstants;
  readonly params: MillParams;
  /** batch size: volume multiplier on the seeded bank (1 = GEOMETRY.bankVolume) */
  readonly batch?: number;
}

export const BATCH_CHOICES: readonly number[] = [0.5, 0.65, 0.75, 1, 1.5, 2, 3];
/** Range accepted from `?batch=` (the panel offers BATCH_CHOICES). */
export const BATCH_LIMITS = { min: 0.25, max: 4 } as const;

/** Physical scale used for the HUD readout only: one sim unit is about 0.2 m. */
export const UNIT_METRES = 0.2;
/** Uncured silicone putty is ~1.1 kg per litre. */
export const SILICONE_KG_PER_LITRE = 1.1;

/** Litres of material for a particle count at cell size h (rest volume h^3/8 per particle). */
export function materialLitres(particles: number, h: number): number {
  const units3 = (particles * h * h * h) / 8;
  return units3 * UNIT_METRES ** 3 * 1000;
}

export function defaultConfig(preset: QualityPreset = 'high'): MillConfig {
  return {
    quality: QUALITY_PRESETS[preset],
    material: DEFAULT_MATERIAL,
    params: { ...DEFAULT_PARAMS, gap: QUALITY_PRESETS[preset].gap },
    batch: QUALITY_PRESETS[preset].batch
  };
}
