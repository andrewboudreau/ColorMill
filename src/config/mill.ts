/**
 * ColorMill v2 — mill configuration and geometry (pure TypeScript, no GPU).
 *
 * This module is the single source of truth for the mill's geometry, the
 * material constants, and the quality presets. See docs/design-v2.md §1, §4, §7.
 * Coordinates: +y up, +z toward the viewer, roller axes parallel to x.
 */

export type QualityPreset = 'low' | 'medium' | 'high' | 'ultra';

export interface QualitySettings {
  readonly preset: QualityPreset;
  /** grid cells per sim unit (h = 1 / cellsPerUnit) */
  readonly cellsPerUnit: number;
  /** fixed substep length in sim seconds */
  readonly dt: number;
  /** substeps per rendered frame (fixed, deterministic) */
  readonly substepsPerFrame: number;
}

export const QUALITY_PRESETS: Readonly<Record<QualityPreset, QualitySettings>> = {
  low: { preset: 'low', cellsPerUnit: 32, dt: 1.6e-3, substepsPerFrame: 8 },
  medium: { preset: 'medium', cellsPerUnit: 48, dt: 1.1e-3, substepsPerFrame: 12 },
  high: { preset: 'high', cellsPerUnit: 64, dt: 8e-4, substepsPerFrame: 16 },
  ultra: { preset: 'ultra', cellsPerUnit: 72, dt: 7.1e-4, substepsPerFrame: 18 }
};

export interface MaterialConstants {
  /** Young's modulus */
  readonly E: number;
  /** Poisson ratio */
  readonly nu: number;
  /** plastic compression threshold (singular values clamped to >= 1 - thetaC) */
  readonly thetaC: number;
  /** plastic stretch threshold (singular values clamped to <= 1 + thetaS) */
  readonly thetaS: number;
}

/*
 * Uncured silicone millbase is a soft yield-stress paste. E = 15 (unit
 * density) lets the bank slump into the nip's V as a rolling bank and keeps
 * the nip fed so the sheet on the front roll is continuous; at E = 60 the
 * bank is a rigid slab that starves the nip and the sheet comes out lacy
 * (validated on the GPU at the low preset, gravity 2, see docs/design-v2.md §4).
 */
export const DEFAULT_MATERIAL: MaterialConstants = {
  E: 15,
  /* nearly incompressible: at 0.35 the nip packed the putty to 1.5x rest density
     and carried 50% more through than the gap allows; at 0.45 it still packed
     the sheet to 1.5x once the viscoplastic drag pulled the bank in (and the
     wrap ran short); at 0.49 (bulk modulus ~500) the sheet came out at
     ~1.0-1.3x rest but the fed pile of a cut & roll blew up (speeds of 10-25,
     dt at ~0.85 of the elastic CFL limit); 0.47 (bulk modulus ~80) is stable
     through a cut & roll. */
  nu: 0.47,
  thetaC: 0.025,
  /* tensile cohesion: at 0.0075 the sheet re-forming after an operator move
     comes out lacy; 0.03 keeps it continuous (validated headlessly) */
  thetaS: 0.03
};

/** Lamé parameters derived from E and nu. */
export function lameParameters(m: MaterialConstants): { mu: number; lambda: number } {
  const mu = m.E / (2 * (1 + m.nu));
  const lambda = (m.E * m.nu) / ((1 + m.nu) * (1 - 2 * m.nu));
  return { mu, lambda };
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
  /** shear-driven dispersion rate k (see design §5) */
  dispersion: number;
  /** back-roller Coulomb friction coefficient */
  backFriction: number;
  /** front-roll adhesion layer thickness in cells (0 = auto: gap/h + 1) */
  tackCells: number;
}

export const DEFAULT_PARAMS: Readonly<MillParams> = {
  omega: 3.0,
  frictionRatio: 1.25,
  gap: 0.04,
  gravity: 2.0,
  /* nearly off: colour should mill as distinct streaks that thin with folding,
     not fade to a tint (measured: at 0.1 a black chunk is 77% grey after 3 s of
     milling; at 0 it stays black and streaks around the roll) */
  dispersion: 0.02,
  backFriction: 0.4,
  tackCells: 0
};

export const PARAM_LIMITS: Readonly<Record<keyof MillParams, { min: number; max: number; step: number }>> = {
  omega: { min: 0, max: 6, step: 0.05 },
  frictionRatio: { min: 1, max: 1.6, step: 0.01 },
  gap: { min: 0.02, max: 0.1, step: 0.002 },
  gravity: { min: 0, max: 6, step: 0.1 },
  dispersion: { min: 0, max: 0.5, step: 0.005 },
  backFriction: { min: 0, max: 1, step: 0.02 },
  tackCells: { min: 0, max: 6, step: 0.1 }
};

/** Fixed geometry (sim units). */
export const GEOMETRY = {
  /** domain size (x, y, z); origin at (0,0,0). y leaves headroom above the rolls
      for the operator's rolled-up sheet standing over the nip while it is fed in,
      and for pigment chunks to drop in. */
  domain: [1.5, 2.25, 1.5] as readonly [number, number, number],
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
  /** x margin the bank keeps from the end guides at seeding */
  bankEndMargin: 0.05,
  /** particles per axis per cell at seeding (8 per cell) */
  seedPerAxis: 2
} as const;

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
  const h = 1 / q.cellsPerUnit;
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

type BankParams = Pick<MillParams, 'gap' | 'omega' | 'frictionRatio'>;

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

/**
 * Top of the seeded bank for a batch size: the height at which the bank slab
 * holds `batch × GEOMETRY.bankVolume` of material. Batch is a volume (mass)
 * multiplier, not a height multiplier, so 2× really is twice the putty.
 */
export function bankTopY(batch: number, params: BankParams = DEFAULT_PARAMS): number {
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

/** Axis-aligned bounds of the initial bank (x, y, z ranges). */
export function bankBounds(params: BankParams, batch = 1): {
  x: [number, number];
  y: [number, number];
  z: [number, number];
} {
  const top = bankTopY(batch, params);
  return {
    x: [GEOMETRY.bankEndMargin, GEOMETRY.length - GEOMETRY.bankEndMargin],
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

/** Radius of one pigment chunk (a dollop of coloured putty dropped on the bank). */
export const PIGMENT_CHUNK_RADIUS = 0.1;

/** Number of fixed pigment drop positions along the roll. A tap sets its chunk
    down over the nip at the chosen slot (left to right as the viewer sees the
    mill) rather than at a random x, so an operator can lay stripes on purpose
    and stack chunks on top of each other. */
export const DROP_SLOTS = 6;

/** x of drop slot `slot` (0-based, clamped): the slots are evenly spaced
    between margins that keep a whole chunk clear of the end guides. */
export function dropSlotX(slot: number, slots = DROP_SLOTS): number {
  const margin = PIGMENT_CHUNK_RADIUS + 0.05;
  const i = Math.min(slots - 1, Math.max(0, Math.round(slot)));
  return margin + ((i + 0.5) * (GEOMETRY.length - 2 * margin)) / slots;
}

/** Estimated particle count for a preset (batch volume / (h^3/8)), for UI/preset selection. */
export function estimateParticleCount(q: QualitySettings, batch = 1): number {
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
  readonly params: MillParams;
  /** batch size: volume multiplier on the seeded bank (1 = GEOMETRY.bankVolume) */
  readonly batch?: number;
}

export const BATCH_CHOICES: readonly number[] = [0.5, 0.75, 1, 1.5, 2, 3];
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
    params: { ...DEFAULT_PARAMS }
  };
}
