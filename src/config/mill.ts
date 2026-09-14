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
  nu: 0.35,
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
  backFriction: 0.4
};

export const PARAM_LIMITS: Readonly<Record<keyof MillParams, { min: number; max: number; step: number }>> = {
  omega: { min: 0, max: 6, step: 0.05 },
  frictionRatio: { min: 1, max: 1.6, step: 0.01 },
  gap: { min: 0.02, max: 0.1, step: 0.002 },
  gravity: { min: 0, max: 6, step: 0.1 },
  dispersion: { min: 0, max: 0.5, step: 0.005 },
  backFriction: { min: 0, max: 1, step: 0.02 }
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
  /** initial bank: half-width in z around nipZ, and height above the roller top */
  bankHalfDepth: 0.25,
  /* tall enough that a gap-thick sheet around the front roll (2πR·gap per unit
     length) leaves a bank in front of the nip instead of consuming it */
  bankHeight: 0.36,
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

/** Axis-aligned bounds of the initial bank (x, y, z ranges). */
export function bankBounds(params: Pick<MillParams, 'gap' | 'omega' | 'frictionRatio'>): {
  x: [number, number];
  y: [number, number];
  z: [number, number];
} {
  void params;
  const top = GEOMETRY.axisY + GEOMETRY.radius + GEOMETRY.bankHeight;
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
export function seedBankPositions(q: QualitySettings, params: MillParams, seed = 1234): Float32Array {
  const { h } = gridDims(q);
  const per = GEOMETRY.seedPerAxis;
  const b = bankBounds(params);
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
export const PIGMENT_CHUNK_RADIUS = 0.14;

/** Estimated particle count for a preset (bank volume / (h^3/8)), for UI/preset selection. */
export function estimateParticleCount(q: QualitySettings): number {
  const { h } = gridDims(q);
  const b = bankBounds(DEFAULT_PARAMS);
  // bank slab minus the roller caps under it; the caps remove roughly a third of the slab volume
  const vol = (b.x[1] - b.x[0]) * (b.y[1] - b.y[0]) * (b.z[1] - b.z[0]) * 0.72;
  return Math.round(vol / (h * h * h / 8));
}

/** Convert roller angular speed to the rpm shown in the HUD. */
export function omegaToRpm(omega: number): number {
  return (omega * 60) / (2 * Math.PI);
}

export interface MillConfig {
  readonly quality: QualitySettings;
  readonly material: MaterialConstants;
  readonly params: MillParams;
}

export function defaultConfig(preset: QualityPreset = 'high'): MillConfig {
  return {
    quality: QUALITY_PRESETS[preset],
    material: DEFAULT_MATERIAL,
    params: { ...DEFAULT_PARAMS }
  };
}
