/**
 * Nip diagnostics (physics review §4.5, tests 1 to 6): CPU measurements on particle snapshots.
 * Pure TypeScript, no GPU; the app's test hook pairs it with the solver's event counters
 * (`window.__colormill.nipDiagnostics`, tools/perf/nip.mjs).
 *
 * What it answers: does the volume the constitutive law sees (J = det F) agree with the density
 * the particles actually have on the grid? Where the two part, pressure is computed from a
 * volume the material does not have.
 */
import { GEOMETRY, rollerPoses, type GridDims, type MillParams } from '../config/mill';

/** What nipDiagnostics reads of a ParticleSnapshot. */
export interface DiagSnapshot {
  readonly count: number;
  readonly positions: Float32Array;
  readonly velocities: Float32Array;
  readonly deformation: Float32Array;
  readonly flags: Uint32Array;
}

export interface Spread { readonly mean: number; readonly p5: number; readonly p95: number }

export interface RegionStats {
  /** particles in the region */
  readonly n: number;
  /** J = det F: the volume ratio the stress is computed from */
  readonly J: Spread;
  /** particle density gathered from the grid (quadratic weights), relative to rest (8 particles a cell) */
  readonly rho: Spread;
  /** the same with the rest mass the rolls displace added to the nodes (rollGhostMass): next to a roll
   *  the plain gather under-reads, since nodes inside the roll hold no particles */
  readonly rhoGhost: Spread;
}

export interface NipDiagnostics {
  /** sim seconds between the two snapshots */
  readonly seconds: number;
  /** F resets and roller push-outs per sim second (the solver's counters; NaN when not supplied) */
  readonly resetsPerSecond: number;
  readonly pushOutsPerSecond: number;
  /** the sheet: particles on the lower half of the front roll, in the middle of the roll's length */
  readonly sheet: RegionStats;
  /** the bank: particles between the roll axes above the nip channel */
  readonly bank: RegionStats;
  /** sheet particles per unit arc over those of a gap-thick sheet at rest density (calendering theory: 1.2 to 1.4) */
  readonly sheetCount: number;
  /** net particles through the nip plane per second over U_front · gap · rho0 · L */
  readonly flux: number;
  /** 90th percentile of the sheet particles' depth off the front roll, in gaps */
  readonly thicknessGaps: number;
  /** fastest free particle, and that over 2 omega R, twice the front roll's surface speed (a stable run stays under 1) */
  readonly maxSpeed: number;
  readonly maxSpeedOverLimit: number;
  /** (c + max |v|) dt / h, c the sound speed handed in (the measured limit is about 1) */
  readonly courant: number;
  /** particles below y = 3h (on the tray) */
  readonly tray: number;
  /** smallest and largest J over every free particle */
  readonly Jmin: number;
  readonly Jmax: number;
}

/** Integral of the quadratic B-spline from -inf to t (t in cells): the share of a node's kernel on one side of a plane. */
export function splineCdf(t: number): number {
  if (t <= -1.5) return 0;
  if (t < -0.5) return (t + 1.5) ** 3 / 6;
  if (t < 0.5) return 1 / 6 + 0.75 * (t + 0.5) - (t * t * t + 0.125) / 3;
  if (t < 1.5) return 1 - (1.5 - t) ** 3 / 6;
  return 1;
}

/**
 * Share (0..1) of a node's kernel that lies inside the rolls, for a node at signed distances sF, sB
 * (in cells, positive outside) from the front and back roll surfaces. The roll is taken as a plane
 * at the node (R is 20 to 46 cells) and the kernel as its one-dimensional profile along the normal.
 */
export function rollGhostShare(sF: number, sB: number): number {
  return Math.min(1, splineCdf(-sF) + splineCdf(-sB));
}

function spread(values: number[] | Float32Array): Spread {
  const n = values.length;
  if (n === 0) return { mean: NaN, p5: NaN, p95: NaN };
  const v = Float32Array.from(values).sort();
  let s = 0;
  for (let i = 0; i < n; i++) s += v[i];
  return { mean: s / n, p5: v[Math.floor(0.05 * (n - 1))], p95: v[Math.ceil(0.95 * (n - 1))] };
}

export interface DiagOptions {
  readonly dims: GridDims;
  readonly params: Pick<MillParams, 'gap' | 'omega' | 'frictionRatio'>;
  /** substep length */
  readonly dt: number;
  /** sim seconds between the snapshots */
  readonly seconds: number;
  /** sound speed for the Courant number */
  readonly soundSpeed: number;
  /** counter differences between the snapshots (optional) */
  readonly resets?: number;
  readonly pushOuts?: number;
}

/** The middle of the roll's length the sheet and flux measures use (clear of the ends of the batch). */
export const DIAG_X_BAND: readonly [number, number] = [0.3, 1.2];
/** The arc of sheet measured, radians round the front axis from the nip in the direction of rotation. */
export const DIAG_ARC: readonly [number, number] = [0.35, Math.PI];

/**
 * Measure the nip between two snapshots of the same particles (`a` earlier; state statistics are
 * taken on `b`, the flux from the crossings between them).
 */
export function nipDiagnostics(a: DiagSnapshot, b: DiagSnapshot, o: DiagOptions): NipDiagnostics {
  const { h, nx, ny, nz } = o.dims;
  const invh = 1 / h;
  const R = GEOMETRY.radius;
  const ay = GEOMETRY.axisY;
  const { back, front } = rollerPoses(o.params);
  const azF = front.axisZ, azB = back.axisZ;
  const gap = o.params.gap;
  const n = b.count;
  const pos = b.positions, F = b.deformation;

  // grid mass, as P2G scatters it (unit particle mass), then the roll's ghost mass on occupied nodes
  const mass = new Float32Array(nx * ny * nz);
  const w = [new Float64Array(3), new Float64Array(3), new Float64Array(3)];
  const base = [0, 0, 0];
  const weights = (p: number): void => {
    for (let c = 0; c < 3; c++) {
      const g = pos[3 * p + c] * invh;
      const b0 = Math.floor(g - 0.5);
      const f = g - b0;
      base[c] = b0;
      w[c][0] = 0.5 * (1.5 - f) * (1.5 - f);
      w[c][1] = 0.75 - (f - 1) * (f - 1);
      w[c][2] = 0.5 * (f - 0.5) * (f - 0.5);
    }
  };
  const free = (p: number): boolean => (b.flags[p] & 1) === 0;
  for (let p = 0; p < n; p++) {
    weights(p);
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
      mass[((base[2] + k) * ny + base[1] + j) * nx + base[0] + i] += w[0][i] * w[1][j] * w[2][k];
    }
  }
  const ghostYZ = new Float32Array(ny * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const y = j * h, z = k * h;
    ghostYZ[k * ny + j] = 8 * rollGhostShare((Math.hypot(y - ay, z - azF) - R) * invh, (Math.hypot(y - ay, z - azB) - R) * invh);
  }

  const sJ: number[] = [], sRho: number[] = [], sGhost: number[] = [], depth: number[] = [];
  const bJ: number[] = [], bRho: number[] = [], bGhost: number[] = [];
  let vmax = 0, tray = 0, Jmin = Infinity, Jmax = -Infinity;
  const tackReach = R + gap + 4 * h;
  for (let p = 0; p < n; p++) {
    const x = pos[3 * p], y = pos[3 * p + 1], z = pos[3 * p + 2];
    if (y < 3 * h) tray++;
    if (!free(p)) continue;
    const vx = b.velocities[3 * p], vy = b.velocities[3 * p + 1], vz = b.velocities[3 * p + 2];
    vmax = Math.max(vmax, Math.hypot(vx, vy, vz));
    const f = 9 * p;
    const J = F[f] * (F[f + 4] * F[f + 8] - F[f + 5] * F[f + 7]) - F[f + 1] * (F[f + 3] * F[f + 8] - F[f + 5] * F[f + 6])
      + F[f + 2] * (F[f + 3] * F[f + 7] - F[f + 4] * F[f + 6]);
    if (J < Jmin) Jmin = J;
    if (J > Jmax) Jmax = J;
    const dy = y - ay, dz = z - azF;
    const d = Math.hypot(dy, dz);
    let th = Math.atan2(-dy, -dz);   // 0 on the nip side, pi/2 at the bottom of the roll
    if (th < 0) th += 2 * Math.PI;
    const inBand = x > DIAG_X_BAND[0] && x < DIAG_X_BAND[1];
    const inSheet = inBand && d < tackReach && th > DIAG_ARC[0] && th < DIAG_ARC[1];
    const inBank = y > ay + 0.06 && z > azB && z < azF;
    if (!inSheet && !inBank) continue;
    weights(p);
    let rho = 0, rhoG = 0;
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) {
      const row = (base[2] + k) * ny + base[1] + j;
      const g = ghostYZ[row];
      for (let i = 0; i < 3; i++) {
        const wt = w[0][i] * w[1][j] * w[2][k];
        const m = mass[row * nx + base[0] + i];
        rho += wt * m;
        rhoG += wt * (m + g);
      }
    }
    if (inSheet) { sJ.push(J); sRho.push(rho / 8); sGhost.push(rhoG / 8); depth.push(d - R); }
    else { bJ.push(J); bRho.push(rho / 8); bGhost.push(rhoG / 8); }
  }

  // flux: net particles through the nip plane (y = axisY between the roll axes) between the snapshots
  let crossed = 0;
  const m = Math.min(a.count, b.count);
  for (let p = 0; p < m; p++) {
    const x = pos[3 * p], z = pos[3 * p + 2];
    if (!(x > DIAG_X_BAND[0] && x < DIAG_X_BAND[1]) || !(z > azB && z < azF)) continue;
    const za = a.positions[3 * p + 2];
    if (!(za > azB && za < azF)) continue;
    const ya = a.positions[3 * p + 1], yb = pos[3 * p + 1];
    if (ya >= ay && yb < ay) crossed++;
    else if (ya < ay && yb >= ay) crossed--;
  }
  const L = DIAG_X_BAND[1] - DIAG_X_BAND[0];
  const rho0 = 8 * invh * invh * invh;
  const Uf = Math.abs(front.omegaX) * R;
  const flux = o.seconds > 0 ? crossed / o.seconds / (Uf * gap * rho0 * L) : NaN;
  const restSheet = (DIAG_ARC[1] - DIAG_ARC[0]) * (R + gap / 2) * gap * L * rho0;
  depth.sort((p, q) => p - q);
  const limit = 2 * Uf;
  return {
    seconds: o.seconds,
    resetsPerSecond: o.resets === undefined ? NaN : o.resets / o.seconds,
    pushOutsPerSecond: o.pushOuts === undefined ? NaN : o.pushOuts / o.seconds,
    sheet: { n: sJ.length, J: spread(sJ), rho: spread(sRho), rhoGhost: spread(sGhost) },
    bank: { n: bJ.length, J: spread(bJ), rho: spread(bRho), rhoGhost: spread(bGhost) },
    sheetCount: sJ.length / restSheet,
    flux,
    thicknessGaps: depth.length ? depth[Math.floor(0.9 * (depth.length - 1))] / gap : NaN,
    maxSpeed: vmax,
    maxSpeedOverLimit: vmax / limit,
    courant: ((o.soundSpeed + vmax) * o.dt) / h,
    tray,
    Jmin,
    Jmax
  };
}
