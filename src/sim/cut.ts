/**
 * Drawn cut (a tear in the sheet on the front roll): pure geometry shared by the
 * solver (GpuMpm.cutAlong), the app's knife gesture and the tests. No GPU code.
 *
 * Sheet coordinates: `x` along the roll (as in the world), `s` the arc around the
 * front roll's axis measured at the roll surface from the crown (the top of the
 * roll), positive down the front face toward the viewer: s = R * atan2(dz, dy)
 * with (dy, dz) the offset from the front axis. The front face at axis height is
 * s = pi R / 2 (about 0.5), the bottom of the roll s = pi R. The front roll turns
 * so that material on the front face moves up, toward smaller s.
 */
import { GEOMETRY, rollerPoses, type MillParams } from '../config/mill';

/** A point on the sheet: [x along the roll, s around the front roll from the crown]. */
export type SheetPoint = readonly [number, number];
export type Vec3 = readonly [number, number, number];

/** Most polyline points the solver's cut uniform holds (mirrors CUT_MAX_POINTS in cut.wgsl). */
export const CUT_MAX_POINTS = 64;
/** Sim seconds a drawn cut stays a tear before the sides are one material again (the nip knits
 *  each particle back sooner, when it passes through it). */
export const CUT_SECONDS = 4.0;
/** Half-width of the knife's kerf, in solver cells: the strip that is pushed aside onto the lips. */
export const CUT_KERF_CELLS = 1.5;
/** Half-width of the band of particles that get a side label, in cells beyond the kerf: wide enough
 *  that no unlabelled particle's stencil (1.5 cells) touches both sides. */
export const CUT_BAND_CELLS = 3.5;

/** Front roll axis (y, z) for the given parameters. */
export function frontAxis(params: MillParams): { axisY: number; axisZ: number } {
  const { front } = rollerPoses(params);
  return { axisY: front.axisY, axisZ: front.axisZ };
}

/** World position -> sheet coordinates (x, s) and the radial depth dr off the front roll surface. */
export function sheetCoords(p: Vec3, params: MillParams): { x: number; s: number; dr: number } {
  const { axisY, axisZ } = frontAxis(params);
  const dy = p[1] - axisY;
  const dz = p[2] - axisZ;
  const R = GEOMETRY.radius;
  return { x: p[0], s: R * Math.atan2(dz, dy), dr: Math.hypot(dy, dz) - R };
}

/** Sheet coordinates (x, s, depth dr off the roll surface) -> world position. */
export function sheetToWorld(x: number, s: number, dr: number, params: MillParams): Vec3 {
  const { axisY, axisZ } = frontAxis(params);
  const R = GEOMETRY.radius;
  const th = s / R;
  const r = R + dr;
  return [x, axisY + r * Math.cos(th), axisZ + r * Math.sin(th)];
}

/**
 * Nearest intersection (t > 0) of the ray origin + t * dir with the cylinder of radius `radius`
 * about the line (y = axisY, z = axisZ) parallel to x. Null when the ray misses it.
 */
export function rayCylinderHit(origin: Vec3, dir: Vec3, axisY: number, axisZ: number, radius: number): { t: number; point: Vec3 } | null {
  const oy = origin[1] - axisY;
  const oz = origin[2] - axisZ;
  const a = dir[1] * dir[1] + dir[2] * dir[2];
  if (a < 1e-12) return null;
  const b = 2 * (oy * dir[1] + oz * dir[2]);
  const c = oy * oy + oz * oz - radius * radius;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  let t = (-b - sq) / (2 * a);
  if (t <= 1e-6) t = (-b + sq) / (2 * a);
  if (t <= 1e-6) return null;
  return { t, point: [origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]] };
}

/**
 * Project a view ray onto the sheet on the front roll: intersect it with the cylinder at the
 * sheet's mid-thickness and return the hit in sheet coordinates (null when it misses the roll or
 * lands beyond the roll's ends).
 */
export function rayToSheet(origin: Vec3, dir: Vec3, params: MillParams, sheetDepth = 0.5 * params.gap): SheetPoint | null {
  const { axisY, axisZ } = frontAxis(params);
  const hit = rayCylinderHit(origin, dir, axisY, axisZ, GEOMETRY.radius + sheetDepth);
  if (!hit) return null;
  const x = hit.point[0];
  if (x < 0 || x > GEOMETRY.length) return null;
  const { s } = sheetCoords(hit.point, params);
  return [x, s];
}

/**
 * Tidy a drawn polyline for the solver: drop non-finite points and points closer than
 * `minSpacing` to the last kept one, then, if more than `maxPoints` remain, resample it evenly by
 * arc length (keeping both ends). Returns [] when fewer than two distinct points remain.
 */
export function simplifyPolyline(points: readonly SheetPoint[], minSpacing: number, maxPoints = CUT_MAX_POINTS): SheetPoint[] {
  const kept: SheetPoint[] = [];
  for (const p of points) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
    const last = kept[kept.length - 1];
    if (last && Math.hypot(p[0] - last[0], p[1] - last[1]) < minSpacing) continue;
    kept.push([p[0], p[1]]);
  }
  // keep the true end of the stroke even if it was within minSpacing of the last kept point
  const end = points.length ? points[points.length - 1] : undefined;
  if (end && kept.length && Number.isFinite(end[0]) && Number.isFinite(end[1])) {
    const last = kept[kept.length - 1];
    if (last[0] !== end[0] || last[1] !== end[1]) {
      if (kept.length >= 2) kept[kept.length - 1] = [end[0], end[1]];
      else if (Math.hypot(end[0] - last[0], end[1] - last[1]) > 0) kept.push([end[0], end[1]]);
    }
  }
  if (kept.length < 2) return [];
  if (kept.length <= maxPoints) return kept;
  const cum = [0];
  for (let i = 1; i < kept.length; i++) cum.push(cum[i - 1] + Math.hypot(kept[i][0] - kept[i - 1][0], kept[i][1] - kept[i - 1][1]));
  const total = cum[cum.length - 1];
  const out: SheetPoint[] = [];
  let j = 0;
  for (let k = 0; k < maxPoints; k++) {
    const target = (total * k) / (maxPoints - 1);
    while (j < kept.length - 2 && cum[j + 1] < target) j++;
    const seg = cum[j + 1] - cum[j];
    const t = seg > 0 ? Math.min(Math.max((target - cum[j]) / seg, 0), 1) : 0;
    out.push([kept[j][0] + t * (kept[j + 1][0] - kept[j][0]), kept[j][1] + t * (kept[j + 1][1] - kept[j][1])]);
  }
  return out;
}

/**
 * Signed distance from q to the polyline (left of the direction of travel is positive), taking only
 * segments whose perpendicular foot falls inside them (beyond the ends there is no tear), plus the
 * arc length along the polyline at the foot. Null when no segment's foot covers q.
 */
export function signedDistanceToPolyline(q: SheetPoint, pts: readonly SheetPoint[]): { d: number; along: number; total: number } | null {
  let best: { d: number; along: number } | null = null;
  let cum = 0;
  for (let i = 0; i + 1 < pts.length; i++) {
    const ax = pts[i][0], as = pts[i][1];
    const ex = pts[i + 1][0] - ax, es = pts[i + 1][1] - as;
    const len2 = ex * ex + es * es;
    const len = Math.sqrt(len2);
    if (len2 > 1e-12) {
      const t = ((q[0] - ax) * ex + (q[1] - as) * es) / len2;
      if (t >= 0 && t <= 1) {
        const d = ((q[0] - ax) * -es + (q[1] - as) * ex) / len;
        if (!best || Math.abs(d) < Math.abs(best.d)) best = { d, along: cum + t * len };
      }
    }
    cum += len;
  }
  return best ? { ...best, total: cum } : null;
}
