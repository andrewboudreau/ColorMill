import { describe, expect, it } from 'vitest';
import { DEFAULT_PARAMS, GEOMETRY } from '../src/config/mill';
import { defaultCamera, projectPoint, rayThroughNdc } from '../src/render/camera';
import {
  CUT_MAX_POINTS, rayCylinderHit, rayToSheet, sheetCoords, sheetToWorld, signedDistanceToPolyline, simplifyPolyline, type SheetPoint
} from '../src/sim/cut';

describe('sheet coordinates', () => {
  it('round-trips world <-> (x, s, dr)', () => {
    for (const [x, s, dr] of [[0.3, 0.1, 0.02], [1.2, 0.5, 0], [0.75, -0.4, 0.05], [0.1, 0.9, 0.01]]) {
      const w = sheetToWorld(x, s, dr, DEFAULT_PARAMS);
      const c = sheetCoords(w, DEFAULT_PARAMS);
      expect(c.x).toBeCloseTo(x, 9);
      expect(c.s).toBeCloseTo(s, 9);
      expect(c.dr).toBeCloseTo(dr, 9);
    }
  });

  it('puts s = 0 on the crown and s = pi R / 2 on the front face at axis height', () => {
    const crown = sheetToWorld(0.5, 0, 0, DEFAULT_PARAMS);
    expect(crown[1]).toBeCloseTo(GEOMETRY.axisY + GEOMETRY.radius, 9);
    const face = sheetToWorld(0.5, (Math.PI * GEOMETRY.radius) / 2, 0, DEFAULT_PARAMS);
    expect(face[1]).toBeCloseTo(GEOMETRY.axisY, 9);
    expect(face[2]).toBeGreaterThan(GEOMETRY.nipZ + GEOMETRY.radius);   // toward the viewer
  });
});

describe('rayCylinderHit', () => {
  it('takes the near side of the cylinder', () => {
    const hit = rayCylinderHit([0.5, 0, 5], [0, 0, -1], 0, 0, 1);
    expect(hit).not.toBeNull();
    expect(hit!.point[2]).toBeCloseTo(1, 9);
    expect(hit!.t).toBeCloseTo(4, 9);
  });
  it('misses when the ray passes by, and when it points away', () => {
    expect(rayCylinderHit([0, 2, 5], [0, 0, -1], 0, 0, 1)).toBeNull();
    expect(rayCylinderHit([0, 0, 5], [0, 0, 1], 0, 0, 1)).toBeNull();
    expect(rayCylinderHit([0, 0, 5], [1, 0, 0], 0, 0, 1)).toBeNull();
  });
});

describe('screen -> sheet', () => {
  it('inverts projectPoint for a point on the sheet in the default view', () => {
    const cam = defaultCamera();
    const aspect = 1.5;
    const depth = 0.5 * DEFAULT_PARAMS.gap;
    for (const [x, s] of [[0.4, 0.3], [0.75, 0.15], [1.1, 0.45]] as SheetPoint[]) {
      const w = sheetToWorld(x, s, depth, DEFAULT_PARAMS);
      const ndc = projectPoint(cam, aspect, w)!;
      const ray = rayThroughNdc(cam, aspect, ndc.x, ndc.y);
      const back = rayToSheet(ray.origin, ray.dir, DEFAULT_PARAMS, depth)!;
      expect(back).not.toBeNull();
      expect(back[0]).toBeCloseTo(x, 6);
      expect(back[1]).toBeCloseTo(s, 6);
    }
  });
  it('returns null off the roll', () => {
    const cam = defaultCamera();
    const ray = rayThroughNdc(cam, 1.5, 0, 0.99);   // the top of the screen looks over the mill
    expect(rayToSheet(ray.origin, ray.dir, DEFAULT_PARAMS)).toBeNull();
  });
});

describe('simplifyPolyline', () => {
  it('drops near-duplicates and keeps the ends', () => {
    const pts: SheetPoint[] = [[0, 0], [0.001, 0], [0.1, 0], [0.1005, 0], [0.2, 0.01]];
    const out = simplifyPolyline(pts, 0.01);
    expect(out[0]).toEqual([0, 0]);
    expect(out[out.length - 1]).toEqual([0.2, 0.01]);
    expect(out.length).toBe(3);
  });
  it('resamples long strokes to the point limit', () => {
    const pts: SheetPoint[] = [];
    for (let i = 0; i <= 500; i++) pts.push([i / 500, 0.2 + 0.05 * Math.sin(i / 20)]);
    const out = simplifyPolyline(pts, 0.0001);
    expect(out.length).toBe(CUT_MAX_POINTS);
    expect(out[0][0]).toBeCloseTo(0, 9);
    expect(out[out.length - 1][0]).toBeCloseTo(1, 9);
  });
  it('returns [] for a tap', () => {
    expect(simplifyPolyline([[0.5, 0.2], [0.5, 0.2]], 0.01)).toEqual([]);
    expect(simplifyPolyline([[0.5, 0.2]], 0.01)).toEqual([]);
  });
});

describe('signedDistanceToPolyline', () => {
  const line: SheetPoint[] = [[0.2, 0.3], [1.0, 0.3]];
  it('is positive to the left of the direction of travel', () => {
    expect(signedDistanceToPolyline([0.5, 0.35], line)!.d).toBeCloseTo(0.05, 9);
    expect(signedDistanceToPolyline([0.5, 0.25], line)!.d).toBeCloseTo(-0.05, 9);
    expect(signedDistanceToPolyline([0.5, 0.25], line)!.along).toBeCloseTo(0.3, 9);
  });
  it('has no tear beyond the ends', () => {
    expect(signedDistanceToPolyline([0.1, 0.3], line)).toBeNull();
    expect(signedDistanceToPolyline([1.1, 0.3], line)).toBeNull();
  });
});
