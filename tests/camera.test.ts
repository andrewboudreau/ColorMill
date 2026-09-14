import { describe, expect, it } from 'vitest';
import { GEOMETRY } from '../src/config/mill';
import {
  CAMERA_LIMITS, cameraBasis, cameraOrigin, clampCamera, defaultCamera, orbitBy, projectPoint, zoomBy
} from '../src/render/camera';
import type { Vec3 } from '../src/render/camera';

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);

describe('orbit camera basis', () => {
  it('is orthonormal and right-handed for a range of orbits', () => {
    for (const yaw of [0, 0.7, -2.1, 3.0]) {
      for (const pitch of [-0.05, 0.3, 1.2]) {
        const c = { ...defaultCamera(), yaw, pitch };
        const b = cameraBasis(c, 1.5);
        expect(len(b.right)).toBeCloseTo(1, 6);
        expect(len(b.up)).toBeCloseTo(1, 6);
        expect(len(b.forward)).toBeCloseTo(1, 6);
        expect(dot(b.right, b.up)).toBeCloseTo(0, 6);
        expect(dot(b.right, b.forward)).toBeCloseTo(0, 6);
        expect(dot(b.up, b.forward)).toBeCloseTo(0, 6);
        // right-handed camera frame: right x up = -forward (the view direction is the frame's -z)
        const cx: Vec3 = [
          b.right[1] * b.up[2] - b.right[2] * b.up[1],
          b.right[2] * b.up[0] - b.right[0] * b.up[2],
          b.right[0] * b.up[1] - b.right[1] * b.up[0]
        ];
        expect(dot(cx, b.forward)).toBeCloseTo(-1, 6);
        expect(b.up[1]).toBeGreaterThan(0);
        expect(b.right[1]).toBeCloseTo(0, 6);
      }
    }
  });

  it('default view is in front of the mill, elevated, looking at the nip', () => {
    const c = defaultCamera();
    expect(c.target[0]).toBeCloseTo(GEOMETRY.length / 2, 6);
    expect(c.target[2]).toBeCloseTo(GEOMETRY.nipZ, 6);
    expect(c.target[1]).toBeGreaterThan(GEOMETRY.axisY);
    const o = cameraOrigin(c);
    expect(o[2]).toBeGreaterThan(GEOMETRY.domain[2]); // in front (viewer side)
    expect(o[1]).toBeGreaterThan(c.target[1]); // slightly above
    const b = cameraBasis(c, 16 / 9);
    // forward points from the eye to the target
    const toTarget: Vec3 = [c.target[0] - o[0], c.target[1] - o[1], c.target[2] - o[2]];
    expect(dot(b.forward, toTarget) / len(toTarget)).toBeCloseTo(1, 6);
    expect(b.forward[2]).toBeLessThan(0);
    expect(b.right[0]).toBeCloseTo(1, 6); // +x is screen right from the front
    // the nip projects to the centre of the screen
    const p = projectPoint(c, 16 / 9, c.target);
    expect(p).not.toBeNull();
    expect(p!.x).toBeCloseTo(0, 6);
    expect(p!.y).toBeCloseTo(0, 6);
    expect(p!.depth).toBeCloseTo(c.distance, 6);
    // the whole mill (roller tops, tray under the front roller, the end guides) is in view
    for (const pt of [
      [GEOMETRY.length / 2, GEOMETRY.axisY + GEOMETRY.radius, GEOMETRY.nipZ + 0.35],
      [GEOMETRY.length / 2, 0, GEOMETRY.nipZ + GEOMETRY.radius + 0.3],
      [0, GEOMETRY.axisY, GEOMETRY.nipZ],
      [GEOMETRY.length, GEOMETRY.axisY, GEOMETRY.nipZ]
    ] as Vec3[]) {
      const q = projectPoint(c, 16 / 9, pt);
      expect(q).not.toBeNull();
      expect(Math.abs(q!.x)).toBeLessThan(1);
      expect(Math.abs(q!.y)).toBeLessThan(1);
    }
  });

  it('projects +x to screen right and +y to screen up from the front view', () => {
    const c = defaultCamera();
    const a = 1.5;
    const t = c.target;
    expect(projectPoint(c, a, [t[0] + 0.3, t[1], t[2]])!.x).toBeGreaterThan(0);
    expect(projectPoint(c, a, [t[0], t[1] + 0.3, t[2]])!.y).toBeGreaterThan(0);
    // behind the eye -> null
    const o = cameraOrigin(c);
    expect(projectPoint(c, a, [o[0], o[1], o[2] + 1])).toBeNull();
  });

  it('clamps pitch, distance and fov and wraps yaw', () => {
    const c = { ...defaultCamera(), pitch: 3, distance: 100, fovY: 9, yaw: 7 };
    clampCamera(c);
    expect(c.pitch).toBe(CAMERA_LIMITS.maxPitch);
    expect(c.distance).toBe(CAMERA_LIMITS.maxDistance);
    expect(c.fovY).toBe(CAMERA_LIMITS.maxFovY);
    expect(c.yaw).toBeGreaterThan(-Math.PI);
    expect(c.yaw).toBeLessThanOrEqual(Math.PI);
    expect(c.yaw).toBeCloseTo(7 - 2 * Math.PI, 9);
    const d = { ...defaultCamera(), pitch: -3, distance: 0, fovY: 0, yaw: NaN };
    clampCamera(d);
    expect(d.pitch).toBe(CAMERA_LIMITS.minPitch);
    expect(d.distance).toBe(CAMERA_LIMITS.minDistance);
    expect(d.fovY).toBe(CAMERA_LIMITS.minFovY);
    expect(d.yaw).toBe(0);
  });

  it('orbit and zoom helpers respect the limits', () => {
    const c = defaultCamera();
    orbitBy(c, 0, 100000, 600);
    expect(c.pitch).toBe(CAMERA_LIMITS.maxPitch);
    orbitBy(c, 0, -100000, 600);
    expect(c.pitch).toBe(CAMERA_LIMITS.minPitch);
    zoomBy(c, 1000);
    expect(c.distance).toBe(CAMERA_LIMITS.maxDistance);
    zoomBy(c, 1e-6);
    expect(c.distance).toBe(CAMERA_LIMITS.minDistance);
    // dragging right swings the view so the scene moves right (camera moves toward -x)
    const e = defaultCamera();
    orbitBy(e, 50, 0, 600);
    expect(cameraOrigin(e)[0]).toBeLessThan(GEOMETRY.length / 2);
  });
});
