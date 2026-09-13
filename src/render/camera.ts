/**
 * Orbit camera for the mill renderer (design §8, §9).
 *
 * Pure math (basis, limits, defaults) lives here and is unit-tested; the DOM
 * controls are a thin layer on top (Pointer Events + wheel) that mutates a
 * CameraState in place and notifies the caller.
 *
 * Conventions: yaw = 0 puts the camera in front of the mill (on the +z side
 * of the target) looking toward -z; positive yaw swings the camera toward
 * +x (the viewer's right); pitch is the elevation above the horizon.
 */
import { GEOMETRY } from '../config/mill';
import type { CameraState } from './types';

export type Vec3 = readonly [number, number, number];

export interface CameraLimits {
  readonly minPitch: number;
  readonly maxPitch: number;
  readonly minDistance: number;
  readonly maxDistance: number;
  readonly minFovY: number;
  readonly maxFovY: number;
}

export const CAMERA_LIMITS: CameraLimits = {
  minPitch: -0.08,
  maxPitch: 1.45,
  minDistance: 0.9,
  maxDistance: 5.0,
  minFovY: 0.2,
  maxFovY: 1.4
};

/** Default front view: slightly elevated, looking at the nip. */
export function defaultCamera(): CameraState {
  return {
    yaw: 0.0,
    pitch: 0.62,
    distance: 2.9,
    target: [GEOMETRY.length / 2, GEOMETRY.axisY + 0.12, GEOMETRY.nipZ],
    fovY: (40 * Math.PI) / 180
  };
}

/** Copy `src` into `dst` (keeps the object identity the renderer exposes). */
export function assignCamera(dst: CameraState, src: CameraState): void {
  dst.yaw = src.yaw;
  dst.pitch = src.pitch;
  dst.distance = src.distance;
  dst.target = [src.target[0], src.target[1], src.target[2]];
  dst.fovY = src.fovY;
}

/** Clamp pitch/distance/fov to the limits and wrap yaw into (-pi, pi]. */
export function clampCamera(c: CameraState, limits: CameraLimits = CAMERA_LIMITS): CameraState {
  c.pitch = Math.min(limits.maxPitch, Math.max(limits.minPitch, c.pitch));
  c.distance = Math.min(limits.maxDistance, Math.max(limits.minDistance, c.distance));
  c.fovY = Math.min(limits.maxFovY, Math.max(limits.minFovY, c.fovY));
  if (!Number.isFinite(c.yaw)) c.yaw = 0;
  const twoPi = Math.PI * 2;
  c.yaw = c.yaw - twoPi * Math.floor((c.yaw + Math.PI) / twoPi);
  return c;
}

export interface CameraBasis {
  /** eye position (sim units) */
  readonly origin: Vec3;
  /** unit vectors: screen right, screen up, view direction */
  readonly right: Vec3;
  readonly up: Vec3;
  readonly forward: Vec3;
  /** tan(fovY / 2) */
  readonly tanHalfFovY: number;
  readonly aspect: number;
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Eye position for an orbit state. */
export function cameraOrigin(c: CameraState): Vec3 {
  const cp = Math.cos(c.pitch);
  return [
    c.target[0] + c.distance * Math.sin(c.yaw) * cp,
    c.target[1] + c.distance * Math.sin(c.pitch),
    c.target[2] + c.distance * Math.cos(c.yaw) * cp
  ];
}

/**
 * Ray basis for the shader: a pixel with NDC (x, y) in [-1, 1] has direction
 * forward + right * x * tanHalfFovY * aspect + up * y * tanHalfFovY.
 */
export function cameraBasis(c: CameraState, aspect: number): CameraBasis {
  const origin = cameraOrigin(c);
  const forward = normalize([c.target[0] - origin[0], c.target[1] - origin[1], c.target[2] - origin[2]]);
  // world up, with a fallback when looking straight down
  const worldUp: Vec3 = Math.abs(forward[1]) > 0.9999 ? [0, 0, -1] : [0, 1, 0];
  const right = normalize(cross(forward, worldUp));
  const up = cross(right, forward);
  return { origin, right, up, forward, tanHalfFovY: Math.tan(c.fovY / 2), aspect };
}

/**
 * Project a world point to normalised device coords ([-1,1] x [-1,1], y up)
 * and the view depth. Returns null when the point is behind the eye. Used by
 * tests to find where a known feature lands on screen.
 */
export function projectPoint(c: CameraState, aspect: number, p: Vec3): { x: number; y: number; depth: number } | null {
  const b = cameraBasis(c, aspect);
  const d: Vec3 = [p[0] - b.origin[0], p[1] - b.origin[1], p[2] - b.origin[2]];
  const z = d[0] * b.forward[0] + d[1] * b.forward[1] + d[2] * b.forward[2];
  if (z <= 1e-6) return null;
  const x = d[0] * b.right[0] + d[1] * b.right[1] + d[2] * b.right[2];
  const y = d[0] * b.up[0] + d[1] * b.up[1] + d[2] * b.up[2];
  return { x: x / (z * b.tanHalfFovY * b.aspect), y: y / (z * b.tanHalfFovY), depth: z };
}

/** Orbit by pixel deltas (used by the drag handler; exposed for tests). */
export function orbitBy(c: CameraState, dxPixels: number, dyPixels: number, heightPixels: number): CameraState {
  const k = 2.2 / Math.max(1, heightPixels);
  c.yaw -= dxPixels * k;
  c.pitch += dyPixels * k;
  return clampCamera(c);
}

/** Multiply the distance (wheel/pinch). */
export function zoomBy(c: CameraState, factor: number): CameraState {
  c.distance *= factor;
  return clampCamera(c);
}

export interface CameraControlsOptions {
  /** limits used for clamping (default CAMERA_LIMITS) */
  readonly limits?: CameraLimits;
  /** state restored on double-tap (default defaultCamera()) */
  readonly home?: () => CameraState;
}

/**
 * Attach orbit / zoom / reset controls to a canvas. Mutates `camera` in
 * place and calls `onChange` after every change. Returns a detach function.
 *
 * - primary pointer drag: orbit
 * - wheel: zoom (trackpad pinch arrives as ctrl+wheel and is handled the same)
 * - two pointers: pinch zoom (+ orbit with the midpoint)
 * - double tap / double click: reset to the home view
 */
export function attachCameraControls(
  canvas: HTMLCanvasElement,
  camera: CameraState,
  onChange: () => void,
  opts: CameraControlsOptions = {}
): () => void {
  const limits = opts.limits ?? CAMERA_LIMITS;
  const home = opts.home ?? defaultCamera;
  canvas.style.touchAction = 'none';
  canvas.style.userSelect = 'none';

  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;
  let lastTapTime = 0;
  let lastTapX = 0;
  let lastTapY = 0;
  let moved = 0;

  const changed = (): void => {
    clampCamera(camera, limits);
    onChange();
  };

  const reset = (): void => {
    assignCamera(camera, home());
    changed();
  };

  const onDown = (e: PointerEvent): void => {
    if (e.button !== undefined && e.button !== 0 && e.pointerType === 'mouse') return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      /* not all environments support capture */
    }
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
    if (pointers.size === 1) moved = 0;
    e.preventDefault();
  };

  const onMove = (e: PointerEvent): void => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const cur = { x: e.clientX, y: e.clientY };
    const dx = cur.x - prev.x;
    const dy = cur.y - prev.y;
    pointers.set(e.pointerId, cur);
    moved += Math.abs(dx) + Math.abs(dy);
    const h = canvas.clientHeight || canvas.height || 1;
    if (pointers.size === 1) {
      orbitBy(camera, dx, dy, h);
      changed();
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0 && d > 0) zoomBy(camera, pinchDist / d);
      pinchDist = d;
      // orbit with half the midpoint motion so a two-finger drag still pans the view
      orbitBy(camera, dx * 0.5, dy * 0.5, h);
      changed();
    }
    e.preventDefault();
  };

  const onUp = (e: PointerEvent): void => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    pinchDist = 0;
    // double-tap detection for touch/pen (mouse gets dblclick)
    if (e.pointerType !== 'mouse' && pointers.size === 0 && moved < 12) {
      const now = performance.now();
      if (now - lastTapTime < 320 && Math.hypot(e.clientX - lastTapX, e.clientY - lastTapY) < 40) {
        lastTapTime = 0;
        reset();
      } else {
        lastTapTime = now;
        lastTapX = e.clientX;
        lastTapY = e.clientY;
      }
    }
  };

  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? 400 : 1;
    const dy = e.deltaY * unit;
    // ctrl+wheel is a trackpad pinch: scale a bit more gently
    const speed = e.ctrlKey ? 0.006 : 0.0016;
    zoomBy(camera, Math.exp(dy * speed));
    changed();
  };

  const onDblClick = (e: MouseEvent): void => {
    e.preventDefault();
    reset();
  };

  const onContextMenu = (e: Event): void => e.preventDefault();

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDblClick);
  canvas.addEventListener('contextmenu', onContextMenu);

  return () => {
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerup', onUp);
    canvas.removeEventListener('pointercancel', onUp);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('dblclick', onDblClick);
    canvas.removeEventListener('contextmenu', onContextMenu);
    pointers.clear();
  };
}
