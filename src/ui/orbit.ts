/**
 * Pointer controls for the orbit camera (design §9): drag orbits, wheel or
 * pinch zooms, double-tap resets to the front view. Works on the
 * `CameraState` object the renderer exposes, so it has no renderer imports.
 */
import type { CameraState } from '../render/types';

export interface OrbitOptions {
  /** radians per CSS pixel of drag */
  readonly sensitivity?: number;
  readonly minPitch?: number;
  readonly maxPitch?: number;
  readonly minDistance?: number;
  readonly maxDistance?: number;
  /** called on any user interaction (used to suspend auto-orbit) */
  readonly onInteract?: () => void;
}

export interface OrbitControls {
  /** true while a pointer is down on the canvas */
  readonly active: boolean;
  destroy(): void;
}

interface Pt { x: number; y: number }

export function installOrbitControls(
  canvas: HTMLElement,
  camera: CameraState,
  resetCamera: () => void,
  opts: OrbitOptions = {}
): OrbitControls {
  // defaults mirror CAMERA_LIMITS in src/render/camera.ts
  const sens = opts.sensitivity ?? 0.006;
  const minPitch = opts.minPitch ?? -0.08;
  const maxPitch = opts.maxPitch ?? 1.45;
  const minDist = opts.minDistance ?? 0.9;
  const maxDist = opts.maxDistance ?? 5.0;
  const pointers = new Map<number, Pt>();
  let lastTapTime = 0;
  let lastTapPos: Pt = { x: 0, y: 0 };
  let downPos: Pt = { x: 0, y: 0 };
  let moved = false;
  let active = false;

  const clampCamera = (): void => {
    camera.pitch = Math.min(maxPitch, Math.max(minPitch, camera.pitch));
    camera.distance = Math.min(maxDist, Math.max(minDist, camera.distance));
  };
  const centroid = (): Pt => {
    let x = 0, y = 0;
    for (const p of pointers.values()) { x += p.x; y += p.y; }
    const n = Math.max(1, pointers.size);
    return { x: x / n, y: y / n };
  };
  const spread = (): number => {
    if (pointers.size < 2) return 0;
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };

  const onDown = (e: PointerEvent): void => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    canvas.setPointerCapture?.(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) {
      downPos = { x: e.clientX, y: e.clientY };
      moved = false;
    }
    active = true;
    opts.onInteract?.();
    e.preventDefault();
  };

  const onMove = (e: PointerEvent): void => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const before = centroid();
    const spreadBefore = spread();
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const after = centroid();
    const dx = after.x - before.x;
    const dy = after.y - before.y;
    if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) > 6) moved = true;
    camera.yaw -= dx * sens;
    camera.pitch += dy * sens;
    if (pointers.size >= 2 && spreadBefore > 0) {
      const s = spread();
      if (s > 0) camera.distance *= spreadBefore / s;
    }
    clampCamera();
    e.preventDefault();
  };

  const onUp = (e: PointerEvent): void => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    canvas.releasePointerCapture?.(e.pointerId);
    if (pointers.size === 0) {
      active = false;
      const now = performance.now();
      const isTap = !moved && e.type !== 'pointercancel';
      if (isTap) {
        const near = Math.hypot(e.clientX - lastTapPos.x, e.clientY - lastTapPos.y) < 30;
        if (now - lastTapTime < 350 && near) {
          resetCamera();
          clampCamera();
          lastTapTime = 0;
        } else {
          lastTapTime = now;
          lastTapPos = { x: e.clientX, y: e.clientY };
        }
      }
    }
  };

  const onWheel = (e: WheelEvent): void => {
    const k = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1;
    camera.distance *= Math.exp(e.deltaY * k * 0.0012);
    clampCamera();
    opts.onInteract?.();
    e.preventDefault();
  };

  const onDblClick = (e: MouseEvent): void => {
    // mouse double-click (touch double-tap is handled in onUp)
    resetCamera();
    clampCamera();
    e.preventDefault();
  };

  const onContext = (e: Event): void => e.preventDefault();

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDblClick);
  canvas.addEventListener('contextmenu', onContext);

  return {
    get active() { return active; },
    destroy() {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDblClick);
      canvas.removeEventListener('contextmenu', onContext);
      pointers.clear();
    }
  };
}
