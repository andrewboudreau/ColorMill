/**
 * Renderer contract (design §8). Keep free of implementation details so the
 * UI/app layer and tests can be written against it.
 */
import type { MillParams } from '../config/mill';
import type { RenderVolumes } from '../sim/types';

export interface CameraState {
  /** orbit angles in radians: yaw around +y, pitch above the horizon */
  yaw: number;
  pitch: number;
  /** distance from the target */
  distance: number;
  /** look-at target in sim units */
  target: readonly [number, number, number];
  /** vertical field of view in radians */
  fovY: number;
}

export interface RenderFrameInfo {
  readonly params: MillParams;
  readonly rollerAngleFront: number;
  readonly rollerAngleBack: number;
  /** seconds since start, for subtle animation */
  readonly timeSeconds: number;
}

export interface Renderer {
  readonly camera: CameraState;
  /** Swap the volumes the ray-marcher samples (called when the sim is rebuilt). */
  setVolumes(volumes: RenderVolumes): void;
  /** Match the canvas backing store to its CSS size * dpr (capped at 2). Returns true if it changed. */
  resize(): boolean;
  /** Encode the frame into `encoder`; the caller submits. */
  render(encoder: GPUCommandEncoder, info: RenderFrameInfo): void;
  /** Reset the camera to the default front view. */
  resetCamera(): void;
  /**
   * Enable/disable presenting to the canvas. When disabled, render() draws
   * into an offscreen target instead (automated browsers whose canvas
   * presentation is broken, e.g. headless SwiftShader Chromium).
   */
  setPresentation?(enabled: boolean): void;
  /** Render one frame offscreen and read back tightly packed RGBA8 pixels (tests/diagnostics). */
  renderToPixels?(info: RenderFrameInfo, width?: number, height?: number): Promise<{ width: number; height: number; data: Uint8Array }>;
  destroy(): void;
}
