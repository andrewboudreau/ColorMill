/**
 * Shared interfaces between the GPU solver, the renderer, the UI and tests.
 * See docs/design-v2.md. Keep this file free of GPU code.
 */
import type { GridDims, MillConfig, MillParams, QualitySettings } from '../config/mill';

/** A 7-float Mixbox latent. */
export type Latent = readonly [number, number, number, number, number, number, number];

export interface SimStats {
  readonly particleCount: number;
  readonly grid: GridDims;
  /** sim seconds advanced so far */
  readonly simTime: number;
  /** sim seconds per rendered frame (dt * substepsPerFrame) */
  readonly simSecondsPerFrame: number;
  /** GPU time of the last step() if timestamp queries are available, else NaN */
  readonly lastStepGpuMs: number;
  /** front roller angle (rad) accumulated for rendering the rotation */
  readonly rollerAngleFront: number;
  readonly rollerAngleBack: number;
}

/** CPU readback of particle state (for tests / diagnostics). */
export interface ParticleSnapshot {
  readonly count: number;
  /** xyz interleaved */
  readonly positions: Float32Array;
  /** xyz interleaved */
  readonly velocities: Float32Array;
  /** 7 floats per particle */
  readonly latents: Float32Array;
  /** 9 floats per particle, row-major F */
  readonly deformation: Float32Array;
  /** 1 uint per particle */
  readonly flags: Uint32Array;
}

/** The render volume produced by the solver each frame (design §3.5 / §8). */
export interface RenderVolumes {
  /** rgba16float 3D texture: (density, lat0, lat1, lat2) */
  readonly volA: GPUTexture;
  /** rgba16float 3D texture: (lat3, lat4, lat5, lat6) */
  readonly volB: GPUTexture;
  readonly dims: GridDims;
}

export interface GpuMpmSimOptions {
  readonly device: GPUDevice;
  readonly config: MillConfig;
}

/**
 * The GPU MLS-MPM two-roll mill solver. One instance per quality preset;
 * changing the preset means constructing a new instance (buffers are sized
 * from the grid and the seeded bank).
 */
export interface GpuMpmSim {
  readonly config: MillConfig;
  readonly quality: QualitySettings;
  readonly dims: GridDims;
  readonly volumes: RenderVolumes;
  /** live parameters; mutate then call step() (uploaded each step) */
  readonly params: MillParams;
  readonly stats: SimStats;
  /** Resolves once every pipeline compiled cleanly; rejects with the WGSL/pipeline errors. */
  readonly ready: Promise<void>;
  paused: boolean;

  /** Advance substepsPerFrame substeps and refresh the render volumes. Encodes into `encoder`; the caller submits. */
  step(encoder: GPUCommandEncoder, frameDtSeconds: number): void;
  /** Re-seed the bank as fresh white silicone. */
  reset(): void;
  /** Blend a pigment into particles inside a sphere (design §5). */
  addPigment(center: readonly [number, number, number], radius: number, latent: Latent, strength?: number): void;
  /** Set every particle's pigment back to the base latent. */
  clearPigment(): void;
  /** Start the scripted operator cut-and-fold move (design §6). No-op if one is running. */
  cutAndFold(): void;
  /** Read particle state back to the CPU (slow; tests and diagnostics only). */
  readParticles(): Promise<ParticleSnapshot>;
  /** Read the packed density volume (volA red channel) back as float32, node-major (i fastest). */
  readDensity(): Promise<Float32Array>;
  destroy(): void;
}

/**
 * Debug/automation hooks the app exposes on `window.__colormill` for the
 * Playwright e2e tests (tests/e2e). Not a public API.
 */
export interface DebugApi {
  readonly sim: GpuMpmSim;
  /** Run exactly n frames (sim step + render) synchronously in a loop and resolve when the GPU is idle. */
  stepFrames(n: number): Promise<void>;
  /** Particle snapshot (readParticles) */
  snapshot(): Promise<ParticleSnapshot>;
  /** Last frame's stats */
  stats(): SimStats;
  /** Inject a named pigment (key of PIGMENTS) at a random bank spot, as a UI tap would. */
  tapPigment(name: string): void;
  /** Switch preset (rebuilds the sim), resolves when ready */
  setQuality(preset: 'low' | 'medium' | 'high' | 'ultra'): Promise<void>;
  /** Render the current frame offscreen and return RGBA8 pixels (works where canvas presentation does not). */
  screenshot(): Promise<{ width: number; height: number; data: Uint8Array }>;
  /** Report whether the app finished initialising */
  readonly ready: boolean;
}
