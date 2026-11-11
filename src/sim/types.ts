/**
 * Shared interfaces between the GPU solver, the renderer, the UI and tests.
 * See docs/design-v2.md. Keep this file free of GPU code.
 */
import type { GridDims, MillConfig, MillParams, QualitySettings, RollStyle } from '../config/mill';
import type { Drop } from '../drops';
import type { MixingMetrics, PigmentInventory } from './metrics';
import type { NipDiagnostics } from './diagnostics';

/** A 7-float Mixbox latent. */
export type Latent = readonly [number, number, number, number, number, number, number];

export interface SimStats {
  readonly particleCount: number;
  /** particle buffer capacity (seeded bank + reserved pigment pool) */
  readonly particleCapacity: number;
  /** material volume in sim units^3 (particles x rest volume); conserved except when chunks are added */
  readonly materialVolume: number;
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
  /** 4 floats per particle: the pigment laminae's unit normal xyz, then q = ln(s / s0), the log of their thickness
   *  over the thickness they arrived with (src/sim/lamina.ts) */
  readonly laminae: Float32Array;
  /** 2 floats per particle: ln s0 (world units), then ln of the thickness the pigment is already blended down to */
  readonly mixState: Float32Array;
  /** 1 float per particle: pigment load (0 = clear base, PIGMENT_LOAD = pure masterbatch) */
  readonly loads: Float32Array;
}

/** The render volume produced by the solver each frame (design §3.5 / §8). */
export interface RenderVolumes {
  /** rgba16float 3D texture: (density, 0, 0, 0) */
  readonly volA: GPUTexture;
  /** Incremented whenever volume contents change. Omit to disable render caching. */
  readonly version?: number;
  /** Incremented when density changes. Omit to rebuild the density mip on every draw. */
  readonly densityVersion?: number;
  readonly dims: GridDims;
  /** colour raster (design §3.5), in a box of fineDims nodes laid over the material: rgba16float (pigment load, then the latent premultiplied by it: load·lat0, load·lat1, load·lat2) */
  readonly finA: GPUTexture;
  /** rgba16float (load·lat3, load·lat4, load·lat5, load·lat6) on fineDims; the reader divides by the sum of the four
   *  concentration channels (load·lat0..3) after filtering */
  readonly finB: GPUTexture;
  /** rgba16float (mass, coverage, 0, 0) on fineDims, same normalisation as finA.x, so finA.x / finC.x is the load per unit mass
   *  and finC.y / finC.x the share of the mass that carries pigment (0..1) */
  readonly finC: GPUTexture;
  /** node counts of the colour raster's textures and its finest cell (the cell when texel 0 is at the origin
   *  and no fineBox says otherwise) */
  readonly fineDims: GridDims;
  /** Where the colour raster's box is this frame (design §3.6), as a 32-byte uniform buffer the solver
   *  writes on the GPU: vec4 (world position of texel 0, cell size), vec4 (pack normalisation, cell in
   *  eighths of the finest, 0, 0). Omit for a raster fixed at the origin with cell fineDims.h. */
  readonly fineBox?: GPUBuffer;
}

/**
 * A picture to stamp onto the material: `width × height` pixels, row-major from the top-left, 8 floats per
 * pixel: the Mixbox latent (7) and the pixel's opacity 0..1 (0 leaves the material alone).
 */
export interface StampImage {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
}

/**
 * Where a stamp lands, seen from above: the image's left and right edges at x0 / x1, its top row at z0 and its
 * bottom row at z1. The back of the bank is the smaller z, so with z0 < z1 the picture reads right from the front.
 */
export interface StampPlacement {
  readonly x0: number;
  readonly x1: number;
  readonly z0: number;
  readonly z1: number;
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
  /** Blend a pigment into particles inside a sphere (design §5). `striation` is the thickness its laminae are
   *  laid down with (world units); 0 or omitted = one blob, the sphere's diameter. */
  addPigment(center: readonly [number, number, number], radius: number, latent: Latent, strength?: number, striation?: number): void;
  /**
   * Blend a pigment into a sphere sitting just under the material surface at
   * (x, z): the GPU probes the highest particle in the column, so the dollop
   * lands on the bank wherever it has slumped or drained to (design §5).
   */
  addPigmentOnSurface(x: number, z: number, radius: number, latent: Latent, strength?: number): void;
  /**
   * Add a chunk of NEW pigmented putty (particles from the reserved pool): a
   * sphere of radius `radius` centred above the bank at (x, z) that drops in
   * under gravity. Returns the number of particles added (0 when the pool is
   * exhausted, in which case the caller may fall back to addPigmentOnSurface).
   */
  addPigmentChunk(x: number, z: number, radius: number, latent: Latent): Promise<number>;
  /** Set every particle's pigment back to the base latent. */
  clearPigment(): void;
  /**
   * Stamp a picture onto the material seen from above: every particle under the image's footprint that lies
   * within `depth` (sim units; default STAMP_DEPTH = 0.06, at least 1.5 cells) of the top of its column takes the pixel's latent, blended by
   * the pixel's opacity times `strength` (default 1), and a masterbatch pigment load. The surface is found on
   * the GPU per column, so it works on the flat seeded bank and on whatever the material has become (design §5).
   */
  stampImage(image: StampImage, place: StampPlacement, options?: { depth?: number; strength?: number }): void;
  /** Start the scripted operator move: cut the sheet off the front roll, roll it into a log, turn it and set it on the bank (design §6). No-op if one is running. */
  cutAndFold(): void;
  /** Start the other operator move: cut the sheet across at the crown of the front roll and take the cut edge
   *  in hand: lift it off the crown, pull it back just past the crown at the roll's speed and drop it, so the flap
   *  flops back onto the front roll folded over itself; the rest of the sheet follows in the solver and the roll
   *  carries the fold into the nip. One press
   *  makes params.foldPasses such passes, each a fresh cut (design §6b). `side` only alternates, as an operator
   *  alternates hands. No-op if a move is running. */
  cutAndFlop(side: 'left' | 'right'): void;
  /** The current (or last) cut & fold's timeline in sim seconds from its start: the hand's speed, the lift, one
   *  pass's length, the gap between passes, how many, when each drops, the end. */
  readonly operatorHandTimes: { readonly speed: number; readonly lift: number; readonly pass: number; readonly gap: number; readonly passes: number; readonly drops: readonly number[]; readonly end: number };
  /** true while an operator move (cut & roll or cut & fold) is running */
  readonly operatorBusy: boolean;
  /** Length of the current (or last) cut & roll's roll phase in sim seconds: the material is wound off
   *  the roll and the log swung up over the nip; after it the log is dropped or lowered in (design §6). */
  readonly operatorRollSeconds: number;
  /** How cut & roll puts the roll back: folded in half ('double', the default) or as one long roll
   *  ('long'); read when the move starts. */
  rollStyle: RollStyle;
  /** Read particle state back to the CPU (slow; tests and diagnostics only). */
  readParticles(): Promise<ParticleSnapshot>;
  /** Read the packed density volume (volA red channel) back as float32, node-major (i fastest). */
  readDensity(): Promise<Float32Array>;
  /** Read where the colour raster's box is (design §3.6; tests and diagnostics): the world position of its
   *  texel 0, its cell size, and that cell in eighths of the finest one (8 = particle resolution). */
  readColourBox(): Promise<{ origin: [number, number, number]; cell: number; eighths: number }>;
  /** Diagnostic event counters since the simulator was built (wrapping u32): G2P's F resets and roller push-outs. */
  readCounters(): Promise<{ resets: number; pushOuts: number }>;
  destroy(): void;
}

/**
 * Debug/automation hooks the app exposes on `window.__colormill` for the
 * Playwright e2e tests (tests/e2e). Not a public API.
 */
export interface DebugApi {
  readonly sim: GpuMpmSim;
  /** the live renderer (camera, iso threshold, colour mode) */
  readonly renderer: { readonly camera: unknown; iso?: number; mode?: 'depth' | 'surface'; renderScale?: number };
  /** Run exactly n frames (sim step + render) synchronously in a loop and resolve when the GPU is idle. */
  stepFrames(n: number): Promise<void>;
  /** Particle snapshot (readParticles) */
  snapshot(): Promise<ParticleSnapshot>;
  /** Last frame's stats */
  stats(): SimStats;
  /** Total pigment and its colour over the particles (sum of load x mass, and of load x mass x latent per channel):
   *  conserved by milling, dispersion and the operator's moves; only a drop, an inject, a stamp or a reset changes it. */
  inventory(): Promise<PigmentInventory>;
  /** Mixing measures of the pigment load: Danckwerts' intensity of segregation at boxes of 1/32, 1/16 and 1/8 of a sim unit (the same at every preset), and the axial segregation. */
  mixing(): Promise<MixingMetrics>;
  /** Step `frames` frames and measure the nip over them (src/sim/diagnostics.ts; tools/perf/nip.mjs prints the table). */
  nipDiagnostics(frames: number): Promise<NipDiagnostics>;
  /** Inject a named pigment (key of PIGMENTS) at a random bank spot, as a UI tap would. */
  tapPigment(name: string, slot?: number, size?: number): void;
  /** Choose the drop slot (0-based, left to right) that taps land on. */
  setDropSlot(slot: number): void;
  /** the drop slot taps currently land on */
  readonly dropSlot: number;
  /** Choose the chunk size (index into PIGMENT_CHUNK_SIZES) taps use. */
  setChunkSize(size: number): void;
  readonly chunkSize: number;
  /** Every pigment chunk set down this session (taps, ?drops=, applyDrops), in order; cleared by Reset and by a rebuild. */
  readonly dropLog: readonly Drop[];
  /** URL of index.html that reproduces this session's start: ?drops= from dropLog, plus batch/preset when set. */
  startLink(): string;
  /** Set the chunks down in sequence (each awaited so they stack), as the ?drops= parameter does at boot. Does not change the current slot/size selection. */
  applyDrops(drops: readonly Drop[]): Promise<void>;
  /** Stamp a line of text on the bank, as ?stamp= does at boot (`ink`: a pigment name or '#rrggbb'; black by default). */
  stampText(text: string, ink?: string): void;
  /** Stamp a picture of latents on the bank (sim.stampImage, fitted to the bank when `place` is omitted). */
  stampImage(image: StampImage, place?: StampPlacement): void;
  /** Switch preset (rebuilds the sim), resolves when ready */
  setQuality(preset: 'low' | 'medium' | 'high' | 'ultra'): Promise<void>;
  /** Render the current frame offscreen and return RGBA8 pixels (works where canvas presentation does not). */
  screenshot(): Promise<{ width: number; height: number; data: Uint8Array }>;
  /** Report whether the app finished initialising */
  readonly ready: boolean;
}
