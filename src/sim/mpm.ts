/**
 * ColorMill v2 — WebGPU compute MLS-MPM solver for the two-roll mill.
 * Implements GpuMpmSim (src/sim/types.ts) per docs/design-v2.md §1–§7.
 *
 * Storage is structure-of-arrays: positions and velocities as vec4 (xyz + pad,
 * one 16-byte load each), C (9, row-major), F (9, row-major), the premultiplied
 * P2G affine matrix −dt·pVol·(4/h²)·stressScale·τ + pMass·C (9, row-major;
 * produced by g2p so p2g reads one matrix and repeats no SVD), latent (7),
 * flags (u32) and the cut-and-fold start pose (vec4: p0 + arc). Grid
 * accumulators are i32 fixed point (mass 2^20, momentum 2^16, latent·mass 2^18)
 * plus a vec4 f32 velocity grid. The render volume is two rgba16float 3D
 * storage textures. readParticles() converts the vec4 arrays back to the
 * xyz-interleaved ParticleSnapshot layout.
 */
import {
  PIGMENT_POOL_FRACTION,
  GEOMETRY, bankTopY, gridDims, lameParameters, rollerPoses, seedBankPositions,
  type GridDims, type MillConfig, type MillParams, type QualitySettings
} from '../config/mill';
import type {
  GpuMpmSim, GpuMpmSimOptions, Latent, ParticleSnapshot, RenderVolumes, SimStats
} from './types';
import commonSrc from './shaders/common.wgsl?raw';
import clearSrc from './shaders/clear.wgsl?raw';
import p2gSrc from './shaders/p2g.wgsl?raw';
import gridSrc from './shaders/grid.wgsl?raw';
import g2pSrc from './shaders/g2p.wgsl?raw';
import rasterSrc from './shaders/raster.wgsl?raw';
import disperseSrc from './shaders/disperse.wgsl?raw';
import packSrc from './shaders/pack.wgsl?raw';
import injectSrc from './shaders/inject.wgsl?raw';
import resetSrc from './shaders/reset.wgsl?raw';
import foldSrc from './shaders/fold.wgsl?raw';

/** Titanium White base latent (src/sim/mixbox.c MB_WHITE). */
export const WHITE_LATENT: Latent = [0, 0, 0, 1, 0.00481862, 0.00021851, 0.00295198];

/** Workgroup sizes shared with the shaders (particleIndex() assumes 128). */
export const PARTICLE_WG = 128;
export const GRID_WG = 4;
/** Uniform ring stride (>= minUniformBufferOffsetAlignment). */
const UNIFORM_STRIDE = 256;
/** Operator move (design §6): roll the sheet into a log (FOLD_ROLL_SECONDS), then
 * put it back: dropped whole onto the nip (params.logFeed = 0) or lowered in
 * end-first at params.logFeed units/s. */
/** Pigment load of a masterbatch chunk (mirrors PIGMENT_LOAD in common.wgsl). The clear base
 * carries load 0: pigment is an opaque colourant in a transparent medium, so a node's colour
 * is the Mixbox mix of the pigments present and its load per unit mass sets the opacity. */
export const PIGMENT_LOAD = 12.0;
export const FOLD_ROLL_SECONDS = 1.2;
/** Lowering speed the design doc and the lowered-in e2e case use (sim units / s along the log axis). */
export const FOLD_FEED_SPEED = 0.15;
export const FOLD_TILT = 0.42;                // log axis tilt from vertical toward the viewer (rad)
/** Arc bins and depth slices of the fold's thickness histogram (mirror NB / NS in fold.wgsl). */
export const FOLD_BINS = 128;
export const FOLD_SLICES = 32;
/** Total script length for a given feed: roll, then either let the whole log go (a
 * substep later) or lower it (the material folded in half: L/2 long, plus the
 * tilted end face of a log up to ~0.7 units across) through the nip. */
export function foldDuration(logFeed: number): number {
  if (!(logFeed > 0)) return FOLD_ROLL_SECONDS + 0.02;
  return FOLD_ROLL_SECONDS + (0.5 * GEOMETRY.length + 0.6) / logFeed + 0.3;
}
/** Script length of the lowered-in move at FOLD_FEED_SPEED (≈ 10.5 s). */
export const FOLD_DURATION = foldDuration(FOLD_FEED_SPEED);
/** Cut & fold (fold the cut flap over the cut onto the sheet beside it; fold.wgsl): the fold takes FLOP_SECONDS. */
export const FLOP_SECONDS = 0.8;
export const FLOP_DURATION = FLOP_SECONDS + 0.02;
/** Operator-move modes carried in P.fold.x: 1 = cut & roll (log), 2 / 3 = cut & fold lifting the x < L/2 / x >= L/2 half. */
export type FoldMode = 0 | 1 | 2 | 3;
/** Arc length along the front roll -> z on the bank (the unrolled sheet is compressed by this factor). */
/**
 * Reference material density. The stress force in P2G is scaled by
 * pMass / (pVol * MATERIAL_DENSITY) so that the elastic constants of §4 act
 * on a material of unit density (c = sqrt(E/rho), the CFL note of §4), while
 * pMass = 1 / pVol = h^3/8 stay as in §2 for the grid mass and the volume.
 */
export const MATERIAL_DENSITY = 1;

/** Split a 1D workgroup count into a (x, y) dispatch that respects the per-dimension limit. */
export function dispatchSize(groups: number, maxPerDim: number): [number, number] {
  if (groups <= 0) return [1, 1];
  if (groups <= maxPerDim) return [groups, 1];
  const y = Math.ceil(groups / maxPerDim);
  return [Math.ceil(groups / y), y];
}

/** Smoothstep progress s(t) and ds/dt for the fold script (design §6). */
export function foldProfile(t: number, T: number): { s: number; dsdt: number } {
  const tau = Math.min(Math.max(t / T, 0), 1);
  return { s: tau * tau * (3 - 2 * tau), dsdt: (6 * tau * (1 - tau)) / T };
}

/** IEEE half -> float (for reading the rgba16float volume back). */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

type Resource =
  | { kind: 'storage'; buffer: GPUBuffer; readOnly?: boolean }
  | { kind: 'uniform'; buffer: GPUBuffer }
  | { kind: 'storageTexture'; view: GPUTextureView };

interface Kernel {
  readonly name: string;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
}

interface MutableStats {
  particleCount: number;
  particleCapacity: number;
  materialVolume: number;
  grid: GridDims;
  simTime: number;
  simSecondsPerFrame: number;
  lastStepGpuMs: number;
  rollerAngleFront: number;
  rollerAngleBack: number;
}

export class GpuMpm implements GpuMpmSim {
  readonly config: MillConfig;
  readonly quality: QualitySettings;
  readonly dims: GridDims;
  readonly volumes: RenderVolumes;
  readonly params: MillParams;
  paused = false;

  /** Resolves once every shader compiled and the pipelines validated; rejects with the messages otherwise. */
  readonly ready: Promise<void>;

  private readonly device: GPUDevice;
  private count: number;
  private readonly batch: number;
  private readonly seedCount: number;
  private readonly capacity: number;
  private readonly seeds: Float32Array;
  private readonly statsData: MutableStats;

  // particle SoA
  private readonly bufPos: GPUBuffer;
  private readonly bufVel: GPUBuffer;
  private readonly bufC: GPUBuffer;
  private readonly bufF: GPUBuffer;
  private readonly bufAff: GPUBuffer;
  private readonly bufLat: GPUBuffer;
  private readonly bufFlags: GPUBuffer;
  private readonly bufFold: GPUBuffer;
  /** cut & fold reductions (live pile top, x range, count, per-bin histogram); zeroed on the CPU before each selection */
  private readonly bufFoldInfo: GPUBuffer;
  /** cut & fold spiral tables (per arc bin) + header, built on the GPU from the histogram */
  private readonly bufFoldTables: GPUBuffer;
  // grid
  private readonly bufGMass: GPUBuffer;
  private readonly bufGMom: GPUBuffer;
  private readonly bufGVel: GPUBuffer;
  private readonly bufPMass: GPUBuffer;
  private readonly bufPLat: GPUBuffer;
  private readonly bufPLoad: GPUBuffer;
  // uniforms
  private readonly bufParams: GPUBuffer;
  private readonly paramsData: Float32Array;
  private readonly paramsU32: Uint32Array;
  private readonly bufInject: GPUBuffer;
  private readonly bufInjectAll: GPUBuffer;
  private readonly bufProbe: GPUBuffer;
  private readonly uniformSlots: number;
  private readonly volA: GPUTexture;
  private readonly volB: GPUTexture;
  private readonly volC: GPUTexture;

  private readonly kernels: Record<string, Kernel>;
  private readonly modules: GPUShaderModule[] = [];
  private particleDispatch: [number, number];
  private readonly gridDispatch: [number, number, number];

  // timestamps
  private readonly querySet: GPUQuerySet | null = null;
  private readonly queryResolve: GPUBuffer | null = null;
  private readonly queryStaging: { buffer: GPUBuffer; state: 'free' | 'copied' | 'mapping' }[] = [];

  // cut & fold script state
  private foldPending = false;
  private foldActive = false;
  private foldTime = 0;
  /** feed of the running / pending cut & roll (captured when the move starts, so a
   *  slider change mid-move cannot change its length under it) */
  private foldFeed = 0;
  private foldMode: FoldMode = 0;

  private readbackStaging: GPUBuffer | null = null;
  private destroyed = false;

  constructor(opts: GpuMpmSimOptions) {
    const { device, config } = opts;
    this.device = device;
    this.config = config;
    this.quality = config.quality;
    this.params = config.params;
    this.dims = gridDims(config.quality);
    const d = this.dims;

    this.batch = config.batch ?? 1;
    this.seeds = seedBankPositions(config.quality, config.params, 1234, this.batch);
    this.clampSeeds();
    this.seedCount = this.seeds.length / 3;
    this.count = this.seedCount;
    // reserve room for pigment chunks (new coloured material added by the user)
    this.capacity = Math.max(1, Math.ceil(this.seedCount * (1 + PIGMENT_POOL_FRACTION)));
    const n = this.capacity;

    const mk = (floats: number, extra = 0): GPUBuffer =>
      device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra });
    this.bufPos = mk(4 * n);
    this.bufVel = mk(4 * n);
    this.bufC = mk(9 * n);
    this.bufF = mk(9 * n);
    this.bufAff = mk(9 * n);
    this.bufLat = mk(7 * n);
    this.bufFlags = mk(n);
    this.bufFold = mk(4 * n);
    this.bufFoldInfo = mk(8 + 2 * FOLD_BINS + 2 * FOLD_BINS * FOLD_SLICES);
    this.bufFoldTables = mk(8 * FOLD_BINS + 8 + 2 * FOLD_BINS * (FOLD_SLICES + 1));
    this.bufGMass = mk(d.nodeCount);
    this.bufGMom = mk(3 * d.nodeCount);
    this.bufGVel = mk(4 * d.nodeCount);
    this.bufPMass = mk(d.nodeCount);
    this.bufPLat = mk(7 * d.nodeCount);
    this.bufPLoad = mk(d.nodeCount);

    this.uniformSlots = config.quality.substepsPerFrame + 1;
    this.bufParams = device.createBuffer({ size: UNIFORM_STRIDE * this.uniformSlots, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsData = new Float32Array((UNIFORM_STRIDE / 4) * this.uniformSlots);
    this.paramsU32 = new Uint32Array(this.paramsData.buffer);
    this.bufInject = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufInjectAll = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufProbe = device.createBuffer({ label: 'GpuMpm-probe', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

    const texUsage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    this.volA = device.createTexture({ size: [d.nx, d.ny, d.nz], dimension: '3d', format: 'rgba16float', usage: texUsage, label: 'volA' });
    this.volB = device.createTexture({ size: [d.nx, d.ny, d.nz], dimension: '3d', format: 'rgba16float', usage: texUsage, label: 'volB' });
    this.volC = device.createTexture({ size: [d.nx, d.ny, d.nz], dimension: '3d', format: 'rgba16float', usage: texUsage, label: 'volC' });
    this.volumes = { volA: this.volA, volB: this.volB, volC: this.volC, dims: d };

    const maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    this.particleDispatch = dispatchSize(Math.ceil(Math.max(this.count, 1) / PARTICLE_WG), maxDim);
    this.gridDispatch = [Math.ceil(d.nx / GRID_WG), Math.ceil(d.ny / GRID_WG), Math.ceil(d.nz / GRID_WG)];

    this.statsData = {
      particleCount: this.count,
      particleCapacity: this.capacity,
      materialVolume: (this.count * d.h * d.h * d.h) / 8,
      grid: d,
      simTime: 0,
      simSecondsPerFrame: config.quality.dt * config.quality.substepsPerFrame,
      lastStepGpuMs: NaN,
      rollerAngleFront: 0,
      rollerAngleBack: 0
    };

    if (device.features.has('timestamp-query')) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.queryResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      for (let i = 0; i < 3; i++) {
        this.queryStaging.push({ buffer: device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), state: 'free' });
      }
    }

    device.pushErrorScope('validation');
    this.kernels = this.buildKernels();
    const scope = device.popErrorScope();
    this.ready = this.checkCompilation(scope);

    this.uploadInitialState();
  }

  get stats(): SimStats {
    return this.statsData;
  }

  // ---------------------------------------------------------------------------
  // construction helpers
  // ---------------------------------------------------------------------------

  private clampSeeds(): void {
    const h = this.dims.h;
    const dom = GEOMETRY.domain;
    for (let i = 0; i < this.seeds.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        const v = this.seeds[i + a];
        this.seeds[i + a] = Math.min(Math.max(v, 1.5 * h), dom[a] - 1.5 * h);
      }
    }
  }

  private makeKernel(name: string, module: GPUShaderModule, entryPoint: string, resources: Resource[]): Kernel {
    const device = this.device;
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } }
    ];
    const bindings: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.bufParams, offset: 0, size: UNIFORM_STRIDE } }
    ];
    resources.forEach((r, i) => {
      const binding = i + 1;
      if (r.kind === 'storage') {
        entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: r.readOnly ? 'read-only-storage' : 'storage' } });
        bindings.push({ binding, resource: { buffer: r.buffer } });
      } else if (r.kind === 'uniform') {
        entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        bindings.push({ binding, resource: { buffer: r.buffer } });
      } else {
        entries.push({ binding, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } });
        bindings.push({ binding, resource: r.view });
      }
    });
    const bgl = device.createBindGroupLayout({ label: `${name}-bgl`, entries });
    const pipeline = device.createComputePipeline({
      label: name,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint }
    });
    const bindGroup = device.createBindGroup({ label: `${name}-bg`, layout: bgl, entries: bindings });
    return { name, pipeline, bindGroup };
  }

  private buildKernels(): Record<string, Kernel> {
    const device = this.device;
    const mod = (name: string, src: string): GPUShaderModule => {
      const m = device.createShaderModule({ label: name, code: commonSrc + '\n' + src });
      this.modules.push(m);
      return m;
    };
    const rw = (buffer: GPUBuffer): Resource => ({ kind: 'storage', buffer });
    const ro = (buffer: GPUBuffer): Resource => ({ kind: 'storage', buffer, readOnly: true });

    const clearMod = mod('clear', clearSrc);
    const foldMod = mod('fold', foldSrc);
    const injectMod = mod('inject', injectSrc);
    const clearRes = [rw(this.bufGMass), rw(this.bufGMom), rw(this.bufGVel), rw(this.bufPMass), rw(this.bufPLat), rw(this.bufPLoad)];
    const foldRes = [rw(this.bufPos), rw(this.bufVel), rw(this.bufC), rw(this.bufF), rw(this.bufAff), rw(this.bufFlags), rw(this.bufFold),
      rw(this.bufFoldInfo), ro(this.bufPMass), rw(this.bufFoldTables)];
    const volAView = this.volA.createView({ dimension: '3d' });
    const volBView = this.volB.createView({ dimension: '3d' });
    const volCView = this.volC.createView({ dimension: '3d' });

    return {
      clearGrid: this.makeKernel('clearGrid', clearMod, 'clearGrid', clearRes),
      clearRaster: this.makeKernel('clearRaster', clearMod, 'clearRaster', clearRes),
      p2g: this.makeKernel('p2g', mod('p2g', p2gSrc), 'main',
        [ro(this.bufPos), ro(this.bufVel), ro(this.bufAff), ro(this.bufFlags), rw(this.bufGMass), rw(this.bufGMom)]),
      grid: this.makeKernel('grid', mod('grid', gridSrc), 'main', [ro(this.bufGMass), ro(this.bufGMom), rw(this.bufGVel), ro(this.bufPMass)]),
      g2p: this.makeKernel('g2p', mod('g2p', g2pSrc), 'main',
        [rw(this.bufPos), rw(this.bufVel), rw(this.bufC), rw(this.bufF), rw(this.bufAff), ro(this.bufFlags), ro(this.bufGVel),
          ro(this.bufGMass), rw(this.bufFoldInfo), ro(this.bufFold)]),
      raster: this.makeKernel('raster', mod('raster', rasterSrc), 'main',
        [ro(this.bufPos), ro(this.bufLat), rw(this.bufPMass), rw(this.bufPLat), rw(this.bufPLoad)]),
      disperse: this.makeKernel('disperse', mod('disperse', disperseSrc), 'main',
        [rw(this.bufPos), ro(this.bufC), rw(this.bufLat), ro(this.bufPMass), ro(this.bufPLat), ro(this.bufFlags), ro(this.bufPLoad)]),
      pack: this.makeKernel('pack', mod('pack', packSrc), 'main',
        [ro(this.bufPMass), ro(this.bufPLat), { kind: 'storageTexture', view: volAView }, { kind: 'storageTexture', view: volBView }, ro(this.bufPLoad),
          { kind: 'storageTexture', view: volCView }]),
      inject: this.makeKernel('inject', injectMod, 'main', [{ kind: 'uniform', buffer: this.bufInject }, rw(this.bufPos), rw(this.bufLat), rw(this.bufProbe)]),
      probe: this.makeKernel('probe', injectMod, 'probeColumn', [{ kind: 'uniform', buffer: this.bufInject }, rw(this.bufPos), rw(this.bufLat), rw(this.bufProbe)]),
      injectAll: this.makeKernel('injectAll', injectMod, 'main', [{ kind: 'uniform', buffer: this.bufInjectAll }, rw(this.bufPos), rw(this.bufLat), rw(this.bufProbe)]),
      reset: this.makeKernel('reset', mod('reset', resetSrc), 'main',
        [rw(this.bufVel), rw(this.bufC), rw(this.bufF), rw(this.bufAff), rw(this.bufFlags)]),
      foldSelect: this.makeKernel('foldSelect', foldMod, 'select_', foldRes),
      foldTables: this.makeKernel('foldTables', foldMod, 'tables_', foldRes),
      foldMove: this.makeKernel('foldMove', foldMod, 'move_', foldRes),
      foldFinish: this.makeKernel('foldFinish', foldMod, 'finish_', foldRes)
    };
  }

  private async checkCompilation(scope: Promise<GPUError | null>): Promise<void> {
    const problems: string[] = [];
    for (const m of this.modules) {
      const info = await m.getCompilationInfo();
      for (const msg of info.messages) {
        const line = `[${m.label}] ${msg.type} at ${msg.lineNum}:${msg.linePos}: ${msg.message}`;
        if (msg.type === 'error') problems.push(line);
        else console.warn('wgsl', line);
      }
    }
    const err = await scope;
    if (err) problems.push(`validation: ${err.message}`);
    if (problems.length) {
      for (const p of problems) console.error('GpuMpm:', p);
      throw new Error('GpuMpm shader/pipeline errors:\n' + problems.join('\n'));
    }
  }

  private uploadInitialState(): void {
    const q = this.device.queue;
    const pos4 = new Float32Array(4 * Math.max(this.count, 1));
    for (let i = 0; i < this.count; i++) {
      pos4[4 * i] = this.seeds[3 * i];
      pos4[4 * i + 1] = this.seeds[3 * i + 1];
      pos4[4 * i + 2] = this.seeds[3 * i + 2];
      pos4[4 * i + 3] = 0; // clear base: no pigment load (colour comes only from masterbatch)
    }
    q.writeBuffer(this.bufPos, 0, pos4);
    // "set all" inject uniform: white base
    const inj = new Float32Array(16);
    inj.set([0, 0, 0, 1e9], 0);
    inj.set(WHITE_LATENT.slice(0, 4), 4);
    inj.set([WHITE_LATENT[4], WHITE_LATENT[5], WHITE_LATENT[6], 1], 8);
    new Uint32Array(inj.buffer)[12] = 1;
    q.writeBuffer(this.bufInjectAll, 0, inj);
    this.writeParams(0, 0);
    q.writeBuffer(this.bufParams, 0, this.paramsData, 0, (UNIFORM_STRIDE / 4) * this.uniformSlots);

    const enc = this.device.createCommandEncoder({ label: 'GpuMpm-init' });
    const pass = enc.beginComputePass();
    this.dispatchParticles(pass, this.kernels.reset, 0);
    this.dispatchParticles(pass, this.kernels.injectAll, 0);
    this.encodeFrameKernels(pass, 0, false);
    pass.end();
    q.submit([enc.finish()]);
  }

  // ---------------------------------------------------------------------------
  // uniforms
  // ---------------------------------------------------------------------------

  /**
   * Fill uniform slot `slot` (fold state: s and ds/dt for that substep). All
   * time-dependent physics uses the fixed substep dt; the per-frame kernels
   * (dispersion) use dt * substepsPerFrame, never the wall-clock frame time.
   */
  private writeParams(slot: number, foldT: number, foldActive = false): void {
    const d = this.dims;
    const q = this.quality;
    const p = this.params;
    const { back, front } = rollerPoses(p);
    const { mu, lambda } = lameParameters(this.config.material);
    const h = d.h;
    const pVol = (h * h * h) / 8;
    const pMass = 1;
    const o = (UNIFORM_STRIDE / 4) * slot;
    const f = this.paramsData;
    const u = this.paramsU32;
    u[o + 0] = d.nx; u[o + 1] = d.ny; u[o + 2] = d.nz; u[o + 3] = this.count;
    f.set([h, 1 / h, q.dt, p.gravity], o + 4);
    f.set([GEOMETRY.domain[0], GEOMETRY.domain[1], GEOMETRY.domain[2], q.dt * q.substepsPerFrame], o + 8);
    f.set([back.axisY, back.axisZ, back.omegaX, p.backFriction], o + 12);
    f.set([front.axisY, front.axisZ, front.omegaX, GEOMETRY.radius], o + 16);
    f.set([mu, lambda, this.config.material.thetaC, this.config.material.thetaS], o + 20);
    f.set([pVol, pMass, pMass / (pVol * MATERIAL_DENSITY), p.dispersion], o + 24);
    f.set([foldActive ? this.foldMode : 0, foldT, FOLD_ROLL_SECONDS, this.foldFeed], o + 28);
    // fold: the live bank top is reduced on the GPU (fold.wgsl); fold2.x is only the fallback
    const yMax = GEOMETRY.domain[1] - 3 * h;
    const bankTopFallback = Math.min(bankTopY(this.batch, this.config.params), yMax);
    const gap = front.axisZ - back.axisZ - 2 * GEOMETRY.radius;
    f.set([bankTopFallback, GEOMETRY.length, gap, GEOMETRY.nipZ], o + 32);
    // front-roll tack band: the adhesion layer is as thick as the sheet the nip
    // produces (the gap) plus one cell of stencil slack, so the whole sheet
    // rides the roll instead of only its innermost layer (design §3.3).
    const tackBand = p.tackCells > 0 ? p.tackCells * h : p.gap + 1.0 * h;
    f.set([tackBand, 0.5 * h, 2 * h, 0.6], o + 36);
    f.set([GEOMETRY.bankHalfDepth, 0.5 * h, FOLD_TILT, yMax], o + 40);
  }

  // ---------------------------------------------------------------------------
  // dispatch helpers
  // ---------------------------------------------------------------------------

  /** Change the live particle count (chunks added / reset) and the dispatch that covers it. */
  private setCount(n: number): void {
    this.count = n;
    this.statsData.particleCount = n;
    const h = this.dims.h;
    this.statsData.materialVolume = (n * h * h * h) / 8;
    const maxDim = this.device.limits.maxComputeWorkgroupsPerDimension;
    this.particleDispatch = dispatchSize(Math.ceil(Math.max(n, 1) / PARTICLE_WG), maxDim);
  }

  private dispatchParticles(pass: GPUComputePassEncoder, k: Kernel, slot: number): void {
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.bindGroup, [slot * UNIFORM_STRIDE]);
    pass.dispatchWorkgroups(this.particleDispatch[0], this.particleDispatch[1], 1);
  }

  private dispatchGrid(pass: GPUComputePassEncoder, k: Kernel, slot: number): void {
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.bindGroup, [slot * UNIFORM_STRIDE]);
    pass.dispatchWorkgroups(this.gridDispatch[0], this.gridDispatch[1], this.gridDispatch[2]);
  }

  /** raster (+ disperse) + pack, using uniform slot `slot`. */
  private encodeFrameKernels(pass: GPUComputePassEncoder, slot: number, disperse: boolean): void {
    const k = this.kernels;
    this.dispatchGrid(pass, k.clearRaster, slot);
    this.dispatchParticles(pass, k.raster, slot);
    if (disperse) this.dispatchParticles(pass, k.disperse, slot);
    this.dispatchGrid(pass, k.pack, slot);
  }

  // ---------------------------------------------------------------------------
  // GpuMpmSim
  // ---------------------------------------------------------------------------

  step(encoder: GPUCommandEncoder, frameDtSeconds: number): void {
    // frameDtSeconds is part of the interface (the app's wall-clock frame time)
    // but the physics is fixed-step: nothing below depends on it.
    void frameDtSeconds;
    if (this.destroyed) return;
    this.pollTimestamps();
    // paused: no uniform write, no substeps, no raster/disperse/pack, no timestamps
    // (addPigment / clearPigment / reset re-raster themselves).
    if (this.paused) return;
    const q = this.quality;
    const k = this.kernels;
    const substeps = q.substepsPerFrame;
    const frameSlot = q.substepsPerFrame;
    const { back, front } = rollerPoses(this.params);

    // uniform ring: one slot per substep (fold script state differs per substep)
    let foldActive = this.foldActive || this.foldPending;
    let foldT = this.foldPending ? 0 : this.foldTime;
    const duration = this.foldMode === 1 ? foldDuration(this.foldFeed) : FLOP_DURATION;
    const finishAt: number[] = [];
    for (let s = 0; s < substeps; s++) {
      this.writeParams(s, foldT, foldActive);
      if (foldActive) {
        foldT += q.dt;
        if (foldT >= duration) { finishAt.push(s); foldActive = false; }
      }
    }
    this.writeParams(frameSlot, foldT, foldActive);
    this.device.queue.writeBuffer(this.bufParams, 0, this.paramsData, 0, (UNIFORM_STRIDE / 4) * this.uniformSlots);
    if (this.foldPending) {
      // reductions: [0] live pile top y bits, [1] max x bits, [2] min x bits (atomicMin, init +inf),
      // [3] count, then the per-(half, bin) particle counts and dr sums
      const init = new Uint32Array(8 + 2 * FOLD_BINS + 2 * FOLD_BINS * FOLD_SLICES);
      init[2] = 0x7f800000;
      this.device.queue.writeBuffer(this.bufFoldInfo, 0, init);
    } else if (this.foldActive) {
      // the pile top under the log is reduced afresh every frame (g2p, every substep)
      this.device.queue.writeBuffer(this.bufFoldInfo, 0, new Uint32Array([0]));
    }

    const timestampWrites: GPUComputePassTimestampWrites | undefined = this.querySet
      ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      : undefined;
    const pass = encoder.beginComputePass({ label: 'GpuMpm-step', timestampWrites });
    if (this.foldPending) {
      this.dispatchParticles(pass, k.foldSelect, 0);
      pass.setPipeline(k.foldTables.pipeline);
      pass.setBindGroup(0, k.foldTables.bindGroup, [0]);
      pass.dispatchWorkgroups(1, 1, 1);
      this.foldPending = false;
      this.foldActive = true;
      this.foldTime = 0;
    }
    for (let s = 0; s < substeps; s++) {
      if (this.foldActive) this.dispatchParticles(pass, k.foldMove, s);
      this.dispatchGrid(pass, k.clearGrid, s);
      this.dispatchParticles(pass, k.p2g, s);
      this.dispatchGrid(pass, k.grid, s);
      this.dispatchParticles(pass, k.g2p, s);
      if (this.foldActive) {
        this.foldTime += q.dt;
        if (finishAt.includes(s)) {
          this.dispatchParticles(pass, k.foldFinish, s);
          this.foldActive = false;
        }
      }
    }
    this.encodeFrameKernels(pass, frameSlot, true);
    pass.end();

    if (this.querySet && this.queryResolve) {
      const slot = this.queryStaging.find((s) => s.state === 'free');
      if (slot) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
        encoder.copyBufferToBuffer(this.queryResolve, 0, slot.buffer, 0, 16);
        slot.state = 'copied';
      }
    }

    const st = this.statsData;
    st.simTime += q.dt * substeps;
    st.rollerAngleFront += front.omegaX * q.dt * substeps;
    st.rollerAngleBack += back.omegaX * q.dt * substeps;
    st.simSecondsPerFrame = q.dt * q.substepsPerFrame;
  }

  /** Map any staging buffer whose copy has been submitted (called at the start of the next step). */
  private pollTimestamps(): void {
    for (const slot of this.queryStaging) {
      if (slot.state !== 'copied') continue;
      slot.state = 'mapping';
      slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
        if (this.destroyed) return;
        const ts = new BigUint64Array(slot.buffer.getMappedRange());
        // an end stamp below the start stamp (timer reset / unordered stamps) is not a sample
        const valid = ts[1] >= ts[0];
        const ns = valid ? Number(ts[1] - ts[0]) : NaN;
        slot.buffer.unmap();
        slot.state = 'free';
        if (valid && Number.isFinite(ns)) this.statsData.lastStepGpuMs = ns / 1e6;
      }).catch(() => { slot.state = 'free'; });
    }
  }

  reset(): void {
    if (this.destroyed) return;
    this.foldPending = false;
    this.foldActive = false;
    this.foldTime = 0;
    this.statsData.simTime = 0;
    this.statsData.rollerAngleFront = 0;
    this.statsData.rollerAngleBack = 0;
    this.setCount(this.seedCount);
    this.uploadInitialState();
  }

  addPigment(center: readonly [number, number, number], radius: number, latent: Latent, strength = 1): void {
    if (this.destroyed) return;
    this.device.queue.writeBuffer(this.bufInject, 0, this.injectData(center, radius, latent, strength, 0));
    this.runParticleKernel(this.kernels.inject);
  }

  /** Highest material in the column |x - cx| < r, |z - cz| < r (GPU probe, one small readback); 0 if the column is empty. */
  private async probeColumnTop(x: number, z: number, radius: number): Promise<number> {
    const q = this.device.queue;
    q.writeBuffer(this.bufInject, 0, this.injectData([x, 0, z], radius, WHITE_LATENT, 0, 2));
    q.writeBuffer(this.bufProbe, 0, new Uint32Array([0, 0, 0, 0]));
    const staging = this.device.createBuffer({ label: 'GpuMpm-probe-read', size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder({ label: 'GpuMpm-probe' });
    const pass = enc.beginComputePass();
    this.dispatchParticles(pass, this.kernels.probe, 0);
    pass.end();
    enc.copyBufferToBuffer(this.bufProbe, 0, staging, 0, 16);
    q.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const top = new Float32Array(staging.getMappedRange())[0];
    staging.unmap();
    staging.destroy();
    return top;
  }

  async addPigmentChunk(x: number, z: number, radius: number, latent: Latent): Promise<number> {
    if (this.destroyed) return 0;
    const h = this.dims.h;
    const per = GEOMETRY.seedPerAxis;
    const dom = GEOMETRY.domain;
    const r = Math.max(radius, 2 * h);
    const cx = Math.min(Math.max(x, 1.5 * h + r), dom[0] - 1.5 * h - r);
    const cz = Math.min(Math.max(z, 1.5 * h + r), dom[2] - 1.5 * h - r);
    // set the chunk down touching whatever is in that column (the bank, the nip
    // floor, an earlier chunk), as an operator does, instead of dropping it from
    // the seeded bank height: falling into the empty V cost ~0.6 s before the
    // rolls could even start on it
    const top = await this.probeColumnTop(cx, cz, r);
    if (this.destroyed) return 0;
    const rest = top > 0 ? top + r + 0.5 * h : bankTopY(this.batch, this.config.params) + r + 0.02;
    const cy = Math.min(Math.max(rest, GEOMETRY.axisY + r), dom[1] - 1.5 * h - r);
    const room = this.capacity - this.count;
    if (room <= 0) return 0;
    const pts: number[] = [];
    const c0 = Math.floor((cx - r) / h), c1 = Math.ceil((cx + r) / h);
    const d0 = Math.floor((cy - r) / h), d1 = Math.ceil((cy + r) / h);
    const e0 = Math.floor((cz - r) / h), e1 = Math.ceil((cz + r) / h);
    let seed = (Math.random() * 0xffffffff) >>> 0;
    const rnd = (): number => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed / 4294967296; };
    outer: for (let k = e0; k < e1; k++) for (let j = d0; j < d1; j++) for (let i = c0; i < c1; i++) {
      for (let sz = 0; sz < per; sz++) for (let sy = 0; sy < per; sy++) for (let sx = 0; sx < per; sx++) {
        const px = (i + (sx + 0.5 + (rnd() - 0.5) * 0.35) / per) * h;
        const py = (j + (sy + 0.5 + (rnd() - 0.5) * 0.35) / per) * h;
        const pz = (k + (sz + 0.5 + (rnd() - 0.5) * 0.35) / per) * h;
        const dx = px - cx, dy = py - cy, dz = pz - cz;
        if (dx * dx + dy * dy + dz * dz > r * r) continue;
        pts.push(px, py, pz);
        if (pts.length / 3 >= room) break outer;
      }
    }
    const m = pts.length / 3;
    if (m === 0) return 0;
    const base = this.count;
    const q = this.device.queue;
    const pos4 = new Float32Array(4 * m);
    const lat = new Float32Array(7 * m);
    const F = new Float32Array(9 * m);
    for (let i = 0; i < m; i++) {
      pos4[4 * i] = pts[3 * i]; pos4[4 * i + 1] = pts[3 * i + 1]; pos4[4 * i + 2] = pts[3 * i + 2];
      pos4[4 * i + 3] = PIGMENT_LOAD; // masterbatch: concentrated pigment
      for (let c = 0; c < 7; c++) lat[7 * i + c] = latent[c];
      F[9 * i] = 1; F[9 * i + 4] = 1; F[9 * i + 8] = 1;
    }
    q.writeBuffer(this.bufPos, 16 * base, pos4);
    q.writeBuffer(this.bufVel, 16 * base, new Float32Array(4 * m));
    q.writeBuffer(this.bufC, 36 * base, new Float32Array(9 * m));
    q.writeBuffer(this.bufF, 36 * base, F);
    q.writeBuffer(this.bufAff, 36 * base, new Float32Array(9 * m));
    q.writeBuffer(this.bufLat, 28 * base, lat);
    q.writeBuffer(this.bufFlags, 4 * base, new Uint32Array(m));
    q.writeBuffer(this.bufFold, 16 * base, new Float32Array(4 * m));
    this.setCount(base + m);
    // refresh the uniform's particle count and the render volume right away
    this.writeParams(0, 0);
    q.writeBuffer(this.bufParams, 0, this.paramsData, 0, UNIFORM_STRIDE / 4);
    const enc = this.device.createCommandEncoder({ label: 'GpuMpm-chunk' });
    const pass = enc.beginComputePass();
    this.encodeFrameKernels(pass, 0, false);
    pass.end();
    q.submit([enc.finish()]);
    return m;
  }

  addPigmentOnSurface(x: number, z: number, radius: number, latent: Latent, strength = 1): void {
    if (this.destroyed) return;
    // fallback centre (used when the column is empty): the seeded bank top
    const yTop = bankTopY(this.batch, this.config.params) - 0.5 * radius;
    this.device.queue.writeBuffer(this.bufInject, 0, this.injectData([x, yTop, z], radius, latent, strength, 2));
    this.device.queue.writeBuffer(this.bufProbe, 0, new Uint32Array([0, 0, 0, 0]));
    const enc = this.device.createCommandEncoder({ label: 'GpuMpm-inject-surface' });
    const pass = enc.beginComputePass();
    this.dispatchParticles(pass, this.kernels.probe, 0);
    this.dispatchParticles(pass, this.kernels.inject, 0);
    this.encodeFrameKernels(pass, 0, false);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private injectData(center: readonly [number, number, number], radius: number, latent: Latent, strength: number, mode: number): Float32Array {
    const inj = new Float32Array(16);
    inj.set([center[0], center[1], center[2], Math.max(radius, 1e-6)], 0);
    inj.set(latent.slice(0, 4), 4);
    inj.set([latent[4], latent[5], latent[6], strength], 8);
    new Uint32Array(inj.buffer)[12] = mode;
    return inj;
  }

  clearPigment(): void {
    if (this.destroyed) return;
    this.runParticleKernel(this.kernels.injectAll);
  }

  /** Submit a standalone particle kernel followed by a raster+pack so the volume reflects it. */
  private runParticleKernel(k: Kernel): void {
    const enc = this.device.createCommandEncoder({ label: `GpuMpm-${k.name}` });
    const pass = enc.beginComputePass();
    this.dispatchParticles(pass, k, 0);
    this.encodeFrameKernels(pass, 0, false);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  cutAndFold(): void {
    if (this.destroyed || this.operatorBusy) return;
    this.foldMode = 1;
    this.foldFeed = Math.max(this.params.logFeed, 0);
    this.foldPending = true;
  }

  cutAndFlop(side: 'left' | 'right'): void {
    if (this.destroyed || this.operatorBusy) return;
    this.foldMode = side === 'left' ? 2 : 3;
    this.foldPending = true;
  }

  get operatorBusy(): boolean {
    return this.foldActive || this.foldPending;
  }

  async readParticles(): Promise<ParticleSnapshot> {
    const n = this.count;
    // positions / velocities are vec4 on the GPU; the snapshot is xyz-interleaved
    const sizes = [4 * n, 4 * n, 7 * n, 9 * n, n].map((f) => f * 4);
    const total = sizes.reduce((a, b) => a + b, 0);
    if (!this.readbackStaging || this.readbackStaging.size < total) {
      this.readbackStaging?.destroy();
      this.readbackStaging = this.device.createBuffer({ size: Math.max(total, 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'GpuMpm-readback' });
    }
    const staging = this.readbackStaging;
    const enc = this.device.createCommandEncoder();
    let off = 0;
    for (const [buf, size] of [[this.bufPos, sizes[0]], [this.bufVel, sizes[1]], [this.bufLat, sizes[2]], [this.bufF, sizes[3]], [this.bufFlags, sizes[4]]] as [GPUBuffer, number][]) {
      if (size > 0) enc.copyBufferToBuffer(buf, 0, staging, off, size);
      off += size;
    }
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, Math.max(total, 16));
    const bytes = staging.getMappedRange(0, Math.max(total, 16));
    off = 0;
    const take = (size: number): ArrayBuffer => { const s = bytes.slice(off, off + size); off += size; return s; };
    const strip = (buf: ArrayBuffer): Float32Array => {
      const v4 = new Float32Array(buf);
      const v3 = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) { v3[3 * i] = v4[4 * i]; v3[3 * i + 1] = v4[4 * i + 1]; v3[3 * i + 2] = v4[4 * i + 2]; }
      return v3;
    };
    const posRaw = new Float32Array(take(sizes[0]));
    const loads = new Float32Array(n);
    for (let i = 0; i < n; i++) loads[i] = posRaw[4 * i + 3];
    const snap: ParticleSnapshot = {
      count: n,
      positions: strip(posRaw.buffer),
      velocities: strip(take(sizes[1])),
      loads,
      latents: new Float32Array(take(sizes[2])),
      deformation: new Float32Array(take(sizes[3])),
      flags: new Uint32Array(take(sizes[4]))
    };
    staging.unmap();
    return snap;
  }

  async readDensity(): Promise<Float32Array> {
    const d = this.dims;
    const bytesPerRow = Math.ceil((d.nx * 8) / 256) * 256;
    const size = bytesPerRow * d.ny * d.nz;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'GpuMpm-density' });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.volA }, { buffer: staging, bytesPerRow, rowsPerImage: d.ny }, [d.nx, d.ny, d.nz]);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(staging.getMappedRange());
    const out = new Float32Array(d.nodeCount);
    const rowU16 = bytesPerRow / 2;
    for (let k = 0; k < d.nz; k++) {
      for (let j = 0; j < d.ny; j++) {
        const row = (k * d.ny + j) * rowU16;
        const dst = (k * d.ny + j) * d.nx;
        for (let i = 0; i < d.nx; i++) out[dst + i] = halfToFloat(u16[row + i * 4]);
      }
    }
    staging.unmap();
    staging.destroy();
    return out;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.bufPos, this.bufVel, this.bufC, this.bufF, this.bufAff, this.bufLat, this.bufFlags, this.bufFold, this.bufFoldInfo, this.bufFoldTables,
      this.bufGMass, this.bufGMom, this.bufGVel, this.bufPMass, this.bufPLat, this.bufParams, this.bufInject, this.bufInjectAll, this.bufProbe, this.bufPLoad]) {
      b.destroy();
    }
    this.readbackStaging?.destroy();
    this.queryResolve?.destroy();
    for (const s of this.queryStaging) s.buffer.destroy();
    this.querySet?.destroy();
    this.volA.destroy();
    this.volB.destroy();
    this.volC.destroy();
  }
}
