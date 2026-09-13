/**
 * ColorMill v2 — WebGPU compute MLS-MPM solver for the two-roll mill.
 * Implements GpuMpmSim (src/sim/types.ts) per docs/design-v2.md §1–§7.
 *
 * Storage is structure-of-arrays (flat f32 buffers): positions (3), velocities
 * (3), C (9, row-major), F (9, row-major), Cauchy stress of the current F
 * (9, row-major; produced by g2p so p2g does not repeat the SVD), latent (7),
 * flags (u32) and the cut-and-fold start pose (4). Grid accumulators are i32
 * fixed point (mass 2^20, momentum 2^16, latent·mass 2^18) plus an f32
 * velocity grid. The render volume is two rgba16float 3D storage textures.
 */
import {
  GEOMETRY, gridDims, lameParameters, rollerPoses, seedBankPositions,
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
/** Cut & fold script length (design §6). */
export const FOLD_DURATION = 1.2;
export const FOLD_LIFT = 0.25;
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
  private readonly count: number;
  private readonly seeds: Float32Array;
  private readonly statsData: MutableStats;

  // particle SoA
  private readonly bufPos: GPUBuffer;
  private readonly bufVel: GPUBuffer;
  private readonly bufC: GPUBuffer;
  private readonly bufF: GPUBuffer;
  private readonly bufS: GPUBuffer;
  private readonly bufLat: GPUBuffer;
  private readonly bufFlags: GPUBuffer;
  private readonly bufFold: GPUBuffer;
  // grid
  private readonly bufGMass: GPUBuffer;
  private readonly bufGMom: GPUBuffer;
  private readonly bufGVel: GPUBuffer;
  private readonly bufPMass: GPUBuffer;
  private readonly bufPLat: GPUBuffer;
  // uniforms
  private readonly bufParams: GPUBuffer;
  private readonly paramsData: Float32Array;
  private readonly paramsU32: Uint32Array;
  private readonly bufInject: GPUBuffer;
  private readonly bufInjectAll: GPUBuffer;
  private readonly uniformSlots: number;
  private readonly volA: GPUTexture;
  private readonly volB: GPUTexture;

  private readonly kernels: Record<string, Kernel>;
  private readonly modules: GPUShaderModule[] = [];
  private readonly particleDispatch: [number, number];
  private readonly gridDispatch: [number, number, number];

  // timestamps
  private readonly querySet: GPUQuerySet | null = null;
  private readonly queryResolve: GPUBuffer | null = null;
  private readonly queryStaging: { buffer: GPUBuffer; state: 'free' | 'copied' | 'mapping' }[] = [];

  // cut & fold script state
  private foldPending = false;
  private foldActive = false;
  private foldTime = 0;

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

    this.seeds = seedBankPositions(config.quality, config.params);
    this.clampSeeds();
    this.count = this.seeds.length / 3;
    const n = Math.max(this.count, 1);

    const mk = (floats: number, extra = 0): GPUBuffer =>
      device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra });
    this.bufPos = mk(3 * n);
    this.bufVel = mk(3 * n);
    this.bufC = mk(9 * n);
    this.bufF = mk(9 * n);
    this.bufS = mk(9 * n);
    this.bufLat = mk(7 * n);
    this.bufFlags = mk(n);
    this.bufFold = mk(4 * n);
    this.bufGMass = mk(d.nodeCount);
    this.bufGMom = mk(3 * d.nodeCount);
    this.bufGVel = mk(3 * d.nodeCount);
    this.bufPMass = mk(d.nodeCount);
    this.bufPLat = mk(7 * d.nodeCount);

    this.uniformSlots = config.quality.substepsPerFrame + 1;
    this.bufParams = device.createBuffer({ size: UNIFORM_STRIDE * this.uniformSlots, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsData = new Float32Array((UNIFORM_STRIDE / 4) * this.uniformSlots);
    this.paramsU32 = new Uint32Array(this.paramsData.buffer);
    this.bufInject = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufInjectAll = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const texUsage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    this.volA = device.createTexture({ size: [d.nx, d.ny, d.nz], dimension: '3d', format: 'rgba16float', usage: texUsage, label: 'volA' });
    this.volB = device.createTexture({ size: [d.nx, d.ny, d.nz], dimension: '3d', format: 'rgba16float', usage: texUsage, label: 'volB' });
    this.volumes = { volA: this.volA, volB: this.volB, dims: d };

    const maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    this.particleDispatch = dispatchSize(Math.ceil(n / PARTICLE_WG), maxDim);
    this.gridDispatch = [Math.ceil(d.nx / GRID_WG), Math.ceil(d.ny / GRID_WG), Math.ceil(d.nz / GRID_WG)];

    this.statsData = {
      particleCount: this.count,
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
    const clearRes = [rw(this.bufGMass), rw(this.bufGMom), rw(this.bufGVel), rw(this.bufPMass), rw(this.bufPLat)];
    const foldRes = [rw(this.bufPos), rw(this.bufVel), rw(this.bufC), rw(this.bufF), rw(this.bufS), rw(this.bufFlags), rw(this.bufFold)];
    const volAView = this.volA.createView({ dimension: '3d' });
    const volBView = this.volB.createView({ dimension: '3d' });

    return {
      clearGrid: this.makeKernel('clearGrid', clearMod, 'clearGrid', clearRes),
      clearRaster: this.makeKernel('clearRaster', clearMod, 'clearRaster', clearRes),
      p2g: this.makeKernel('p2g', mod('p2g', p2gSrc), 'main',
        [ro(this.bufPos), ro(this.bufVel), ro(this.bufC), ro(this.bufS), ro(this.bufFlags), rw(this.bufGMass), rw(this.bufGMom)]),
      grid: this.makeKernel('grid', mod('grid', gridSrc), 'main', [ro(this.bufGMass), ro(this.bufGMom), rw(this.bufGVel)]),
      g2p: this.makeKernel('g2p', mod('g2p', g2pSrc), 'main',
        [rw(this.bufPos), rw(this.bufVel), rw(this.bufC), rw(this.bufF), rw(this.bufS), ro(this.bufFlags), ro(this.bufGVel)]),
      raster: this.makeKernel('raster', mod('raster', rasterSrc), 'main',
        [ro(this.bufPos), ro(this.bufLat), rw(this.bufPMass), rw(this.bufPLat)]),
      disperse: this.makeKernel('disperse', mod('disperse', disperseSrc), 'main',
        [ro(this.bufPos), ro(this.bufC), rw(this.bufLat), ro(this.bufPMass), ro(this.bufPLat), ro(this.bufFlags)]),
      pack: this.makeKernel('pack', mod('pack', packSrc), 'main',
        [ro(this.bufPMass), ro(this.bufPLat), { kind: 'storageTexture', view: volAView }, { kind: 'storageTexture', view: volBView }]),
      inject: this.makeKernel('inject', injectMod, 'main', [{ kind: 'uniform', buffer: this.bufInject }, ro(this.bufPos), rw(this.bufLat)]),
      injectAll: this.makeKernel('injectAll', injectMod, 'main', [{ kind: 'uniform', buffer: this.bufInjectAll }, ro(this.bufPos), rw(this.bufLat)]),
      reset: this.makeKernel('reset', mod('reset', resetSrc), 'main',
        [rw(this.bufVel), rw(this.bufC), rw(this.bufF), rw(this.bufS), rw(this.bufFlags)]),
      foldSelect: this.makeKernel('foldSelect', foldMod, 'select_', foldRes),
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
    q.writeBuffer(this.bufPos, 0, this.seeds);
    // "set all" inject uniform: white base
    const inj = new Float32Array(16);
    inj.set([0, 0, 0, 1e9], 0);
    inj.set(WHITE_LATENT.slice(0, 4), 4);
    inj.set([WHITE_LATENT[4], WHITE_LATENT[5], WHITE_LATENT[6], 1], 8);
    new Uint32Array(inj.buffer)[12] = 1;
    q.writeBuffer(this.bufInjectAll, 0, inj);
    this.writeParams(0, this.quality.dt, 0);
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

  /** Fill uniform slot `slot` (fold state: s and ds/dt for that substep). */
  private writeParams(slot: number, frameDt: number, foldT: number, foldActive = false): void {
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
    f.set([GEOMETRY.domain[0], GEOMETRY.domain[1], GEOMETRY.domain[2], frameDt], o + 8);
    f.set([back.axisY, back.axisZ, back.omegaX, p.backFriction], o + 12);
    f.set([front.axisY, front.axisZ, front.omegaX, GEOMETRY.radius], o + 16);
    f.set([mu, lambda, this.config.material.thetaC, this.config.material.thetaS], o + 20);
    f.set([pVol, pMass, pMass / (pVol * MATERIAL_DENSITY), p.dispersion], o + 24);
    const fp = foldProfile(foldT, FOLD_DURATION);
    f.set([foldActive ? 1 : 0, fp.s, fp.dsdt, FOLD_LIFT], o + 28);
    const bankTop = GEOMETRY.axisY + GEOMETRY.radius + GEOMETRY.bankHeight + 2 * h;
    f.set([bankTop, GEOMETRY.length, 0.25 * GEOMETRY.length, GEOMETRY.nipZ], o + 32);
    f.set([1.5 * h, 0.5 * h, 2 * h, 0.6], o + 36);
  }

  // ---------------------------------------------------------------------------
  // dispatch helpers
  // ---------------------------------------------------------------------------

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
    if (this.destroyed) return;
    this.pollTimestamps();
    const q = this.quality;
    const k = this.kernels;
    const substeps = this.paused ? 0 : q.substepsPerFrame;
    const frameSlot = q.substepsPerFrame;
    const { back, front } = rollerPoses(this.params);

    // uniform ring: one slot per substep (fold script state differs per substep)
    let foldActive = this.foldActive || this.foldPending;
    let foldT = this.foldPending ? 0 : this.foldTime;
    const finishAt: number[] = [];
    for (let s = 0; s < substeps; s++) {
      this.writeParams(s, frameDtSeconds, foldT, foldActive);
      if (foldActive) {
        foldT += q.dt;
        if (foldT >= FOLD_DURATION) { finishAt.push(s); foldActive = false; }
      }
    }
    this.writeParams(frameSlot, frameDtSeconds, foldT, foldActive);
    this.device.queue.writeBuffer(this.bufParams, 0, this.paramsData, 0, (UNIFORM_STRIDE / 4) * this.uniformSlots);

    const timestampWrites: GPUComputePassTimestampWrites | undefined = this.querySet
      ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      : undefined;
    const pass = encoder.beginComputePass({ label: 'GpuMpm-step', timestampWrites });
    if (this.foldPending && substeps > 0) {
      this.dispatchParticles(pass, k.foldSelect, 0);
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
    this.encodeFrameKernels(pass, frameSlot, substeps > 0);
    pass.end();

    if (this.querySet && this.queryResolve) {
      const slot = this.queryStaging.find((s) => s.state === 'free');
      if (slot) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
        encoder.copyBufferToBuffer(this.queryResolve, 0, slot.buffer, 0, 16);
        slot.state = 'copied';
      }
    }

    if (substeps > 0) {
      const st = this.statsData;
      st.simTime += q.dt * substeps;
      st.rollerAngleFront += front.omegaX * q.dt * substeps;
      st.rollerAngleBack += back.omegaX * q.dt * substeps;
      st.simSecondsPerFrame = q.dt * q.substepsPerFrame;
    }
  }

  /** Map any staging buffer whose copy has been submitted (called at the start of the next step). */
  private pollTimestamps(): void {
    for (const slot of this.queryStaging) {
      if (slot.state !== 'copied') continue;
      slot.state = 'mapping';
      slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
        if (this.destroyed) return;
        const ts = new BigUint64Array(slot.buffer.getMappedRange());
        const ns = Number(ts[1] - ts[0]);
        slot.buffer.unmap();
        slot.state = 'free';
        if (Number.isFinite(ns) && ns >= 0) this.statsData.lastStepGpuMs = ns / 1e6;
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
    this.uploadInitialState();
  }

  addPigment(center: readonly [number, number, number], radius: number, latent: Latent, strength = 1): void {
    if (this.destroyed) return;
    const inj = new Float32Array(16);
    inj.set([center[0], center[1], center[2], Math.max(radius, 1e-6)], 0);
    inj.set(latent.slice(0, 4), 4);
    inj.set([latent[4], latent[5], latent[6], strength], 8);
    new Uint32Array(inj.buffer)[12] = 0;
    this.device.queue.writeBuffer(this.bufInject, 0, inj);
    this.runParticleKernel(this.kernels.inject);
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
    if (this.destroyed || this.foldActive || this.foldPending) return;
    this.foldPending = true;
  }

  async readParticles(): Promise<ParticleSnapshot> {
    const n = this.count;
    const sizes = [3 * n, 3 * n, 7 * n, 9 * n, n].map((f) => f * 4);
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
    const snap: ParticleSnapshot = {
      count: n,
      positions: new Float32Array(take(sizes[0])),
      velocities: new Float32Array(take(sizes[1])),
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
    for (const b of [this.bufPos, this.bufVel, this.bufC, this.bufF, this.bufS, this.bufLat, this.bufFlags, this.bufFold,
      this.bufGMass, this.bufGMom, this.bufGVel, this.bufPMass, this.bufPLat, this.bufParams, this.bufInject, this.bufInjectAll]) {
      b.destroy();
    }
    this.readbackStaging?.destroy();
    this.queryResolve?.destroy();
    for (const s of this.queryStaging) s.buffer.destroy();
    this.querySet?.destroy();
    this.volA.destroy();
    this.volB.destroy();
  }
}
