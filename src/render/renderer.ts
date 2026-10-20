/**
 * RayMarchRenderer — the fullscreen ray-march renderer of design §8.
 *
 * Owns: the canvas configuration, one render pipeline (fullscreen triangle),
 * a compute pipeline that rebuilds a coarse max-density mip of the live
 * volume when density changes (empty-space skipping in the march), a uniform buffer
 * (camera ray basis, volume dims, roller poses/angles, lights, mip dims,
 * end-guide plates), a filtering sampler and the bind groups that reference
 * the render volumes (rebuilt by setVolumes()).
 */
import type { GpuContext } from '../gpu/device';
import { configureCanvas } from '../gpu/device';
import { GEOMETRY, bankTopY, rollerPoses } from '../config/mill';
import type { RenderVolumes } from '../sim/types';
import { assignCamera, cameraBasis, defaultCamera } from './camera';
import type { RenderMode, CameraState, RenderFrameInfo, Renderer } from './types';
import mixboxSrc from './shaders/mixbox.wgsl?raw';
import raymarchSrc from './shaders/raymarch.wgsl?raw';

/** Density threshold that defines the putty surface (design §8). */
export const ISO_THRESHOLD = 0.32; // a gap-thick sheet is ~1.3 cells at `low`; 0.45 punched holes in it
/** Cap on the backing-store scale (design §8). */
export const MAX_DPR = 2;
const GUIDE_TOP = bankTopY(1) + 0.15;

/** Light rig; directions point toward the light, colours are linear. */
export interface LightRig {
  keyDir: [number, number, number];
  keyColor: [number, number, number];
  keyIntensity: number;
  fillDir: [number, number, number];
  fillColor: [number, number, number];
  fillIntensity: number;
  rimDir: [number, number, number];
  rimColor: [number, number, number];
  rimIntensity: number;
  exposure: number;
}

export function defaultLightRig(): LightRig {
  return {
    keyDir: [0.42, 0.74, 0.52],
    keyColor: [1.0, 0.93, 0.82],
    keyIntensity: 1.7,
    fillDir: [-0.72, 0.3, 0.55],
    fillColor: [0.6, 0.7, 0.9],
    fillIntensity: 0.65,
    rimDir: [0.15, 0.55, -0.82],
    rimColor: [0.85, 0.9, 1.0],
    rimIntensity: 1.1,
    exposure: 1.0
  };
}

const UNIFORM_FLOATS = 18 * 4; // 18 vec4f, see Uniforms in raymarch.wgsl

/** Edge length (in texels) of the blocks of the coarse max-density mip used for empty-space skipping. */
export const MIP_BLOCK = 4;
/** Workgroup size of the mip compute pass (one invocation per coarse cell), see csMip in raymarch.wgsl. */
const MIP_WG = 4;

/** Coarse mip dimensions for a volume. */
export function mipDims(dims: { nx: number; ny: number; nz: number }): [number, number, number] {
  return [Math.ceil(dims.nx / MIP_BLOCK), Math.ceil(dims.ny / MIP_BLOCK), Math.ceil(dims.nz / MIP_BLOCK)];
}

function normalize3(v: readonly [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

export class RayMarchRenderer implements Renderer {
  readonly camera: CameraState = defaultCamera();
  /** density iso-surface threshold (fraction of rest density); debug-tunable via ?iso= */
  iso = ISO_THRESHOLD;
  /** colour mode: a few cells into the material (default), or the top-most material only (design §8) */
  mode: RenderMode = 'depth';
  private scale = 1;
  get renderScale(): number { return this.scale; }
  set renderScale(value: number) {
    if (Number.isFinite(value)) this.scale = Math.min(1, Math.max(0.5, value));
  }
  readonly lights: LightRig = defaultLightRig();
  /** Draw the translucent end-guide plates at x = 0 and x = L (design §8, optional). */
  endGuides = true;

  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly module: GPUShaderModule;
  private readonly pipeline: GPURenderPipeline;
  private readonly mipPipeline: GPUComputePipeline;
  private readonly uniformBuffer: GPUBuffer;
  private mipBuffer: GPUBuffer;
  private mipBindGroup: GPUBindGroup;
  private mipSize: [number, number, number] = [1, 1, 1];
  private readonly uniformData = new Float32Array(UNIFORM_FLOATS);
  private readonly sampler: GPUSampler;
  private readonly placeholder: GPUTexture;
  private bindGroup: GPUBindGroup;
  private volumes: RenderVolumes | null = null;
  private destroyed = false;
  private present = true;
  private offscreen: GPUTexture | null = null;
  private lastFrameData: Float32Array | null = null;
  private lastVolumeVersion: number | undefined;
  private mipDensityVersion: number | undefined;

  constructor(ctx: GpuContext, canvas: HTMLCanvasElement) {
    this.device = ctx.device;
    this.canvas = canvas;
    this.format = ctx.canvasFormat;
    this.context = configureCanvas(ctx, canvas);

    this.module = this.device.createShaderModule({
      label: 'raymarch',
      code: mixboxSrc + '\n' + raymarchSrc
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'raymarch-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } }
      ]
    });

    const mipBgl = this.device.createBindGroupLayout({
      label: 'raymarch-mip-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });
    this.mipPipeline = this.device.createComputePipeline({
      label: 'raymarch-mip-pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [mipBgl] }),
      compute: { module: this.module, entryPoint: 'csMip' }
    });

    this.pipeline = this.device.createRenderPipeline({
      label: 'raymarch-pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: { module: this.module, entryPoint: 'vsMain' },
      fragment: { module: this.module, entryPoint: 'fsMain', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' }
    });

    this.uniformBuffer = this.device.createBuffer({
      label: 'raymarch-uniforms',
      size: UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    this.sampler = this.device.createSampler({
      label: 'volume-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge'
    });

    // 1x1x1 empty volume so the bind group is valid before setVolumes()
    this.placeholder = this.device.createTexture({
      label: 'volume-placeholder',
      size: { width: 1, height: 1, depthOrArrayLayers: 1 },
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    this.mipBuffer = this.makeMipBuffer(1);
    this.bindGroup = this.makeBindGroup(this.placeholder, this.placeholder, this.placeholder, this.placeholder);
    this.mipBindGroup = this.makeMipBindGroup(this.placeholder);
    this.resize();
  }

  private makeMipBuffer(cells: number): GPUBuffer {
    return this.device.createBuffer({
      label: 'raymarch-density-mip',
      size: Math.max(16, cells * 4),
      usage: GPUBufferUsage.STORAGE
    });
  }

  private makeMipBindGroup(volA: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'raymarch-mip-bg',
      layout: this.mipPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: volA.createView({ dimension: '3d' }) },
        { binding: 5, resource: { buffer: this.mipBuffer } }
      ]
    });
  }

  /** WGSL compile diagnostics (for the dev harness). */
  compilationInfo(): Promise<GPUCompilationInfo> {
    return this.module.getCompilationInfo();
  }

  private makeBindGroup(volA: GPUTexture, finA: GPUTexture, finB: GPUTexture, finC: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'raymarch-bg',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: volA.createView({ dimension: '3d' }) },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: { buffer: this.mipBuffer } },
        { binding: 7, resource: finA.createView({ dimension: '3d' }) },
        { binding: 8, resource: finB.createView({ dimension: '3d' }) },
        { binding: 9, resource: finC.createView({ dimension: '3d' }) }
      ]
    });
  }

  setVolumes(volumes: RenderVolumes): void {
    this.volumes = volumes;
    this.lastFrameData = null;
    this.mipDensityVersion = undefined;
    const m = mipDims(volumes.dims);
    const cells = m[0] * m[1] * m[2];
    if (cells !== this.mipSize[0] * this.mipSize[1] * this.mipSize[2] || this.mipBuffer.size < cells * 4) {
      this.mipBuffer.destroy();
      this.mipBuffer = this.makeMipBuffer(cells);
    }
    this.mipSize = m;
    this.bindGroup = this.makeBindGroup(volumes.volA, volumes.finA, volumes.finB, volumes.finC);
    this.mipBindGroup = this.makeMipBindGroup(volumes.volA);
  }

  resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR) * this.scale;
    const cssW = this.canvas.clientWidth || this.canvas.width / dpr || 1;
    const cssH = this.canvas.clientHeight || this.canvas.height / dpr || 1;
    const maxDim = this.device.limits.maxTextureDimension2D;
    const w = Math.max(1, Math.min(maxDim, Math.round(cssW * dpr)));
    const h = Math.max(1, Math.min(maxDim, Math.round(cssH * dpr)));
    if (w === this.canvas.width && h === this.canvas.height) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    this.lastFrameData = null;
    return true;
  }

  resetCamera(): void {
    assignCamera(this.camera, defaultCamera());
  }

  private writeUniforms(info: RenderFrameInfo, w: number, hgt: number): void {
    const u = this.uniformData;
    const basis = cameraBasis(this.camera, w / hgt);
    const set4 = (i: number, a: number, b: number, c: number, d: number): void => {
      u[i * 4] = a; u[i * 4 + 1] = b; u[i * 4 + 2] = c; u[i * 4 + 3] = d;
    };
    const vol = this.volumes;
    const hCell = vol ? vol.dims.h : 1 / 32;
    const poses = rollerPoses(info.params);
    const L = this.lights;
    const kd = normalize3(L.keyDir);
    const fd = normalize3(L.fillDir);
    const rd = normalize3(L.rimDir);
    let flags = 0;
    if (vol) flags |= 1;
    if (this.format.endsWith('-srgb')) flags |= 2;
    if (this.mode === 'surface') flags |= 4;

    set4(0, basis.origin[0], basis.origin[1], basis.origin[2], basis.tanHalfFovY);
    set4(1, basis.right[0], basis.right[1], basis.right[2], basis.aspect);
    set4(2, basis.up[0], basis.up[1], basis.up[2], info.timeSeconds);
    set4(3, basis.forward[0], basis.forward[1], basis.forward[2], hCell);
    set4(4, vol ? vol.dims.nx : 1, vol ? vol.dims.ny : 1, vol ? vol.dims.nz : 1, this.iso);
    set4(5, GEOMETRY.domain[0], GEOMETRY.domain[1], GEOMETRY.domain[2], GEOMETRY.length);
    set4(6, poses.back.axisY, poses.back.axisZ, poses.back.radius, info.rollerAngleBack);
    set4(7, poses.front.axisY, poses.front.axisZ, poses.front.radius, info.rollerAngleFront);
    set4(8, kd[0], kd[1], kd[2], L.keyIntensity);
    set4(9, L.keyColor[0], L.keyColor[1], L.keyColor[2], 0);
    set4(10, fd[0], fd[1], fd[2], L.fillIntensity);
    set4(11, L.fillColor[0], L.fillColor[1], L.fillColor[2], 0);
    set4(12, rd[0], rd[1], rd[2], L.rimIntensity);
    set4(13, L.rimColor[0], L.rimColor[1], L.rimColor[2], 0);
    set4(14, w, hgt, L.exposure, flags);
    const m = this.mipSize;
    set4(15, m[0], m[1], m[2], MIP_BLOCK);
    // end-guide plates: plate top (a little above the settled bank), half depth in z, thickness, enabled
    set4(16, GUIDE_TOP, GEOMETRY.bankHalfDepth, 0.012, this.endGuides ? 1 : 0);
    // the colour raster's grid (design §3.5): node counts and cell size
    const cg = vol ? vol.fineDims : undefined;
    set4(17, cg ? cg.nx : 1, cg ? cg.ny : 1, cg ? cg.nz : 1, cg ? cg.h : hCell);
  }

  /** Rebuild the coarse max-density mip from the live volume (design §8: empty-space skipping). */
  private encodeMipPass(encoder: GPUCommandEncoder): void {
    if (!this.volumes) return;
    const version = this.volumes.densityVersion;
    if (version !== undefined && version === this.mipDensityVersion) return;
    const m = this.mipSize;
    const pass = encoder.beginComputePass({ label: 'raymarch-density-mip' });
    pass.setPipeline(this.mipPipeline);
    pass.setBindGroup(0, this.mipBindGroup);
    pass.dispatchWorkgroups(Math.ceil(m[0] / MIP_WG), Math.ceil(m[1] / MIP_WG), Math.ceil(m[2] / MIP_WG));
    pass.end();
    this.mipDensityVersion = version;
  }

  private encodePass(encoder: GPUCommandEncoder, view: GPUTextureView): void {
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
    this.encodeMipPass(encoder);
    const pass = encoder.beginRenderPass({
      label: 'raymarch-pass',
      colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }

  setPresentation(enabled: boolean): void {
    if (enabled !== this.present) this.lastFrameData = null;
    this.present = enabled;
  }

  private offscreenView(w: number, h: number): GPUTextureView {
    if (!this.offscreen || this.offscreen.width !== w || this.offscreen.height !== h) {
      this.offscreen?.destroy();
      this.offscreen = this.device.createTexture({
        label: 'raymarch-offscreen-frame',
        size: { width: w, height: h },
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
      });
    }
    return this.offscreen.createView();
  }

  render(encoder: GPUCommandEncoder, info: RenderFrameInfo): boolean {
    if (this.destroyed) return false;
    const w = Math.max(1, this.canvas.width), h = Math.max(1, this.canvas.height);
    this.writeUniforms(info, w, h);
    const version = this.volumes?.version;
    // Uniforms include camera, lighting, roller poses, colour mode and size.
    // camUp.w is reserved wall-clock time, currently unused by the shader.
    // Unknown volume versions deliberately keep drawing for external callers.
    if (version !== undefined && version === this.lastVolumeVersion && this.lastFrameData
      && this.uniformData.every((v, i) => i === 11 || v === this.lastFrameData![i])) return false;
    const view = this.present ? this.context.getCurrentTexture().createView() : this.offscreenView(w, h);
    this.encodePass(encoder, view);
    if (!this.lastFrameData) this.lastFrameData = new Float32Array(UNIFORM_FLOATS);
    this.lastFrameData.set(this.uniformData);
    this.lastVolumeVersion = version;
    return true;
  }

  /**
   * Render one frame into an offscreen texture of the given size (default:
   * the canvas size) and read the pixels back as tightly packed RGBA8.
   * For tests and diagnostics: it does not touch the canvas, which matters
   * in headless browsers whose canvas presentation is broken.
   */
  async renderToPixels(info: RenderFrameInfo, width = this.canvas.width, height = this.canvas.height): Promise<{ width: number; height: number; data: Uint8Array }> {
    const w = Math.max(1, width | 0);
    const h = Math.max(1, height | 0);
    const target = this.device.createTexture({
      label: 'raymarch-offscreen',
      size: { width: w, height: h },
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const readback = this.device.createBuffer({
      label: 'raymarch-readback',
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: 'raymarch-offscreen' });
      this.writeUniforms(info, w, h);
      this.encodePass(encoder, target.createView());
      encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow, rowsPerImage: h }, { width: w, height: h });
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(readback.getMappedRange());
      const data = new Uint8Array(w * h * 4);
      const bgra = this.format.startsWith('bgra');
      for (let y = 0; y < h; y++) {
        const row = src.subarray(y * bytesPerRow, y * bytesPerRow + w * 4);
        if (bgra) {
          for (let i = 0; i < w * 4; i += 4) {
            const o = y * w * 4 + i;
            data[o] = row[i + 2]; data[o + 1] = row[i + 1]; data[o + 2] = row[i]; data[o + 3] = row[i + 3];
          }
        } else {
          data.set(row, y * w * 4);
        }
      }
      readback.unmap();
      return { width: w, height: h, data };
    } finally {
      readback.destroy();
      target.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniformBuffer.destroy();
    this.mipBuffer.destroy();
    this.placeholder.destroy();
    this.offscreen?.destroy();
    this.offscreen = null;
    this.volumes = null;
    try {
      this.context.unconfigure();
    } catch {
      /* ignore */
    }
  }
}
