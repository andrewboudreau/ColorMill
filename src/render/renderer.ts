/**
 * RayMarchRenderer — the fullscreen ray-march renderer of design §8.
 *
 * Owns: the canvas configuration, one render pipeline (fullscreen triangle),
 * a uniform buffer (camera ray basis, volume dims, roller poses/angles,
 * lights), a filtering sampler and the bind group that references the
 * render volumes (rebuilt by setVolumes()).
 */
import type { GpuContext } from '../gpu/device';
import { configureCanvas } from '../gpu/device';
import { GEOMETRY, rollerPoses } from '../config/mill';
import type { RenderVolumes } from '../sim/types';
import { assignCamera, cameraBasis, defaultCamera } from './camera';
import type { CameraState, RenderFrameInfo, Renderer } from './types';
import mixboxSrc from './shaders/mixbox.wgsl?raw';
import raymarchSrc from './shaders/raymarch.wgsl?raw';

/** Density threshold that defines the putty surface (design §8). */
export const ISO_THRESHOLD = 0.45;
/** Cap on the backing-store scale (design §8). */
export const MAX_DPR = 2;

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
    keyIntensity: 2.3,
    fillDir: [-0.72, 0.3, 0.55],
    fillColor: [0.6, 0.7, 0.9],
    fillIntensity: 0.75,
    rimDir: [0.15, 0.55, -0.82],
    rimColor: [0.85, 0.9, 1.0],
    rimIntensity: 1.1,
    exposure: 1.0
  };
}

const UNIFORM_FLOATS = 15 * 4; // 15 vec4f, see Uniforms in raymarch.wgsl

function normalize3(v: readonly [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

export class RayMarchRenderer implements Renderer {
  readonly camera: CameraState = defaultCamera();
  readonly lights: LightRig = defaultLightRig();

  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly module: GPUShaderModule;
  private readonly pipeline: GPURenderPipeline;
  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformData = new Float32Array(UNIFORM_FLOATS);
  private readonly sampler: GPUSampler;
  private readonly placeholder: GPUTexture;
  private bindGroup: GPUBindGroup;
  private volumes: RenderVolumes | null = null;
  private destroyed = false;

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
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }
      ]
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
    this.bindGroup = this.makeBindGroup(this.placeholder, this.placeholder);
    this.resize();
  }

  /** WGSL compile diagnostics (for the dev harness). */
  compilationInfo(): Promise<GPUCompilationInfo> {
    return this.module.getCompilationInfo();
  }

  private makeBindGroup(volA: GPUTexture, volB: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'raymarch-bg',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: volA.createView({ dimension: '3d' }) },
        { binding: 2, resource: volB.createView({ dimension: '3d' }) },
        { binding: 3, resource: this.sampler }
      ]
    });
  }

  setVolumes(volumes: RenderVolumes): void {
    this.volumes = volumes;
    this.bindGroup = this.makeBindGroup(volumes.volA, volumes.volB);
  }

  resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    const cssW = this.canvas.clientWidth || this.canvas.width / dpr || 1;
    const cssH = this.canvas.clientHeight || this.canvas.height / dpr || 1;
    const maxDim = this.device.limits.maxTextureDimension2D;
    const w = Math.max(1, Math.min(maxDim, Math.round(cssW * dpr)));
    const h = Math.max(1, Math.min(maxDim, Math.round(cssH * dpr)));
    if (w === this.canvas.width && h === this.canvas.height) return false;
    this.canvas.width = w;
    this.canvas.height = h;
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

    set4(0, basis.origin[0], basis.origin[1], basis.origin[2], basis.tanHalfFovY);
    set4(1, basis.right[0], basis.right[1], basis.right[2], basis.aspect);
    set4(2, basis.up[0], basis.up[1], basis.up[2], info.timeSeconds);
    set4(3, basis.forward[0], basis.forward[1], basis.forward[2], hCell);
    set4(4, vol ? vol.dims.nx : 1, vol ? vol.dims.ny : 1, vol ? vol.dims.nz : 1, ISO_THRESHOLD);
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
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);
  }

  private encodePass(encoder: GPUCommandEncoder, info: RenderFrameInfo, view: GPUTextureView, w: number, h: number): void {
    this.writeUniforms(info, w, h);
    const pass = encoder.beginRenderPass({
      label: 'raymarch-pass',
      colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }

  render(encoder: GPUCommandEncoder, info: RenderFrameInfo): void {
    if (this.destroyed) return;
    const view = this.context.getCurrentTexture().createView();
    this.encodePass(encoder, info, view, this.canvas.width, this.canvas.height);
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
      this.encodePass(encoder, info, target.createView(), w, h);
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
    this.placeholder.destroy();
    this.volumes = null;
    try {
      this.context.unconfigure();
    } catch {
      /* ignore */
    }
  }
}
