/**
 * Renderer dev harness: builds a procedural RenderVolumes pair (a bank on the
 * nip plus a sheet wrapped around the front roller, pigment bands along x)
 * and renders it with RayMarchRenderer + the orbit camera controls.
 * Served by `npx vite` at /ColorMill/harness/render.html; not part of the
 * production build. Exposes `window.__render` for the e2e test.
 */
import { createGpuContext } from '../src/gpu/device';
import { DEFAULT_PARAMS, GEOMETRY, QUALITY_PRESETS, gridDims, rollerPoses } from '../src/config/mill';
import type { GridDims } from '../src/config/mill';
import type { Latent, RenderVolumes } from '../src/sim/types';
import { RayMarchRenderer } from '../src/render/renderer';
import { attachCameraControls, projectPoint } from '../src/render/camera';

// Latents from src/sim/mixbox.c (official Mixbox rgbToLatent of the pigment colours).
const LAT = {
  white: [0.0, 0.0, 0.0, 1.0, 0.00481862, 0.00021851, 0.00295198] as Latent,
  red: [0.0, 0.33960806, 0.65860828, 0.00178365, 0.08873356, -0.01747544, -0.06755477] as Latent,
  blue: [0.86413725, 0.00441961, 0.02987264, 0.1015705, -0.05379499, -0.01226009, 0.00350272] as Latent,
  yellow: [0.00392157, 0.8595098, 0.0, 0.13656863, 0.03300247, 0.10249173, -0.08066935] as Latent
};

function mixLatents(parts: Array<[Latent, number]>): Latent {
  const out = [0, 0, 0, 0, 0, 0, 0];
  let total = 0;
  for (const [, w] of parts) total += w;
  for (const [z, w] of parts) for (let i = 0; i < 7; i++) out[i] += (z[i] * w) / total;
  return out as unknown as Latent;
}

/** Pigment bands along x (mass-averaged mixes between the pure pigments). */
export const BANDS: Array<{ name: string; latent: Latent }> = [
  { name: 'white', latent: LAT.white },
  { name: 'blue', latent: LAT.blue },
  { name: 'green', latent: mixLatents([[LAT.blue, 1], [LAT.yellow, 1.2]]) },
  { name: 'yellow', latent: LAT.yellow },
  { name: 'orange', latent: mixLatents([[LAT.yellow, 1], [LAT.red, 0.6]]) },
  { name: 'red', latent: LAT.red }
];

const params = { ...DEFAULT_PARAMS };
const query = new URLSearchParams(location.search);
const preset = query.get('quality') === 'low' ? QUALITY_PRESETS.low : QUALITY_PRESETS.medium;
/** sheet thickness in cells (the sim's sheet after the nip is ~gap/h, 1.6 cells at low) */
const sheetCells = Number(query.get('sheet') || 3);
/** amplitude of per-node density noise (emulates MPM particle-count fluctuation) */
const noiseAmp = Number(query.get('noise') || 0);

function toHalf(f: number): number {
  // IEEE 754 binary16 with round-to-nearest-even
  const buf = new Float32Array(1);
  const u = new Uint32Array(buf.buffer);
  buf[0] = f;
  const x = u[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let m = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (m & 1))) m++;
    return sign | m;
  }
  let m = mant >>> 13;
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (m & 1))) m++;
  if (m === 0x400) { m = 0; exp++; if (exp >= 0x1f) return sign | 0x7c00; }
  return sign | (exp << 10) | m;
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function latentAt(x: number): Latent {
  const bw = GEOMETRY.length / BANDS.length;
  const f = x / bw - 0.5; // band centres at (i + 0.5) * bw
  const i0 = Math.max(0, Math.min(BANDS.length - 1, Math.floor(f)));
  const i1 = Math.min(BANDS.length - 1, i0 + 1);
  // blend over a narrow strip around the band border (what shear mixing would do)
  const t = smoothstep(0.42, 0.58, f - i0);
  if (t <= 0) return BANDS[i0].latent;
  if (t >= 1) return BANDS[i1].latent;
  return mixLatents([[BANDS[i0].latent, 1 - t], [BANDS[i1].latent, t]]);
}

/** Fill the two rgba16float volumes with the procedural bank + sheet. */
function buildVolumes(device: GPUDevice, dims: GridDims): RenderVolumes {
  const { h, nx, ny, nz } = dims;
  const { back, front } = rollerPoses(params);
  const R = GEOMETRY.radius;
  const L = GEOMETRY.length;
  const b = GEOMETRY.bankEndMargin;
  const bankCy = GEOMETRY.axisY + R * 0.69;
  const sheetT = sheetCells * h;
  const sheetMargin = 0.14;
  const a = new Uint16Array(nx * ny * nz * 4);
  const bb = new Uint16Array(nx * ny * nz * 4);
  const edge = 1.2 * h;
  for (let k = 0; k < nz; k++) {
    const z = k * h;
    for (let j = 0; j < ny; j++) {
      const y = j * h;
      for (let i = 0; i < nx; i++) {
        const x = i * h;
        // bank: a rolling roll of material sitting in the V between the rolls, radius
        // slightly uneven along x, ends rounded off; minus the rollers
        const bankR = 0.22 + 0.016 * Math.sin(x * 7.0) + 0.01 * Math.sin(x * 19.0 + 1.0) + 0.03 * Math.cos(Math.atan2(y - bankCy, z - GEOMETRY.nipZ) * 2.0 + x * 3.0) * 0.5;
        let sd = Math.hypot(y - bankCy, z - GEOMETRY.nipZ) - bankR;
        const endClip = Math.abs(x - L / 2) - (L / 2 - b - 0.06);
        sd = Math.max(sd, endClip) + 0.04 * Math.max(0, Math.min(1, 1 + endClip / 0.06)); // soften the ends
        const dB = Math.hypot(y - back.axisY, z - back.axisZ);
        const dF = Math.hypot(y - front.axisY, z - front.axisZ);
        sd = Math.max(sd, R - dB, R - dF);
        // sheet wrapped around the front roller (slightly wavy thickness), clipped in x
        const wobble = 1 + 0.025 * Math.sin(x * 9.0 + Math.atan2(y - front.axisY, z - front.axisZ) * 2.0);
        const sheetR = R + sheetT * wobble;
        let sheet = Math.max(dF - sheetR, R - dF);
        // the sheet is narrower than the roll: bare metal shows at both ends
        sheet = Math.max(sheet, Math.abs(x - L / 2) - (L / 2 - sheetMargin));
        sd = Math.min(sd, sheet);
        let dens = Math.min(1, Math.max(0, 0.5 - sd / edge));
        if (noiseAmp > 0 && dens > 0) {
          // deterministic per-node hash noise
          let hsh = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791);
          hsh = Math.imul(hsh ^ (hsh >>> 13), 0x5bd1e995);
          hsh ^= hsh >>> 15;
          dens = Math.min(1, Math.max(0, dens * (1 + noiseAmp * (((hsh >>> 0) / 4294967296) * 2 - 1))));
        }
        const idx = ((k * ny + j) * nx + i) * 4;
        if (dens > 0) {
          const lat = latentAt(x);
          a[idx] = toHalf(dens);
          a[idx + 1] = toHalf(lat[0]);
          a[idx + 2] = toHalf(lat[1]);
          a[idx + 3] = toHalf(lat[2]);
          bb[idx] = toHalf(lat[3]);
          bb[idx + 1] = toHalf(lat[4]);
          bb[idx + 2] = toHalf(lat[5]);
          bb[idx + 3] = toHalf(lat[6]);
        }
      }
    }
  }
  const make = (label: string, data: Uint16Array): GPUTexture => {
    const tex = device.createTexture({
      label,
      size: { width: nx, height: ny, depthOrArrayLayers: nz },
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: nx * 8, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    return tex;
  };
  return { volA: make('harness-volA', a), volB: make('harness-volB', bb), dims };
}

/**
 * World points just inside the sheet / the bank for each band (for the e2e
 * colour checks): the camera ray toward an interior point hits the surface in
 * front of it, so its projection lands on that band's surface.
 */
function samplePoints(dims: GridDims): Record<string, [number, number, number][]> {
  const { front } = rollerPoses(params);
  const out: Record<string, [number, number, number][]> = {};
  const bw = GEOMETRY.length / BANDS.length;
  const bankCy = GEOMETRY.axisY + GEOMETRY.radius * 0.69;
  const toward = 1.05; // angle from +z toward +y: the upper front of the bank, clear of the sheet and the highlight
  BANDS.forEach((band, i) => {
    // band centre, kept away from the bare roll ends and the rounded bank ends
    const x = Math.min(GEOMETRY.length - 0.2, Math.max(0.2, (i + 0.5) * bw));
    out[band.name] = [
      // inside the sheet on the front roller, a little below the axis, facing the viewer
      [x, front.axisY - 0.08, front.axisZ + Math.sqrt((front.radius + 1.5 * dims.h) ** 2 - 0.08 ** 2)],
      // inside the rolling bank, on the viewer's side
      [x, bankCy + 0.14 * Math.sin(toward), GEOMETRY.nipZ + 0.14 * Math.cos(toward)]
    ];
  });
  return out;
}

async function main(): Promise<void> {
  const hud = document.getElementById('hud') as HTMLDivElement;
  const err = document.getElementById('err') as HTMLDivElement;
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const ctx = await createGpuContext();
  const device = ctx.device;
  device.addEventListener('uncapturederror', (e) => {
    err.textContent += (e as GPUUncapturedErrorEvent).error.message + '\n';
  });

  device.pushErrorScope('validation');
  const renderer = new RayMarchRenderer(ctx, canvas);
  const dims = gridDims(preset);
  const t0 = performance.now();
  const volumes = buildVolumes(device, dims);
  const buildMs = performance.now() - t0;
  renderer.setVolumes(volumes);
  const info = await renderer.compilationInfo();
  for (const m of info.messages) {
    const line = `[wgsl:${m.type}] ${m.lineNum}:${m.linePos} ${m.message}`;
    if (m.type === 'error') err.textContent += line + '\n';
    console[m.type === 'error' ? 'error' : 'warn'](line);
  }

  let angleFront = 0;
  let angleBack = 0;
  let time = 0;
  let lastFrameMs = 0;
  let needsRender = true;

  // Headless test browsers (SwiftShader) lose the device when a pass targets the
  // canvas texture, so under webdriver (or ?offscreen) frames go through the
  // offscreen readback path and are blitted onto a 2D canvas for the eye.
  const offscreen = navigator.webdriver || new URLSearchParams(location.search).has('offscreen');
  let lastPixels: { width: number; height: number; data: Uint8Array } | null = null;
  let blit: CanvasRenderingContext2D | null = null;
  if (offscreen) {
    const c2 = document.createElement('canvas');
    c2.id = 'blit';
    c2.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none';
    document.body.insertBefore(c2, hud);
    blit = c2.getContext('2d');
  }

  function frameInfo(): { params: typeof params; rollerAngleFront: number; rollerAngleBack: number; timeSeconds: number } {
    const { back, front } = rollerPoses(params);
    const dt = 1 / 60;
    time += dt;
    angleFront += front.omegaX * dt;
    angleBack += back.omegaX * dt;
    return { params, rollerAngleFront: angleFront, rollerAngleBack: angleBack, timeSeconds: time };
  }

  function updateHud(): void {
    hud.textContent =
      `ColorMill render harness — ${ctx.caps.adapterDescription}${ctx.caps.isSoftware ? ' (software)' : ''}${offscreen ? ' (offscreen readback)' : ''}\n` +
      `grid ${dims.nx}x${dims.ny}x${dims.nz} (${preset.preset}), canvas ${canvas.width}x${canvas.height}\n` +
      `frame ${lastFrameMs.toFixed(1)} ms   volume build ${buildMs.toFixed(0)} ms\n` +
      `drag: orbit · wheel/pinch: zoom · double-tap: reset`;
  }

  async function frame(): Promise<void> {
    renderer.resize();
    const info = frameInfo();
    const t = performance.now();
    if (offscreen) {
      lastPixels = await renderer.renderToPixels(info);
      lastFrameMs = performance.now() - t;
      if (blit) {
        const { width, height, data } = lastPixels;
        blit.canvas.width = width;
        blit.canvas.height = height;
        blit.putImageData(new ImageData(new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength), width, height), 0, 0);
      }
    } else {
      const enc = device.createCommandEncoder();
      renderer.render(enc, info);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      lastFrameMs = performance.now() - t;
    }
    updateHud();
  }

  /** Pixels of the last frame as base64 RGBA8 (tests pull this out of the page). */
  async function readPixels(): Promise<{ width: number; height: number; base64: string }> {
    const px = lastPixels ?? (await renderer.renderToPixels(frameInfo()));
    let bin = '';
    const chunk = 0x8000;
    for (let i = 0; i < px.data.length; i += chunk) bin += String.fromCharCode.apply(null, Array.from(px.data.subarray(i, i + chunk)));
    return { width: px.width, height: px.height, base64: btoa(bin) };
  }

  await frame();
  const validation = await device.popErrorScope();
  if (validation) err.textContent += 'validation: ' + validation.message + '\n';

  attachCameraControls(canvas, renderer.camera, () => { needsRender = true; });
  window.addEventListener('resize', () => { needsRender = true; });

  const project = (p: [number, number, number]): { x: number; y: number } | null => {
    const r = projectPoint(renderer.camera, canvas.width / canvas.height, p);
    if (!r) return null;
    // canvas pixel coords (CSS pixels: the canvas fills the viewport)
    return { x: ((r.x + 1) / 2) * canvas.clientWidth, y: ((1 - r.y) / 2) * canvas.clientHeight };
  };

  const api = {
    renderer,
    dims,
    bands: BANDS.map((b) => b.name),
    samplePoints: samplePoints(dims),
    project,
    frame,
    readPixels,
    offscreen,
    lastFrameMs: () => lastFrameMs,
    validationError: validation ? validation.message : null,
    compileMessages: info.messages.map((m) => ({ type: m.type, message: m.message })),
    animate: !navigator.webdriver,
    ready: true
  };
  (window as unknown as { __render: typeof api }).__render = api;

  const loop = async (): Promise<void> => {
    if (api.animate || needsRender) {
      needsRender = false;
      await frame();
    }
    requestAnimationFrame(() => { void loop(); });
  };
  requestAnimationFrame(() => { void loop(); });
}

main().catch((e: unknown) => {
  const err = document.getElementById('err') as HTMLDivElement;
  err.textContent = String(e instanceof Error ? e.stack ?? e.message : e);
  (window as unknown as { __render: unknown }).__render = { ready: false, error: String(e) };
});
