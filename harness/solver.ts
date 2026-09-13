/**
 * Dev harness for the GPU solver (not part of the production build).
 * Serve with `npx vite` and open /ColorMill/harness/solver.html?preset=low.
 * Exposes window.__solver for the e2e tests (tests/e2e/solver.spec.mjs).
 */
import { createGpuContext } from '../src/gpu/device';
import { defaultConfig, type QualityPreset } from '../src/config/mill';
import { GpuMpm } from '../src/sim/mpm';
import type { ParticleSnapshot, SimStats } from '../src/sim/types';

interface SolverApi {
  sim: GpuMpm;
  ready: boolean;
  error: string | null;
  frames: number;
  /** wall-clock ms spent in stepFrames per frame (last call) */
  msPerFrame: number;
  stepFrames(n: number): Promise<void>;
  snapshot(): Promise<ParticleSnapshot>;
  density(): Promise<Float32Array>;
  stats(): SimStats;
  drawSlice(): Promise<void>;
}

declare global {
  interface Window { __solver: SolverApi }
}

const hud = document.getElementById('hud') as HTMLDivElement;
const canvas = document.getElementById('slice') as HTMLCanvasElement;

function log(msg: string): void {
  hud.textContent = msg;
}

async function main(): Promise<void> {
  const presetParam = new URLSearchParams(location.search).get('preset') ?? 'low';
  const preset = (['low', 'medium', 'high', 'ultra'].includes(presetParam) ? presetParam : 'low') as QualityPreset;
  const ctx = await createGpuContext();
  const device = ctx.device;
  device.addEventListener('uncapturederror', (ev) => {
    const e = ev as GPUUncapturedErrorEvent;
    console.error('WebGPU uncaptured error:', e.error.message);
    api.error = (api.error ? api.error + '\n' : '') + e.error.message;
  });

  const config = defaultConfig(preset);
  device.pushErrorScope('validation');
  const sim = new GpuMpm({ device, config });
  const api: SolverApi = {
    sim,
    ready: false,
    error: null,
    frames: 0,
    msPerFrame: NaN,
    async stepFrames(n: number): Promise<void> {
      const t0 = performance.now();
      for (let i = 0; i < n; i++) {
        device.pushErrorScope('validation');
        const enc = device.createCommandEncoder();
        sim.step(enc, 1 / 60);
        device.queue.submit([enc.finish()]);
        const err = await device.popErrorScope();
        if (err) {
          console.error('step validation error:', err.message);
          api.error = err.message;
        }
        await device.queue.onSubmittedWorkDone();
        api.frames++;
      }
      api.msPerFrame = n > 0 ? (performance.now() - t0) / n : NaN;
      const st = sim.stats;
      log(`preset=${preset} particles=${st.particleCount} grid=${st.grid.nx}x${st.grid.ny}x${st.grid.nz} ` +
        `frames=${api.frames} simTime=${st.simTime.toFixed(3)}s gpuMs=${st.lastStepGpuMs.toFixed(2)} wallMs/frame=${api.msPerFrame.toFixed(1)}`);
    },
    snapshot: () => sim.readParticles(),
    density: () => sim.readDensity(),
    stats: () => sim.stats,
    async drawSlice(): Promise<void> {
      // y-z slice through the middle of x, drawn on the 2D canvas
      const d = sim.dims;
      const dens = await sim.readDensity();
      const g = canvas.getContext('2d');
      if (!g) return;
      const sx = canvas.width / d.nz;
      const sy = canvas.height / d.ny;
      g.fillStyle = '#000';
      g.fillRect(0, 0, canvas.width, canvas.height);
      const i = Math.floor(d.nx / 2);
      for (let k = 0; k < d.nz; k++) {
        for (let j = 0; j < d.ny; j++) {
          const v = dens[(k * d.ny + j) * d.nx + i];
          if (v <= 0.01) continue;
          const c = Math.min(255, Math.round(v * 200 + 40));
          g.fillStyle = `rgb(${c},${c},${c})`;
          g.fillRect(k * sx, canvas.height - (j + 1) * sy, Math.ceil(sx), Math.ceil(sy));
        }
      }
    }
  };
  window.__solver = api;

  try {
    await sim.ready;
    const err = await device.popErrorScope();
    if (err) throw new Error(err.message);
    api.ready = true;
    log(`ready: preset=${preset} particles=${sim.stats.particleCount} grid=${sim.dims.nx}x${sim.dims.ny}x${sim.dims.nz} ` +
      `adapter=${ctx.caps.adapterDescription} timestamps=${ctx.caps.timestampQuery}`);
  } catch (e) {
    api.error = e instanceof Error ? e.message : String(e);
    log('solver failed: ' + api.error);
    console.error(e);
  }
}

main().catch((e) => {
  log('harness failed: ' + (e instanceof Error ? e.message : String(e)));
  console.error(e);
});
