/**
 * ColorMill v2 — app glue: canvas + UI, WebGPU init, preset auto-select,
 * the frame loop (one command encoder per frame: sim.step then
 * renderer.render, then queue.submit), stats, and the window.__colormill
 * debug API used by the e2e tests.
 *
 * The solver and the renderer are injected as factories so this module
 * compiles against the contracts in src/sim/types.ts and src/render/types.ts
 * only. See src/main.ts for the wiring.
 *
 * Query parameters: ?preset=low|medium|high|ultra overrides the auto-select,
 * ?paused=1 starts paused (deterministic driving via __colormill.stepFrames),
 * ?orbit=1 starts with auto-orbit on.
 */
import './styles.css';
import { BASE_LATENT, findPigment, rgbToLatentAsync } from './color/pigments';
import {
  BATCH_LIMITS, DEFAULT_MATERIAL, DEFAULT_PARAMS, DROP_SLOTS, GEOMETRY, PARAM_LIMITS, PIGMENT_CHUNK_RADIUS, QUALITY_PRESETS, SILICONE_KG_PER_LITRE, dropSlotX, estimateParticleCount, gridDims, materialLitres,
  type MaterialConstants, type MillConfig, type MillParams, type QualityPreset
} from './config/mill';
import { WebGpuUnavailableError, createGpuContext, type GpuCapabilities, type GpuContext } from './gpu/device';
import type { CameraState, Renderer } from './render/types';
import type { DebugApi, GpuMpmSim, Latent, ParticleSnapshot, SimStats } from './sim/types';
import { Hud } from './ui/hud';
import { installKeyboard } from './ui/keys';
import { installOrbitControls, type OrbitControls } from './ui/orbit';
import { Overlay } from './ui/overlay';
import { Palette } from './ui/palette';
import { Panel } from './ui/panel';

export interface BootOptions {
  /** element the app renders into (emptied first) */
  root: HTMLElement;
  makeSim: (device: GPUDevice, config: MillConfig) => GpuMpmSim;
  makeRenderer: (ctx: GpuContext, canvas: HTMLCanvasElement) => Renderer;
  /**
   * Optional pointer-controls factory (e.g. `attachCameraControls` from
   * src/render/camera.ts). Must return a detach function. Defaults to the
   * built-in src/ui/orbit.ts controls.
   */
  attachCameraControls?: (canvas: HTMLCanvasElement, camera: CameraState, onChange: () => void) => () => void;
}

export interface AppHandle {
  /** the current solver (replaced when the quality preset changes) */
  readonly sim: GpuMpmSim;
  readonly renderer: Renderer;
  readonly ctx: GpuContext;
  readonly preset: QualityPreset;
  destroy(): void;
}

declare global {
  interface Window {
    __colormill?: DebugApi;
  }
}

const PRESETS: readonly QualityPreset[] = ['low', 'medium', 'high', 'ultra'];

const AUTO_ORBIT_RATE = 0.12; // rad/s
const STATS_WINDOW = 60;
/** Adaptive quality: step the preset down when frames average above this for a sustained period. */
const ADAPT_SLOW_MS = 45;
const ADAPT_SETTLE_SECONDS = 3;
const PRESET_ORDER: readonly QualityPreset[] = ['low', 'medium', 'high', 'ultra'];
const MAX_REPORTED_GPU_ERRORS = 5;

function isPreset(s: string | null): s is QualityPreset {
  return s !== null && (PRESETS as readonly string[]).includes(s);
}

function isMobileUserAgent(): boolean {
  const ua = navigator.userAgent || '';
  if (/Android|iPhone|iPad|iPod|Mobile|Silk|Windows Phone/i.test(ua)) return true;
  // iPadOS reports itself as a Mac but has touch
  return /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
}

/**
 * Quality auto-select (design §7): software adapters and mobile devices get
 * `low`; adapters whose vendor/architecture look integrated get `medium`;
 * everything else `high`. `?preset=` overrides.
 */
export function choosePreset(caps: GpuCapabilities, search = location.search): { preset: QualityPreset; reason: string } {
  const q = new URLSearchParams(search).get('preset');
  if (isPreset(q)) return { preset: q, reason: 'query parameter' };
  if (caps.isSoftware) return { preset: 'low', reason: 'software adapter' };
  if (isMobileUserAgent()) return { preset: 'low', reason: 'mobile device' };
  const id = `${caps.vendor} ${caps.architecture} ${caps.adapterDescription}`.toLowerCase();
  // discrete-class hints win over vendor names (e.g. "amd rdna-3" is discrete, "intel gen-12lp" is not)
  const discrete = /rdna|gcn|vega|navi|turing|ampere|ada|blackwell|pascal|volta|hopper|maxwell|kepler|geforce|radeon rx|quadro|arc-a|arc a\d/i;
  const integrated = /intel|gen-\d|xe-lp|xe-hpg|iris|uhd|adreno|mali|powervr|qualcomm|arm|apple|metal|imagination|mediatek|samsung|vivante/i;
  if (discrete.test(id)) return { preset: 'high', reason: `discrete GPU (${caps.architecture || caps.vendor})` };
  if (integrated.test(id)) return { preset: 'medium', reason: `integrated GPU (${caps.architecture || caps.vendor})` };
  return { preset: 'high', reason: 'default' };
}

function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message || e.name;
  return String(e);
}

/** Let the browser paint (the overlay progress text) before heavy work. */
function nextPaint(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

function clampParam(key: keyof MillParams, v: number): number {
  const l = PARAM_LIMITS[key];
  const snapped = Math.round(v / l.step) * l.step;
  return Math.min(l.max, Math.max(l.min, parseFloat(snapped.toFixed(6))));
}

export async function bootApp(opts: BootOptions): Promise<AppHandle> {
  const root = opts.root;
  root.innerHTML = '';
  root.classList.add('cm-app');

  const canvas = document.createElement('canvas');
  canvas.className = 'cm-canvas';
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Two-roll mill viewport');
  root.appendChild(canvas);

  const hud = new Hud(root);
  const overlay = new Overlay(root);
  overlay.showLoading('Requesting a WebGPU device…');
  hud.setText('starting…');

  // --- GPU --------------------------------------------------------------------
  let ctx: GpuContext;
  try {
    ctx = await createGpuContext();
  } catch (e) {
    const msg = errorMessage(e);
    overlay.showUnavailable(msg);
    hud.setText('WebGPU unavailable');
    throw e instanceof WebGpuUnavailableError ? e : new WebGpuUnavailableError(msg);
  }
  const { device } = ctx;
  const search = location.search;
  const query = new URLSearchParams(search);
  const chosen = choosePreset(ctx.caps, search);
  let preset: QualityPreset = chosen.preset;
  // batch size multiplier (?batch=1.5); scales the seeded bank
  const batchQuery = parseFloat(query.get('batch') ?? '');
  let batch = Number.isFinite(batchQuery) && batchQuery > 0
    ? Math.min(Math.max(batchQuery, BATCH_LIMITS.min), BATCH_LIMITS.max)
    : 1;
  let autoOrbit = query.get('orbit') === '1';
  const startPaused = query.get('paused') === '1';
  // Unless the preset was pinned by the user or the URL, drop a level when the
  // GPU cannot keep up (frame time sustained above ADAPT_SLOW_MS).
  let adaptive = chosen.reason !== 'query parameter' && !navigator.webdriver;
  let slowSince = 0;
  // Automated browsers (Playwright) on software WebGPU cannot present to the
  // canvas; render offscreen there and let the debug API read pixels back.
  const offscreen = navigator.webdriver || query.get('offscreen') === '1';

  // --- state ------------------------------------------------------------------
  let sim: GpuMpmSim | undefined;
  let renderer: Renderer | undefined;
  let orbit: OrbitControls | undefined;
  let detachControls: (() => void) | undefined;
  let destroyed = false;
  let ready = false;
  let stepping = false; // __colormill.stepFrames is driving frames
  let rebuilding = false;
  let rafId = 0;
  let lastFrameTime = 0;
  let startTime = performance.now();
  let reportedGpuErrors = 0;
  const frameTimes: number[] = [];
  let frameTimeSum = 0;
  let lastStatsPaint = 0;

  const reportError = (what: string, e: unknown): void => {
    const msg = `${what}: ${errorMessage(e)}`;
    console.error(msg, e);
    hud.setError(msg);
  };

  device.addEventListener('uncapturederror', (ev: GPUUncapturedErrorEvent) => {
    if (reportedGpuErrors++ < MAX_REPORTED_GPU_ERRORS) reportError('WebGPU error', ev.error.message);
  });
  device.lost.then((info) => {
    if (destroyed) return;
    reportError('GPU device lost', `${info.reason}: ${info.message}`);
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
  });

  // --- UI ---------------------------------------------------------------------
  const wideScreen = window.matchMedia('(min-width: 900px)').matches;
  const panel = new Panel(root, {
    onParam: (key, value) => {
      if (sim) sim.params[key] = value;
    },
    onQuality: (p) => { adaptive = false; void rebuildSim(p, batch); },
    onBatch: (b) => { void rebuildSim(preset, b); },
    onAutoOrbit: (on) => { autoOrbit = on; }
  }, { params: { ...DEFAULT_PARAMS }, preset, batch, autoOrbit, open: wideScreen });

  // where along the roll a tap lands: one of DROP_SLOTS fixed positions, the
  // operator picks which (strip in the palette, [ and ], ?slot=)
  const clampSlot = (slot: number, fallback: number): number =>
    Number.isFinite(slot) ? Math.min(DROP_SLOTS - 1, Math.max(0, Math.round(slot))) : fallback;
  let dropSlot = clampSlot(parseInt(query.get('slot') ?? '', 10) - 1, Math.floor((DROP_SLOTS - 1) / 2));
  const setDropSlot = (slot: number): void => {
    dropSlot = clampSlot(slot, dropSlot);
    palette.setSlot(dropSlot);
  };

  const injectLatent = (latent: Latent, slot = dropSlot): void => {
    if (!sim) return;
    const x = dropSlotX(clampSlot(slot, dropSlot));
    // a dollop on top of whatever material is over the nip at this x (the GPU
    // finds the surface, so chunks stack on each other and it works after the
    // bank has slumped or drained)
    const z = GEOMETRY.nipZ;
    palette.flashSlot();
    // a chunk of coloured putty set down on the bank; when the reserved pool is
    // used up, tint the material at the surface instead
    const s = sim;
    s.addPigmentChunk(x, z, PIGMENT_CHUNK_RADIUS, latent).then((added) => {
      if (added === 0 && sim === s) {
        s.addPigmentOnSurface(x, z, PIGMENT_CHUNK_RADIUS, latent);
        hud.showHint('Pigment pool used up: tinting the bank instead (Reset to refill)', 4000);
      }
    }).catch((e) => reportError('addPigment failed', e));
  };

  const tapPigment = (name: string, slot?: number): void => {
    const p = findPigment(name);
    if (!p) throw new Error(`unknown pigment "${name}"`);
    palette.flashPigment(p.key);
    injectLatent(p.latent, slot);
  };
  const withSim = (what: string, fn: (s: GpuMpmSim) => void): void => {
    if (!sim) return;
    try { fn(sim); } catch (e) { reportError(what, e); }
  };
  const togglePause = (): void => {
    if (!sim) return;
    sim.paused = !sim.paused;
    palette.setPaused(sim.paused);
    paintStats(true);
  };

  // cut & fold alternates ends, as an operator does: cut from the left, then from the right
  let flopSide: 'left' | 'right' = 'left';
  const cutFlop = (): void => withSim('cutAndFlop failed', (s) => {
    if (s.operatorBusy) return;
    s.cutAndFlop(flopSide);
    hud.showHint(`Cut & fold: the sheet is cut from the ${flopSide} end and the flap folds over the cut toward the middle`, 3000);
    flopSide = flopSide === 'left' ? 'right' : 'left';
  });

  const palette = new Palette(root, {
    onPigment: (key) => injectLatent(findPigment(key)?.latent ?? BASE_LATENT),
    onCutFlop: cutFlop,
    onSlot: (i) => {
      dropSlot = i;
      hud.showHint(`Pigment drops at slot ${i + 1} of ${DROP_SLOTS} (left to right along the roll)`, 2500);
    },
    onCustom: (hex) => {
      hud.setError(null);
      rgbToLatentAsync(hex).then(injectLatent).catch((e) => reportError('Custom colour failed', e));
    },
    onCutFold: () => withSim('cutAndFold failed', (s) => s.cutAndFold()),
    onClear: () => withSim('clearPigment failed', (s) => s.clearPigment()),
    onReset: () => withSim('reset failed', (s) => { s.reset(); hud.setError(null); }),
    onTogglePause: togglePause
  }, undefined, dropSlot);

  const nudge = (key: 'omega' | 'gap', dir: 1 | -1): void => {
    if (!sim) return;
    const notch = key === 'omega' ? 0.25 : 0.005;
    sim.params[key] = clampParam(key, sim.params[key] + dir * notch);
    panel.setParams(sim.params);
  };
  const removeKeys = installKeyboard(window, {
    togglePause,
    reset: () => withSim('reset failed', (s) => s.reset()),
    cutFold: () => withSim('cutAndFold failed', (s) => s.cutAndFold()),
    cutFlop,
    pigment: (i) => { const key = palette.order[i]; if (key) tapPigment(key); },
    speed: (d) => nudge('omega', d),
    gap: (d) => nudge('gap', d),
    slot: (d) => {
      setDropSlot(dropSlot + d);
      hud.showHint(`Pigment drops at slot ${dropSlot + 1} of ${DROP_SLOTS} (left to right along the roll)`, 2500);
    },
    togglePanel: () => panel.toggle()
  });

  // --- stats --------------------------------------------------------------------
  const gridText = (s: GpuMpmSim): string => `${s.dims.nx}×${s.dims.ny}×${s.dims.nz}`;
  const fpsNow = (): number => (frameTimes.length ? frameTimes.length / frameTimeSum : 0);
  const paintStats = (force = false): void => {
    if (!sim) return;
    const now = performance.now();
    if (!force && now - lastStatsPaint < 250) return;
    lastStatsPaint = now;
    const fps = fpsNow();
    const msFrame = frameTimes.length ? (frameTimeSum / frameTimes.length) * 1000 : 0;
    const st = sim.stats;
    const simSpeed = sim.paused ? 0 : st.simSecondsPerFrame * fps;
    const litres = materialLitres(st.particleCount, sim.dims.h);
    hud.setStatus({ preset, particles: st.particleCount, grid: gridText(sim), fps, msFrame, simSpeed, paused: sim.paused, litres, kg: litres * SILICONE_KG_PER_LITRE });
    panel.setStats({
      particles: st.particleCount, grid: gridText(sim), fps, msFrame, simSpeed,
      gpuMs: st.lastStepGpuMs, simTime: st.simTime, adapter: ctx.caps.adapterDescription
    });
  };
  const recordFrameTime = (dt: number): void => {
    frameTimes.push(dt);
    frameTimeSum += dt;
    if (frameTimes.length > STATS_WINDOW) frameTimeSum -= frameTimes.shift() ?? 0;
  };

  // --- frame ----------------------------------------------------------------------
  /**
   * One frame: resize, (optionally) step the sim, render, submit. A throwing
   * sim.step is reported in the HUD, pauses the sim and the frame is still
   * rendered on a fresh encoder; a throwing render is reported and skipped.
   */
  const runFrame = (frameDt: number, forceStep = false): void => {
    if (destroyed || !sim || !renderer) return;
    try { renderer.resize(); } catch (e) { reportError('resize failed', e); }
    if (autoOrbit && !(orbit?.active ?? false)) renderer.camera.yaw += frameDt * AUTO_ORBIT_RATE;

    let encoder = device.createCommandEncoder({ label: 'colormill frame' });
    const wasPaused = sim.paused;
    if (forceStep) sim.paused = false;
    if (!sim.paused) {
      try {
        sim.step(encoder, frameDt);
      } catch (e) {
        reportError('Simulation step failed', e);
        sim.paused = true;
        palette.setPaused(true);
        encoder = device.createCommandEncoder({ label: 'colormill frame (render only)' });
      }
    }
    if (forceStep && !sim.paused) sim.paused = wasPaused;

    const st = sim.stats;
    try {
      renderer.render(encoder, {
        params: sim.params,
        rollerAngleFront: st.rollerAngleFront,
        rollerAngleBack: st.rollerAngleBack,
        timeSeconds: (performance.now() - startTime) / 1000
      });
    } catch (e) {
      reportError('Render failed', e);
      return;
    }
    device.queue.submit([encoder.finish()]);
  };

  const frame = (now: number): void => {
    if (destroyed) return;
    rafId = requestAnimationFrame(frame);
    if (document.hidden || stepping || rebuilding) { lastFrameTime = now; return; }
    const dt = lastFrameTime ? Math.min(0.1, Math.max(0.001, (now - lastFrameTime) / 1000)) : 1 / 60;
    lastFrameTime = now;
    runFrame(dt);
    recordFrameTime(dt);
    paintStats();
    adaptQuality(now);
  };

  const adaptQuality = (now: number): void => {
    if (!adaptive || !sim || rebuilding || sim.paused || frameTimes.length < STATS_WINDOW) return;
    const msFrame = (frameTimeSum / frameTimes.length) * 1000;
    if (msFrame < ADAPT_SLOW_MS) { slowSince = 0; return; }
    if (!slowSince) { slowSince = now; return; }
    if (now - slowSince < ADAPT_SETTLE_SECONDS * 1000) return;
    const idx = PRESET_ORDER.indexOf(preset);
    if (idx <= 0) { adaptive = false; return; }
    const next = PRESET_ORDER[idx - 1];
    slowSince = 0;
    hud.showHint(`Frame time ${msFrame.toFixed(0)} ms: lowering quality to ${next} (choose a preset in the panel to pin it)`, 9000);
    void setQuality(next);
  };

  const onVisibility = (): void => {
    if (document.hidden) {
      lastFrameTime = 0;
    } else {
      // drop the stats window so the pause does not show as a huge frame
      frameTimes.length = 0;
      frameTimeSum = 0;
    }
  };
  document.addEventListener('visibilitychange', onVisibility);

  // --- build / rebuild --------------------------------------------------------------
  // material constants can be overridden from the URL for experiments (?E=30&thetaS=0.02)
  const materialOverride: { -readonly [K in keyof MaterialConstants]: number } = { ...DEFAULT_MATERIAL };
  for (const k of ['E', 'nu', 'thetaC', 'thetaS'] as const) {
    const v = parseFloat(query.get(k) ?? '');
    if (Number.isFinite(v) && v > 0) materialOverride[k] = v;
  }
  const material: MaterialConstants = materialOverride;
  const tackQuery = parseFloat(query.get('tack') ?? '');
  const tackCells = Number.isFinite(tackQuery) && tackQuery > 0 ? tackQuery : undefined;
  const buildConfig = (p: QualityPreset, params: MillParams, b: number = batch): MillConfig => ({
    quality: QUALITY_PRESETS[p],
    material,
    params: { ...params, ...(tackCells !== undefined ? { tackCells } : {}) },
    batch: b
  });

  /** Rebuild the sim at a preset and batch size (both re-seed the bank). */
  async function rebuildSim(p: QualityPreset, b: number): Promise<void> {
    if (!sim || !renderer || rebuilding || (p === preset && b === batch)) { panel.setQuality(preset); panel.setBatch(batch); return; }
    rebuilding = true;
    panel.setQualityEnabled(false);
    const old = sim;
    hud.setText(`rebuilding at ${p}, batch ${b}× (~${estimateParticleCount(QUALITY_PRESETS[p], b).toLocaleString()} particles)…`);
    await nextPaint();
    try {
      const next = opts.makeSim(device, buildConfig(p, old.params, b));
      try {
        await next.ready;
      } catch (e) {
        next.destroy();
        throw e;
      }
      next.paused = old.paused;
      renderer.setVolumes(next.volumes);
      sim = next;
      preset = p;
      batch = b;
      old.destroy();
      frameTimes.length = 0;
      frameTimeSum = 0;
      hud.setError(null);
      runFrame(1 / 60);
      await device.queue.onSubmittedWorkDone();
    } catch (e) {
      reportError(`Could not switch to ${p} / batch ${b}×`, e);
      if (sim === old) { try { renderer.setVolumes(old.volumes); } catch { /* keep going */ } }
    } finally {
      rebuilding = false;
      panel.setQualityEnabled(true);
      panel.setQuality(preset);
      panel.setBatch(batch);
      panel.setParams(sim.params);
      paintStats(true);
    }
  }
  const setQuality = (p: QualityPreset): Promise<void> => rebuildSim(p, batch);

  // --- construct sim + renderer -------------------------------------------------------
  overlay.setProgress(`Building the ${preset} simulation (${chosen.reason}, ~${estimateParticleCount(QUALITY_PRESETS[preset]).toLocaleString()} particles)…`);
  hud.setText(`${preset} · ${gridDims(QUALITY_PRESETS[preset]).nx}³-ish grid · building…`);
  await nextPaint();
  try {
    sim = opts.makeSim(device, buildConfig(preset, DEFAULT_PARAMS));
    await sim.ready;
    sim.paused = startPaused;
  } catch (e) {
    reportError('Could not build the simulation', e);
    overlay.showError('Could not start the simulation', errorMessage(e));
    throw e;
  }
  overlay.setProgress('Compiling the renderer…');
  await nextPaint();
  try {
    renderer = opts.makeRenderer(ctx, canvas);
    if (offscreen) renderer.setPresentation?.(false);
    const isoQuery = parseFloat(query.get('iso') ?? '');
    if (Number.isFinite(isoQuery) && isoQuery > 0.05 && isoQuery < 1) renderer.iso = isoQuery;   // debug: iso-surface threshold
    renderer.setVolumes(sim.volumes);
  } catch (e) {
    reportError('Could not build the renderer', e);
    overlay.showError('Could not start the renderer', errorMessage(e));
    throw e;
  }
  const stopAutoOrbit = (): void => {
    if (autoOrbit) { autoOrbit = false; panel.setAutoOrbit(false); }
  };
  if (opts.attachCameraControls) {
    detachControls = opts.attachCameraControls(canvas, renderer.camera, stopAutoOrbit);
  } else {
    orbit = installOrbitControls(canvas, renderer.camera, () => renderer?.resetCamera(), { onInteract: stopAutoOrbit });
  }
  panel.setParams(sim.params);
  palette.setPaused(sim.paused);

  // --- debug api ------------------------------------------------------------------------
  const api: DebugApi = {
    get renderer() { return renderer as Renderer; },
    get sim(): GpuMpmSim { return sim as GpuMpmSim; },
    async stepFrames(n: number): Promise<void> {
      stepping = true;
      try {
        for (let i = 0; i < n; i++) runFrame(1 / 60, true);
        await device.queue.onSubmittedWorkDone();
        paintStats(true);
      } finally {
        stepping = false;
      }
    },
    snapshot(): Promise<ParticleSnapshot> { return (sim as GpuMpmSim).readParticles(); },
    stats(): SimStats { return (sim as GpuMpmSim).stats; },
    tapPigment,
    setDropSlot,
    get dropSlot(): number { return dropSlot; },
    setQuality,
    async screenshot() {
      const r = renderer as Renderer;
      if (!r.renderToPixels) throw new Error('renderer has no offscreen readback');
      const st = (sim as GpuMpmSim).stats;
      return r.renderToPixels({
        params: (sim as GpuMpmSim).params,
        rollerAngleFront: st.rollerAngleFront,
        rollerAngleBack: st.rollerAngleBack,
        timeSeconds: (performance.now() - startTime) / 1000
      });
    },
    get ready(): boolean { return ready; }
  };
  window.__colormill = api;

  // --- first frame, then go -----------------------------------------------------------
  overlay.setProgress('First frame…');
  startTime = performance.now();
  try {
    runFrame(1 / 60);
    await device.queue.onSubmittedWorkDone();
  } catch (e) {
    reportError('First frame failed', e);
  }
  overlay.hide();
  ready = true;
  paintStats(true);
  hud.showHint('Tap a colour to drop pigmented putty on the bank (the strip picks where along the roll) · Cut & fold / Cut & roll mix across the width · drag to orbit');
  canvas.focus({ preventScroll: true });
  rafId = requestAnimationFrame(frame);

  const handle: AppHandle = {
    get sim() { return sim as GpuMpmSim; },
    get renderer() { return renderer as Renderer; },
    ctx,
    get preset() { return preset; },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      ready = false;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
      document.removeEventListener('visibilitychange', onVisibility);
      removeKeys();
      orbit?.destroy();
      detachControls?.();
      if (window.__colormill === api) delete window.__colormill;
      try { renderer?.destroy(); } catch (e) { console.error(e); }
      try { sim?.destroy(); } catch (e) { console.error(e); }
      palette.destroy();
      panel.destroy();
      hud.destroy();
      overlay.destroy();
      canvas.remove();
      root.classList.remove('cm-app');
    }
  };
  return handle;
}
