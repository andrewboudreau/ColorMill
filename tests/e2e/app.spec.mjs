/**
 * End-to-end test of the real app (index.html -> src/main.ts) driven through
 * the DebugApi the app exposes on `window.__colormill` (src/sim/types.ts).
 *
 * Runs at `?preset=low` in headless Chromium + SwiftShader, so frame counts
 * are kept small and every wait is bounded by a timeout. Checks:
 *   1. the app initialises (`__colormill.ready`) and honours ?preset=low
 *   2. after N frames the particle state has no NaN/Inf, every particle is
 *      inside the domain and outside both rollers
 *   3. reset() keeps the particle count
 *   4. tapping two pigments then stepping advances simTime and changes latents
 *   5. a screenshot (tests/e2e/out/app.png) shows something other than the
 *      background at the centre of the canvas
 */
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, assert, withBrowser, encodePng } from './lib.mjs';

const OUT_DIR = path.join(ROOT, 'tests', 'e2e', 'out');
const SCREENSHOT = path.join(OUT_DIR, 'app.png');

/** Fixed mill geometry — mirrors GEOMETRY in src/config/mill.ts (e2e is plain Node, no TS import). */
const DOMAIN = [1.5, 2.25, 1.5];
const ROLLER_R = 0.32;
const AXIS_Y = 0.55;
const NIP_Z = 0.75;

const FRAMES_A = 30; // frames before the first state check
const FRAMES_B = 30; // frames after pigment injection
const FRAMES_AFTER_RESET = 5;

const NOT_INTEGRATED_GRACE_MS = 20_000; // window.__colormill must appear within this
const READY_TIMEOUT_MS = 120_000; // ...and become ready within this (pipeline compile on SwiftShader is slow)
const STEP_TIMEOUT_MS = 240_000; // a stepFrames() call
const EVAL_TIMEOUT_MS = 180_000; // any other page.evaluate (SwiftShader on a 2-core runner is slow)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function withTimeout(promise, ms, label) {
  let t;
  const timeout = new Promise((_, reject) => {
    t = setTimeout(() => reject(new Error(`timeout after ${ms / 1000}s: ${label}`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(t));
}


/** Mean RGB (0..1) over a rectangle of the decoded image. */
function meanRgb(img, x0, y0, x1, y1) {
  const acc = [0, 0, 0];
  let n = 0;
  for (let y = Math.max(0, y0 | 0); y < Math.min(img.height, y1 | 0); y++) {
    for (let x = Math.max(0, x0 | 0); x < Math.min(img.width, x1 | 0); x++) {
      const p = img.px(x, y);
      acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2];
      n++;
    }
  }
  if (!n) throw new Error('empty sample rectangle');
  return acc.map((v) => v / n / 255);
}

const luma = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const rgbDist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** Wait until the DebugApi is present and ready; fail with a targeted message otherwise. */
async function waitForApp(page) {
  const t0 = Date.now();
  let state = { present: false, ready: false, text: '' };
  while (Date.now() - t0 < READY_TIMEOUT_MS) {
    state = await withTimeout(page.evaluate(() => {
      const api = window.__colormill;
      const text = (document.body?.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 300);
      if (!api) return { present: false, ready: false, text };
      return { present: true, ready: api.ready === true, text };
    }), EVAL_TIMEOUT_MS, 'probe window.__colormill');
    if (state.present && state.ready) return;
    if (!state.present && Date.now() - t0 > NOT_INTEGRATED_GRACE_MS) break;
    await sleep(250);
  }
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  if (!state.present) {
    throw new Error(
      `app not integrated: window.__colormill (DebugApi, see src/sim/types.ts) was not exposed by the page ` +
      `within ${elapsed}s. Either src/main.ts does not attach the debug API yet, or the app threw before doing so ` +
      `(check any [pageerror] lines above). Page text: "${state.text}"`
    );
  }
  throw new Error(
    `app exposed window.__colormill but never became ready within ${elapsed}s ` +
    `(WebGPU init or sim build failed?). Page text: "${state.text}"`
  );
}

/** Snapshot the particle state in-page and reduce it to a small summary. */
async function analyseParticles(page) {
  return withTimeout(page.evaluate(async ({ DOMAIN, ROLLER_R, AXIS_Y, NIP_Z }) => {
    const api = window.__colormill;
    const s = await api.snapshot();
    const gap = api.sim.params.gap;
    const h = api.sim.dims.h;
    const half = ROLLER_R + gap / 2;
    const backZ = NIP_Z - half, frontZ = NIP_Z + half;
    const n = s.count;
    const r = {
      count: n,
      posLen: s.positions.length, velLen: s.velocities.length, latLen: s.latents.length, defLen: s.deformation.length, flagLen: s.flags.length,
      nonFinitePos: 0, nonFiniteVel: 0, nonFiniteLat: 0, nonFiniteF: 0,
      outsideDomain: 0, insideRoller: 0, minRollerDist: Infinity, kinematic: 0,
      maxSpeed: 0, latentSpread: 0, pigmented: 0,
      bbox: [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]
    };
    const inRollerTol = ROLLER_R - 0.5 * h; // deeper than half a cell inside a roller counts as "inside"
    for (let i = 0; i < n; i++) {
      const x = s.positions[3 * i], y = s.positions[3 * i + 1], z = s.positions[3 * i + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { r.nonFinitePos++; continue; }
      r.bbox[0] = Math.min(r.bbox[0], x); r.bbox[1] = Math.min(r.bbox[1], y); r.bbox[2] = Math.min(r.bbox[2], z);
      r.bbox[3] = Math.max(r.bbox[3], x); r.bbox[4] = Math.max(r.bbox[4], y); r.bbox[5] = Math.max(r.bbox[5], z);
      if (x < 0 || x > DOMAIN[0] || y < 0 || y > DOMAIN[1] || z < 0 || z > DOMAIN[2]) r.outsideDomain++;
      const db = Math.hypot(y - AXIS_Y, z - backZ);
      const df = Math.hypot(y - AXIS_Y, z - frontZ);
      const d = Math.min(db, df);
      if (d < r.minRollerDist) r.minRollerDist = d;
      if (d < inRollerTol) r.insideRoller++;
    }
    for (let i = 0; i < 3 * n; i += 3) {
      const vx = s.velocities[i], vy = s.velocities[i + 1], vz = s.velocities[i + 2];
      if (!Number.isFinite(vx) || !Number.isFinite(vy) || !Number.isFinite(vz)) { r.nonFiniteVel++; continue; }
      r.maxSpeed = Math.max(r.maxSpeed, Math.hypot(vx, vy, vz));
    }
    for (let i = 0; i < 9 * n; i++) if (!Number.isFinite(s.deformation[i])) { r.nonFiniteF++; }
    // latent statistics. The "base" latent is a trimmed mean (particles with
    // below-median deviation from the plain mean), so a few pigmented blobs do
    // not drag the reference away from the untouched white bank.
    const devOf = (i, ref) => {
      let dev = 0;
      for (let c = 0; c < 7; c++) {
        const v = s.latents[7 * i + c];
        if (Number.isFinite(v)) dev = Math.max(dev, Math.abs(v - ref[c]));
      }
      return dev;
    };
    const mean = new Float64Array(7);
    for (let i = 0; i < n; i++) for (let c = 0; c < 7; c++) {
      const v = s.latents[7 * i + c];
      if (!Number.isFinite(v)) { r.nonFiniteLat++; continue; }
      mean[c] += v;
    }
    for (let c = 0; c < 7; c++) mean[c] /= Math.max(1, n);
    const devs = new Float32Array(n);
    for (let i = 0; i < n; i++) devs[i] = devOf(i, mean);
    const median = n ? Float32Array.from(devs).sort()[n >> 1] : 0;
    const base = new Float64Array(7);
    let nb = 0;
    for (let i = 0; i < n; i++) {
      if (devs[i] > median) continue;
      for (let c = 0; c < 7; c++) base[c] += s.latents[7 * i + c];
      nb++;
    }
    for (let c = 0; c < 7; c++) base[c] /= Math.max(1, nb);
    for (let i = 0; i < n; i++) {
      const dev = devOf(i, base);
      r.latentSpread = Math.max(r.latentSpread, dev);
      if (dev > 0.02) r.pigmented++;
    }
    for (let i = 0; i < n; i++) if (s.flags[i] & 1) r.kinematic++;
    r.minRollerDist = Number.isFinite(r.minRollerDist) ? r.minRollerDist : -1;
    return r;
  }, { DOMAIN, ROLLER_R, AXIS_Y, NIP_Z }), EVAL_TIMEOUT_MS, 'snapshot() + analysis');
}

function assertStateSane(a, label) {
  assert(a.count > 0, `${label}: particle count > 0 (got ${a.count})`);
  assert(a.posLen >= 3 * a.count && a.velLen >= 3 * a.count && a.latLen >= 7 * a.count && a.defLen >= 9 * a.count && a.flagLen >= a.count,
    `${label}: snapshot arrays sized for ${a.count} particles (pos ${a.posLen}, vel ${a.velLen}, lat ${a.latLen}, F ${a.defLen}, flags ${a.flagLen})`);
  assert(a.nonFinitePos === 0, `${label}: no NaN/Inf positions (${a.nonFinitePos} bad)`);
  assert(a.nonFiniteVel === 0, `${label}: no NaN/Inf velocities (${a.nonFiniteVel} bad)`);
  assert(a.nonFiniteLat === 0, `${label}: no NaN/Inf latents (${a.nonFiniteLat} bad)`);
  assert(a.nonFiniteF === 0, `${label}: no NaN/Inf deformation gradients (${a.nonFiniteF} bad)`);
  assert(a.outsideDomain === 0, `${label}: all particles inside the domain ${DOMAIN.join('x')} (${a.outsideDomain} outside; bbox ${a.bbox.map((v) => v.toFixed(3)).join(' ')})`);
  assert(a.insideRoller === 0, `${label}: no particle inside a roller (${a.insideRoller} inside; min axis distance ${a.minRollerDist.toFixed(4)}, R=${ROLLER_R})`);
}

async function stepFrames(page, n) {
  const t0 = Date.now();
  await withTimeout(page.evaluate((k) => window.__colormill.stepFrames(k), n), STEP_TIMEOUT_MS, `stepFrames(${n})`);
  console.log(`  stepped ${n} frames in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

async function stats(page) {
  const s = await withTimeout(page.evaluate(() => {
    const st = window.__colormill.stats();
    return { particleCount: st.particleCount, simTime: st.simTime, simSecondsPerFrame: st.simSecondsPerFrame, grid: st.grid };
  }), EVAL_TIMEOUT_MS, 'stats()');
  assert(Number.isFinite(s.simTime) && Number.isFinite(s.simSecondsPerFrame), `stats() numbers are finite (${JSON.stringify(s)})`);
  return s;
}

export default async function run() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  await withBrowser(async ({ page, baseUrl }) => {
    // paused: the animation loop must not keep queuing sim frames between the
    // driven steps, or readbacks (snapshot/screenshot) starve behind them on a
    // software GPU. stepFrames() forces steps while paused.
    await page.goto(`${baseUrl}?preset=low&paused=1`, { waitUntil: 'load', timeout: 60_000 });
    await waitForApp(page);

    // --- 1. configuration honoured -------------------------------------------------
    const cfg = await withTimeout(page.evaluate(() => {
      const api = window.__colormill;
      return { preset: api.sim.quality.preset, dims: api.sim.dims, gap: api.sim.params.gap, particleCount: api.stats().particleCount };
    }), EVAL_TIMEOUT_MS, 'read sim config');
    console.log(`  preset ${cfg.preset}, grid ${cfg.dims.nx}x${cfg.dims.ny}x${cfg.dims.nz} (h=${cfg.dims.h}), ${cfg.particleCount} particles`);
    assert(cfg.preset === 'low', `?preset=low selected the low preset (got ${cfg.preset})`);
    assert(cfg.dims.nx === 49 && cfg.dims.ny === 73 && cfg.dims.nz === 65, `low preset grid is 49x73x65 nodes (got ${cfg.dims.nx}x${cfg.dims.ny}x${cfg.dims.nz})`);
    assert(cfg.particleCount > 10_000, `low preset seeds a bank (particleCount ${cfg.particleCount})`);

    // --- 2. run and check the particle state ---------------------------------------
    await stepFrames(page, FRAMES_A);
    const st1 = await stats(page);
    assert(st1.simTime > 0, `simTime advanced after ${FRAMES_A} frames (${st1.simTime})`);
    assert(st1.particleCount === cfg.particleCount, `particle count unchanged after stepping (${st1.particleCount} vs ${cfg.particleCount})`);
    const a1 = await analyseParticles(page);
    assertStateSane(a1, `after ${FRAMES_A} frames`);
    console.log(`  state ok: maxSpeed ${a1.maxSpeed.toFixed(3)}, min roller distance ${a1.minRollerDist.toFixed(4)}, kinematic ${a1.kinematic}`);

    // --- 3. reset keeps the particle count -----------------------------------------
    await withTimeout(page.evaluate(() => window.__colormill.sim.reset()), EVAL_TIMEOUT_MS, 'sim.reset()');
    await stepFrames(page, FRAMES_AFTER_RESET);
    const st2 = await stats(page);
    assert(st2.particleCount === cfg.particleCount, `particle count unchanged after reset (${st2.particleCount} vs ${cfg.particleCount})`);
    const a2 = await analyseParticles(page);
    assert(a2.count === cfg.particleCount, `snapshot count unchanged after reset (${a2.count} vs ${cfg.particleCount})`);
    assertStateSane(a2, 'after reset');

    // --- 4. pigments: tap two colours, run, check time advanced and latents changed ----
    await withTimeout(page.evaluate(() => {
      window.__colormill.tapPigment('cobaltBlue');
      window.__colormill.tapPigment('cadmiumYellow');
    }), EVAL_TIMEOUT_MS, "tapPigment('cobaltBlue') / tapPigment('cadmiumYellow')");
    await stepFrames(page, FRAMES_B);
    const st3 = await stats(page);
    const expected = FRAMES_B * st3.simSecondsPerFrame;
    assert(st3.simTime > st2.simTime, `simTime advanced after pigment injection (${st2.simTime} -> ${st3.simTime})`);
    assert(st3.simTime - st2.simTime >= 0.5 * expected, `simTime advanced by roughly ${FRAMES_B} frames (${(st3.simTime - st2.simTime).toFixed(4)}s vs expected ${expected.toFixed(4)}s)`);
    const a3 = await analyseParticles(page);
    assertStateSane(a3, `after pigment + ${FRAMES_B} frames`);
    assert(a3.latentSpread > 1e-3, `pigment injection changed particle latents (max deviation from mean ${a3.latentSpread})`);
    console.log(`  pigment ok: ${a3.pigmented} particles visibly pigmented, simTime ${st3.simTime.toFixed(3)}s`);

    // --- 5. screenshot: rendered offscreen through the debug API (canvas presentation
    // is unavailable in headless SwiftShader Chromium); the centre must be lit ---------
    const shot = await withTimeout(page.evaluate(async () => {
      const s = await window.__colormill.screenshot();
      return { width: s.width, height: s.height, data: Array.from(s.data) };
    }), EVAL_TIMEOUT_MS, 'screenshot');
    const data = Uint8Array.from(shot.data);
    const img = { width: shot.width, height: shot.height, data, px: (x, y) => { const i = (y * shot.width + x) * 4; return [data[i], data[i + 1], data[i + 2]]; } };
    fs.mkdirSync(OUT_DIR, { recursive: true });
    fs.writeFileSync(SCREENSHOT, encodePng(img.width, img.height, img.data));
    const cx = img.width / 2, cy = img.height / 2;
    const centre = meanRgb(img, cx - 6, cy - 6, cx + 6, cy + 6);
    const edge = meanRgb(img, 3, cy - 6, 9, cy + 6);
    const whole = meanRgb(img, 0, 0, img.width, img.height);
    const cl = luma(centre), dEdge = rgbDist(centre, edge);
    console.log(`  screenshot ${img.width}x${img.height} -> ${path.relative(ROOT, SCREENSHOT)}; centre rgb ${centre.map((v) => v.toFixed(3)).join(',')} (luma ${cl.toFixed(3)}), left-edge rgb ${edge.map((v) => v.toFixed(3)).join(',')}, mean luma ${luma(whole).toFixed(3)}`);
    assert(cl > 0.08, `centre of the frame is lit, not near-black (luma ${cl.toFixed(3)})`);
    assert(cl > 0.35 || dEdge > 0.08, `centre of the frame is not the background colour (luma ${cl.toFixed(3)}, distance to same-row left edge ${dEdge.toFixed(3)})`);
  });
}
