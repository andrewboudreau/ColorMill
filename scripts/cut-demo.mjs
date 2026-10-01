/**
 * Headless check of the drawn cut (GpuMpmSim.cutAlong, src/sim/shaders/cut.wgsl): mill a few
 * seconds at `low` with pigment dropped, draw a cut across the sheet on the front roll through the
 * same screen -> sheet projection the knife gesture uses (__colormill.cutScreen), step on, and
 * save frames plus particle statistics that show whether the sheet has a gap along the cut.
 *
 *   node scripts/cut-demo.mjs [outDir]          (default tests/e2e/out/cut)
 *
 * Env: CUT_MILL_SECONDS (3), CUT_KERF (kerf half-width in cells, default the solver's),
 *      CUT_S (target arc of the cut's middle, default 0.45: the front face at about axis height),
 *      CUT_NONE=1 (control: everything the same but no cut is drawn),
 *      CUT_DROPS (the ?drops= list), CUT_CLOSE=0 (skip the close-up frames),
 *      CUT_TEAR_SECONDS (how long the two sides stay separate bodies; 0 opens the kerf with no tear),
 *      CUT_TIMES (comma-separated checkpoints in seconds after the cut).
 */
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, encodePng, withBrowser } from '../tests/e2e/lib.mjs';

const OUT = path.resolve(process.argv[2] || path.join(ROOT, 'tests', 'e2e', 'out', 'cut'));
fs.mkdirSync(OUT, { recursive: true });
const MILL_SECONDS = Number(process.env.CUT_MILL_SECONDS || 3);
const KERF = process.env.CUT_KERF !== undefined ? Number(process.env.CUT_KERF) : undefined;
const TEAR = process.env.CUT_TEAR_SECONDS !== undefined ? Number(process.env.CUT_TEAR_SECONDS) : undefined;
const cutOpts = { ...(KERF !== undefined ? { kerfCells: KERF } : {}), ...(TEAR !== undefined ? { seconds: TEAR } : {}) };
const S_MID = Number(process.env.CUT_S || 0.45);
const NO_CUT = process.env.CUT_NONE === '1';
const DROPS = process.env.CUT_DROPS || 'phthaloBlue@2.m,hansaYellow@4.m,naphtholRed@5.m';
const CLOSE = process.env.CUT_CLOSE !== '0';
const SEC_PER_FRAME = 1.6e-3 * 8;   // low preset: dt * substeps
const framesFor = (sec) => Math.max(1, Math.round(sec / SEC_PER_FRAME));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

async function shot(page, name) {
  const img = await page.evaluate(async () => {
    const s = await window.__colormill.screenshot();
    return { width: s.width, height: s.height, data: Array.from(s.data) };
  });
  const file = path.join(OUT, `${name}.png`);
  fs.writeFileSync(file, encodePng(img.width, img.height, Uint8Array.from(img.data)));
  log('saved', file);
  return file;
}

/** Camera views: the default front view and a close-up of the front face. */
async function setView(page, view) {
  await page.evaluate((v) => {
    const c = window.__colormill.renderer.camera;
    if (v === 'close') { c.yaw = 0.0; c.pitch = 0.3; c.distance = 1.35; c.target = [0.75, 0.5, 1.15]; c.fovY = 0.7; }
    else { c.yaw = 0.0; c.pitch = 0.62; c.distance = 2.9; c.target = [0.75, 0.67, 0.75]; c.fovY = (40 * Math.PI) / 180; }
  }, view);
}

/**
 * Statistics of the sheet across the cut, in the material's coordinates at the moment of the cut:
 * on the front roll the sheet rides rigidly at the roll's speed, so s at the cut = s now + omega R dt.
 * Histogram of signed distance (in cells) to the cut polyline for sheet particles (dr < depth) whose
 * foot lies inside the cut, away from its tapered ends; plus side-label counts.
 */
async function stats(page, poly, dtSinceCut) {
  return page.evaluate(async ([poly, dt]) => {
    const api = window.__colormill;
    const sim = api.sim;
    const snap = await api.snapshot();
    const R = 0.32;
    const gap = sim.params.gap;
    const h = sim.dims.h;
    const omega = sim.params.omega;
    const axisY = 0.55;
    const axisZ = 0.75 + R + gap / 2;
    const depth = 3 * gap + h;
    const NB = 24;   // bins of 0.5 cell from -6 to +6 cells
    const hist = new Array(NB).fill(0);
    let total = 0, side1 = 0, side2 = 0, onSheet = 0;
    // sheet profile around the front roll, as it is NOW: particles and mean depth per 0.05 of arc s
    const PS = 0.05, P0 = -0.3, PN = 30;
    const prof = new Array(PN).fill(0), profDr = new Array(PN).fill(0);
    // polyline length for the end taper
    const cum = [0];
    for (let i = 1; i < poly.length; i++) cum.push(cum[i - 1] + Math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]));
    const L = cum[cum.length - 1];
    for (let p = 0; p < snap.count; p++) {
      const f = snap.flags[p];
      const sd = (f >> 2) & 3;
      if (sd === 1) side1++; else if (sd === 2) side2++;
      const x = snap.positions[3 * p], y = snap.positions[3 * p + 1], z = snap.positions[3 * p + 2];
      const dy = y - axisY, dz = z - axisZ;
      const dr = Math.hypot(dy, dz) - R;
      if (dr < -0.5 * h || dr > depth) continue;
      const sNow = R * Math.atan2(dz, dy);
      const pb = Math.floor((sNow - P0) / PS);
      if (pb >= 0 && pb < PN && x > 0.1 && x < 1.4) { prof[pb]++; profDr[pb] += dr; }
      const s = sNow + omega * R * dt;
      onSheet++;
      let best = null;
      for (let i = 0; i + 1 < poly.length; i++) {
        const ex = poly[i + 1][0] - poly[i][0], es = poly[i + 1][1] - poly[i][1];
        const len2 = ex * ex + es * es;
        if (len2 < 1e-12) continue;
        const t = ((x - poly[i][0]) * ex + (s - poly[i][1]) * es) / len2;
        if (t < 0 || t > 1) continue;
        const len = Math.sqrt(len2);
        const d = ((x - poly[i][0]) * -es + (s - poly[i][1]) * ex) / len;
        if (!best || Math.abs(d) < Math.abs(best.d)) best = { d, along: cum[i] + t * len };
      }
      if (!best || best.along < 0.15 || best.along > L - 0.15) continue;
      const b = Math.floor((best.d / h + 6) * 2);
      if (b >= 0 && b < NB) { hist[b]++; total++; }
    }
    const profile = prof.map((n, i) => ({ s: +(P0 + (i + 0.5) * PS).toFixed(3), n, dr: n ? +(profDr[i] / n).toFixed(4) : 0 }));
    return { hist, total, side1, side2, onSheet, count: snap.count, simTime: sim.stats.simTime, cutActive: sim.cutActive, h, profile };
  }, [poly, dtSinceCut]);
}

function printHist(label, st) {
  const cells = st.hist.map((n, i) => `${((i / 2) - 6).toFixed(1).padStart(5)}:${String(n).padStart(4)}`);
  log(`${label}: t=${st.simTime.toFixed(2)} s, sheet particles ${st.onSheet}, side labels ${st.side1}/${st.side2}, cut live ${st.cutActive}`);
  log('  distance to the cut (cells) : particles');
  for (let i = 0; i < cells.length; i += 6) log('  ' + cells.slice(i, i + 6).join('  '));
  log('  sheet profile (s: particles per 0.05 of arc over x 0.1..1.4, mean depth):');
  for (let i = 0; i < st.profile.length; i += 6) log('  ' + st.profile.slice(i, i + 6).map((b) => `${b.s.toFixed(2).padStart(5)}:${String(b.n).padStart(5)}/${b.dr.toFixed(3)}`).join('  '));
  const inner = st.hist.slice(10, 14).reduce((a, b) => a + b, 0);   // |d| < 1 cell
  const outer = st.hist.slice(0, 4).reduce((a, b) => a + b, 0) + st.hist.slice(20, 24).reduce((a, b) => a + b, 0);   // 4..6 cells out
  log(`  |d| < 1 cell: ${inner} particles; 4-6 cells out (same total width): ${(outer / 2).toFixed(0)}  -> ratio ${(inner / Math.max(outer / 2, 1)).toFixed(2)}`);
  return { inner, outer: outer / 2, hist: st.hist, profile: st.profile, side1: st.side1, side2: st.side2 };
}

await withBrowser(async ({ page, baseUrl }) => {
  page.setDefaultTimeout(900000);
  // the app's frame loop ray-marches every animation frame even while paused, which on SwiftShader
  // starves everything else; once the app is up and the drops are in, stop scheduling it
  // (stepFrames and screenshot do not need it)
  await page.addInitScript(() => {
    const raf = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (cb) => (window.__noRaf ? 0 : raf(cb));
  });
  const url = `${baseUrl}index.html?preset=low&paused=1&drops=${DROPS}`;
  const nDrops = DROPS.split(',').length;
  log('open', url);
  await page.goto(url);
  await page.waitForFunction(() => window.__colormill && window.__colormill.ready, null, { timeout: 180000 });
  // the linked drops are set down asynchronously once the first frame is up
  await page.waitForFunction((n) => window.__colormill.dropLog.length >= n, nDrops, { timeout: 120000 });
  // wait until every chunk is in (the particle count stops growing)
  await page.evaluate(async (nDrops) => {
    // each chunk raises the particle count once; wait for all of them (or give up after 90 s)
    let last = window.__colormill.stats().particleCount, rises = 0;
    const seed = window.__colormill.sim.stats.particleCapacity / 1.5;
    if (last > seed + 1) rises = Math.round((last - seed) / 1000);
    const t0 = Date.now();
    while (rises < nDrops && Date.now() - t0 < 90000) {
      await new Promise((r) => setTimeout(r, 250));
      const n = window.__colormill.stats().particleCount;
      if (n !== last) { rises++; last = n; }
    }
    window.__noRaf = true;
  }, nDrops);
  log('particles', await page.evaluate(() => window.__colormill.stats().particleCount));
  log('ready; milling', MILL_SECONDS, 's');
  const t0 = Date.now();
  await page.evaluate(async (n) => { await window.__colormill.stepFrames(n); }, framesFor(MILL_SECONDS));
  log(`milled in ${((Date.now() - t0) / 1000).toFixed(0)} s wall`);

  // the sheet at `low` is not uniform around the roll (it can have a thin or nearly bare stretch a
  // few cells long); so that the gap measured is the cut's and not one the sheet already had, step
  // on (up to ADAPT_MAX s) until the stretch of front face the cut will cross is a full sheet
  if (!process.env.CUT_S) {
    const ADAPT_MAX = 1.5;
    for (let waited = 0; ; waited += 0.05) {
      const st = await stats(page, [[0.2, 0], [1.3, 0]], 0);
      const full = st.profile.filter((b) => b.s > -0.2 && b.s < 1.0).map((b) => b.n).sort((a, b) => a - b);
      const ref = full[Math.floor(0.8 * (full.length - 1))];
      const win = st.profile.filter((b) => Math.abs(b.s - S_MID) <= 0.13).map((b) => b.n);
      const minWin = Math.min(...win);
      log(`placement: window around s=${S_MID} min ${minWin} per bin vs a full sheet ~${ref} (waited ${waited.toFixed(2)} s)`);
      if (minWin >= 0.75 * ref || waited >= ADAPT_MAX) break;
      await page.evaluate(async (k) => { await window.__colormill.stepFrames(k); }, framesFor(0.05));
    }
  }
  await setView(page, 'default');
  await shot(page, '00-before-default');
  if (CLOSE) { await setView(page, 'close'); await shot(page, '01-before-close'); }

  // find the screen line of the front face at arc S_MID (default view), then draw a slanted cut across it
  await setView(page, 'default');
  const line = await page.evaluate((sMid) => {
    const api = window.__colormill;
    const find = (x, target) => {
      let best = null;
      for (let y = -0.9; y <= 0.9; y += 0.0025) {
        const p = api.screenToSheet(x, y);
        if (p && (!best || Math.abs(p[1] - target) < Math.abs(best.p[1] - target))) best = { y, p };
      }
      return best;
    };
    const a = find(-0.5, sMid - 0.04);
    const b = find(0.5, sMid + 0.04);
    return { a, b };
  }, S_MID);
  log('cut from screen', JSON.stringify(line));
  const ndc = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    ndc.push([-0.5 + t, line.a.y + t * (line.b.y - line.a.y)]);
  }
  const before = await stats(page, [line.a.p, line.b.p], 0);

  // the sheet polyline the knife would draw (projected without cutting, for the control)
  const drawn = await page.evaluate((ndc) => ndc.map(([x, y]) => window.__colormill.screenToSheet(x, y)).filter(Boolean), ndc);
  const poly = NO_CUT ? drawn : await page.evaluate(([ndc, opts]) => window.__colormill.cutScreen(ndc, opts), [ndc, cutOpts]);
  log(NO_CUT ? 'CONTROL (no cut); line' : 'cut polyline (sheet coords):', poly.length, 'points, from', JSON.stringify(poly[0]), 'to', JSON.stringify(poly[poly.length - 1]));
  if (poly.length < 2) throw new Error('nothing was cut');
  const tCut = await page.evaluate(() => window.__colormill.sim.stats.simTime);
  fs.writeFileSync(path.join(OUT, 'cut-polyline.json'), JSON.stringify({ ndc, poly, tCut }, null, 1));

  const results = { before: printHist('before the cut (same line)', before) };
  const checkpoints = (process.env.CUT_TIMES || "0,0.1,0.2,0.3,0.45,0.7,1.2").split(",").map(Number);
  let tPrev = 0;
  for (const [i, t] of checkpoints.entries()) {
    const n = t === 0 ? 1 : framesFor(t - tPrev);
    await page.evaluate(async (k) => { await window.__colormill.stepFrames(k); }, n);
    tPrev = t === 0 ? SEC_PER_FRAME : t;
    const now = await page.evaluate(() => window.__colormill.sim.stats.simTime);
    const st = await stats(page, poly, now - tCut);
    results[`t+${t}`] = printHist(`after the cut, +${(now - tCut).toFixed(2)} s`, st);
    const tag = `${String(i + 2).padStart(2, '0')}-${NO_CUT ? 'nocut' : 'cut'}+${(now - tCut).toFixed(2)}s`;
    await setView(page, 'default');
    await shot(page, `${tag}-default`);
    if (CLOSE) { await setView(page, 'close'); await shot(page, `${tag}-close`); }
  }
  fs.writeFileSync(path.join(OUT, 'stats.json'), JSON.stringify(results, null, 1));
  log('done');
}, { mode: 'dev' });
