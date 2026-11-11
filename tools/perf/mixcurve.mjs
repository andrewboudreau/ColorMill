// Mixing against work on the real GPU: a fixed pigment recipe, a fixed sequence of operator moves (or none),
// and at fixed sim times the intensity of segregation at three physical box sizes, the axial segregation, how
// much of the material carries pigment, and the striation thickness. The preset-independence check of the
// lamellar mixing model (design §5): the same table should come out at every preset.
//
// usage: node tools/perf/mixcurve.mjs [low medium high ...] [--seconds=60] [--every=5] [--moves=1] [--k=1] [--omega=3] [--query=&frictionRatio=1.6]
//                                      [--shots=<dir>] [--shot-at=5,20,60] [--tag=new] [--json=<file>]
//   --moves=1 (default): cut & fold from the left at 5 s, from the right at 12 s, cut & roll at 20 s, and again
//             every 20 s. --moves=0: steady milling, nobody touches it.
// Everything is measured from the particle snapshot in this script (boxes of 1/32, 1/16 and 1/8 of a sim unit,
// whatever the grid), so it also runs against a build from before the lamina state existed: the striation
// columns are then empty.
import { mkdirSync, writeFileSync } from 'node:fs';
import { launch, openApp } from './lib.mjs';
import { encodePng } from '../../tests/e2e/lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const a = args.find((x) => x.startsWith(`--${k}=`)); return a ? a.slice(k.length + 3) : d; };
const presets = args.filter((a) => !a.startsWith('--'));
if (!presets.length) presets.push('low', 'medium', 'high');
const SECONDS = Number(opt('seconds', 60)), EVERY = Number(opt('every', 5)), MOVES = opt('moves', '1') !== '0';
const K = Number(opt('k', NaN)), SHOTS = opt('shots', ''), TAG = opt('tag', 'new'), JSON_OUT = opt('json', '');
const OMEGA = Number(opt('omega', NaN));
const SHOT_AT = opt('shot-at', '5,20,60').split(',').map(Number);
const BOXES = [1 / 32, 1 / 16, 1 / 8];
/** the moves, by sim second within each 20 s cycle */
const CYCLE = 20;
const MOVE_AT = [[5, 'flopLeft'], [12, 'flopRight'], [20, 'roll']];

/** Runs in the page: the measures of one snapshot. */
async function measure({ BOXES }) {
  const api = window.__colormill;
  const snap = await api.snapshot();
  const n = snap.count, FULL = 12;
  const conc = new Float64Array(n);
  let total = 0, load = 0, pigmented = 0, tinted = 0;
  for (let p = 0; p < n; p++) {
    const w = snap.loads[p];
    const c = Math.min(Math.max(w / FULL, 0), 1);
    conc[p] = Number.isFinite(c) ? c : 0;
    total += conc[p];
    load += w;
    if (w > 0) pigmented++;
    if (w > 0.3) tinted++;
  }
  const mean = total / n, sigma0 = mean * (1 - mean);
  const variance = (keyOf) => {
    const bins = new Map();
    for (let p = 0; p < n; p++) {
      const key = keyOf(p);
      const b = bins.get(key);
      if (b) { b[0] += conc[p]; b[1]++; } else bins.set(key, [conc[p], 1]);
    }
    let acc = 0;
    for (const [sum, count] of bins.values()) { const d = sum / count - mean; acc += count * d * d; }
    return sigma0 > 1e-12 ? acc / n / sigma0 : 0;
  };
  const X = snap.positions;
  const I = BOXES.map((box) => variance((p) => `${Math.floor(X[3 * p] / box)},${Math.floor(X[3 * p + 1] / box)},${Math.floor(X[3 * p + 2] / box)}`));
  const axial = variance((p) => Math.floor(X[3 * p] / BOXES[0]));
  // striation thickness, load-weighted, over the pigment (the lamina state; absent in the old model)
  const h = api.stats().grid.h, spacing = 0.5 * h;
  let stri = null;
  if (snap.laminae && snap.mixState) {
    const items = [];
    let wsum = 0, below = 0, thin = 0;
    for (let p = 0; p < n; p++) {
      const w = snap.loads[p];
      if (!(w > 0)) continue;
      const q = snap.laminae[4 * p + 3];
      const s = Math.exp(q + snap.mixState[2 * p]);
      if (!Number.isFinite(s)) continue;
      items.push([s, w]);
      wsum += w;
      thin += -q * w;
      if (s < spacing) below += w;
    }
    items.sort((a, b) => a[0] - b[0]);
    const quant = (f) => { let acc = 0; for (const [s, w] of items) { acc += w; if (acc >= f * wsum) return s; } return NaN; };
    stri = { p10: quant(0.1), median: quant(0.5), p90: quant(0.9), below: wsum > 0 ? below / wsum : 0, meanLogThinning: wsum > 0 ? thin / wsum : 0 };
  }
  return { t: api.stats().simTime, particles: n, load, pigmented: pigmented / n, tinted: tinted / n, I, axial, stri, spacing };
}

const ctx = await launch();
const out = {};
try {
  if (SHOTS) mkdirSync(SHOTS, { recursive: true });
  for (const preset of presets) {
    const page = await openApp(ctx.browser, ctx.server.baseUrl, preset, { width: 1280, height: 800, query: opt('query', '') });
    const info = await page.evaluate(({ K, OMEGA }) => {
      const api = window.__colormill;
      if (Number.isFinite(OMEGA)) api.sim.params.omega = OMEGA;
      if (Number.isFinite(K)) api.sim.params.dispersion = K;
      // the chunks' particle jitter comes from Math.random: the same recipe every run
      let seed = 12345;
      Math.random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
      return { spf: api.stats().simSecondsPerFrame, k: api.sim.params.dispersion, omega: api.sim.params.omega };
    }, { K, OMEGA });
    const frames = (seconds) => Math.round(seconds / info.spf);
    let frame = 0;
    const stepTo = async (seconds) => {
      const target = frames(seconds);
      while (frame < target) { const n = Math.min(100, target - frame); await page.evaluate((n) => window.__colormill.stepFrames(n), n); frame += n; }
    };
    const shoot = async (name) => {
      const shot = await page.evaluate(async () => {
        const s = await window.__colormill.screenshot();
        let bin = '';
        for (let i = 0; i < s.data.length; i += 0x8000) bin += String.fromCharCode.apply(null, Array.from(s.data.subarray(i, i + 0x8000)));
        return { width: s.width, height: s.height, base64: btoa(bin) };
      });
      const file = `${SHOTS}/${name}.png`;
      writeFileSync(file, encodePng(shot.width, shot.height, new Uint8Array(Buffer.from(shot.base64, 'base64'))));
    };
    // settle, then the recipe; t = 0 is the drop
    await page.evaluate(() => window.__colormill.stepFrames(30));
    await page.evaluate(() => window.__colormill.applyDrops([
      { pigment: 'naphtholRed', slot: 2, size: 'm' }, { pigment: 'phthaloBlue', slot: 4, size: 'l' }, { pigment: 'hansaYellow', slot: 5, size: 'm' }]));
    const rows = [await page.evaluate(measure, { BOXES })];
    const t0 = rows[0].t;
    // the events in order: measurements, moves, stills
    const events = [];
    for (let t = EVERY; t <= SECONDS + 1e-9; t += EVERY) events.push([t, 'measure']);
    if (MOVES) for (let c = 0; c * CYCLE < SECONDS; c++) for (const [at, what] of MOVE_AT) if (c * CYCLE + at < SECONDS) events.push([c * CYCLE + at, what]);
    if (SHOTS) for (const t of SHOT_AT) if (t <= SECONDS) events.push([t, 'shot']);
    const order = { measure: 0, shot: 1, flopLeft: 2, flopRight: 2, roll: 2 };
    events.sort((a, b) => a[0] - b[0] || order[a[1]] - order[b[1]]);
    for (const [t, what] of events) {
      await stepTo(t);
      if (what === 'measure') rows.push(await page.evaluate(measure, { BOXES }));
      else if (what === 'shot') await shoot(`${TAG}-${preset}-${MOVES ? 'moves' : 'steady'}-${t}s`);
      else await page.evaluate((what) => {
        const sim = window.__colormill.sim;
        if (what === 'roll') sim.cutAndFold(); else sim.cutAndFlop(what === 'flopLeft' ? 'left' : 'right');
      }, what);
    }
    out[preset] = { info, rows };
    const f = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '   -');
    const e = (v) => (Number.isFinite(v) ? v.toExponential(1) : '   -');
    console.log(`\n${TAG} ${preset}: ${MOVES ? 'with the operator moves' : 'steady milling, no moves'}; k=${info.k}; ${rows[0].particles} particles, spacing ${rows[0].spacing.toFixed(4)}; load ${rows[0].load.toFixed(1)}`);
    console.log('   t(s)  revs  I(1/32) I(1/16)  I(1/8)   axial  pigmented tinted | s p10     median  p90     s<spacing  ln(s0/s) | load drift');
    for (const r of rows) {
      const t = r.t - t0;
      const s = r.stri;
      console.log(`  ${f(t, 1).padStart(5)} ${f(t * info.omega / (2 * Math.PI), 1).padStart(5)}  ${r.I.map((v) => f(v)).join('   ')}   ${f(r.axial)}   ${f(r.pigmented)}    ${f(r.tinted)} | ` +
        (s ? `${e(s.p10)} ${e(s.median)} ${e(s.p90)} ${f(s.below).padStart(7)}   ${f(s.meanLogThinning, 2).padStart(6)}` : '      -       -       -       -        -') +
        ` | ${((r.load - rows[0].load) / rows[0].load).toExponential(1)}`);
    }
    await page.close();
  }
} finally {
  await ctx.close();
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(out));
